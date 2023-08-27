import argparse
import json
import logging
import multiprocessing
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch import optim
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import libs.commons as commons
import libs.so_vits_rvc.utils as utils
from libs.so_vits_rvc.data_utils import TextAudioCollate, TextAudioSpeakerLoader
from libs.so_vits_rvc.models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from libs.so_vits_rvc.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from libs.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from accelerate import Accelerator
from libs.str_animation import StrAnimator
import libs.train_utils as train_utils 
import accelerate
import time


logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

def main(args, hps):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = hps.train.port

    # mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.init_trackers("train.vc_vits")
    accelerator.print("Initialization Accelerator.")

    run(accelerator, args, 0, n_gpus=n_gpus, hps=hps)


def run(accelerator : Accelerator, args, rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    
    # for pytorch on win, backend use gloo    
    # dist.init_process_group(backend=  'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    accelerator.free_memory()
    accelerate.utils.set_seed(hps.train.seed)

    if args.custom_dataset != "":
        training_files, validation_files, speakers = train_utils.load_custom_dataset(args.custom_dataset)
        if speakers is None:
            n_speakers = 1
        else:
            n_speakers = len(speakers)
    else:
        training_files = train_utils.load_filepaths_and_text(hps.data.training_files)
        validation_files = train_utils.load_filepaths_and_text(hps.data.validation_files)
        n_speakers = int(hps.data.n_speakers)
        speakers = None

    # torch.cuda.set_device(rank)
    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem   # If you have enough memory, turn on this option to avoid disk IO and speed up training.
    train_dataset = TextAudioSpeakerLoader(training_files, hps, all_in_mem=all_in_mem)
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    if all_in_mem:
        num_workers = 0
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
                              batch_size=hps.train.batch_size, collate_fn=collate_fn)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(validation_files, hps, all_in_mem=all_in_mem,vol_aug = False)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=1, pin_memory=False,
                                 drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    accelerator.print("Initialization SynthesizerTrn model.")
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    accelerator.print("Initialization MultiPeriodDiscriminator model.")

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    # net_g = DDP(net_g, device_ids=[rank])  # , find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank])

    skip_optimizer = False
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g, skip_optimizer)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        name=utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        global_step=int(name[name.rfind("_")+1:name.rfind(".")])+1
        #global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    warmup_epoch = hps.train.warmup_epochs
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    

    net_g, optim_g, train_dataset, eval_dataset, scheduler_g = accelerator.prepare(net_g, optim_g, train_dataset, eval_dataset, scheduler_g)
    net_d, optim_d, train_dataset, eval_dataset, scheduler_d = accelerator.prepare(net_d, optim_d, train_dataset, eval_dataset, scheduler_g)
    
    accelerator.print("===============================================================")
    accelerator.print("-Epochs:", hps.train.epochs)
    accelerator.print("-Batch size:", hps.train.batch_size)
    accelerator.print("-Max train steps:", f"{(hps.train.epochs) * len(train_loader)} Steps")
    
    # accelerator.print("-Optimizer:", optim_class.__name__)
    accelerator.print("-Learning rate:", hps.train.learning_rate)
    accelerator.print("-Sampling rate:", hps.data.sampling_rate)
    accelerator.print("==================All looks good, ready to train===============")

    process_bar = tqdm.tqdm(range(0, (hps.train.epochs) * len(train_loader)), desc='Warmup...')
    
    for epoch in range(epoch_str, hps.train.epochs + 1):
        # set up warm-up learning rate
        if epoch <= warmup_epoch:
            for param_group in optim_g.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
            for param_group in optim_d.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
        # training
        if rank == 0:
            train_and_evaluate(process_bar, accelerator, rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d],
                               [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(process_bar, accelerator, rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d],
                               [train_loader, None], None, None)
        # update learning rate
        accelerator.wait_for_everyone()
        scheduler_g.step()
        scheduler_d.step()
    process_bar.close()
    accelerator.print("Train done")
    accelerator.end_training()

def train_and_evaluate(process_bar : tqdm.tqdm, accelerator : Accelerator, rank, epoch, hps, nets, optims : list[optim.Optimizer], schedulers, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers
    
    # half_type = torch.bfloat16 if hps.train.half_type=="bf16" else torch.float16

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    descs =  StrAnimator(["-    ", " -   ", "  -  ", "   - ", "    -"])
    for batch_idx, items in enumerate(train_loader):
        c, f0, spec, y, spk, lengths, uv, volume = items
        g = spk.to(accelerator.device, non_blocking=True)
        spec, y = spec.to(accelerator.device, non_blocking=True), y.to(accelerator.device, non_blocking=True)
        c = c.to(accelerator.device, non_blocking=True)
        f0 = f0.to(accelerator.device, non_blocking=True)
        uv = uv.to(accelerator.device, non_blocking=True)
        lengths = lengths.to(accelerator.device, non_blocking=True)
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)
        
        with accelerator.accumulate(net_d):
            with accelerator.autocast():
                y_hat, ids_slice, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(c, f0, uv, spec, g=g, c_lengths=lengths,
                                                                                    spec_lengths=lengths,vol = volume)

                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )
                y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
            
            optim_d.zero_grad()
            accelerator.backward(loss_disc_all)
            # scaler.scale(loss_disc_all).backward()

            accelerator.unscale_gradients(optim_d)
            # scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        
            optim_d.step()
        
        with accelerator.accumulate(net_g):
            with accelerator.autocast():
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                # loss_lf0 = F.mse_loss(pred_lf0, lf0) if net_g.module.use_automatic_f0_prediction else 0
                loss_lf0 = F.mse_loss(pred_lf0, lf0) if net_g.use_automatic_f0_prediction else 0
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0

        optim_g.zero_grad()

        accelerator.backward(loss_gen_all)
        accelerator.unscale_gradients(optim_g)
        # scaler.scale(loss_gen_all).backward()
        # scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        # scaler.step(optim_g)
        optim_g.step()

        if accelerator.sync_gradients:
            losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                
                # reference_loss=0
                # for i in losses:
                #     reference_loss += i
                # logger.info('Train Epoch: {} [{:.0f}%]'.format(
                #     epoch,
                #     100. * batch_idx / len(train_loader)))
                # logger.info(f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}")

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl,
                                    "loss/g/lf0": loss_lf0})

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
                }

                if net_g.use_automatic_f0_prediction:
                    image_dict.update({
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                              pred_lf0[0, 0, :].detach().cpu().numpy()),
                        "all/norm_lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                                   norm_lf0[0, 0, :].detach().cpu().numpy())
                    })

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(accelerator, hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

            process_bar.desc = f'Epoch {epoch}/{hps.train.epochs}{descs.next()}Steps'
            
            process_bar.set_postfix({"avg_loss" : train_utils.mean([x.item() for x in losses])})

        process_bar.update(1)
        global_step += 1

    # if rank == 0:
    #     global start_time
    #     now = time.time()
    #     durtaion = format(now - start_time, '.2f')
    #     logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
    #     start_time = now


def evaluate(accelerator, hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv,volume = items
            g = spk[:1].to(accelerator.device)
            spec, y = spec[:1].to(accelerator.device), y[:1].to(accelerator.device)
            c = c[:1].to(accelerator.device)
            f0 = f0[:1].to(accelerator.device)
            uv= uv[:1].to(accelerator.device)
            if volume is not None:
                volume = volume[:1].to(accelerator.device)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            # y_hat,_ = generator.module.infer(c, f0, uv, g=g,vol = volume)
            y_hat,_ = generator.infer(c, f0, uv, g=g,vol = volume)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat[0],
                f"gt/audio_{batch_idx}": y[0]
            })
        image_dict.update({
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

def parser_args():
    parser = argparse.ArgumentParser()

    parser = train_utils.train_args(parser)

    parser.add_argument('-c', '--config', default="./configs/base.json",
                        help='JSON file for configuration')
    parser.add_argument('-m', '--model_dir', required=True,
                        help='Model name')
    args = parser.parse_args()

    config_path = args.config
    config_save_path = os.path.join(args.model_dir, "config.json")

    with open(config_path, "r", encoding='utf-8') as f:
        data = f.read()
    os.makedirs(args.model_dir, exist_ok=True)
    with open(config_save_path, "w", encoding='utf-8') as f:
        f.write(data)

    config = json.loads(data)

    hparams = utils.HParams(**config)

    if not args.batch_size is None:
        hparams.train.batch_size = int(args.batch_size)

    if not args.epochs is None:
        hparams.train.epochs = int(args.epochs)

    if not args.learning_rate is None:
        hparams.train.learning_rate = int(args.learning_rate)

    if not args.save_every_n_epochs is None:
        hparams.train.save_every_n_epochs= int(args.save_every_n_epochs)
    hparams.model_dir = args.model_dir
    return args, hparams

if __name__ == "__main__":
    args, hps = parser_args()
    main(args, hps)