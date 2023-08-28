import os
import json
import argparse
import itertools
import math
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import torch.multiprocessing as mp
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
from scipy.io import wavfile
from datetime import datetime
import logging
import tqdm
import libs.train_utils as train_utils
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from torch import optim
import libs.commons as commons
import libs.utils as utils
from libs.str_animation import StrAnimator

from libs.data_utils import (
  TextAudioSpeakerCollate
)

from libs.models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from libs.losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from libs.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
from text import text_to_sequence

torch.backends.cudnn.benchmark = True


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def gen_preview(net_g, hps, args):
    stn_tst = get_text(hps.preview.text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([int(hps.preview.sid)])
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        now = datetime.now()
        formatted_time = now.strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.join(args.model_dir, "preview"), exist_ok=True)
        wavfile.write(os.path.join(args.model_dir, "preview", f"{formatted_time}.wav"), rate=hps.data.sampling_rate, data=audio)


def main(args, hps):
    # """Assume Single Node Multi GPUs Training Only"""
    # assert torch.cuda.is_available(), "CPU training is not allowed."

    # n_gpus = torch.cuda.device_count()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    accelerator.init_trackers("train.vits")
    accelerator.print("Initialization Accelerator.")
    # mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
    run(hps=hps, args=args, accelerator=accelerator)


def caching_fn(items):
    wav = utils.load_wav_to_torch(items[0])
    name = os.path.basename(items[0])

    return name, {'audio': wav, 
                    'spectrogram': train_utils.caching_spectrogram(items[0], hps.data)}

def run(hps, args, accelerator : Accelerator):
    global global_step

    if accelerator.is_main_process:
        logs_dir = os.path.join(args.model_dir, "logs")
        logger = utils.get_logger(logs_dir)
        # logger.info(hps)
        utils.check_git_hash(args.model_dir)
        writer = SummaryWriter(log_dir=logs_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(logs_dir, "eval"))

    # dist.init_process_group(backend='gloo', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    # torch.cuda.set_device(rank)
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

    caching_spectrogram_files = training_files 
    caching_spectrogram_files.extend(validation_files)

    cache_bucket = None
    if accelerator.is_main_process:
        if args.cache_spectrogram == True:
            
            
            cache_bucket = train_utils.CacheBucket(caching_spectrogram_files, caching_fn=caching_fn)

            if os.path.exists(os.path.join(args.model_dir, "cache_spectrogram.ckpt")):
                print("Load caching spectrogram.")
                cache_bucket.load_cache_from_file(os.path.join(args.model_dir, "cache_spectrogram.ckpt"))
            else:
                cache_bucket.caching()
                if args.cache_spectrogram_to_disk == True:
                    cache_bucket.save_to_disk(os.path.join(args.model_dir, "cache_spectrogram.ckpt"))
    
    train_dataset = train_utils.TextAudioSpeakerLoader(training_files, hps.data, cache_bucket=cache_bucket)

    # train_sampler = DistributedBucketSampler(
    #     train_dataset,
    #     hps.train.batch_size,
    #     [32,300,400,500,600,700,800,900,1000],
    #     num_replicas=n_gpus,
    #     rank=0,
    #     shuffle=True)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=2, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, batch_size=hps.train.batch_size)
    
    if accelerator.is_main_process:
        eval_dataset = train_utils.TextAudioSpeakerLoader(validation_files, hps.data, cache_bucket=cache_bucket)
        eval_loader = DataLoader(eval_dataset, num_workers=2, shuffle=False,
            batch_size=hps.train.batch_size, pin_memory=True,
            drop_last=False, collate_fn=collate_fn)

    


    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=n_speakers,
        **hps.model)
    accelerator.print("Initialization SynthesizerTrn model.")
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    accelerator.print("Initialization MultiPeriodDiscriminator model.")
    
    # optim_g = torch.optim.AdamW(
    #     net_g.parameters(), 
    #     hps.train.learning_rate, 
    #     betas=hps.train.betas, 
    #     eps=hps.train.eps)
    # optim_d = torch.optim.AdamW(
    #     net_d.parameters(),
    #     hps.train.learning_rate, 
    #     betas=hps.train.betas, 
    #     eps=hps.train.eps)

    # _, optim_class = train_utils.get_optimizer(args=args)

    optim_g = optim.AdamW(
        net_g.parameters(), 
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps
    )
    
    optim_d = optim.AdamW(
        net_d.parameters(), 
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps
    )
    
    
    # net_g = DataParallel(net_g)
    # net_d = DataParallel(net_d)
    
    try:
        _, _, _, epoch_str, fine_tune = utils.load_checkpoint(utils.latest_checkpoint_path(args.model_dir, "G_*.pth"), net_g, optim_g)
        
        accelerator.print("Load latest", utils.latest_checkpoint_path(args.model_dir, "G_*.pth"))
        _, _, _, epoch_str, fine_tune = utils.load_checkpoint(utils.latest_checkpoint_path(args.model_dir, "D_*.pth"), net_d, optim_d)
        accelerator.print("Load latest", utils.latest_checkpoint_path(args.model_dir, "D_*.pth"))
        global_step = (epoch_str - 1) * len(train_loader)
        
    except Exception as e:
        epoch_str = 1
        global_step = 0
        fine_tune = False

    if fine_tune == True or args.fine_tune == True:
        accelerator.print("Start Fine-tune...")
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    

    net_g, optim_g, train_dataset, eval_dataset, scheduler_g = accelerator.prepare(net_g, optim_g, train_dataset, eval_dataset, scheduler_g)
    net_d, optim_d, train_dataset, eval_dataset, scheduler_d = accelerator.prepare(net_d, optim_d, train_dataset, eval_dataset, scheduler_g)
    # print(optim_g.scaler)
    accelerator.print("===============================================================")
    accelerator.print("-Epochs:", hps.train.epochs)
    accelerator.print("-Batch size:", hps.train.batch_size)
    accelerator.print("-Max train steps:", f"Epochs: {hps.train.epochs} * (Datas: {len(training_files)} / Batch size: {hps.train.batch_size}) * Repeat: {args.repeat} = {(hps.train.epochs) * len(train_loader) * args.repeat} Steps")
    if not speakers is None:
        accelerator.print("-Speaker: ", ", ".join([f'{speaker[0]}: {speaker[1]}' for speaker in speakers]))
    else:
        accelerator.print(f"-Speaker: {n_speakers}")
    # accelerator.print("-Optimizer:", optim_class.__name__)
    accelerator.print("-Learning rate:", hps.train.learning_rate)
    accelerator.print("-Text cleaners:", hps.data.text_cleaners)
    accelerator.print("-Sampling rate:", hps.data.sampling_rate)
    accelerator.print("==================All looks good, ready to train===============")

    # scaler = GradScaler(enabled=hps.train.fp16_run)
    # scaler = accelerator.scaler = GradScaler(enabled=hps.train.fp16_run)
    
    
    process_bar = tqdm.tqdm(range(0, (hps.train.epochs) * len(train_loader) * args.repeat), desc='Warmup...')

    for epoch in range(1, hps.train.epochs + 1):
        
        if accelerator.is_main_process:
            train_and_evaluate(args, process_bar, accelerator, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(args,process_bar, accelerator, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], [train_loader, None], None, None)
            scheduler_g.step()
            scheduler_d.step()
            accelerator.wait_for_everyone()
        
    process_bar.close()
    train_utils.save_model(accelerator, args, (net_g, net_d), (optim_g, optim_d), hps.train.learning_rate, epoch)
    accelerator.print("Train done")
    accelerator.end_training()


def train_and_evaluate(args, process_bar : tqdm.tqdm, accelerator : Accelerator, epoch, hps, nets, optims : list[optim.Optimizer], schedulers, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step
    
    net_g.train()
    net_d.train()

    descs =  StrAnimator(["-    ", " -   ", "  -  ", "   - ", "    -"])
    
    for repeat in range(0, args.repeat):
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(train_loader):
            x, x_lengths = x.to(accelerator.device, non_blocking=True), x_lengths.to(accelerator.device, non_blocking=True)
            spec, spec_lengths = spec.to(accelerator.device, non_blocking=True), spec_lengths.to(accelerator.device, non_blocking=True)
            y, y_lengths = y.to(accelerator.device, non_blocking=True), y_lengths.to(accelerator.device, non_blocking=True)
            speakers = speakers.to(accelerator.device, non_blocking=True)

            # x, x_lengths = x, x_lengths
            # spec, spec_lengths = spec, spec_lengths
            # y, y_lengths = y, y_lengths

            # autocast()
            with accelerator.accumulate(net_d):
                with accelerator.autocast(cache_enabled=True):
                    y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
                    (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)

                    mel = spec_to_mel_torch(
                        spec, 
                        hps.data.filter_length, 
                        hps.data.n_mel_channels, 
                        hps.data.sampling_rate,
                        hps.data.mel_fmin, 
                        hps.data.mel_fmax)
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

                    y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

                    # Discriminator
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                    
                    with accelerator.autocast():
                        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                        loss_disc_all = loss_disc
                
                
                optim_d.zero_grad()

                # scaler.scale(loss_disc_all).backward()
                accelerator.backward(loss_disc_all)

                accelerator.unscale_gradients(optim_d)
                # scaler.unscale_(optim_d)
                
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)

                # grad_norm_d = accelerator.clip_grad_value_(net_d.parameters(), None)
                # accelerator.scaler.step(optim_d)
                optim_d.step()

            with accelerator.accumulate(net_g):
                with accelerator.autocast(cache_enabled=True):
                # Generator
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                    with accelerator.autocast():
                        loss_dur = torch.sum(l_length.float())
                        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                        loss_fm = feature_loss(fmap_r, fmap_g)
                        loss_gen, losses_gen = generator_loss(y_d_hat_g)
                        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

                optim_g.zero_grad()
                # scaler.scale(loss_gen_all).backward()
                # accelerator.scaler.scale(loss_gen_all).backward()
                accelerator.backward(loss_gen_all)
                # scaler.unscale_(optim_g)
                accelerator.unscale_gradients(optim_g)
                # accelerator.scaler.unscale_(optim_g)

                grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
                # grad_norm_g= accelerator.clip_grad_value_(net_g.parameters(), None)

                # scaler.step(optim_g)
                # scaler.update()
                optim_g.step()
                # accelerator.scaler.step(optim_g)
            # accelerator.scaler.update()

            if accelerator.sync_gradients:
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                if global_step % hps.train.log_interval == 0:
                    lr = optim_g.param_groups[0]['lr']
                    # logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    #   epoch,
                    #   100. * batch_idx / len(train_loader)))
                    # logger.info([x.item() for x in losses] + [global_step, lr])
                    # logger.info([x.item() for x in losses] + [global_step, lr])
                    scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "loss/avg_totals/": train_utils.mean([x.item() for x in losses]), "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}

                    scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})
                    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                    scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                    scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                    image_dict = { 
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
                        "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                        "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
                    }
                    utils.summarize(
                        writer=writer,
                        global_step=global_step, 
                        images=image_dict,
                        scalars=scalar_dict)
                
                if hps.train.eval_interval > 0:
                    if global_step % hps.train.eval_interval == 0 and global_step != 0:
                        evaluate(accelerator, hps, net_g, eval_loader, writer_eval)
                        # utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                        # utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                        train_utils.save_model(accelerator, global_step, args, (net_g, net_d), (optim_g, optim_d), hps.train.learning_rate, epoch)

                        if hps.preview.enable == True:
                            accelerator.print("Generation Preview")
                            gen_preview(net_g=net_g, hps=hps, args=args)

                if global_step % (len(train_loader) * args.repeat * hps.train.save_every_n_epochs - 1) == 0 and global_step != 0:
                    evaluate(accelerator, hps, net_g, eval_loader, writer_eval)

                    train_utils.save_model(accelerator, global_step, args, (net_g, net_d), (optim_g, optim_d), hps.train.learning_rate, epoch)

                if global_step % (len(train_loader)* args.repeat * hps.preview.preview_n_epochs - 1) == 0 and global_step != 0:
                    if hps.preview.enable == True:
                        accelerator.print("Generation Preview")
                        gen_preview(net_g=net_g, hps=hps, args=args)
                process_bar.desc = f'Epoch {epoch}/{hps.train.epochs}{descs.next()}Steps'
                process_bar.set_postfix({"avg_loss" : train_utils.mean([x.item() for x in losses])})
                process_bar.update(1)
            global_step += 1

def evaluate(accelerator : Accelerator, hps, generator : SynthesizerTrn, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
            # x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            # spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            # y, y_lengths = y.cuda(0), y_lengths.cuda(0)

            x, x_lengths = x.to(accelerator.device), x_lengths.to(accelerator.device)
            spec, spec_lengths = spec.to(accelerator.device), spec_lengths.to(accelerator.device)
            y, y_lengths = y.to(accelerator.device), y_lengths.to(accelerator.device)
            speakers = speakers.to(accelerator.device)

            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            speakers = speakers[:1]
            break
        y_hat, attn, mask, *_ = generator.infer(x, x_lengths, speakers, max_len=1000)
        # y_hat, attn, mask, *_ = generator.infer(x, x_lengths, max_len=1000)
        y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

        mel = spec_to_mel_torch(
            spec, 
            hps.data.filter_length, 
            hps.data.n_mel_channels, 
            hps.data.sampling_rate,
            hps.data.mel_fmin, 
            hps.data.mel_fmax)
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
        image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
        }
        audio_dict = {
        "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
        }
        if global_step == 0:
            image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
            audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

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

    return args, hparams

if __name__ == "__main__":
    args, hps = parser_args()
    main(args, hps)
