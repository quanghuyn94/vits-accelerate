import argparse
import hashlib
import re
from torch import optim
import torch
import ast

import tqdm
import libs.mel_processing as mel_processing
from scipy.io.wavfile import read
import numpy as np
from libs.mel_processing import spectrogram_torch
from libs.utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence
import random
import libs.commons as commons
import os
import glob
import safetensors_utils
import libs.so_vits_rvc.utils as utils
from accelerate import Accelerator


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def caching_spectrogram(filename, hparams):

    audio, sampling_rate = load_wav_to_torch(filename)
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)

    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, hparams.sampling_rate))

    spec = mel_processing.spectrogram_torch(audio_norm, hparams.filter_length,
            hparams.sampling_rate, hparams.hop_length, hparams.win_length,
            center=False)
    spec = torch.squeeze(spec, 0)

    return spec

def val_speaker_format(dir : str):
    speaker = dir.split("_")
    try:
        int(speaker[0])
        return len(speaker) > 1
    except:
        return False
    
def load_speakers(full_path : str):
    dirs = os.listdir(full_path)
    speakers = []
    for dir in dirs:
        if val_speaker_format(dir=dir):
            speaker = dir.split("_")
            speakers.append((speaker[0], speaker[1]))
    return speakers

def load_audio_caption(full_path, id = 0, etx=".txt.cleaned"):
    filepaths_and_text = []
    audio_paths = glob.glob(full_path + "/**/*.wav", recursive=True)

    for audio_path in audio_paths:
        base_path = os.path.splitext(audio_path)[0]

        with open(base_path + etx, encoding='utf-8') as f:
            caption = f.read()

        filepaths_and_text.append((audio_path, caption, id))

    return filepaths_and_text

def delete_range(input_list, start_index, length):
    if start_index < 0 or start_index >= len(input_list) or length <= 0:
        raise ValueError("Invalid start index or length")

    end_index = start_index + length - 1

    if end_index >= len(input_list):
        raise ValueError("Range to delete exceeds list length")

    deleted_items = input_list[start_index:end_index + 1]
    remaining_items = input_list[:start_index] + input_list[end_index + 1:]

    return remaining_items, deleted_items

def load_eval_dataset(full_path, id = 0, etx=".txt.cleaned"):
    return load_audio_caption(full_path=os.path.join(full_path + "/eval"), id=id, etx=etx)

def load_train_dataset(full_path, id = 0, etx=".txt.cleaned"):
    return load_audio_caption(full_path=os.path.join(full_path + "/train"), id=id, etx=etx)

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def load_custom_dataset(full_path : str, etx=".txt.cleaned"):
    training_files = []
    validation_files = []

    speakers = load_speakers(full_path)

    if len(speakers) > 0:
        print("Load speakers format.")
        for speaker in speakers:
            speaker_id, speaker_name = speaker
            speaker_validation_files = []
            target_path = f'{full_path}/{speaker_id}_{speaker_name}'
            speaker_validation_files = []
            speaker_training_files = []
            if os.path.exists(target_path + '/eval'):
                speaker_validation_files = load_eval_dataset(target_path, id=speaker_id, etx=etx)

            if os.path.exists(target_path + '/train'):
                speaker_training_files = load_train_dataset(target_path, id=speaker_id, etx=etx)

            if len(speaker_training_files) <= 0 or len(speaker_validation_files) <= 0:
                print("Train folder or Eval folder not exist. Auto create dataset.")
                audio_files = load_audio_caption(full_path=target_path, id=speaker_id, etx=etx)

                speaker_training_files, speaker_validation_files = delete_range(audio_files, random.randint(0, len(audio_files) - 101), 100)

            training_files.extend(speaker_training_files)
            validation_files.extend(speaker_validation_files)

        return training_files, validation_files, speakers
    
    if os.path.exists(full_path + '/eval'):
        validation_files = load_eval_dataset(full_path, etx=etx)

    if os.path.exists(full_path + '/train'):
        training_files = load_train_dataset(full_path, etx=etx)

    if len(training_files) <= 0 or len(validation_files) <= 0:
        print("Train folder or Eval folder not exist. Auto create dataset.")
        audio_files = load_audio_caption(full_path=full_path, etx=etx)

        training_files, validation_files = delete_range(audio_files, random.randint(0, len(audio_files) - 101), 100)
    
    return training_files, validation_files, None

def load_checkpoint(checkpoint_path : str, model, optimizer=None, fine_tune=False):
    assert os.path.isfile(checkpoint_path)

    if checkpoint_path.endswith(".safetensors"):
        tensors_dict = safetensors_utils.load_tensors(checkpoint_path)
        checkpoint_dict_state_dict = safetensors_utils.from_safetensors(tensors_dict)
        del tensors_dict
    else:
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    
    try:
        if optimizer is not None and fine_tune == False:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
    except:
        fine_tune = True
        pass

    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict= {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
        new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
        
    return model, optimizer, learning_rate, iteration, fine_tune

def save_model(accelerator : Accelerator, global_step, args, nets, optims, learning_rate, epoch):
    net_g, net_d = nets
    optim_g, optim_d = optims
    net_g_checkpoint = get_checkpoint(net_g, optim_g, learning_rate, epoch, args.pruned)
    net_d_checkpoint = get_checkpoint(net_d, optim_d, learning_rate, epoch, args.pruned)

    net_g_save_path = os.path.join(args.model_dir, "G_{}{}.pth".format(global_step, "-pruned" if args.pruned == True else ""))
    net_d_save_path = os.path.join(args.model_dir, "D_{}{}.pth".format(global_step, "-pruned" if args.pruned == True else ""))
    if args.save_safetensors == True:
        net_g_checkpoint = safetensors_utils.to_safetensors(net_g_checkpoint['state_dict'])
        net_d_checkpoint = safetensors_utils.to_safetensors(net_d_checkpoint['state_dict'])

        net_g_save_path = net_g_save_path.replace(".pth", ".safetensors")
        net_d_save_path = net_d_save_path.replace(".pth", ".safetensors")

        safetensors_utils.save_tensors(net_g, net_g_save_path)
        accelerator.print("Save checkpoint:", net_g_save_path)

        safetensors_utils.save_tensors(net_d, net_d_save_path)
        accelerator.print("Save checkpoint:", net_d_save_path)

    else:
        torch.save(net_g_checkpoint, net_g_save_path)
        accelerator.print("Save checkpoint:", net_g_save_path)

        
        torch.save(net_d_checkpoint, net_d_save_path)
        accelerator.print("Save checkpoint:", net_d_save_path)
    
    if args.keep_n_checkpoints > 0:
        clean_checkpoints(args.model_dir, int(args.keep_n_checkpoints))
        
def clean_checkpoints(path_to_models, n_ckpts_to_keep=2, sort_by_time=True, etx=".pth"):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                            False -> lexicographically delete ckpts
    """
    ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
    def name_key(_f):
        return int(re.compile(f"._(\\d+)\\{etx}").match(_f).group(1))
    def time_key(_f):
        return os.path.getmtime(os.path.join(path_to_models, _f))
    sort_key = time_key if sort_by_time else name_key
    def x_sorted(_x):
        return sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")], key=sort_key)
    to_del = [os.path.join(path_to_models, fn) for fn in
                (x_sorted('G')[:-n_ckpts_to_keep] + x_sorted('D')[:-n_ckpts_to_keep])]
    def del_routine(x):
        return [os.remove(x)]
    
    [del_routine(fn) for fn in to_del]

def train_args(parser : argparse.ArgumentParser):

    parser.add_argument("--epochs", default=None)
    parser.add_argument("--learning_rate", default=None)
    parser.add_argument("--batch_size", default=None)
    parser.add_argument("--save_every_n_epochs", default=None)
    parser.add_argument("--repeat", default=1, type=int, help="How many repeats of dataset per epoch?")

    parser.add_argument('--cache_spectrogram_to_disk', action="store_true", default=False, help='Store all spectrogram to disk.')
    parser.add_argument('--cache_spectrogram', action="store_true", default=True, help='Create cache spectrogram.')
    parser.add_argument('--pruned', action="store_true", default=False, help='Pruned you model.')
    parser.add_argument('--mixed_precision', default="fp16")
    parser.add_argument('--fine-tune', action="store_true", default=False)

    parser.add_argument('--custom_dataset', default="")

    parser.add_argument('--save_safetensors', action="store_true", default=False)
    parser.add_argument('--keep_n_checkpoints', default=3)
    return parser

class CacheBucket():
    """
    A class for managing data storage and retrieval using an in-memory cache.
    """

    def __init__(self, files=[], caching_fn=None):
        """
        Initializes a CacheBucket instance.

        Args:
            files (list): List of items to be cached.
            caching_fn (function): A function that converts items to key-value pairs for caching.

        Attributes:
            files (list): List of items to be cached.
            caching_fn (function): A function that converts items to key-value pairs for caching.
            cache (dict): In-memory cache to store key-value pairs.
        """
        self.files = files
        self.caching_fn = caching_fn
        self.cache = {}

    def caching(self):
        """
        Perform caching of items in the cache.

        Returns:
            None
        """
        for index, items in tqdm.tqdm(enumerate(self.files), desc="Caching", total=len(self.files)):
            key, data = self.caching_fn(items)
            self.cache[key] = data

    def save_to_disk(self, path):
        """
        Save cache to disk.

        Args:
            path (str): Save cache to the specified path on disk.
        Returns:
            None
        """
        torch.save(self.cache, path)

    def load_cache_from_file(self, path):
        """
        Load cache data from a file into the cache.

        Args:
            path (str): Path to the cache file.

        Returns:
            None
        """
        self.cache = torch.load(path)

    def get_item(self, key: str):
        """
        Retrieve an item from the cache based on the provided key.

        Args:
            key (str): Key associated with the desired item.

        Returns:
            The cached item.
        """
        return self.cache[key]
    
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs, sid
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, cache_bucket = None):
        # self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.audiopaths_and_text = audiopaths_and_text
        self.text_cleaners  = hparams.text_cleaners
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length 
        self.hop_length     = hparams.hop_length 
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate 

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        self.cache_bucket : CacheBucket = cache_bucket

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()


    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, text, sid in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, text,  sid])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text, sid = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid=sid)
        
        return (text, spec, wav, sid)

    def get_audio(self, filename):
        name = os.path.basename(filename)

        if not self.cache_bucket is None:
            audio, sampling_rate = self.cache_bucket.get_item(name)['audio']
        else:
            audio, sampling_rate = load_wav_to_torch(filename)

        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        # spec_filename = filename.replace(".wav", ".spec.pt")

        if not self.cache_bucket is None:
            spec = self.cache_bucket.get_item(name)['spectrogram']
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            # torch.save(spec, spec_filename)
            
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm
    
    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid
    
    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

def calculate_sha256(data_bytes):
    # Tạo đối tượng băm SHA-256
    sha256_hash = hashlib.sha256()
    
    # Cập nhật dữ liệu vào đối tượng băm
    sha256_hash.update(data_bytes)
    
    # Lấy giá trị mã băm dưới dạng chuỗi hex
    hashed_data = sha256_hash.hexdigest()
    
    return hashed_data

class TextAudioSpeakerRVCLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths, hparams, all_in_mem: bool = False, vol_aug: bool = True):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.hparams = hparams
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.unit_interpolate_mode = hparams.data.unit_interpolate_mode
        self.sampling_rate = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.spec_len = hparams.train.max_speclen
        self.spk_map = hparams.spk
        self.vol_emb = hparams.model.vol_embedding
        self.vol_aug = hparams.train.vol_aug and vol_aug
        random.seed(1234)
        random.shuffle(self.audiopaths)
        
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]

    def get_audio(self, filename):
        filename = filename.replace("\\", "/")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "Sample Rate not match. Expect {} but got {} from {}".format(
                    self.sampling_rate, sampling_rate, filename))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")

        # Ideally, all data generated after Mar 25 should have .spec.pt
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                                     self.sampling_rate, self.hop_length, self.win_length,
                                     center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        spk = filename.split("/")[-2]
        spk = torch.LongTensor([self.spk_map[spk]])

        f0, uv = np.load(filename + ".f0.npy", allow_pickle=True)
        
        f0 = torch.FloatTensor(np.array(f0,dtype=float))
        uv = torch.FloatTensor(np.array(uv,dtype=float))

        c = torch.load(filename+ ".soft.pt")
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[0], mode=self.unit_interpolate_mode)
        if self.vol_emb:
            volume_path = filename + ".vol.npy"
            volume = np.load(volume_path)
            volume = torch.from_numpy(volume).float()
        else:
            volume = None

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (c.size(-1), spec.size(-1), f0.shape, filename)
        assert abs(audio_norm.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        spec, c, f0, uv = spec[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin]
        audio_norm = audio_norm[:, :lmin * self.hop_length]
        if volume is not None:
            volume = volume[:lmin]
        return c, f0, spec, audio_norm, spk, uv, volume

    def random_slice(self, c, f0, spec, audio_norm, spk, uv, volume):
        # if spec.shape[1] < 30:
        #     print("skip too short audio:", filename)
        #     return None

        if random.choice([True, False]) and self.vol_aug and volume is not None:
            max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
            max_shift = min(1, np.log10(1/max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            audio_norm = audio_norm * (10 ** log10_vol_shift)
            volume = volume * (10 ** log10_vol_shift)
            spec = spectrogram_torch(audio_norm,
            self.hparams.data.filter_length,
            self.hparams.data.sampling_rate,
            self.hparams.data.hop_length,
            self.hparams.data.win_length,
            center=False)[0]

        if spec.shape[1] > 800:
            start = random.randint(0, spec.shape[1]-800)
            end = start + 790
            spec, c, f0, uv = spec[:, start:end], c[:, start:end], f0[start:end], uv[start:end]
            audio_norm = audio_norm[:, start * self.hop_length : end * self.hop_length]
            if volume is not None:
                volume = volume[start:end]
        return c, f0, spec, audio_norm, spk, uv, volume

    def __getitem__(self, index):
        if self.all_in_mem:
            return self.random_slice(*self.cache[index])
        else:
            return self.random_slice(*self.get_audio(self.audiopaths[index][0]))

    def __len__(self):
        return len(self.audiopaths)

class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths

def get_checkpoint(model, optimizer, learning_rate, iteration, pruned=False):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    optimizer_dict = None

    if pruned == False:
        optimizer_dict = optimizer.state_dict()
        
    return {'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer_dict,
              'learning_rate': learning_rate}

class MaxSizeList:
    def __init__(self, max_size):
        self.max_size = max_size
        self._list = []

    def append(self, value):
        self._list.append(value)
        if len(self._list) > self.max_size:
            self._list.pop(0) 

    def __str__(self):
        return str(self._list)
    
def mean(array):
    """
    Calculates the mean of an array.

    Args:
        array: A list or array of numbers.

    Returns:
        The mean of the array.
    """
    sum = 0
    for number in array:
        sum += number
    return sum / len(array)