import argparse
from torch import optim
import torch
import ast
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


# def get_optimizer(args):
#     # "Optimizer to use: AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, Lion8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, Adafactor"

#     optimizer_type = args.optimizer_type
#     if args.use_8bit_adam:
#         assert (
#             not args.use_lion_optimizer
#         ), "both option use_8bit_adam and use_lion_optimizer are specified."
#         assert (
#             optimizer_type is None or optimizer_type == ""
#         ), "both option use_8bit_adam and optimizer_type are specified."
#         optimizer_type = "AdamW8bit"

#     elif args.use_lion_optimizer:
#         assert (
#             optimizer_type is None or optimizer_type == ""
#         ), "both option use_lion_optimizer and optimizer_type are specified."
#         optimizer_type = "Lion"

#     if optimizer_type is None or optimizer_type == "":
#         optimizer_type = "AdamW"
#     optimizer_type = optimizer_type.lower()


#     if optimizer_type == "AdamW8bit".lower():
#         try:
#             import bitsandbytes as bnb
#         except ImportError:
#             raise ImportError("No bitsandbytes")
        
#         print(f"use 8-bit AdamW optimize")
#         optimizer_class = bnb.optim.AdamW8bit
#         # optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr, **optimizer_kwargs)

#     elif optimizer_type == "SGDNesterov8bit".lower():
#         try:
#             import bitsandbytes as bnb
#         except ImportError:
#             raise ImportError("No bitsand bytes / bitsandbytesがインストールされていないようです")
#         print(f"use 8-bit SGD with Nesterov optimizer")

#         optimizer_class = bnb.optim.SGD8bit
#         # optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)
#     else:
#         optimizer_class = optim.AdamW


#     optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__

#     return optimizer_name, optimizer_class

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

def load_audio_caption(full_path, etx=".txt.cleaned"):
    filepaths_and_text = []
    audio_paths = glob.glob(full_path + "/**/*.wav", recursive=True)

    for audio_path in audio_paths:
        base_path = os.path.splitext(audio_path)[0]

        with open(base_path + etx, encoding='utf-8') as f:
            caption = f.read()

        filepaths_and_text.append((audio_path, caption))

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

def load_eval_dataset(full_path, etx=".txt.cleaned"):
    return load_audio_caption(full_path=os.path.join(full_path + "/eval"), etx=etx)

def load_train_dataset(full_path, etx=".txt.cleaned"):
    return load_audio_caption(full_path=os.path.join(full_path + "/train"), etx=etx)

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def load_custom_dataset(full_path, etx=".txt.cleaned"):
    training_files = []
    validation_files = []

    if os.path.exists(full_path + '/eval'):
        validation_files = load_eval_dataset(full_path, etx=etx)

    if os.path.exists(full_path + '/train'):
        training_files = load_train_dataset(full_path, etx=etx)

    if len(training_files) <= 0 or len(validation_files) <= 0:
        print("Train folder or Eval folder not exist. Auto create dataset.")
        audio_files = load_audio_caption(full_path=full_path, etx=etx)

        training_files, validation_files = delete_range(audio_files, random.randint(0, len(audio_files) - 101), 100)
    
    return training_files, validation_files

def train_args(parser : argparse.ArgumentParser):

    parser.add_argument("--epochs", default=None)
    parser.add_argument("--learning_rate", default=None)
    parser.add_argument("--batch_size", default=None)
    parser.add_argument("--save_every_n_epochs", default=None)
    parser.add_argument("--repeat", default=1, type=int)

    parser.add_argument('--cache_spectrogram_to_disk', action="store_true", default=False, help='Store all spectrogram to disk.')
    parser.add_argument('--cache_spectrogram', action="store_true", default=True, help='Create cache spectrogram.')
    parser.add_argument('--pruned', action="store_true", default=False, help='Pruned you model.')
    parser.add_argument('--mixed_precision', default="fp16")
    parser.add_argument('--fine-tune', action="store_true", default=False)

    parser.add_argument('--custom_dataset', default="")

    return parser

class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, caching_spectrograms = None):
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
        self.caching_spectrograms = caching_spectrograms

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

        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        # spec_filename = filename.replace(".wav", ".spec.pt")

        if filename in self.caching_spectrograms:
            spec = self.caching_spectrograms[filename]
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

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


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