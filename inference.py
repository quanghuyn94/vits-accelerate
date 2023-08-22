import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io import wavfile
from datetime import datetime

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def gen_preview(text, net_g, hps, out):

  stn_tst = get_text(text, hps)
  with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([0]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d-%H%M%S")
    wavfile.write(out, rate=hps.data.sampling_rate, data=audio)


def main():
   hps = utils.get_hparams_from_file("configs/isla_base.json")

   net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
   net_g.eval()
   utils.load_checkpoint("isla_base/G_5532-pruned.pth", net_g, None)
   gen_preview("こんにちは", net_g=net_g, hps=hps, out="test.wav")

if __name__ == "__main__":
   main()