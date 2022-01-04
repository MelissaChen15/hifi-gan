from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
import numpy as np

h = None
device = None

def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)



def main():
    print('Extracting Mels..')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='mels_hifigan')
    parser.add_argument('--checkpoint_file', required=True)

    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    filelist = os.listdir(a.input_wavs_dir)
    os.makedirs(a.output_dir, exist_ok=True)

    with torch.no_grad():
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav)
            mel = get_mel(wav.unsqueeze(0))
            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '.npy')
            np.save(output_file, mel)
            # print(output_file, mel.shape)


if __name__ == '__main__':
    main()

