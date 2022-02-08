from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import MultiPeriodDiscriminator4Spoof, MultiScaleDiscriminator4Spoof,feature_loss4spoof, discriminator_loss4spoof
from pynvml import *


h = None
device = None

def framing(y, frame_size, hop_size, n_frames):
    N = len(y)
    n_frames = np.min(((N - frame_size) // hop_size + 1, n_frames))
    frames = []

    for i in range(n_frames):
        frame = y[i*hop_size : i*hop_size + frame_size]
        frames.append(frame)
        
    return torch.stack(frames)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    mpd = MultiPeriodDiscriminator4Spoof().to(device)
    msd = MultiScaleDiscriminator4Spoof().to(device)

    state_dict_do = load_checkpoint(a.checkpoint_file, device)
    mpd.load_state_dict(state_dict_do['mpd'])
    msd.load_state_dict(state_dict_do['msd'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    mpd.eval()
    msd.eval()
    with torch.no_grad():
        for i, filname in enumerate(tqdm(filelist)):
            try:
                torch.cuda.empty_cache()
                wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
                wav = wav / MAX_WAV_VALUE
                wav = torch.FloatTensor(wav).to(device)
                frames = framing(wav, h.segment_size, h.hop_size, 300)
                frames = frames.unsqueeze(1)
                del wav; torch.cuda.empty_cache()

                # MPD              
                y_df_hat_r, fmap_df_r = mpd(frames)
                feature_loss_f = feature_loss4spoof(fmap_df_r)

                del y_df_hat_r, fmap_df_r; torch.cuda.empty_cache()

                # MSD
                y_ds_hat_r, fmap_ds_r = msd(frames)
                feature_loss_s = feature_loss4spoof(fmap_ds_r)

                feature_loss_all = torch.cat((feature_loss_f, feature_loss_s), 0)

                # print(feature_loss_all.shape) # torch.Size([54, 92])
                # print(feature_loss_all.cpu().numpy().shape) (feats, t)
                # exit(1)

                output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '.npy')
                np.save(output_file, feature_loss_all.cpu().numpy())

                del y_ds_hat_r, fmap_ds_r, feature_loss_f, feature_loss_s, feature_loss_all; torch.cuda.empty_cache()
            except Exception as e:
                print(filname, frames.shape, e)
        
            # import gc
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             print(type(obj), obj.size())
            #     except:
            #         pass
            # exit(1)


def main():
    print('Initializing Feature Extraction Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='/home/dotdot/data/ASVspoof2019_LA/ASVspoof2019_LA_eval/wav')
    parser.add_argument('--output_dir', default='/home/dotdot/data/ASVspoof2019_LA/ASVspoof2019_LA_eval/gan_feats')
    parser.add_argument('--checkpoint_file',default = 'checkpoints/16k/do_01000000')
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

    inference(a)


if __name__ == '__main__':
    main()

