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


h = None
device = None

def framing(y, frame_size, hop_size):
    N = len(y)
    n_frames = (N - frame_size) // hop_size + 1
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
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            frames = framing(wav, h.segment_size, h.hop_size)
            frames = frames.unsqueeze(1)

            # MPD              
            y_df_hat_r, fmap_df_r = mpd(frames)
            loss_disc_f, losses_disc_f_r = discriminator_loss4spoof(y_df_hat_r)
            feature_loss_f = feature_loss4spoof(fmap_df_r)

    

            # MSD
            y_ds_hat_r, fmap_ds_r = msd(frames)
            loss_disc_s, losses_disc_s_r = discriminator_loss4spoof(y_ds_hat_r)
            feature_loss_s = feature_loss4spoof(fmap_ds_r)

            feature_loss_all = torch.cat((feature_loss_f, feature_loss_s), 0)
            loss_disc_all = loss_disc_s + loss_disc_f

            # print(feature_loss_all.shape) # torch.Size([54, 92])
            # print(feature_loss_all.cpu().numpy().shape) (feats, t)
            # exit(1)

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '.npy')
            np.save(output_file, feature_loss_all.cpu().numpy())
            
            # del wav, y_df_hat_r, fmap_df_r, y_ds_hat_r, fmap_ds_r, feature_loss_f, feature_loss_s
            # torch.cuda.empty_cache()



def main():
    print('Initializing Feature Extraction Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='/home/dotdot/data/ASVspoof2019_LA/ASVspoof2019_LA_train/wav')
    parser.add_argument('--output_dir', default='/home/dotdot/data/ASVspoof2019_LA/ASVspoof2019_LA_train/gan_feats')
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

