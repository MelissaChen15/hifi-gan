import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import MultiPeriodDiscriminator4Spoof, MultiScaleDiscriminator4Spoof, feature_loss,feature_loss4spoof, discriminator_loss4spoof
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
import numpy as np

torch.backends.cudnn.benchmark = True


from collections.abc import Sequence


def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """

    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst), )
    
    # recurse
    shape = get_shape(lst[0], shape)

    return shape

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    # generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator4Spoof().to(device)
    msd = MultiScaleDiscriminator4Spoof().to(device)

    if rank == 0:
        # print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if  cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_do = load_checkpoint(cp_do, device)
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
                    

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        # sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    # generator.train()
    mpd.train()
    msd.train()

    disc_losses = []
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        try:
            for i, batch in enumerate(train_loader):
                if rank == 0:
                    start_b = time.time()
                x, y, filename, y_mel = batch
                x = torch.autograd.Variable(x.to(device, non_blocking=True))
                y = torch.autograd.Variable(y.to(device, non_blocking=True))
                y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                y = y.unsqueeze(1) # (b, 1, samples) torch.Size([16, 1, 8192])

                # y_g_hat = generator(x)
                # y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                            #   h.fmin, h.fmax_for_loss)

                optim_d.zero_grad()

                # MPD
                y_df_hat_r, fmap_df_r = mpd(y)
                loss_disc_f, losses_disc_f_r = discriminator_loss4spoof(y_df_hat_r)
                feature_loss_f = feature_loss4spoof(fmap_df_r)
               
    

                # MSD
                y_ds_hat_r, fmap_ds_r = msd(y)
                loss_disc_s, losses_disc_s_r = discriminator_loss4spoof(y_ds_hat_r)
                feature_loss_s = feature_loss4spoof(fmap_ds_r)
                exit(1)

                loss_disc_all = loss_disc_s + loss_disc_f

                # print(len(feature_loss_f), len(feature_loss_s)) #30 24
                # exit(1)

                
                # print(get_shape(y_df_hat_r), get_shape(fmap_df_r), get_shape(y_ds_hat_r), get_shape(fmap_ds_r))
                # (5,) (5, 6) (3,) (3, 8) (#discs, #conv_layers) feature map 包含了最后的x输出(flatten前)
                # print(fmap_ds_r[1][5].shape) #torch.Size([1, 1, 51, 2])  #torch.Size([1, 1024, 128])
                # exit(1)

                
                # (5, 102), (3, 128)

                # print(y_df_hat_r[0].shape, fmap_df_r[0][0].shape,y_ds_hat_r[0].shape, fmap_ds_r[0][0].shape)
                #torch.Size([1, 102]) torch.Size([1, 32, 1366, 2]) torch.Size([1, 128]) torch.Size([1, 128, 8192])

                # print(losses_disc_f_r, losses_disc_s_r)

                
                utt = filename[0].split("/")[7]
                label = int("bonafide" in utt)
                info =  losses_disc_f_r + losses_disc_s_r + [label] + [utt.split(".")[0]]
                disc_losses.append(info)

                if i % 1000 == 0: print(str(i) + "/", str(len(train_loader)))

                # loss_disc_all.backward()
                # optim_d.step()

                # # Generator
                # optim_g.zero_grad()

                # # L1 Mel-Spectrogram Loss
                # loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                # y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                # y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                # loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                # loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                # loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                # loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                # loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

                # loss_gen_all.backward()
                # optim_g.step()

                # if rank == 0:
                #     # STDOUT logging
                #     if steps % a.stdout_interval == 0:
                #         with torch.no_grad():
                #             mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                #         print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                #               format(steps, loss_gen_all, mel_error, time.time() - start_b))

                #     # checkpointing
                #     if steps % a.checkpoint_interval == 0 and steps != 0:
                #         checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                #         save_checkpoint(checkpoint_path,
                #                         {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                #         checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                #         save_checkpoint(checkpoint_path, 
                #                         {'mpd': (mpd.module if h.num_gpus > 1
                #                                              else mpd).state_dict(),
                #                          'msd': (msd.module if h.num_gpus > 1
                #                                              else msd).state_dict(),
                #                          'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                #                          'epoch': epoch})

                #     # Tensorboard summary logging
                #     if steps % a.summary_interval == 0:
                #         sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                #         sw.add_scalar("training/mel_spec_error", mel_error, steps)

                #     # Validation
                #     if steps % a.validation_interval == 0:  # and steps != 0:
                #         generator.eval()
                #         torch.cuda.empty_cache()
                #         val_err_tot = 0
                #         with torch.no_grad():
                #             for j, batch in enumerate(validation_loader):
                #                 x, y, _, y_mel = batch
                #                 y_g_hat = generator(x.to(device))
                #                 y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                #                 y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                #                                               h.hop_size, h.win_size,
                #                                               h.fmin, h.fmax_for_loss)
                #                 val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                #                 if j <= 4:
                #                     if steps == 0:
                #                         sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                #                         sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                #                     sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                #                     y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                #                                                  h.sampling_rate, h.hop_size, h.win_size,
                #                                                  h.fmin, h.fmax)
                #                     sw.add_figure('generated/y_hat_spec_{}'.format(j),
                #                                   plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                #             val_err = val_err_tot / (j+1)
                #             sw.add_scalar("validation/mel_spec_error", val_err, steps)

                #         generator.train()

                steps += 1
        except Exception as err:
            print(filename, err)
            pass

        # np.save("disc_loss_eval.npy", disc_losses)
        # exit(1)
        # scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='/home/dotdot/data/ASVspoof2019_LA/ASVspoof2019_LA_eval/wav')
    parser.add_argument('--input_mels_dir', default='/home/dotdot/data/ASVspoof2019_LA/ASVspoof2019_LA_eval/mels_hifigan')
    parser.add_argument('--input_training_file', default='ASVSpoof19LA/eval.txt')
    parser.add_argument('--input_validation_file', default='ASVSpoof19LA/validation.txt')
    parser.add_argument('--checkpoint_path', default='checkpoints/16k')
    parser.add_argument('--config', default='checkpoints/16k/config.json')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
