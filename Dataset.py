import os
import numpy as np
from torch.utils.data import Dataset
from utils import two_to_three
import torch

def save_checkpt(filename, outfile):
    arr = np.loadtxt(filename)
    A = two_to_three(arr)
    A = np.transpose(A, (2, 0, 1))
    np.save(outfile, A)
    pass


def get_norms(dataloader):
    n_batches = len(dataloader)
    batch = next(iter(dataloader))
    batch_size, channels, xdim, ydim = batch[0].shape 
    channels_per_object = channels // 3
    norms_in = torch.zeros((2, channels_per_object))
    for channel_idx in range(channels_per_object):
        print(f'Channel {channel_idx}/{channels_per_object}')
        arrays = list()
        for idx, batch in enumerate(dataloader):
            print(f'{idx}/{n_batches}')
            arr = batch[0]
            channel_subset = arr[:, [channel_idx, channel_idx + channels_per_object]]
            arrays.append(channel_subset)
        arrays = torch.cat(arrays, dim=0)
        norms_in[0, channel_idx] = arrays.mean().item()
        norms_in[1, channel_idx] = arrays.std().item()
        del arrays
    norms_in = norms_in.repeat(1,2)
    norms_out = torch.zeros((2,channels_per_object))
    norms_out[1] = 1

    norms = torch.cat([norms_in, norms_out], dim=1)
    return norms


class IPGDataset(Dataset):
    def __init__(self, processed_path, cached=True, energy='all', res='512x512', max_samples=-1):
        self.data_path = processed_path
        self.cached = cached
        self.energy = energy
        self.res = res
        energies = ['193', '2760', '5020']

        if energy in energies:
            e_path = os.path.join(processed_path, res, energy)
            files = os.listdir(e_path)
            self.filelist = [os.path.join(e_path, file) for file in files][:max_samples]

        elif energy == 'all':
            self.filelist = list()
            for e in energies:
                e_path = os.path.join(processed_path, res, e)
                files = os.listdir(e_path)
                e_list = [os.path.join(e_path, file) for file in files][:max_samples]
                self.filelist = self.filelist + e_list

        self.n_samples = len(self.filelist)
        if self.cached:
            self.ABC_fn = list()
            for file in self.filelist:
                file_nb = os.path.split(file)[1][:-4]
                ABC = np.load(file)
                self.ABC_fn.append((ABC, file_nb))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.cached:
            ABC_fn = self.ABC_fn[idx]
        else:
            file = self.filelist[idx]
            file_nb = os.path.split(file)[1][:-4]
            ABC = np.load(file)
            ABC_fn = (ABC, file_nb)
        return ABC_fn
