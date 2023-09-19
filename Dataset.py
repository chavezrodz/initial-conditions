import os
import numpy as np
from torch.utils.data import Dataset
from utils import two_to_three
import torch
import random

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


def preprocess(datapath, res, energy):
    # convert data types to tensors & saving them under processed
    energy_src = os.path.join(datapath, 'raw', res, energy)
    energy_chpt = os.path.join(datapath, 'processed', res, energy)
    os.makedirs(energy_chpt, exist_ok=True)
    files = os.listdir(energy_src)
    random.shuffle(files)
    for file in files:
        outfile = os.path.join(energy_chpt, file[:-4]+'.npy')
        if not os.path.exists(outfile):
            infile = os.path.join(energy_src, file)
            print(f'\t missing {energy} {outfile}')
            save_checkpt(infile, outfile)
    # If norms dont exist make them
    if not os.path.exists(self.norms_file):
        print("Norms dont exist, computing them")
        os.makedirs(os.path.split(self.norms_file)[0], exist_ok=True)
        dataset = IPGDataset(
            os.path.join(self.datapath, 'processed'),
            cached=self.cached,
            energy=self.energy,
            max_samples=self.max_samples,
            res=self.res
            )
        total_len = len(dataset)
        print(total_len)
        test_size = int(self.test_split * total_len)
        train_size = total_len - test_size
        val_size = int(self.val_split*train_size)
        train_size = train_size - val_size

        train_ds, _, _ = random_split(
            dataset, [train_size, val_size, test_size]
            )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, num_workers=self.num_workers)
        np.save(self.norms_file, get_norms(train_dl))
    self.norms = torch.tensor(np.load(self.norms_file))



class IPGDataset(Dataset):
    def __init__(self, processed_path, cached, energy, res, max_samples=-1):
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
