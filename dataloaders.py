import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils import two_to_three
import torch

def save_checkpt(filename, outfile):
    arr = np.loadtxt(filename)
    A = two_to_three(arr)
    A = np.transpose(A, (2, 0, 1))
    np.save(outfile, A)
    pass


def get_norms(iterator):
    arrays = list()
    for x in iterator:
        arrays.append(x[0])
    arrays = torch.cat(arrays, dim=0)
    aa, bb, cc = arrays[:, :16], arrays[:, 16:32], arrays[:, 32:48]

    arrays_in = torch.cat([aa, bb], dim=0)
    norms_in = torch.zeros((2, 16))
    norms_in[0] = arrays_in.mean(dim=(0, 2, 3))
    norms_in[1] = arrays_in.std(dim=(0, 2, 3))
    norms_in = norms_in.repeat(1,2)

    norms_out = torch.zeros((2,16))
    norms_out[1] = 1
    norms = torch.cat([norms_in, norms_out], dim=1)
    return norms

class IPGDataset(Dataset):
    def __init__(self, arrays_path, cached=True):
        self.arrays_path = arrays_path
        self.cached = cached

        self.filelist = os.listdir(arrays_path)
        self.n_samples = len(self.filelist)
        if self.cached:
            self.ABC_fn = list()
            for file in self.filelist:
                file_nb = file[:-4]
                ABC = np.load(os.path.join(self.arrays_path, file))
                self.ABC_fn.append((ABC, file_nb))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.cached:
            ABC_fn = self.ABC_fn[idx]
        else:
            file = self.filelist[idx]
            file_nb = file[:-4]
            ABC = np.load(os.path.join(self.arrays_path, file))
            ABC_fn = (ABC, file_nb)
        return ABC_fn


def get_iterators(
    datapath,
    cached,
    batch_size,
    n_workers,
    energy_subdir='5020',
    test_split=0.1,
    val_split=0.1
    ):

    data_dir = os.path.join(datapath, '128x128')
    energies = os.listdir(data_dir)
    checkpt_path = os.path.join(datapath, 'processed')
    checkpt_exists = os.path.exists(checkpt_path)

    if not checkpt_exists:
        print("Making Preprocess Checkpoint")
        for energy in energies:
            print(f'processing energy {energy}')
            energy_src = os.path.join(data_dir, energy)
            energy_chpt = os.path.join(checkpt_path, energy)
            os.makedirs(os.path.join(energy_chpt, 'arrays'), exist_ok=True)
            for file in os.listdir(energy_src):
                print(f'\t {file}')
                filename = os.path.join(energy_src, file)
                outfile = os.path.join(energy_chpt, 'arrays', file[:-4]+'.npy')
                save_checkpt(filename, outfile)
    else:
        print("Using Checkpoint")

    arrays_path = os.path.join(checkpt_path, energy_subdir, 'arrays')
    dataset = IPGDataset(arrays_path, cached=cached)
    total_len = len(dataset)
    test_size = int(test_split * total_len)
    train_size = total_len - test_size
    val_size = int(val_split*train_size)
    train_size = train_size - val_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=n_workers
        )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=n_workers
        )

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=n_workers
        )

    norms = get_norms(train_dl)

    return (train_dl, val_dl, test_dl), norms
