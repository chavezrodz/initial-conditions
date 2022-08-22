import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from utils import two_to_three
import pytorch_lightning as pl
import torch

def save_checkpt(filename, outfile):
    arr = np.loadtxt(filename)
    A = two_to_three(arr)
    A = np.transpose(A, (2, 0, 1))
    np.save(outfile, A)
    pass


def get_norms(dataset):
    arrays = list()
    for idx in range(len(dataset)):
        arr = dataset.__getitem__(idx)[0]
        arrays.append(torch.tensor(arr))
    arrays = torch.stack(arrays)
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
    

class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.cached = args.cached
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.test_split=0.1
        self.val_split=0.1

        self.input_dim, self.output_dim = 32, 16
        self.energy = args.energy


        self.data_dir = os.path.join(args.datapath, args.res)
        self.checkpt_path = os.path.join(args.datapath, 'processed')
        self.energies = os.listdir(self.data_dir)

    def prepare_data(self):
        if not os.path.exists(self.checkpt_path):
            print("Making Preprocess Checkpoint")
            for energy in self.energies:
                print(f'processing energy {energy}')
                energy_src = os.path.join(self.data_dir, energy)
                energy_chpt = os.path.join(self.checkpt_path, energy)
                os.makedirs(os.path.join(energy_chpt, 'arrays'), exist_ok=True)
                for file in os.listdir(energy_src):
                    print(f'\t {energy} {file}')
                    filename = os.path.join(energy_src, file)
                    outfile = os.path.join(energy_chpt, 'arrays', file[:-4]+'.npy')
                    save_checkpt(filename, outfile)
        else:
            print("Using Checkpoint")


    def setup(self, stage=None):
        arrays_path = os.path.join(self.checkpt_path, self.energy, 'arrays')
        self.dataset = IPGDataset(arrays_path, cached=self.cached)

        total_len = len(self.dataset)
        test_size = int(self.test_split * total_len)
        train_size = total_len - test_size
        val_size = int(self.val_split*train_size)
        train_size = train_size - val_size

        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.dataset, [train_size, val_size, test_size]
            )
        self.norms = get_norms(self.train_ds)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    