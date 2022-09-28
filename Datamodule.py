import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from utils import two_to_three
import pytorch_lightning as pl
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


class IPGDataset(Dataset):
    def __init__(
        self, processed_path, cached=True, energy='all', res='512x512', max_samples=-1
        ):
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
    

class DataModule(pl.LightningDataModule):
    def __init__(self, args, stage):
        super().__init__()
        self.cached = args.cached
        self.max_samples = args.max_samples
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.data_stage = stage
        self.test_split=0.1
        self.val_split=0.1

        self.input_dim, self.output_dim = 32, 16
        self.res = args.train_res if self.data_stage == 'train' else args.test_res
        self.energy = args.train_energy if self.data_stage == 'train' else args.test_energy
        self.datapath = args.datapath
        self.checkpt_path = os.path.join(args.datapath, 'processed')
        self.norms_file = os.path.join(self.checkpt_path, 'norms', self.res +'_'+ self.energy + '.npy')

    def prepare_data(self):
        # convert data types to tensors & saving them under processed
        if self.data_stage == 'train':
            for energy in ['193', '2760', '5020']:
                energy_src = os.path.join(self.datapath, self.res, energy)
                energy_chpt = os.path.join(self.checkpt_path, self.res, energy)
                if not os.path.exists(energy_chpt):
                    os.makedirs(energy_chpt, exist_ok=True)
                    files = os.listdir(energy_src)
                    random.shuffle(files)
                    for file in files:
                        filename = os.path.join(energy_src, file)
                        outfile = os.path.join(energy_chpt, file[:-4]+'.npy')
                        if not os.path.exists(outfile):
                            print(f'\t missing {energy} {file}')
                            save_checkpt(filename, outfile)

            # If norms dont exist make them
            if not os.path.exists(self.norms_file):
                print("Norms dont exist, computing them")
                os.makedirs(os.path.join(self.checkpt_path, 'norms'), exist_ok=True)
                dataset = IPGDataset(
                    self.checkpt_path,
                    cached=self.cached,
                    energy=self.energy,
                    max_samples=self.max_samples
                    )
                total_len = len(dataset)
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

    def setup(self, stage=None):
        dataset = IPGDataset(self.checkpt_path, cached=self.cached, energy=self.energy, max_samples=self.max_samples)

        total_len = len(dataset)
        test_size = int(self.test_split * total_len)
        train_size = total_len - test_size
        val_size = int(self.val_split*train_size)
        train_size = train_size - val_size

        train_ds, val_ds, test_ds = random_split(
            dataset, [train_size, val_size, test_size]
            )

        if stage in (None, "fit"):
            self.train_ds, self.val_ds = train_ds, val_ds

        if stage in (None, "test"):
            self.test_ds = test_ds
        
        if stage in (None, "predict"):
            self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
