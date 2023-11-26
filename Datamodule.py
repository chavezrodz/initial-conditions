import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from Dataset import IPGDataset
import pytorch_lightning as pl
import torch
import random


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
            # print(f'{idx}/{n_batches}')
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


class DataModule(pl.LightningDataModule):
    def __init__(self, datapath, max_samples, batch_size, num_workers, res, energy, cached):
        super().__init__()
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datapath = datapath
        self.res = res
        self.energy = energy
        self.cached=cached

        self.norms_file = os.path.join(self.datapath, 'processed', 'norms', self.res, self.energy + '.npy')

        self.test_split=0.1
        self.val_split=0.1
        self.input_dim, self.output_dim = 32, 16

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
        print("Executing setup")
        dataset = IPGDataset(
            os.path.join(self.datapath, 'processed'),
            cached=self.cached, energy=self.energy, res=self.res, max_samples=self.max_samples
            )

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
