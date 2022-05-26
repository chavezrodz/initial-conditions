from functools import cache
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch

def value_to_idx(x):
    unique_values = np.unique(x)
    indices = np.argsort(unique_values)
    for idx,value in enumerate(unique_values):
        x[np.where(x == value)] = indices[idx]
    return x.astype(int)


def read_file(file, n_channels=16):
    arr = np.loadtxt(file)
    len_arr = len(arr)
    pts_per_dim = int(np.sqrt(len_arr))

    y = value_to_idx(arr[:, 1])
    x = value_to_idx(arr[:, 0])

    A = np.zeros((pts_per_dim, pts_per_dim, n_channels))
    B = np.zeros((pts_per_dim, pts_per_dim, n_channels))
    C = np.zeros((pts_per_dim, pts_per_dim, n_channels))

    A[x, y] = arr[:, 2:18]
    B[x, y] = arr[:, 18:34]
    C[x, y] = arr[:, 34:50]

    A = torch.tensor(A).float()
    B = torch.tensor(B).float()
    C = torch.tensor(C).float()

    return A, B, C


class IPGDataset(Dataset):
    def __init__(self, img_dir, cached=True, max_samples=None):
        self.img_dir = img_dir
        self.cached = cached
        if max_samples is None:
            self.filelist = os.listdir(img_dir)
        else:
            self.filelist = os.listdir(img_dir)[:max_samples]

        self.n_samples = len(self.filelist)
        if self.cached:
            self.A, self.B, self.C = list(), list(), list()
            for file in self.filelist:
                A, B, C = read_file(os.path.join(img_dir, file))
                self.A.append(A)
                self.B.append(B)
                self.C.append(C)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.cached:
            A, B, C = self.A[idx], self.B[idx], self.C[idx]
        else:
            A, B, C = read_file(os.path.join(self.img_dir, self.filelist[idx]))
        return A, B, C



def get_iterators(
    datapath,
    cached,
    max_samples,
    batch_size,
    n_workers
    ):

    train_dir=os.path.join(datapath, '2760')

    dataset = IPGDataset(train_dir, cached=cached, max_samples=max_samples)

    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        # collate_fn=collate_batch,
        shuffle=True,
        num_workers=n_workers
        )

    return train_dataloader