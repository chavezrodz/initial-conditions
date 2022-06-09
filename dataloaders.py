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

    A = torch.permute(A, (2, 0, 1))
    B = torch.permute(B, (2, 0, 1))
    C = torch.permute(C, (2, 0, 1))

    return A, B, C

def save_checkpt(filename, outfile):
    arr = np.loadtxt(filename)
    len_arr, cols  = arr.shape
    pts_per_dim = int(np.sqrt(len_arr))

    y = value_to_idx(arr[:, 1])
    x = value_to_idx(arr[:, 0])

    A = np.zeros((pts_per_dim, pts_per_dim, cols - 2))
    A[x, y] = arr[:, 2:]
    A = np.transpose(A, (2, 0, 1))
    np.save(outfile, A)
    pass


def read_checkpt(filename, norms):
    abc = np.transpose(np.load(filename), (1,2,0))
    abc = (abc - norms[0])/norms[1]
    abc = torch.tensor(np.transpose(abc, (2,0,1))).float()
    A, B, C = abc[:16], abc[16:32], abc[32:48] 
    return A, B, C

datafolder = 'data/128x128/5020/checkpt'

def make_norms(arrays_path, outfile):
    files = os.listdir(arrays_path)
    arrays = []
    for file in files:
        arr = np.load(os.path.join(arrays_path, file))
        arrays.append(arr)
    arrays = np.stack(arrays)
    norms = np.zeros((2, 48))
    norms[0] = arrays.mean(axis=(0, 2, 3))
    norms[1] = arrays.std(axis=(0, 2, 3))
    np.save(outfile, norms)


class IPGDataset(Dataset):
    def __init__(self, img_dir, cached=True, max_samples=None):
        self.img_dir = img_dir
        self.cached = cached
        checkpt_path = os.path.join(img_dir, 'checkpt')
        arrays_path = os.path.join(checkpt_path, 'arrays')
        os.makedirs(arrays_path, exist_ok=True)
        checkpt_exists = os.path.exists(checkpt_path)

        if not checkpt_exists:
            print("Making Preprocess Checkpoint")
            filelist = os.listdir(img_dir)
            for file in filelist:
                filename = os.path.join(img_dir, file)
                outfile = os.path.join(arrays_path, file[:-4]+'.npy')
                save_checkpt(filename, outfile)
            norm_file = os.path.join(checkpt_path, 'norms')
            make_norms(arrays_path, norm_file)
        else:
            print("Using Checkpoint")

        self.norms = np.load(os.path.join(checkpt_path, 'norms.npy'))
        self.filelist = os.listdir(arrays_path)[:max_samples]
        self.n_samples = len(self.filelist)

        if self.cached:
            self.A, self.B, self.C = list(), list(), list()
            for file in self.filelist:
                A, B, C = read_checkpt(os.path.join(arrays_path, file), self.norms)
                self.A.append(A)
                self.B.append(B)
                self.C.append(C)


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.cached:
            A, B, C = self.A[idx], self.B[idx], self.C[idx]
        else:
            A, B, C = read_checkpt(os.path.join(self.checkpt_path, self.filelist[idx]), self.norms)
        return A, B, C


def get_iterators(
    datapath,
    cached,
    max_samples,
    batch_size,
    n_workers
    ):

    train_dir=os.path.join(datapath, '128x128/5020')

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
