import os
import numpy as np
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


def get_norms(iterator):
    arrays = list()
    for x in iterator:
        arrays.append(x)
    arrays = torch.cat(arrays, dim=0)
    norms = torch.zeros((2, 48))
    norms[0] = arrays.mean(dim=(0, 2, 3))
    norms[1] = arrays.std(dim=(0, 2, 3))
    return norms

class IPGDataset(Dataset):
    def __init__(self, checkpt_path, cached=True):
        self.checkpt_path = checkpt_path
        self.cached = cached

        self.filelist = filelist = os.listdir(checkpt_path)
        self.n_samples = len(filelist)

        if self.cached:
            self.ABC = list()
            for file in filelist:
                ABC = np.load(os.path.join(checkpt_path, file))
                self.ABC.append(ABC)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.cached:
            ABC = self.ABC[idx]
        else:
            ABC = np.load(os.path.join(self.checkpt_path, self.filelist[idx]))
        return ABC


def get_iterators(
    datapath,
    cached,
    batch_size,
    n_workers,
    test_split=0.1,
    val_split=0.1
    ):

    data_dir=os.path.join(datapath, '128x128/5020')
    checkpt_path = os.path.join(data_dir, 'checkpt')
    checkpt_exists = os.path.exists(checkpt_path)
    arrays_path = os.path.join(checkpt_path, 'arrays')

    if not checkpt_exists:
        print("Making Preprocess Checkpoint")
        os.makedirs(arrays_path, exist_ok=True)
        filelist = os.listdir(data_dir)
        for file in filelist:
            filename = os.path.join(data_dir, file)
            outfile = os.path.join(arrays_path, file[:-4]+'.npy')
            save_checkpt(filename, outfile)
    else:
        print("Using Checkpoint")

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
