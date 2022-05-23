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

    # plt.imshow(C[..., 4])
    # plt.show()

    return A, B, C


class IPGDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        filelist = os.listdir(img_dir)
        self.n_samples = 2 * len(filelist)
        self.A, self.B, self.C = list(), list(), list()
        for file in filelist:
            A, B, C = read_file(os.path.join(img_dir, file))
            self.A.append(A)
            self.B.append(B)
            self.A.append(B)
            self.B.append(A)
            self.C.append(C)
            self.C.append(C)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        a = torch.tensor(self.A[idx]).float()
        b = torch.tensor(self.B[idx]).float()
        c = torch.tensor(self.C[idx]).float()
        return a, b, c



def get_iterators(
    datapath,
    batch_size,
    n_workers
    ):

    train_dir=os.path.join(datapath, 'train')

    dataset = IPGDataset(train_dir)

    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        # collate_fn=collate_batch,
        shuffle=True,
        num_workers=n_workers
        )

    return train_dataloader