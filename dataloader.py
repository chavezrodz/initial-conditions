import numpy as np
import matplotlib.pyplot as plt


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


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label





file = 'GaugeFields.dat'
read_file(file)