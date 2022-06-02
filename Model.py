import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        ):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim


        self.bn0 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        x = (projectile(A, B), target)
        shapes are batch x gauge fields x x_dim x y_dims, 
        ex: (16, (16 or 32), 101, 101)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)

        return x
