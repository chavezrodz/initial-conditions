import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


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

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_dim)


        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        x = (projectile(A, B), target)
        shapes are batch x gauge fields x x_dim x y_dims, 
        ex: (16, (8 or 16), 101, 101)
        """
        x = torch.cat(x, dim=-3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        return x
