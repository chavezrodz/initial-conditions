import torch.nn as nn

class conv_layer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        n_layers,
        kernel_size
        ):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        pad = (kernel_size - 1) // 2

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, padding=pad)
            ])
        self.conv_layers.append(nn.BatchNorm2d(hidden_dim))
        self.conv_layers.append(nn.ReLU())
        for i in range(n_layers):
            self.conv_layers.append(conv_layer(hidden_dim, hidden_dim))
        self.conv_layers.append(
            nn.Conv2d(hidden_dim, output_dim, kernel_size=kernel_size, padding=pad)
            )

    def forward(self, x):
        """
        x = (projectile(A, B), target)
        shapes are batch x gauge fields x x_dim x y_dims,
        ex: (16, (16 or 32), 101, 101)
        """
        for layer in self.conv_layers:
            x = layer(x)
        return x
