import torch.optim
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        ):
        super(Model, self).__init__()
        self.in_dim = input_dim
        self.hid_dim = hidden_dim
        self.n_layers = n_layers

        self.af = nn.ReLU()
        self.l1 = nn.Bilinear(8, 8, 8)
        self.l2 = nn.Linear(8, 8)
    
    def forward(self, x):
        """
        x = (projectile, target)
        shapes are batch x x_dim, y_dims, gauge fields
        ex: (16, 101, 101, 8)
        """
        (a, b) = x
        out = self.af(self.l1(a, b))
        out = self.l2(out)
        return out