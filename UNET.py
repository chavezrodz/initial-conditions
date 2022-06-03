import torch
import torch.nn as nn
import numpy as np

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


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNET(nn.Module):
    def __init__(self,
                input_dim,
                hidden_dim,
                n_layers,
                output_dim):
        super().__init__()
        
        """ Encoder """
        initial_pow = int(np.log2(hidden_dim))
        self.encoders = nn.ModuleList([encoder_block(input_dim, 2**initial_pow)])
        for i in range(n_layers):
            self.encoders.append(encoder_block(2**(initial_pow + i), 2**(initial_pow + i + 1)))

        """ Bottleneck """
        self.b = conv_block(2**(initial_pow + n_layers), 2**(initial_pow + n_layers + 1))

        """ Decoder """
        self.decoders = nn.ModuleList()
        for i in range(n_layers + 1)[::-1]:
            self.decoders.append(decoder_block(2**(initial_pow + i + 1), 2**(initial_pow + i)))

        """ Fully Connected """
        self.outputs = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        """ Encoder """
        ss = []
        for encoder in self.encoders:
            s, x = encoder(x)
            ss.append(s)
        ss.reverse()

        """ Bottleneck """
        b = self.b(x)

        """ Decoder """
        out = self.decoders[0](b, ss[0])
        for i in range(len(ss) - 1):
            out = self.decoders[i+1](out, ss[i+1])
        """ Fully Connected """
        outputs = self.outputs(out)
        return outputs
