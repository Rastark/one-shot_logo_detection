
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Downscaler(nn.Module):
    """Double conv 3x3, then max pool 2x2 stride 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Upscaler(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, in_channels)
        self.upscale = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.double_conv(x)
        return self.upscale(x)        


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):
        return self.conv(x)


# Conditioning Branch
class OneOneConv(nn.Module):
    def __init__(self, in_channels):
        super(OneOneConv, self).__init__()
        self.in_channels = in_channels

    def forward(self, x):
        return nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)


class ConditioningConcat(nn.Module):
    def __init__(self, tile_concat):
        super(ConditioningConcat, self).__init__()
        self.filtered_tile = OneOneConv(self.tile_concat)       # Here or in forward method?

    def forward(self, x):
        return torch.cat(x, self.filtered_tile, dim=-1)


class VGG16(nn.Module):
    def __init__(self, features, batch_norm=False, cfg='A'):
        super(VGG16, self).__init__()
        self.features = features
        self.batch_norm = batch_norm

        self.cfgs = {
            'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 512, 'M'],     # 6 "evenly" distributed maxpools to reduce dims to 1x1x512 
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 'M'],     # 6 maxpools to reduce dims to 1x1x512
        }

        self.cfg = self.cfgs[cfg]

    def forward(self, x):
        layers = []
        in_channels = 3
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        model = nn.Sequential(*layers)
        return model(x)