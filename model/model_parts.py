
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, batch_norm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if batch_norm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class Downscaler(nn.Module):
    """Double conv 3x3, then max pool 2x2 stride 2"""

    def __init__(self, in_channels, out_channels, batch_norm):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels, batch_norm=batch_norm),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Upscaler(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        self.upscale = nn.Sequential(
            DoubleConv(in_channels, in_channels, batch_norm=batch_norm),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.upscale(x)        


# Conditioning Branch
class OneOneConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(OneOneConv, self).__init__()
        if batch_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

# class ConditioningConcat(nn.Module):
#     def __init__(self, tile_concat):
#         super(ConditioningConcat, self).__init__()
#         self.filtered_tile = OneOneConv(self.tile_concat)       # Here or in forward method?

#     def forward(self, x):
#         return torch.cat(x, self.filtered_tile, dim=-1)


class VGG16(nn.Module):
    def __init__(self, batch_norm=True, cfg='A', in_channels=3):
        super(VGG16, self).__init__()
        self.batch_norm = batch_norm

        self.cfgs = {
            'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 512, 'M'],     # 6 "evenly" distributed maxpools to reduce dims to 1x1x512 
            'B': [32, 32, 'M', 64, 64, 'M', 128, 'M', 256, 'M', 512, 'M', 'OutConv'], # From the paper's git
            'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 'M']     # 6 maxpools to reduce dims to 1x1x512
        }
        self.cfg = self.cfgs[cfg]

        self.layers = []
        for v in self.cfg:
            if v == 'M':
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'OutConv':
                outConv = nn.Conv2d(in_channels, 512, kernel_size=2, stride=1)
                if self.batch_norm:
                    self.layers += [outConv, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
                else:
                    self.layers += [outConv, nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    self.layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    self.layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.vgg16 = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.vgg16(x)