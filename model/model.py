import torch
from torch import nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np

from .model_parts import *
from ..utils.dataset_loader import BasicDataset


class LogoDetectionModel(nn.Module):
    def __init__(self,
                 dataset: BasicDataset,
                 batch_norm=False,
                 vgg_cfg: str = 'A'):
        super(LogoDetectionModel, self).__init__()
        self.dataset = dataset
        self.n_channels = 3
        self.img_dim = dataset.mask_img_dim
        self.logo_dim = dataset.query_dim

        # Encoder steps
        self.input_layer = Downscaler(self.n_channels, 64)
        # self.input_layer.wei
        self.down_layer1 = Downscaler(64, 128)
        self.down_layer2 = Downscaler(128, 256)
        self.down_layer3 = Downscaler(256, 512)
        self.down_layer4 = Downscaler(512, 512)  # 1/(2^5)*(width x height) x 512

        # Conditioning Module
        self.latent_repr = VGG16(batch_norm, vgg_cfg)
        self.one_conv1 = OneOneConv(576, 64)  # 64+512
        self.one_conv2 = OneOneConv(640, 128)  # 128+512
        self.one_conv3 = OneOneConv(768, 256)  # 256+512
        self.one_conv4 = OneOneConv(1024, 512)  # 512+512

        # Decoder steps
        self.up1 = Upscaler(1024, 512)
        self.up2 = Upscaler(512, 256)
        self.up3 = Upscaler(256, 128)
        self.up4 = Upscaler(128, 64)
        self.up5 = Upscaler(64, 1)  # How many output channels?
        self.output_layer = OutSoftmax()

        # with torch.no_grad():
        #     self.input_layer.weight = torch.nn.Parameter()

    def forward(self, samples):
        query = samples[:, 0]
        target = samples[:, 1]
        z = self.latent_repr(query)

        # Encoder + Conditioning
        x = self.input_layer(target)

        tile = z.repeat(128, 128)
        x1 = torch.cat(x, tile, dim=-1)
        x = self.down_layer1(x)

        tile = z.repeat(64, 64)
        x2 = torch.cat(x, tile, dim=-1)
        x = self.down_layer2(x)

        tile = z.repeat(32, 32)
        x3 = torch.cat(x, tile, dim=-1)
        x = self.down_layer3(x)

        tile = z.repeat(16, 16)
        x4 = torch.cat(x, tile, dim=-1)
        x = self.down_layer4(x)

        tile = z.repeat(8, 8)
        x5 = torch.cat(x, tile, dim=-1)

        # Decoder + Conditioning
        x = torch.cat(x, x5, dim=-1)
        x = self.up1(x)

        x4 = self.one_conv4(x4)
        x = torch.cat(x, x4, dim=-1)
        x = self.up2(x)

        x3 = self.one_conv3(x3)
        x = torch.cat(x, x3, dim=-1)
        x = self.up3(x)

        x2 = self.one_conv2(x2)
        x = torch.cat(x, x2, dim=-1)
        x = self.up4(x)

        x1 = self.one_conv1(x1)
        x = torch.cat(x, x1, dim=-1)
        x = self.up5(x)

        output = self.output_layer(x)
        return output

    def predict_mask(self, query):
        return self.forward(query)
