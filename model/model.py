import torch
from torch import nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import logging

from .model_parts import *


class LogoDetection(nn.Module):
    def __init__(self,
                 n_channels: int = 3,
                 batch_norm=True,
                 vgg_cfg: str = 'A'):
        super(LogoDetection, self).__init__()
        self.n_channels = n_channels

        # Encoder steps
        self.input_layer = Downscaler(self.n_channels, 64, batch_norm)
        self.down_layer1 = Downscaler(64, 128, batch_norm)
        self.down_layer2 = Downscaler(128, 256, batch_norm)
        self.down_layer3 = Downscaler(256, 512, batch_norm)
        self.down_layer4 = Downscaler(512, 512, batch_norm)  # 1/(2^5)*(width x height) x 512

        # Conditioning Module
        self.latent_repr = VGG16(batch_norm, vgg_cfg)
        self.one_conv1 = OneOneConv(576, 64, batch_norm)  # 64+512
        self.one_conv2 = OneOneConv(640, 128, batch_norm)  # 128+512
        self.one_conv3 = OneOneConv(768, 256, batch_norm)  # 256+512
        self.one_conv4 = OneOneConv(1024, 512, batch_norm)  # 512+512

        # Decoder steps
        self.up1 = Upscaler(1024, 512, batch_norm)  # 512+512
        self.up2 = Upscaler(1024, 256, batch_norm)  # 512*2
        self.up3 = Upscaler(512, 128, batch_norm)  # 256*2
        self.up4 = Upscaler(256, 64, batch_norm)  # 128*2
        self.output_layer = Upscaler(128, 1, batch_norm)  # 64*2
        # self.output_layer = OutSoftmax()

        # with torch.no_grad():
        #     self.input_layer.weight = torch.nn.Parameter()

    def forward(self, query, target):
        # query = samples[:, 0]
        # target = samples[:, 1]
#         logging.info(f"query: {query}")
#        print(f"query: {query}")
#         logging.info(f"target: {target}")
#        print(f"target: {target}")
        z = self.latent_repr(query)
#         logging.info(f"z: {z}")
#         print(f"z: {z}")
        # print(z.shape)

        # Encoder + Conditioning
        x = self.input_layer(target)
#         logging.info(f"input_layer: {x}")

        tile = z.expand(z.shape[0], z.shape[1], 128, 128)
        # print(tile.shape)
#         logging.info(f"tile1: {tile}")
        x1 = torch.cat((x, tile), dim=1)
#         logging.info(f"x1: {x1}")
        x = self.down_layer1(x)
#         logging.info(f"down1: {x}")

        tile = z.expand(z.shape[0], z.shape[1], 64, 64)
        x2 = torch.cat((x, tile), dim=1)
#         logging.info(f"x2: {x2}")
        x = self.down_layer2(x)
#         logging.info(f"down2: {x}")

        tile = z.expand(z.shape[0], z.shape[1], 32, 32)
        x3 = torch.cat((x, tile), dim=1)
#         logging.info(f"x3: {x3}")
        x = self.down_layer3(x)
#         logging.info(f"down3: {x}")

        tile = z.expand(z.shape[0], z.shape[1], 16, 16)
        x4 = torch.cat((x, tile), dim=1)
#         logging.info(f"x4: {x4}")
        x = self.down_layer4(x)
#         logging.info(f"down4: {x}")

        tile = z.expand(z.shape[0], z.shape[1], 8, 8)
        x5 = torch.cat((x, tile), dim=1)
#         logging.info(f"x5: {x5}")
        # print(x.shape)

        # Decoder + Conditioning
#        x = torch.cat((x, x5), dim=1)
#        logging.info(f"cond5: {x}")
        # print(x.shape)
        x = self.up1(x5)
#         logging.info(f"up1: {x}")
        # print(x.shape)

        x4 = self.one_conv4(x4)
#         logging.info(f"cnv4: {x4}")
        x = torch.cat((x, x4), dim=1)
#         logging.info(f"cond4: {x}")
        # del x4
        x = self.up2(x)
#         logging.info(f"up2: {x}")

        x3 = self.one_conv3(x3)
#         logging.info(f"cnv3: {x3}")
        x = torch.cat((x, x3), dim=1)
#         logging.info(f"cond3: {x}")
        # del x3
        x = self.up3(x)
#         logging.info(f"up3: {x}")

        x2 = self.one_conv2(x2)
#         logging.info(f"cnv2: {x2}")
        x = torch.cat((x, x2), dim=1)
#         logging.info(f"cond2: {x}")
        # del x2
        x = self.up4(x)
#         logging.info(f"up4: {x}")

        x1 = self.one_conv1(x1)
#         logging.info(f"cnv1: {x1}")
        x = torch.cat((x, x1), dim=1)
#         logging.info(f"cond1: {x}")
        # del x1
#        output = self.up5(x)
#         logging.info(f"up5: {x}")

        output = self.output_layer(x)
#         logging.info(f"output: {output}")
        return output

    def predict_mask(self, query, target):
        y = self.forward(query, target)
        output_layer = nn.Sequential(
            nn.batchNorm2d(1),
            nn.ReLU(),
            nn.Sigmoid()
            )
        return output_layer(y)
