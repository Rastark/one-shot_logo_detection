import torch
import torch.nn.functional as F

from .model_parts import *
from .conditioning import *
from .vgg16 import *

class LogoDetectionModel(nn.Module):
    def __init__(self, 
                n_channels, 
                n_classes, 
                dataset,
                logo,
                bilinear=True,
                cond_branch_nn: str = "vgg16_pre"):
        super(LogoDetectionModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dataset = dataset
        self.logo = logo
        # self.cond_branch = ConditioningBranch(cond_branch_nn) 
        self.cond_branch = vgg16()

        # Encoder steps
        self.input_layer = Downscaler(n_channels, 64)
        self.down_layer1 = Downscaler(64, 128)
        self.down_layer2 = Downscaler(128, 256)
        self.down_layer3 = Downscaler(256, 512)
        self.down_layer4 = Downscaler(512, 512)      # 1/(2^5)*(width x height) x 512

        # Decoder steps
        self.up1 = Upscaler(1024, 512, bilinear)
        self.up2 = Upscaler(512, 256, bilinear)
        self.up3 = Upscaler(256, 128, bilinear)
        self.up4 = Upscaler(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # cond_out = self.cond_branch.get_vgg16_pre_out(self.logo)

        tile128 = self.cond_branch.repeat(128,128)
        tile64 = self.cond_branch.repeat(64,64)
        tile32 = self.cond_branch.repeat(32,32)
        tile16 = self.cond_branch.repeat(16,16)
        tile8 = self.cond_branch.repeat(8,8)

        x = self.input_layer(x)
        x1 = torch.cat(x, tile128, dim=-1)
        
        x = self.down_layer1(x)
        x2 = torch.cat(x, tile64, dim=-1)

        x = self.down_layer2(x)
        x3 = torch.cat(x, tile32, dim=-1)

        x = self.down_layer3(x)
        x4 = torch.cat(x, tile16, dim=-1)
        
        x = self.down_layer4(x)
        x5 = torch.cat(x, tile8, dim=-1)
        


        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits