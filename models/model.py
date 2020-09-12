import torch
import torch.nn.functional as F

from .model_parts import *
from .conditioning import *

class LogoDetectionModel(nn.Module):
    def __init__(self, 
                n_channels, 
                n_classes, 
                dataset,
                logo,
                cond_branch_nn: str = "vgg16_pre"):
        super(LogoDetectionModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dataset = dataset
        self.logo = logo
        self.cond_branch = ConditioningBranch(cond_branch_nn) 


        # Encoder steps
        self.input_layer = Downsampler(n_channels, 64)
        self.down_layer1 = Downsampler(64, 128)
        self.down_layer2 = Downsampler(128, 256)
        self.down_layer3 = Downsampler(256, 512)
        self.down_layer4 = Downsampler(512, 512)      # 1/(2^5)*(width x height) x 512

        # Decoder steps
        self.up1 = Upsampler(1024, 512)
        self.up2 = Upsampler(512, 256)
        self.up3 = Upsampler(256, 128)
        self.up4 = Upsampler(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        cond_out = self.cond_branch.get_vgg16_pre_out(self.logo)

        x1 = self.input_layer(x)
        tile1 = cond_out.expand(x1.shape[0], x1.shape[1])
        ms1 = torch.cat(x1, tile1, dim=-1)
        
        x2 = self.down_layer1(x1)
        tile2 = cond_out.expand(x2.shape[0], x2.shape[1])
        ms2 = torch.cat(x2, tile2, dim=-1)

        x3 = self.down_layer2(x2)
        tile3 = cond_out.expand(x3.shape[0], x3.shape[1])
        ms3 = torch.cat(x3, tile3, dim=-1)

        x4 = self.down_layer3(x3)
        tile4 = cond_out.expand(x4.shape[0], x4.shape[1])
        ms4 = torch.cat(x4, tile4, dim=-1)
        
        x5 = self.down_layer4(x4)
        tile5 = cond_out.expand(x5.shape[0], x5.shape[1])
        ms5 = torch.cat(x5, tile5, dim=-1)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits