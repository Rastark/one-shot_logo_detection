import torch.nn.functional as F

from .model_parts import *
from .conditioning import *

class LogoDetectionModel(nn.Module):
    def __init__(self, 
                n_channels, 
                n_classes, 
                dataset,
                logos,
                cond_branch_nn: str = "vgg16_pre"
                bilinear=True):
        super(LogoDetectionModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dataset = dataset
        
        self.cond_branch = ConditioningBranch(logos, cond_branch_nn) 
        self.cond_out = cond_branch.get_vgg16_pre_out()

        # Encoder steps
        self.step1 = Encoder(n_channels, 64)

        
        self.step2 = Encoder(64, 128)
        self.step3 = Encoder(128, 256)
        self.step4 = Encoder(256, 512)
        self.step5 = Encoder(512, 512)      # 1/(2^5)*(width x height) x 512

        # Decoder steps
        self.up1 = Decoder(1024, 512)
        self.up2 = Decoder(512, 256)
        self.up3 = Decoder(256, 128)
        self.up4 = Decoder(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits