import torch
import torch.nn.functional as F

from .model_parts import *

class LogoDetectionModel(nn.Module):
    def __init__(self, 
                n_channels: int, 
                n_classes: int, 
                dataset: Dataset,
                logos,
                batch_norm=False,
                cfg='A'):
        super(LogoDetectionModel, self).__init__()
        self.dataset = dataset
        self.n_channels = dataset.n_channels
        self.n_classes = dataset.n_classes
        self.logos = dataset.logos
        # self.cond_branch = ConditioningBranch(cond_branch_nn) 

        # Encoder steps
        self.input_layer = Downscaler(n_channels, 64)
        self.down_layer1 = Downscaler(64, 128)
        self.down_layer2 = Downscaler(128, 256)
        self.down_layer3 = Downscaler(256, 512)
        self.down_layer4 = Downscaler(512, 512)      # 1/(2^5)*(width x height) x 512

        # Conditioning Module
        self.vgg16 = VGG16(batch_norm, cfg)     
        self.one_conv1 = OneOneConv(576, 64)      # 64+512
        self.one_conv2 = OneOneConv(640, 128)     # 128+512
        self.one_conv3 = OneOneConv(768, 256)     # 256+512
        self.one_conv4 = OneOneConv(1024, 512)    # 512+512          

        # Decoder steps
        self.up1 = Upscaler(1024, 512)
        self.up2 = Upscaler(512, 256)
        self.up3 = Upscaler(256, 128)
        self.up4 = Upscaler(128, 64)
        self.up5 = Upscaler(64, 1)      # How many output channels?


    def forward(self, x, z):

        latent_repr = self.vgg16(z)
        
        # Encoder + Conditioning
        x = self.input_layer(x)
        tile = latent_repr.repeat(128,128)
        x1 = torch.cat(x, tile, dim=-1)

        x = self.down_layer1(x)
        tile = latent_repr.repeat(64,64)
        x2 = torch.cat(x, tile, dim=-1)

        x = self.down_layer2(x)
        tile = latent_repr.repeat(32,32)
        x3 = torch.cat(x, tile, dim=-1)

        x = self.down_layer3(x)
        tile = latent_repr.repeat(16,16)
        x4 = torch.cat(x, tile, dim=-1)

        x = self.down_layer4(x)
        tile = latent_repr.repeat(8,8)
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

        output = nn.Softmax(x)
        return output