import torch
from torch import nn
import torch.nn.functional as F

class VGG(nn.Module):

    # Logos must be 64x64
    def __init__(self, logos, dataset):
        super(VGG, self).__init__()
        self.logos = logos
        self.dataset = dataset
        self.model = torch.hub.load('pytorch/vision:0.6.0', 'vgg16_bn', pretrained=True)



        



