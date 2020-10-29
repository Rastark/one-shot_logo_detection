import argparse
import logging
import os
import sys
import tqdm
import yaml

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

from model.model import LogoDetection
from utils.dataset_loader import BasicDataset

def test(model,
          device,
          batch_size,
          max_epochs,
        #   save_path,
          verbose,
          val_percent=0.1):
    model.eval()

    #TODO dataset preprocessing

    # with torch.no_grad():
    #     output = model(queries, targets)

        probs = output.squeeze(0)

    return None