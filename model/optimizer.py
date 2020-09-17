import tqdm
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from .model import LogoDetectionModel

class Optimizer:

    def __init__(self,
        model: LogoDetectionModel,
        optimizer_name: str = "SGD",
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        adam_decay_1: float = 0.9,
        adam_decay_2: float = 0.99,
        l2: float = 0.0,      # L2 regularization
        label_smooth: float = 0.1,      # If we want to implement label smoothing (?)
        verbose: bool = True):

    self.model = model
    self.batch_size = batch_size
    self.label_smooth = label_smooth
    self.verbose = verbose

    # Optimizer selection
    # build all the supported optimizers using the passed params (learning rate and decays if Adam)
    supported_optimizers = {
        'Adam': optim.Adam(params=self.model.parameters(), lr=learning_rate, betas=(decay_adam_1, decay_adam_2), weight_decay=weight_decay),
        'SGD': optim.SGD(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    }

    # choose which Torch Optimizer object to use, based on the passed name
    self.optimizer = supported_optimizers[optimizer_name]

    def adam(learning_rate, decay1, decay2, weight_decay):
        return optim.Adam(params=self.model.parameters(), lr=learning_rate, betas=(decay1, decay2), weight_decay=l2)

    def SGD(l)

