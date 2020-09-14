import tqdm
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from .model import LogoDetectionModel

class Optimizer:

    def __init__(self,
        model: LogoDetectionModel,
        optimizer: str = "SGD",
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        adam_decay_1: float = 0.9,
        adam_decay_2: float = 0.99,
        weight_decay: float = 0.0,      # L2 regularization
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
    self.optimizer = supported_optimizers[optimizer]

    # Evaluator selection

    def train(self,
        train_samples,
        max_epochs: int,
        save_path: str = None,
        evaluate_every: int = -1,
        valid_samples = None)

    for e in range(max_epochs):
        self.model.train()
        self.epoch(batch_size, training_samples)

        # WIP
        # Launches evaluation on the model every evaluate_every steps.
        # We need to change to appropriate evaluation metrics.
        if evaluate_every > 0 and valid_samples is not None and (e + 1) % evaluate_every == 0:
            self.model.eval()
            with torch.no_grad():
                mrr, h1 = self.evaluator.eval(samples=valid_samples, write_otuput= False) 
            
            # Metrics printing
            print("\tValidation: %f" % h1)
            
        if save_path is not None:
            print("\tSaving model...")
            torch.save(self.model.state_dict(), save_path)
        print("\tDone.")

    def epoch(self,
        batch_size: int,
        training_samples: np.array):

        samples_number = training_samples.shape[0]

        # Moving samples to GPU and random shuffling them
        training_samples = torch.from_numpy(training_samples).cuda()
        randomized_samples = training_samples[torch.randperm(samples_number), :]

        loss = torch.nn.BCELoss()

        # Training over batches

        # Progress bar
        with tqdm.tqdm(total=samples_number, unit="ex", disable=not self.verbose) as bar:
            bar.set_description(f'train loss')

            batch_start = 0
            while batch_start < samples_number:
                batch_end = min(batch_start + batch_size, samples_number)
                batch = randomized_samples[batch]

                l = self.step_on_batch(loss, batch)

                batch_start += self.batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.5f}')

    # Computing the loss over a single batch
    def step_on_batch(self, loss, batch):
        prediction = self.model.forward(batch)
        truth = batch[:, 2]

        # # Label smoothing (?)
        # truth = (1.0 - self.label_smooth)*truth + (1.0 / truth.shape[1])
        
        # Compute loss
        l = loss(prediction, truth)

        # Compute loss gradients and run optimization step
        self.optimizer.zero_grad()
        l.backward()
        self.optmizer.step()

        #return loss
        return l
        

