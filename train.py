import argparse
import os
import tqdm
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from .model.model import LogoDetectionModel
from .utils.dataset_loader import BasicDataset

# todo: when we add more models, we should move these variables to another location
MODEL_HOME = os.path.abspath("./stored_models/")
ALL_MODEL_NAMES = ["LogoDetectionModel"]
ALL_DATASET_NAMES = ["FlickrLogos-32"]

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',
                    choices=ALL_DATASET_NAMES,
                    help="Dataset in {}".format(ALL_DATASET_NAMES),
                    required=True
                    )

parser.add_argument('--model',
                    choices=ALL_MODEL_NAMES,
                    help="Model in {}".format(ALL_MODEL_NAMES)
                    )

optimizers = ['Adam', 'SGD']
parser.add_argument('--optimizer',
                    choices=optimizers,
                    default='Adam',
                    help="Optimizer in {}".format(optimizers)
                    )

parser.add_argument('--max_epochs',
                    default=500,
                    type=int,
                    help="Number of epochs"
                    )

parser.add_argument('--valid',
                    default=-1,
                    type=float,
                    help="Number of epochs before valid"
                    )

parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help="Number of samples in each mini-batch in SGD and Adam optimization"
                    )

parser.add_argument('--weight_decay',
                    default=5e-4,
                    type=float,
                    help="L2 weight regularization of the optimizer"
                    )

parser.add_argument('--learning_rate',
                    default=4e-4,
                    type=float,
                    help="Learning rate of the optimizer"
                    )

parser.add_argument('--label_smooth',
                    default=0.1,
                    type=float,
                    help="Label smoothing for true labels"
                    )

parser.add_argument('--decay1',
                    default=0.9,
                    type=float,
                    help="Decay rate for the first momentum estimate in Adam"
                    )

parser.add_argument('--decay2',
                    default=0.999,
                    type=float,
                    help="Decay rate for second momentum estimate in Adam"
                    )

parser.add_argument('--verbose',
                    default=True,
                    type=bool,
                    help="Verbose"
                    )

parser.add_argument('--load',
                    type=bool,
                    required=False,
                    help="Path to the model to load"
                    )

parser.add_argument('--batch_norm',
                    default=False,
                    type=bool,
                    required=False,
                    help="If True, apply batch normalization",
                    )

parser.add_argument('--vgg-cfg',
                    type=str,
                    default='A',
                    help="VGG architecture config",
                    required=False
                    )

args = parser.parse_args()

model_path = "./stored_models/" + "_".join(["LogoDetection", args.dataset]) + ".pt"
if args.load is not None:
    model_path = args.load

print("Loading %s dataset..." % args.dataset)
imgs_dir = os.path.abspath("./data/dataset/FlickrLogos-v2/classes/jpg")
masks_dir = os.path.abspath("./data/dataset/FlickrLogos-v2/classes/masks")
dataset = BasicDataset(imgs_dir=imgs_dir, masks_dir=masks_dir)

print("Initializing model...")
model = LogoDetectionModel(dataset=dataset,
                           batch_norm=args.batch_norm,
                           vgg_cfg=args.vgg_cfg)
model.to('cuda')
if args.load is not None:
    model.load_state_dict(torch.load(model_path))

print("Training model...")

# Optimizer selection
# build all the supported optimizers using the passed params (learning rate and decays if Adam)
supported_optimizers = {
    'Adam': optim.Adam(params=model.parameters(), lr=args.learning_rate, betas=(args.decay_adam_1, args.decay_adam_2),
                       weight_decay=args.weight_decay),
    'SGD': optim.SGD(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
}

# Choose which Torch Optimizer object to use, based on the passed name
optimizer = supported_optimizers[args.optimizer]


# Apply Gaussian normalization to the model
def weights_init(model):
    if isinstance(model, nn.Module):
        nn.init.normal_(model.weight.data, mean=0.0, std=0.01)


def train(model,
          train_samples,
          valid_samples,
          init_normal,
          batch_size,
          max_epochs,
          save_path,
          evaluate_every,
          optimizer,
          label_smooth,
          verbose):
    if (init_normal == True):
        weights_init(model)

    for e in range(max_epochs):
        model.train()
        epoch(batch_size, train_samples)

        # # WIP
        # # Launches evaluation on the model every evaluate_every steps.
        # # We need to change to appropriate evaluation metrics.
        # if evaluate_every > 0 and valid_samples is not None and (e + 1) % evaluate_every == 0:
        #     self.model.eval()
        #     with torch.no_grad():
        #         mrr, h1 = self.evaluator.eval(samples=valid_samples, write_output= False) 

        #     # Metrics printing
        #     print("\tValidation: %f" % h1)

        # if save_path is not None:
        #     print("\tSaving model...")
        #     torch.save(self.model.state_dict(), save_path)
        # print("\tDone.")

    def epoch(self,
              batch_size: int,
              train_samples: np.array):

        n_samples = train_samples.shape[0]

        # Moving samples to GPU and random shuffling them
        train_samples = torch.from_numpy(train_samples).cuda()
        randomized_samples = train_samples[torch.randperm(n_samples), :]

        loss = torch.nn.BCELoss() * (1 / 256 * 256)

        # Training over batches

        # Progress bar
        with tqdm.tqdm(total=n_samples, unit="ex", disable=not self.verbose) as bar:
            bar.set_description(f'train loss')

            batch_start = 0
            while batch_start < n_samples:
                batch_end = min(batch_start + batch_size, n_samples)
                batch = randomized_samples[batch]

                l = self.step_on_batch(loss, batch)

                batch_start += self.batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.5f}')

    # Computing the loss over a single batch
    def step_on_batch(self, loss, batch):
        prediction = self.model.forward(batch)
        truth = batch[:, 3]

        # # Label smoothing (?)
        # truth = (1.0 - self.label_smooth)*truth + (1.0 / truth.shape[1])

        # Compute loss
        l = loss(prediction, truth)

        # Compute loss gradients and run optimization step
        self.optimizer.zero_grad()
        l.backward()
        self.optmizer.step()

        # return loss
        return l


print("\nEvaluating model...")
model.eval()
mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=False)
print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)

# class Optimizer:

#     def __init__(self,
#         model: LogoDetectionModel,
#         optimizer: str = "SGD",
#         batch_size: int = 32,
#         learning_rate: float = 1e-4,
#         adam_decay_1: float = 0.9,
#         adam_decay_2: float = 0.99,
#         weight_decay: float = 0.0,      # L2 regularization
#         label_smooth: float = 0.1,      # If we want to implement label smoothing (?)
#         verbose: bool = True):

#         self.model = model
#         self.batch_size = batch_size
#         self.label_smooth = label_smooth
#         self.verbose = verbose


#     # Evaluator selection

#     def train(self,
#         train_samples,
#         max_epochs: int,
#         save_path: str = None,
#         evaluate_every: int = -1,
#         valid_samples = None)

#     for e in range(max_epochs):
#         self.model.train()
#         self.epoch(batch_size, train_samples)

#         # WIP
#         # Launches evaluation on the model every evaluate_every steps.
#         # We need to change to appropriate evaluation metrics.
#         if evaluate_every > 0 and valid_samples is not None and (e + 1) % evaluate_every == 0:
#             self.model.eval()
#             with torch.no_grad():
#                 mrr, h1 = self.evaluator.eval(samples=valid_samples, write_output= False) 

#             # Metrics printing
#             print("\tValidation: %f" % h1)

#         if save_path is not None:
#             print("\tSaving model...")
#             torch.save(self.model.state_dict(), save_path)
#         print("\tDone.")

#     def epoch(self,
#         batch_size: int,
#         train_samples: np.array):

#         n_samples = train_samples.shape[0]

#         # Moving samples to GPU and random shuffling them
#         train_samples = torch.from_numpy(train_samples).cuda()
#         randomized_samples = train_samples[torch.randperm(n_samples), :]

#         loss = torch.nn.BCELoss()

#         # Training over batches

#         # Progress bar
#         with tqdm.tqdm(total=n_samples, unit="ex", disable=not self.verbose) as bar:
#             bar.set_description(f'train loss')

#             batch_start = 0
#             while batch_start < n_samples:
#                 batch_end = min(batch_start + batch_size, n_samples)
#                 batch = randomized_samples[batch]

#                 l = self.step_on_batch(loss, batch)

#                 batch_start += self.batch_size
#                 bar.update(batch.shape[0])
#                 bar.set_postfix(loss=f'{l.item():.5f}')

#     # Computing the loss over a single batch
#     def step_on_batch(self, loss, batch):
#         prediction = self.model.forward(batch)
#         truth = batch[:, 2]

#         # # Label smoothing (?)
#         # truth = (1.0 - self.label_smooth)*truth + (1.0 / truth.shape[1])

#         # Compute loss
#         l = loss(prediction, truth)

#         # Compute loss gradients and run optimization step
#         self.optimizer.zero_grad()
#         l.backward()
#         self.optmizer.step()

#         #return loss
#         return l
