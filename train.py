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

from .model.model import LogoDetectionModel
from .utils.dataset_loader import BasicDataset

# todo: when we add more models, we should move these variables to another location
MODEL_HOME = os.path.abspath("./stored_models/")
ALL_MODEL_NAMES = ["LogoDetectionModel"]
ALL_DATASET_NAMES = ["FlickrLogos-32"]

with open(os.path.abspath("./config/config.yaml")) as config:
    config_list = yaml.load_all(config)

# # Appl.load_aly Gaussian normalization to the model
# def weights_init(model):
#     if isinstance(model, nn.Module):
#         nn.init.normal_(model.weight.data, mean=0.0, std=0.01)


def train(model,
          device,
        #   init_normal,
          batch_size,
          max_epochs,
          save_path,
          evaluate_every,
          optimizer,
          label_smooth,
          verbose,
          dir_checkpoint,
          save_cp=True,
          val_percent=0.1):
    # if (init_normal == True):
    #     weights_init(model)

    dataset = BasicDataset(imgs_dir, masks_dir)
    
    # Splitting dataset
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    # Loading dataset
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # Logging for TensorBoard
    writer = SummaryWriter(comment=f'LR_{optimizer.lr}_BS_{batch_size}_OPT_{type(optimizer).__name__}')  # does optimizer.lr work?
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:             {max_epochs}
        Batch size:         {batch_size}
        Learning rate:      {optimizer.lr}
        Training size:      {n_train}
        Validation size:    {n_val}
        Device:             {device.type}
    ''')

    criterion = nn.BCELoss() / (256*256)        # L = (1/(H*W)) * BCELoss

    for epoch in range(max_epochs):
        model.train()   # set the model in training flag to True
        epoch_loss = 0  # resets the loss for the current epoch
        # epoch(batch_size, train_samples)

        # TODO
        with tqdm.tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{max_epochs}', unit="img") as bar:
            bar.set_description(f'train loss')

            for batch in train_loader:
                queries = batch[:, 0]       # Correct dimensions?
                targets = batch[:, 1]
                true_masks = batch[:, 2]

                queries = queries.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred = model(queries, targets)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                # TensorBoard logging
                writer.add_scalar('Loss/train', loss.item(), global_step)

                bar.set_postfix(loss=f'{loss.item():.5f}')

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1) Gradient Clipping
                optimizer.step()

                bar.update(queries.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.cpu().numpy(), global_step)
                    # TODO Eval
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    writer.add_images('query_images', queries, global_step)
                    writer.add_images('target_images', targets, global_step)
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/pred', masks_pred, global_step)

            if save_cp:
                try:
                    os.mkdir()
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(model.state_dict(),
                    dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved!')

            writer.close()

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

print("\nEvaluating model...")
model.eval()
mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=False)
print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)


def get_args():
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

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # TODO Modularize paths with respect to the current Dataset
    imgs_dir = os.path.abspath("./data/dataset/FlickrLogos-v2/classes/jpg")
    masks_dir = os.path.abspath("./data/dataset/FlickrLogos-v2/classes/masks")
    
    model_path = "./stored_models/" + "_".join(["LogoDetection", args.dataset]) + ".pt"
    if args.load is not None:
        model_path = args.load

    print("Loading %s dataset..." % args.dataset)
    dataset = BasicDataset(imgs_dir=imgs_dir, masks_dir=masks_dir)

    print("Initializing model...")
    model = LogoDetectionModel(dataset=dataset,
                            batch_norm=args.batch_norm,
                            vgg_cfg=args.vgg_cfg)
    model.to(device=device, dtype=torch.float32)    # ???
    if args.load is not None:
        model.load_state_dict(torch.load(model_path))


    # Change here to adapt your data
    model = LogoDetectionModel(n_channels=3)

    # Optimizer selection
    # build all the supported optimizers using the passed params (learning rate and decays if Adam)
    supported_optimizers = {
        'Adam': optim.Adam(params=model.parameters(), lr=args.learning_rate, betas=(args.decay_adam_1, args.decay_adam_2),
                        weight_decay=args.weight_decay),
        'SGD': optim.SGD(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    }
    # Choose which Torch Optimizer object to use, based on the passed name
    optimizer = supported_optimizers[args.optimizer]

    try:
        train(model=model,
              device=device,
              batch_size=args.batch_size,
              max_epochs=args.max_epochs,
              save_path=args.sa 
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.ph')
        logging.info('Interrupt saved')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)