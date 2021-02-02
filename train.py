import argparse
import logging
import os
import sys
from tqdm import tqdm
import yaml

# import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
# import torch.nn.functional as F

from eval import eval_net

from model.model import LogoDetection
from utils.dataset_loader import BasicDataset

# todo: when we add more models, we should move these variables to another location

MODEL_HOME = os.path.abspath("./stored_models/")
ALL_MODEL_NAMES = ["LogoDetection"]
ALL_DATASET_NAMES = ["FlickrLogos-32", "FlickrLogos-32-test", "TopLogos-10"]

with open(os.path.abspath("./config/config.yaml")) as config:
    config_list = yaml.load(config, Loader=yaml.FullLoader)


# # Appl.load_aly Gaussian normalization to the model
# def weights_init(model):
#     if isinstance(model, nn.Module):
#         nn.init.normal_(model.weight.data, mean=0.0, std=0.01)

def train(model,
          device,
          train_loader,
          val_loader,
          max_epochs,
          optimizer,
          verbose,
          checkpoint_dir,
          model_path,
          save_cp,
          n_train,
          n_val,
          step_eval,
          early_stop=False,
          bad_val_limit=100,
          keep_single_cp=False
          ):
    batch_size = train_loader.batch_size

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:             {max_epochs}
        Batch size:         {batch_size}
        Learning rate:      
        Training size:      {n_train}
        Validation size:    {n_val}
        Device:             {device.type}
    ''')

    criterion = nn.BCEWithLogitsLoss()

    last_val_loss = sys.maxsize
    bad_val_counter = 0
    for epoch in range(max_epochs):
        logging.info(f"Epoch number {epoch + 1}")
        model.train()  # set the model in training flag to True
        mean_train_epoch_loss = 0  # resets the loss for the current epoch

        # training
        with tqdm(total=n_train, unit='img', disable=not verbose) as bar:
            bar.set_description(f'train loss epoch {epoch + 1}/{max_epochs}')

            for batch in train_loader:
                loss, items = get_loss(batch, device, model, criterion)
                mean_train_epoch_loss += loss.detach().item()  # is the .detach() needed?

                bar.set_postfix(loss=f'{loss.item():.5f}')
                logging.info(f"Train loss: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1) Gradient Clipping
                optimizer.step()

                bar.update(items)
                global_step += 1

            mean_train_epoch_loss = mean_train_epoch_loss / len(train_loader)
            logging.info(f"Train epoch loss: {mean_train_epoch_loss}")

        # validating
        mean_val_loss = 0
        with torch.no_grad():
            with tqdm(total=n_val, desc="validation loss", unit='img', disable=not verbose) as bar:
                for val_batch in val_loader:
                    val_loss, items = get_loss(val_batch, device, model, criterion)

                    logging.info(f"Validation loss: {val_loss}")
                    bar.set_postfix(loss=f'{val_loss.item():.5f}')

                    mean_val_loss += val_loss.detach().item()
                    bar.update(items)

                mean_val_loss = mean_val_loss / len(val_loader)
                out = f"Validation epoch loss: {mean_val_loss}"
                logging.info(out)
                print(out)

            if save_cp:
                logging.info(f"Saving model")
                try:
                    os.mkdir(checkpoint_dir)
                    logging.info('Created checkpoint directory')
                except OSError:  # Maybe FileExistsError ?
                    pass

                if keep_single_cp:
                    # Deletes old checkpoints
                    model_files = [f for f in os.listdir(checkpoint_dir) if
                                   os.path.isfile(os.path.join(checkpoint_dir, f))]
                    for model_file in model_files:
                        os.remove(f'{checkpoint_dir}{os.path.sep}{model_file}')
                torch.save(model.state_dict(), checkpoint_dir + os.path.sep + f'CP_epoch{epoch + 1}.pt')
                logging.info(f'Checkpoint {epoch + 1} saved!')

            if mean_val_loss < last_val_loss:
                logging.info(f"This model is better the last one")
                last_val_loss = mean_val_loss
                bad_val_counter = 0
            else:
                logging.info(f"This model isn't better the last one")
                bad_val_counter += 1

            if (epoch + 1) % step_eval == 0:
                eval_net(model, val_loader, device, bbox=False, verbose=True)

            if bad_val_counter > bad_val_limit and early_stop:
                logging.info(f"Train stopped because validation isn't becoming better since {bad_val_limit} epochs")
                break

    # save last epoch model
    torch.save(model.state_dict(), model_path)


def get_loss(batch, device, model, criterion):
    queries = batch['query']
    targets = batch['target']
    true_masks = batch['mask']

    queries = queries.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.float32)
    true_masks = true_masks.to(device=device, dtype=torch.float32)

    pred_masks = model(queries, targets)
    loss = criterion(pred_masks, true_masks)
    return loss, queries.shape[0]


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset',
#                         choices=ALL_DATASET_NAMES,
#                         default="FlickrLogos-32",
#                         type=str,
#                         help="Dataset in {}".format(ALL_DATASET_NAMES)
#                         )

#     parser.add_argument('--model',
#                         choices=ALL_MODEL_NAMES,
#                         default="LogoDetection",
#                         type=str,
#                         help="Model in {}".format(ALL_MODEL_NAMES)
#                         )

#     optimizers = ['Adam', 'SGD']
#     parser.add_argument('--optimizer',
#                         choices=optimizers,
#                         default='Adam',
#                         help="Optimizer in {}".format(optimizers)
#                         )

#     parser.add_argument('--max_epochs',
#                         default=500,
#                         type=int,
#                         help="Number of epochs"
#                         )

#     parser.add_argument('--valid',
#                         default=-1,
#                         type=float,
#                         help="Number of epochs before valid"
#                         )

#     parser.add_argument('--batch_size',
#                         default=32,
#                         type=int,
#                         help="Number of samples in each mini-batch in SGD and Adam optimization"
#                         )

#     parser.add_argument('--weight_decay',
#                         default=5e-4,
#                         type=float,
#                         help="L2 weight regularization of the optimizer"
#                         )

#     parser.add_argument('--learning_rate',
#                         default=4e-4,
#                         type=float,
#                         help="Learning rate of the optimizer"
#                         )

#     parser.add_argument('--label_smooth',
#                         default=0.1,
#                         type=float,
#                         help="Label smoothing for true labels"
#                         )

#     parser.add_argument('--decay1',
#                         default=0.9,
#                         type=float,
#                         help="Decay rate for the first momentum estimate in Adam"
#                         )

#     parser.add_argument('--decay2',
#                         default=0.999,
#                         type=float,
#                         help="Decay rate for second momentum estimate in Adam"
#                         )

#     parser.add_argument('--verbose',
#                         default=True,
#                         type=bool,
#                         help="Verbose"
#                         )

#     parser.add_argument('--load',
#                         type=str,
#                         required=False,
#                         help="Path to the model to load"
#                         )

#     parser.add_argument('--batch_norm',
#                         default=False,
#                         type=bool,
#                         help="If True, apply batch normalization",
#                         )

#     parser.add_argument('--vgg_cfg',
#                         type=str,
#                         default='A',
#                         help="VGG architecture config",
#                         )

#     parser.add_argument('--step_eval',
#                         type=int,
#                         default=0,
#                         help="Enables automatic evaluation checks every X step",
#                         )

#     parser.add_argument('--val_split',
#                         type=float,
#                         default=0.1,
#                         help="Forces the validation subset to be split according to the set value. Must a value in the [0-1] or the software WILL break",
#                         )

#     parser.add_argument('--save_cp',
#                         type=bool,
#                         default=True,
#                         help="If True, saves model checkpoints",
#                         )

#     return parser.parse_args()


def train_main(dataset='FlickrLogos-32',
               model='LogoDetection',
               optimizer='Adam',
               vgg_cfg='A',
               max_epochs=1,
               batch_size=4,
               weight_decay=5e-4,
               learning_rate=4e-4,
               decay1=0.9,
               decay2=0.999,
               verbose=True,
               batch_norm=True,
               load=None,
               val_split=0.1,
               step_eval=10,
               save_cp=True,
               ):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s", filename='oneshot.log',
                        filemode='w')
    # args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Modularized paths with respect to the current Dataset
    imgs_dir = config_list['datasets'][dataset]['paths']['images']
    masks_dir = config_list['datasets'][dataset]['paths']['masks']
    checkpoint_dir = config_list['models'][model]['paths']['train_cp']

    model_path = config_list['models'][model]['paths']['model'] + os.path.sep + "_".join([model, dataset]) + ".pt"

    # create checkpoint dir
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
    except FileExistsError:
        pass

    # create the model dir
    try:
        os.makedirs(config_list['models'][model]['paths']['model'], exist_ok=True)
    except FileExistsError:
        pass

    out = f"Loading {dataset} dataset"
    print(out)
    logging.info(f"Loading {dataset} dataset")
    dataset = BasicDataset(imgs_dir=imgs_dir, masks_dir=masks_dir, dataset_name=dataset)

    # Splitting dataset
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # Loading dataset
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                            drop_last=False)

    # Change here to adapt your data
    out = "Initializing model"
    print(out)
    logging.info(out)
    model = LogoDetection(batch_norm=batch_norm, vgg_cfg=vgg_cfg)

    # Optimizer selection
    # build all the supported optimizers using the passed params (learning rate and decays if Adam)
    supported_optimizers = {
        'Adam': optim.Adam(params=model.parameters(), lr=learning_rate, betas=(decay1, decay2),
                           weight_decay=weight_decay),
        'SGD': optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    }
    # Choose which Torch Optimizer object to use, based on the passed name
    optimizer = supported_optimizers[optimizer]

    if load:
        model.load_state_dict(
            torch.load(load, map_location=device)
        )
        logging.info(f'Model loaded from {load}')
    model.to(device=device)

    try:
        train(model=model,
              device=device,
              train_loader=train_loader,
              val_loader=val_loader,
              max_epochs=max_epochs,
              optimizer=optimizer,
              verbose=verbose,
              checkpoint_dir=checkpoint_dir,
              model_path=model_path,
              save_cp=save_cp,
              n_train=n_train,
              step_eval=step_eval,
              n_val=n_val,
              early_stop=False,
              bad_val_limit=100,
              keep_single_cp=False
              )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), config_list['models'][model]['paths']['model'] + os.path.sep + 'INTERRUPTED.pt')
        logging.info('Interrupt saved')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
