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
from eval import eval

# todo: when we add more models, we should move these variables to another location
MODEL_HOME = os.path.abspath("./stored_models/")
ALL_MODEL_NAMES = ["LogoDetection"]
ALL_DATASET_NAMES = ["FlickrLogos-32", "TopLogos-10"]

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
          ):

    batch_size = train_loader.batch_size

    # Logging for TensorBoard
    writer = SummaryWriter(comment=f'LR__BS_{batch_size}_OPT_{type(optimizer).__name__}')  # does optimizer.lr work? we're gonne find out
    global_step = 0

    ### ERROR: n_train e n_val? ###
    logging.info(f'''Starting training:
        Epochs:             {max_epochs}
        Batch size:         {batch_size}
        Learning rate:      
        Training size:      {n_train}
        Validation size:    {n_val}
        Device:             {device.type}
    ''')

    def criterion(pred, true):
        return torch.div(nn.BCELoss()(pred, true), 256 * 256)  # L = (1/(H*W)) * BCELoss
        # TypeError: unsupported operand type(s) for /: 'BCELoss' and 'int'

    last_epoch_val_score = 0
    for epoch in range(max_epochs):
        model.train()  # set the model in training flag to True
        epoch_loss = 0  # resets the loss for the current epoch
        # epoch(batch_size, train_samples)

        # TODO
        with tqdm.tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{max_epochs}', unit='img', disable=not verbose) as bar:
            bar.set_description(f'train loss')

            for batch in train_loader:
                queries = batch['query']  # Correct dimensions?
                targets = batch['target']
                true_masks = batch['mask']

                queries = queries.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                pred_masks = model(queries, targets)
                # print(pred_masks.shape)
                loss = criterion(pred_masks, true_masks)
                epoch_loss += loss.detach().item()  # is the .detach() needed?

                # TensorBoard logging
                writer.add_scalar('Loss/train', loss.item(), global_step)

                bar.set_postfix(loss=f'{loss.item():.5f}')

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1) Gradient Clipping
                optimizer.step()

                bar.update(queries.shape[0])
                global_step += 1

                # if n_train % batch_size == 0: 
                #     n_batch = n_train // batch_size
                # else:
                #     n_batch = n_train // batch_size + 1

                ### DOMANDA: Dove lo volevamo usare? ###
                ### ALTRA DOMANDA: Non conviene farlo fuori dai cicli? ###
                n_batch = len(train_loader)
                # Deve farlo sia in mezzo ai batch che a fine epoca. Modifica la condizione dell'if
                if global_step % (n_train // (10 * batch_size)) == 0 or global_step == n_batch:
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.cpu().numpy(), global_step)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    writer.add_images('query_images', queries, global_step)
                    writer.add_images('target_images', targets, global_step)
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/pred', pred_masks, global_step)

            # TODO: Se save_cp è false e non viene cambiato il val_split di default non va in train il 10% del dataset. Si potrebbe fare in modo che non sia così
            if save_cp:
                val_score = eval(model, val_loader, device, bbox=False, verbose=True)
                if val_score > last_epoch_val_score:
                    try:
                        os.mkdir()
                        logging.info('Created checkpoint directory')
                    except OSError: # Maybe FileExistsError ?
                        pass
                    model_files = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
                    torch.save(model.state_dict()
                    ,
                            checkpoint_dir + f'CP_epoch{epoch + 1}.pt')
                    for model_file in model_files:
                        os.remove(f'{checkpoint_dir}{model_file}')
                    logging.info(f'Checkpoint {epoch + 1} saved!')
                    last_epoch_val_score = val_score

    writer.close()
    torch.save(model.state_dict(), model_path)

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


# print("\nEvaluating model...")
# model.eval()
# mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=False)
# print("\tTest Hits@1: %f" % h1)
# print("\tTest Mean Reciprocal Rank: %f" % mrr)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=ALL_DATASET_NAMES,
                        default="FlickrLogos-32",
                        type=str,
                        help="Dataset in {}".format(ALL_DATASET_NAMES)
                        )

    parser.add_argument('--model',
                        choices=ALL_MODEL_NAMES,
                        default="LogoDetection",
                        type=str,
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
                        type=str,
                        required=False,
                        help="Path to the model to load"
                        )

    parser.add_argument('--batch_norm',
                        default=False,
                        type=bool,
                        help="If True, apply batch normalization",
                        )

    parser.add_argument('--vgg_cfg',
                        type=str,
                        default='A',
                        help="VGG architecture config",
                        )

    parser.add_argument('--step_eval',
                        type=int,
                        default=0,
                        help="Enables automatic evaluation checks every X step",
                        )

    parser.add_argument('--val_split',
                        type=float,
                        default=0.1,
                        help="Forces the validation subset to be split according to the set value. Must a value in the [0-1] or the sofware WILL break",
                        )

    parser.add_argument('--save_cp',
                        type=bool,
                        default=True,
                        help="If True, saves model checkponts",
                        )

    return parser.parse_args()


if __name__ == '__main__':
    # TODO: Add filename
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Modularized paths with respect to the current Dataset
    imgs_dir = config_list['datasets'][args.dataset]['paths']['images']
    masks_dir = config_list['datasets'][args.dataset]['paths']['masks']
    checkpoint_dir = config_list['models'][args.model]['paths']['train_cp']

    model_path = config_list['models'][args.model]['paths']['model']+ "_".join([args.model, args.dataset]) + ".pt"

    print("Loading %s dataset..." % args.dataset)
    # you can delete this "save_to_disk" to preserve the ssd :like:
    dataset = BasicDataset(imgs_dir=imgs_dir, masks_dir=masks_dir, dataset_name=args.dataset)

    # Splitting dataset
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    # TODO: Il validation set dovrebbe avere il 10% di ogni classe e non il 10% del totale altrimenti verrebbe sbilanciato
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # Loading dataset
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                            drop_last=True)

    # Change here to adapt your data
    print("Initializing model...")
    model = LogoDetection(batch_norm=args.batch_norm,
                          vgg_cfg=args.vgg_cfg)

    # Optimizer selection
    # build all the supported optimizers using the passed params (learning rate and decays if Adam)
    supported_optimizers = {
        'Adam': optim.Adam(params=model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2),
                           weight_decay=args.weight_decay),
        'SGD': optim.SGD(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    }
    # Choose which Torch Optimizer object to use, based on the passed name
    optimizer = supported_optimizers[args.optimizer]

    # stiamo dando ad "args.load" due compiti, quello di dirci il path e quello di dirci se caricare vecchi checkpoint
    if args.load is not None:
        model.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    model.to(device=device)

    try:
        train(model=model,
              device=device,
              train_loader=train_loader,
              val_loader=val_loader,
              max_epochs=args.max_epochs,
              optimizer=optimizer,
              verbose=args.verbose,
              checkpoint_dir=checkpoint_dir,
              model_path=model_path,
              save_cp=args.save_cp,
              )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.ph')
        logging.info('Interrupt saved')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
