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


# todo: when we add more models, we should move these variables to another location
MODEL_HOME = os.path.abspath("./stored_models/")
ALL_MODEL_NAMES = ["LogoDetection"]
ALL_DATASET_NAMES = ["FlickrLogos-32"]

with open(os.path.abspath("./config/config.yaml")) as config:
    config_list = yaml.load(config, Loader=yaml.FullLoader)

def pred(model,
         sample,
         device,
         threshold=0.5):
    if(model.eval==False):
        model.eval()

    queries, targets = torch.from_numpy(BasicDataset.preprocess(index=index, file:files_path))
    queries = queries.unsqueeze(0)
    queries = queries.to(device=device, dtype=torch.float32)
    targets = targets.unsqueeze(0)
    targets = targets.to(device=device, dtype=torch.float32)


def test(model,
         device,
         dataset,
         batch_size,
        #   save_path,
         verbose,
         threshold=0.5):
    model.eval()

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    #TODO dataset preprocessing
    with torch.no_grad():
        output = model(queries, targets)

        probs = output.squeeze(0)
    
    logging.info("\nPredicting image{} ... ")

    with tqdm.tqdm(total=len(dataset), desc=f'Testing dataset', unit='img') as bar:
            bar.set_description(f'model test')

            for batch in test_loader:
                queries = batch['query']  # Correct dimensions?
                targets = batch['target']
                true_masks = batch['mask']

                queries = queries.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                pred_masks = model(queries, targets)
                # print(pred_masks.shape)

                bar.update(queries.shape[0])
                global_step += 1
                if len(dataset) % batch_size == 0: 
                    n_batch = n_train // batch_size
                else:
                    n_batch = n_train // batch_size + 1
                len(train_loader)
                # Deve farlo sia in mezzo ai batch che a fine epoca. Modifica la condizione dell'if
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.cpu().numpy(), global_step)
                    val_score = eval(model, val_loader, device)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    writer.add_images('query_images', queries, global_step)
                    writer.add_images('target_images', targets, global_step)
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/pred', pred_masks, global_step)


    return None


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

    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help="Number of samples in each mini-batch in SGD and Adam optimization"
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

    return parser.parse_args()


    if __name__ == '__main__':
      logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
      args = get_args()
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      logging.info(f'Using device {device}')

    # TODO Modularize paths with respect to the current Dataset
    imgs_dir = os.path.abspath("data/dataset/FlickrLogos-v2/classes/jpg")
    masks_dir = os.path.abspath("data/dataset/FlickrLogos-v2/classes/masks")
    checkpoint_dir = os.path.abspath("checkpoints")

    model_path = config_list['models']['LogoDetection']['path'] + "_".join(["LogoDetection", args.dataset]) + ".pt"

    # print("Loading %s dataset..." % args.dataset)
    # dataset = BasicDataset(imgs_dir=imgs_dir, masks_dir=masks_dir)

    # Change here to adapt your data
    print("Initializing model...")
    model = LogoDetection(batch_norm=args.batch_norm,
                          vgg_cfg=args.vgg_cfg)
    # stiamo dando ad "args.load" due compiti, quello di dirci il path e quello di dirci se caricare vecchi checkpoint
    if args.load is not None:
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        logging.info(f'Model loaded from {model_path}')
    model.to(device=device)

    try:
        test(model=args.model,
             device=device,
             dataset=args.dataset,
             batch_size=args.batch_size,
             verbose=args.verbose
            )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.ph')
        logging.info('Interrupt saved')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
