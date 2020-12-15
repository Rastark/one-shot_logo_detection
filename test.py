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
ALL_MODEL_NAMES = ["LogoDetection"]
ALL_DATASET_NAMES = ["FlickrLogos-32, TopLogos-10"]

with open(os.path.abspath("./config/config.yaml")) as config:
    config_list = yaml.load(config, Loader=yaml.FullLoader)

# def pred(model,
#          sample,
#          device,
#          threshold=0.5):
#     if(model.eval==False):
#         model.eval()

#     queries, targets = torch.from_numpy(BasicDataset.preprocess(index=index, file:files_path))
#     queries = queries.unsqueeze(0)
#     queries = queries.to(device=device, dtype=torch.float32)
#     targets = targets.unsqueeze(0)
#     targets = targets.to(device=device, dtype=torch.float32)


def test(model,
         device,
         dataset,
         batch_size,
        #   save_path,
         verbose: bool,
         threshold=0.5):
    
    model.eval()

    # #TODO dataset preprocessing
    # with torch.no_grad():
    #     output = model(queries, targets)

    #     probs = output.squeeze(0)
    
    logging.info("\nPredicting image{} ... ")

    with tqdm.tqdm(total=len(dataset), desc=f'Testing dataset', unit='test-img', disable=not verbose) as bar:
            bar.set_description(f'model testing')

            for batch in test_loader:
                queries = batch['query']  # Correct dimensions?
                targets = batch['target']
                bboxes = batch['bbox']

                queries = queries.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    pred_masks = model(queries, targets)
                    # print(pred_masks.shape)

                bar.update(queries.shape[0])
                global_step += 1
                val_score += eval(model, batch, device)
                # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                writer.add_images('query_images', queries, global_step)
                writer.add_images('target_images', targets, global_step)
                # writer.add_images('bboxes/true', bboxes, global_step)
                writer.add_images('masks/pred', pred_masks, global_step)
                writer.close()
    return val_score / global_step


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
                        required=True,
                        help="Path to the model to load"
                        )

    return parser.parse_args()


if __name__ == '__main__':
    # Logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Modularized paths with respect to the current Dataset
    imgs_dir = config_list['datasets'][args.dataset]['images']
    masks_dir = config_list['datasets'][args.dataset]['masks']
    # TODO: Controlla che ste due liste hanno le stesse sottocartelle
    imgs_classes = [f.name for f in os.scandir(imgs_dir) if f.is_dir()]
    mask_classes = [f.name for f in os.scandir(masks_dir) if f.is_dir()]
    # checkpoint_dir = config_list['models'][args.model]['train_cp']

    model_path = config_list['models'][args.model]['paths']['model']+ "_".join([args.model, args.dataset]) + ".pt"

    # print("Loading %s dataset..." % args.dataset)
    # dataset = BasicDataset(imgs_dir=imgs_dir, masks_dir=masks_dir)

    # Change here to adapt your data
    print("Initializing model...")
    model = LogoDetection(batch_norm=args.batch_norm,
                        vgg_cfg=args.vgg_cfg)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    logging.info(f'Model loaded from {model_path}')
    model.to(device=device)

    # Neo, enter in Metrics
    metrics = []

    for img_class_idx, img_class_path in enumerate(imgs_classes):
        dataset = BasicDataset(imgs_dir=f"{imgs_dir}{os.path.sep}{img_class_path}", masks_dir=f"{masks_dir}{os.path.sep}{masks_dir[img_class_idx]}", dataset_name=args.dataset, skip_bbox_lines=1)
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        try:
            metrics.append(test(model=model,
                                device=device,
                                dataset=test_loader,
                                batch_size=args.batch_size,
                                verbose=args.verbose
                                ))
        except KeyboardInterrupt:
            # torch.save(model.state_dict(), 'INTERRUPTED.ph')
            # logging.info('Interrupt saved')
            logging.info("Test interrupted")
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
