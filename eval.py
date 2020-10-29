import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import jaccard_score as jsc

from tqdm import tqdm

def eval(model, loader, device):
    model.eval()
    n_val = len(loader) # Number of batches
    matches = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as bar:
        for batch in loader:
            queries, targets, true_masks = batch['query'], batch['target'], batch['mask']
            
            queries = queries.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred_masks = model(queries, targets)
            
    pred_bbox = get_bbox_batch(pred_masks.numpy())
    true_bbox = get_bbox_batch(true_masks.numpy())
    iou = jsc(pred_bbox, true_bbox)
    if (iou > 0.5):
        matches += 1

    model.train()
    return matches/n_val

def get_bbox_batch(img):
    bbox = np.empty([img.shape[0], 4])
    for i in range(img.shape[0]):
        bbox[i] = get_bbox(img[i])
    return bbox


def get_bbox(img):
    print(img)
    a = np.where(img != 0)
    bbox = np.min(a[-2]), np.max(a[-2]), np.min(a[-1]), np.max(a[-1])
    return bbox


# def get_bbox(img):
#     # img.shape = [batch_size, 1, 256, 256]
#     rows = np.any(img, axis=-1)
#     cols = np.any(img, axis=-2)
#     rmin, rmax = np.where(rows)[0][[0, -1]]
#     cmin, cmax = np.where(cols)[0][[0, -1]]
#     # X, Y, Width, Height
#     return [cmin, rmin, cmax-cmin, rmax-rmin]
