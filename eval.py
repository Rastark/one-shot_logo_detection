import functools, operator, collections
import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import average_precision_score as avg_pr
from sklearn.cluster import DBSCAN
from sklearn import metrics

import matplotlib.pyplot as plt

from tqdm import tqdm

from skimage.measure import label, regionprops


def eval(model,
         loader, 
         device, 
         bbox: bool, 
         verbose: bool,
         iou_thr: int = 0.5
         ):
    model.eval()

    # Number of batches
    n_val = len(loader)
    
    matches = 0

    # Number of bboxes
    max_matches = 0

    if bbox:
        truth_type = "bbox"
    else:
        truth_type = "mask"

    precisions, recalls, accuracies = [], [], []

    batch_results = []

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False, disable=not verbose) as bar:
        for batch in loader:
            queries, targets, truth = batch['query'], batch['target'], batch[truth_type]
            queries = queries.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            truth = truth.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred_masks = model(queries, targets)
                # assunzione: gli indici della true e pred masks sono gli stessi
                for mask_index in range(pred_masks.shape[0]):
                    pred_mask = np.asarray(pred_masks[mask_index])
                    pred_mask = masks_as_image(rle_encode(pred_mask))
                    
                    # mask labeling and conversion to bboxes
                    pred_labels = label(pred_mask)
                    pred_bboxes_coords = np.array(list(map(lambda x: x.bbox, regionprops(pred_labels))))
                    pred_bboxes = calc_bboxes_from_coords(pred_bboxes_coords)

                    # computes truth bboxes in the same way as the pred
                    if bbox:
                        truth_bboxes = np.array(truth[mask_index])
                    else:
                        truth_mask = np.asarray(truth[mask_index])
                        truth_mask = masks_as_image(rle_encode(truth_mask))
                        true_mask_labels = label(truth_mask)
                        truth_bboxes_coords = np.array(list(map(lambda x: x.bbox, regionprops(true_mask_labels))))
                        truth_bboxes = calc_bboxes_from_coords(truth_bboxes_coords)

                    max_matches += len(truth_bboxes)
                    
                    b_result = get_pred_results(truth_bboxes, pred_bboxes, iou_thr)
                    batch_results.append(b_result)

    # Should not be here, since the eval method is used in both validation and test -> TODO: better handling of the flag.
    model.train()
    
    result = dict(functools.reduce(operator.add, map(collections.Counter, batch_results)))
    print("Validation output: ", str(result))
    true_pos = result['true_pos']
    false_pos = result['false_pos']
    false_neg = result['false_neg']

    precision = calc_precision(true_pos, false_pos)
    recall = calc_recall(true_pos, false_neg)
    accuracy = calc_accuracy(true_pos, false_pos, false_neg)
    print(f"Precision: {precision}    Recall: {recall}    Accuracy: {accuracy}")

    return accuracy


def calc_bboxes_from_coords(bboxes_coords):
    """Calculate all bounding boxes from a set of bounding boxes coordinates"""
    bboxes = []
    for coord_idx in range(len(bboxes_coords)):
        coord = bboxes_coords[coord_idx]
        bbox = (coord[1], coord[0], int(coord[4])-int(coord[1]), int(coord[3])-int(coord[0]))
        bboxes.append(bbox)
    return bboxes


def get_pred_results(truth_bboxes, pred_bboxes, iou_thr = 0.5):
    """Calculates true_pos, false_pos and false_neg from the input bounding boxes. """
    n_pred_idxs = range(len(pred_bboxes))
    n_truth_idxs = range(len(truth_bboxes))
    if len(n_pred_idxs) == 0:
        tp = 0
        fp = 0
        fn = len(truth_bboxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(n_truth_idxs) == 0:
        tp = 0
        fp = len(pred_bboxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    truth_idx_thr = []
    pred_idx_thr = []
    ious = []
    for pred_idx, pred_bbox in enumerate(pred_bboxes):
        for truth_idx, truth_bbox in enumerate(truth_bboxes):
            iou = get_jaccard(pred_bbox, truth_bbox)
            if iou > iou_thr:
                truth_idx_thr.append(truth_idx)
                pred_idx_thr.append(pred_idx)
                ious.append(iou)
    # ::-1 reverses the list
    ious_desc = np.argsort(ious)[::-1]
    if len(ious_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_bboxes)
        fn = len(truth_bboxes)
    else:
        truth_match_idxes = []
        pred_match_idxes = []
        for idx in ious_desc:
            truth_idx = truth_idx_thr[idx]
            pred_idx = pred_idx_thr[idx]
            # If the bboxes are unmatched, add them to matches
            if (truth_idx not in truth_match_idxes) and (pred_idx not in pred_match_idxes):
                truth_match_idxes.append(truth_idx)
                pred_match_idxes.append(pred_match_idxes)
        tp = len(truth_match_idxes)
        fp = len(pred_bboxes) - len(pred_match_idxes)
        fn = len(truth_bboxes) - len(truth_match_idxes)
    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def calc_precision(true_pos, false_neg):
    try:
        precision = true_pos / (true_pos + false_neg)
    except ZeroDivisionError:
        precision = 0.0
    return precision


def calc_recall(true_pos, false_pos):
    try:
        recall = true_pos / (true_pos + false_pos)
    except ZeroDivisionError:
        recall = 0.0
    return recall


def calc_accuracy(true_pos, false_pos, false_neg):
    try:
        accuracy = true_pos / (true_pos + false_pos + false_neg)
    except ZeroDivisionError:
        accuracy = 0.0
    return accuracy 


def calc_mavg_precision(precision_array):
    return 


def get_jaccard(pred_bbox, truth_bbox):
    pred_mask = get_mask_from_bbox(pred_bbox)
    truth_mask = get_mask_from_bbox(truth_bbox)
    return get_jaccard_from_mask(pred_mask, truth_mask)


def get_jaccard_from_mask(pred_masks, truth):
    return jsc(pred_masks, truth)


def get_mask_from_bbox(bbox):
    mask = np.zeros((256, 256))
    x = bbox[0]
    y = bbox[1]
    for width in range(bbox[2] + 1):
        for height in range(bbox[3] + 1):
            mask[x + width, y + height] = 1
    return mask


def get_bbox_batch(img):
    bbox = np.empty((img.shape[0], 4))
    for i in range(img.shape[0]):
        bbox[i] = get_bbox(img[i])
    return bbox


def get_bbox(img):
    # img.shape = [batch_size, 1, 256, 256]
    rows = np.any(img, axis=-1)
    cols = np.any(img, axis=-2)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # X, Y, Width, Height
    return [cmin, rmin, cmax-cmin, rmax-rmin]


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels_old = img.T.flatten()
    pixels = img.T.flatten()
    for x in range(len(pixels_old)):
        if pixels_old[x] > 0.5:
            pixels[x] = 1
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(256, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list, all_masks=None):
    """
    Take the complete rle_encoded mask and create a mask array of the single masks
    """
    if all_masks is None:
        all_masks = np.zeros((256, 256), dtype=np.int16)
    # if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)