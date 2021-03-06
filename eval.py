import torch
import numpy as np
from sklearn.metrics import jaccard_score as jsc
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from skimage.measure import label, regionprops


def eval_net(model,
             loader,
             device,
             bbox: bool,
             verbose: bool,
             iou_thr: int = 0.5,
             coco_map: bool = True
             ):
    model.eval()
    logging.info("Validating")

    # Number of batches
    n_val = len(loader)

    # Number of bboxes
    max_matches = 0

    if bbox:
        truth_type = "bbox"
    else:
        truth_type = "mask"

    batch_results = []

    with tqdm(total=n_val, desc='Validation round', unit='samples', disable=not verbose) as bar:
        for batch in loader:
            queries, targets, truth = batch['query'], batch['target'], batch[truth_type]
            queries = queries.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred = model(queries, targets)
                pred_masks = pred.cpu().numpy()
                for mask_index in range(pred_masks.shape[0]):
                    pred_mask = np.asarray(pred_masks[mask_index])
                    pred_mask = masks_as_image([rle_encode(pred_mask)])

                    # mask labeling and conversion to bboxes
                    pred_labels = label(pred_mask)
                    pred_bboxes_coords = list(map(lambda x: x.bbox, regionprops(pred_labels)))
                    pred_bboxes = calc_bboxes_from_coords(pred_bboxes_coords)

                    # computes truth bboxes in the same way as the pred
                    if bbox:
                        truth_bboxes = np.array(truth[mask_index])
                    else:
                        truth_mask = np.asarray(truth[mask_index])
                        truth_mask = masks_as_image([rle_encode(truth_mask)])
                        true_mask_labels = label(truth_mask)
                        truth_bboxes_coords = list(map(lambda x: x.bbox, regionprops(true_mask_labels)))
                        truth_bboxes = calc_bboxes_from_coords(truth_bboxes_coords)
                        print("truth_bboxes: ", truth_bboxes)

                    max_matches += len(truth_bboxes)

                    logging.info(f"pred_bboxes: {pred_bboxes}")
                    logging.info(f"truth_bboxes: {truth_bboxes}")

                    if coco_map is True:
                        b_result = get_pred_results_thresholds(truth_bboxes, pred_bboxes)
                    else:
                        b_result = get_pred_results(truth_bboxes, pred_bboxes, iou_thr)
                    logging.info(f"b_result: {b_result}")
                    batch_results.append(b_result)
                bar.update(queries.shape[0])

    # Should not be here, since the eval method is used in both validation and test -> TODO: better handling of the flag.
    model.train()

    results = sum(batch_results, 0)
    logging.info(f"results: {results}")
    n_thr = results.shape[0]
    try:
        true_pos_list = results[:, 0]
    except:
        for i in range(n_thr):
            true_pos_list.append(0)
    try:
        false_pos_list = results[:, 1]
    except:
        for i in range(n_thr):
            false_pos_list.append(0)
    try:
        false_neg_list = results[:, 2]
    except:
        for i in range(n_thr):
            false_neg_list.append(0)
    precisions, recalls, accuracies = precision_recall_curve(true_pos_list, false_pos_list, false_neg_list)
    output = f"Precisions: {precisions}    Recalls: {recalls}    Accuracies: {accuracies}"
    print(output)
    logging.info(output)

    # P-R curve
    plt.plot(recalls, precisions, linewidth=4, color="red")
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.show()

    precisions.append(1)
    recalls.append(0)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    # accuracies = np.array(accuracies)

    AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    logging.info(f'Avg Precision: {AP}')
    print(AP)

    return AP


def calc_bboxes_from_coords(bboxes_coords):
    """Calculate all bounding boxes from a set of bounding boxes coordinates"""
    bboxes = []
    for coord_idx in range(len(bboxes_coords)):
        coord = bboxes_coords[coord_idx]
        bbox = (coord[1], coord[0], int(coord[4]) - int(coord[1]), int(coord[3]) - int(coord[0]))
        bboxes.append(bbox)
    return bboxes


def get_pred_results(truth_bboxes, pred_bboxes, iou_thr=0.5):
    """Calculates true_pos, false_pos and false_neg from the input bounding boxes. """
    n_pred_idxs = range(len(pred_bboxes))
    n_truth_idxs = range(len(truth_bboxes))
    if len(n_pred_idxs) == 0:
        tp = 0
        fp = 0
        fn = len(truth_bboxes)
        return tp, fp, fn
    if len(n_truth_idxs) == 0:
        tp = 0
        fp = len(pred_bboxes)
        fn = 0
        return tp, fp, fn

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
    return tp, fp, fn


def calc_precision(true_pos, false_pos):
    try:
        precision = true_pos / (true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    return precision


def calc_recall(true_pos, false_neg):
    try:
        recall = true_pos / (true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0
    return recall


def calc_accuracy(true_pos, false_pos, false_neg):
    try:
        accuracy = true_pos / (true_pos + false_pos + false_neg)
    except ZeroDivisionError:
        accuracy = 0.0
    return accuracy


def get_pred_results_thresholds(y_true, pred_scores, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(start=0.50, stop=0.95, step=0.05)
    output = []
    for threshold in thresholds:
        true_pos, false_pos, false_neg = get_pred_results(truth_bboxes=y_true, pred_bboxes=pred_scores,
                                                          iou_thr=threshold)
        output.append((true_pos, false_pos, false_neg))
    return np.array(output)


def precision_recall_curve(true_pos_list, false_pos_list, false_neg_list):
    precisions, recalls, accuracies = [], [], []
    for true_pos, false_pos, false_neg in zip(true_pos_list, false_pos_list, false_neg_list):
        precisions.append(calc_precision(true_pos, false_pos))
        recalls.append(calc_recall(true_pos, false_neg))
        accuracies.append(calc_accuracy(true_pos, false_pos, false_neg))
    return precisions, recalls, accuracies


def get_jaccard(pred_bbox, truth_bbox):
    pred_mask = get_mask_from_bbox(pred_bbox)
    truth_mask = get_mask_from_bbox(truth_bbox)
    return jsc(y_true=truth_mask, y_pred=pred_mask, average='micro')


def get_mask_from_bbox(bboxes):
    mask = np.zeros((256, 256))
    if type(bboxes) is list:
        for bbox in bboxes:
            x = bbox[0]
            y = bbox[1]
            for width in range(bbox[2]):
                for height in range(bbox[3]):
                    mask[(int(y) + int(height)), (int(x) + int(width))] = 1
    else:
        bbox = bboxes
        x = bbox[0]
        y = bbox[1]
        for width in range(bbox[2]):
            for height in range(bbox[3]):
                mask[int(x) + int(width), int(y) + int(height)] = 1
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
    return [cmin, rmin, cmax - cmin, rmax - rmin]


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    #     logging.info(f"pred_img: {img}")
    pixels_old = img.T.flatten()
    pixels = img.T.flatten()
    for x in range(len(pixels_old)):
        if pixels_old[x] > 0.5:
            pixels[x] = 1
        else:
            pixels[x] = 0
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
