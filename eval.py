import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import jaccard_score as jsc
from sklearn.cluster import DBSCAN
from sklearn import metrics

import matplotlib.pyplot as plt

from tqdm import tqdm

from skimage.measure import label, regionprops


def eval(model, loader, device):
    model.eval()

    # Number of batches
    n_val = len(loader)
    
    matches = 0

    # Number of bboxes
    max_matches = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as bar:
        for batch in loader:
            queries, targets, true_masks = batch['query'], batch['target'], batch['mask']

            queries = queries.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred_masks = model(queries, targets)
                # assunzione: gli indici della true e pred masks sono gli stessi
                for mask_index in range(len(pred_masks)):
                    # Sarà scritto bene? Da verificare mentre gli fondiamo la titan
                    pred_mask = np.asarray(pred_masks[mask_index])
                    pred_mask = masks_as_image(rle_encode(pred_mask))
                    true_mask = np.asarray(true_masks[mask_index])
                    true_mask = masks_as_image(rle_encode(true_mask))

                    pred_mask_labels = label(pred_mask)
                    pred_mask_props = regionprops(pred_mask_labels)

                    true_mask_labels = label(true_mask)
                    true_mask_props = regionprops(true_mask_labels)

                    max_matches += len(true_mask_props)

                    # ious = np.array([])
                    ious = []
                    for pred_prop_index in range(len(pred_mask_props)):
                        for true_prop_index in range(len(true_mask_props)):
                            pred_bbox = pred_mask_props[pred_prop_index].bbox
                            fixed_pred_bbox = (pred_bbox[1], pred_bbox[0], int(pred_bbox[4])-int(pred_bbox[1]), int(pred_bbox[3])-int(pred_bbox[0]))
                            true_bbox = true_mask_props[true_prop_index].bbox
                            fixed_true_bbox = (true_bbox[1], true_bbox[0], int(true_bbox[4])-int(true_bbox[1]), int(true_bbox[3])-int(true_bbox[0]))

                            # ious = np.append((pred_bbox, true_bbox, iou()))
                            # ious[true_bbox] = ious[pred_bbox].append(iou())
                            ious.append((f"p{pred_prop_index}", f"t{true_prop_index}", get_jaccard(pred_bbox, true_bbox)))
                
                    ious.sort(key=lambda x: x[2])
                    used_bboxes = []
                    best_ious = []

                    # custom_index = -1
                    for index in range(len(true_mask_props)):
                        # if custom_index >= index:
                        #     continue
                        # elif custom_index < index:
                        #     custom_index = index
                        if f"p{ious[index][0]}" not in used_bboxes and f"t{ious[index][1]}" not in used_bboxes:
                            best_ious.append(ious[index][2])
                            used_bboxes.append(f"p{ious[index][0]}")
                            used_bboxes.append(f"t{ious[index][1]}")
                        else:
                            continue
                            # custom_index = custom_index + 1
                            # while custom_index <= len(true_mask_props):
                            #     if f"p{ious[custom_index][0]}" not in used_bboxes and f"t{ious[custom_index][1]}" not in used_bboxes:
                            #         best_ious.append(ious[custom_index])
                            #         used_bboxes.append(f"p{ious[custom_index][0]}")
                            #         used_bboxes.append(f"t{ious[custom_index][1]}")
                            #         break
                            #     custom_index += 1
                        # for _ in range(len(ious)):
                        #     best_iou = max(ious, key=lambda x:x[2])
                        #     if f"p{best_iou[0]}" not in used_bboxes and f"t{best_iou[1]}" not in used_bboxes:
                        #         best_ious.append(best_iou)
                        #         used_bboxes.append(f"p{best_iou[0]}")
                        #         used_bboxes.append(f"t{best_iou[1]}")
                        #     else:
                        #         continue

                    # for prop in pred_mask_props:
                        # do something
                        # IoU between pred and true mask


            # iou = get_jaccard(pred_masks[0].numpy(), true_masks[0].numpy())
    for iou in best_ious:
        if (iou > 0.5):
            matches += 1

    model.train()
    return matches / max_matches

# def get_clusters(mask):
#     data = DBSCAN(eps=0.3, min_samples=2).fit(mask)
#     core_samples_mask = np.zeros_like(data.labels_, dtype=bool)
#     core_samples_mask[data.core_sample_indices_] = True
#     labels = data.labels_

#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)

#     print('Estimated number of clusters: %d' % n_clusters_)
#     print('Estimated number of noise points: %d' % n_noise_)
#     # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#     # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#     # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#     # print("Adjusted Rand Index: %0.3f"
#     #     % metrics.adjusted_rand_score(labels_true, labels))
#     # print("Adjusted Mutual Information: %0.3f"
#     #     % metrics.adjusted_mutual_info_score(labels_true, labels))
#     print("Silhouette Coefficient: %0.3f"
#         % metrics.silhouette_score(mask, labels))

#     # Plot result
#     # Black removed and is used for noise instead.
#     unique_labels = set(labels)
#     colors = [plt.cm.Spectral(each)
#             for each in np.linspace(0, 1, len(unique_labels))]
#     for k, col in zip(unique_labels, colors):
#         if k == -1:
#             # Black used for noise.
#             col = [0, 0, 0, 1]

#         class_member_mask = (labels == k)

#         xy = mask[class_member_mask & core_samples_mask]
#         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                 markeredgecolor='k', markersize=14)

#         xy = mask[class_member_mask & ~core_samples_mask]
#         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                 markeredgecolor='k', markersize=6)

#     plt.title('Estimated number of clusters: %d' % n_clusters_)
#     plt.show()    


# n_noise_ = list(labels).count(-1)

# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))

# #############################################################################
# # Plot result
# import matplotlib.pyplot as plt

# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]

#     class_member_mask = (labels == k)

#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)

#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()



def get_jaccard(pred_bbox, true_bbox):
    # return get_jaccard_from_mask(pred_masks, true_masks)
    return get_jaccard_from_bboxes(pred_bbox, true_bbox)


def get_jaccard_from_bboxes(pred_bbox, true_bbox):
    # pred_bbox = get_bbox_batch(pred_masks[0].numpy())
    # true_bbox = get_bbox_batch(true_masks[0].numpy())
    pred_mask = get_mask_from_bbox(pred_bbox)
    true_mask = get_mask_from_bbox(true_bbox)
    return get_jaccard_from_mask(pred_mask, true_mask)


def get_jaccard_from_mask(pred_masks, true_masks):
    return jsc(pred_masks, true_masks)


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


# def get_bbox(img):
#     print(img)
#     a = np.where(img != 0)
#     bbox = np.min(a[-2]), np.max(a[-2]), np.min(a[-1]), np.max(a[-1])
#     return bbox

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

def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((256, 256), dtype=np.int16)
    # if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)