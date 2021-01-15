import logging
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Path config
rootpath = os.path.abspath('../')
log_path = os.path.join(rootpath, 'logs/test/')
log_file = os.path.join(log_path, 'oneshot_test(4).log')

regex_pred = 'pred_bboxes: \[[0-9]*\.[0-9]*\]'
regex_truth = 'truth_bboxes: \[[0-9]*\.[0-9]*\]'
regex_avgp = 'Avg Precision: [0-9]*\.[0-9]*'
regex_epoch = 'Model loaded: ./data/stored/models/checkpoints/'
n_class_samples = 70

# match_list = []
all_preds = []
all_truths = []
all_avgp = []

with open(log_file, 'r') as log:
    n_classes = 0
    n_samples = 1
    for line in log:
        for match in re.finditer(regex_pred, line, re.S):
            match_text = match.group()
            print(match_text)
            # match_list.append(match_text)
            pred_bboxes = match_text.split(' ')[-1]
            all_preds.append(pred_bboxes)
            logging.info(f'pred_bboxes: sample {n_samples}: {pred_bboxes}')
        for match in re.finditer(regex_truth, line, re.S):
            match_text = match.group()
            print(match_text)
            # match_list.append(match_text)
            truth_bboxes = match_text.split(' ')[-1]
            all_truths.append(truth_bboxes)
            logging.info(f'truth_bboxes: sample {n_samples}: {pred_bboxes}')
            n_saples += 1
        for match in re.finditer(regex_avgp, line, re.S):
            n_classes += 1
            match_text = match.group()
            print(match_text)
            avgp = match_text.split(' ')[-1]
            all_avgp.append(float(avgp))
            logging.info(f'AP: class {n_classes}: {avgp}')

mAP = sum(all_avgp)/10

print(f'Preds: {all_preds}\nTruths: {all_truths}\nAP: {all_avgp}\n mAP: {mAP}')            

# n_epoch_batches = len(all_losses)/n_epochs
# step = 1/n_epoch_batches
# n_samples = n_epoch_batches * 32 # n_batch * batch_size

# # Data for plotting
# # loss_range = np.arange(0.0, max(all_losses), 0.001)
# epoch_range = np.arange(0.0, n_epochs, step)

# fig, ax = plt.subplots()
# ax.plot(epoch_range, all_losses)

# ax.set(xlabel='epoch', ylabel='BCELoss', title='Training Loss')
# ax.grid()

# fig.savefig("train_loss.png")
# plt.show()

