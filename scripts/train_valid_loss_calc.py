import logging
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage.filters import gaussian_filter1d

# Path config
rootpath = os.path.abspath('../')
log_path = os.path.join(rootpath, 'logs/train/')
log_file = os.path.join(log_path, 'oneshot_20_cfgB.log')

regex_train_loss = 'Train epoch loss: [0-9]*\.[0-9]*'
regex_valid_loss = 'Validation epoch loss: [0-9]*\.[0-9]*'
regex_epoch = 'Epoch number [0-9]'

match_list = []
all_train_losses = []
all_valid_losses = []

with open(log_file, 'r') as log:
    n_train_losses = 0
    n_valid_losses = 0
    n_epochs = 0
    for line in log:
        for match in re.finditer(regex_train_loss, line, re.S):
            n_train_losses += 1
            match_text = match.group()
            # match_list.append(match_text)
            loss = match_text.split(' ')[-1]
            all_train_losses.append(float(loss))
            logging.info(f'Training Loss {n_train_losses}: {loss}')
        for match in re.finditer(regex_valid_loss, line, re.S):
            n_valid_losses += 1
            match_text = match.group()
            # match_list.append(match_text)
            loss = match_text.split(' ')[-1]
            all_valid_losses.append(float(loss))
            logging.info(f'Validation Loss {n_valid_losses}: {loss}')
        for match in re.finditer(regex_epoch, line, re.S):
            n_epochs += 1

if n_epochs > len(all_train_losses):
    n_epochs = n_epochs - 1

n_epoch_batches = len(all_train_losses)/n_epochs
# step = 1/n_epoch_batches
n_samples = n_epoch_batches * 32 # n_batch * batch_size

# Data for plotting
# loss_range = np.arange(0.0, max(all_losses), 0.001)
epoch_range = np.arange(1, n_epochs+1, 1)

fig, ax = plt.subplots()

# spl = make_interp_spline(epoch_range, all_train_losses, k=5)
# train_smooth = spl(epoch_range)
# ax.plot(epoch_range, train_smooth)

# y_train_smoothed = gaussian_filter1d(all_train_losses, sigma=1.5)
# ax.plot(epoch_range, y_train_smoothed)
# y_valid_smoothed = gaussian_filter1d(all_valid_losses, sigma=1.5)
# ax.plot(epoch_range, y_valid_smoothed)


# valid_smooth = BSpline(epoch_range,all_valid_losses) 
# ax.plot(epoch_range, valid_smooth)

ax.plot(epoch_range, all_train_losses)
ax.plot(epoch_range, all_valid_losses)

ax.legend(['Training', 'Validation'])

ax.set(xlabel='Epoch', ylabel='BCELoss', title='Loss functions')
ax.grid()

fig.savefig("train_valid_loss.png")
plt.show()

