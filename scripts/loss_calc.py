import logging
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Path config
rootpath = os.path.abspath('../')
log_path = os.path.join(rootpath, 'logs/')
log_file = os.path.join(log_path, 'oneshot.log')

regex_loss = 'Loss: [0-9]*\.[0-9]*'
regex_epoch = 'Epoch number [0-9]'

match_list = []
all_losses = []
with open(log_file, 'r') as log:
    n_losses = 0
    n_epochs = 0
    for line in log:
        for match in re.finditer(regex_loss, line, re.S):
            n_losses += 1
            match_text = match.group()
            match_list.append(match_text)
            loss = match_text.split(' ')[-1]
            all_losses.append(float(loss))
            logging.info(f'Loss {n_losses}: {loss}')
        for match in re.finditer(regex_epoch, line, re.S):
            n_epochs += 1

n_epoch_batches = len(all_losses)/n_epochs
step = 1/n_epoch_batches
n_samples = n_epoch_batches * 16 # n_batch * batch_size

# Data for plotting
# loss_range = np.arange(0.0, max(all_losses), 0.001)
epoch_range = np.arange(0.0, n_epochs, step)

fig, ax = plt.subplots()
ax.plot(epoch_range, all_losses)

ax.set(xlabel='epoch', ylabel='BCELoss', title='Training Loss')
ax.grid()

fig.savefig("train_loss.png")
plt.show()

