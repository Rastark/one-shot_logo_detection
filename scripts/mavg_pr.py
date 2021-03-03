import logging
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Path config
rootpath = os.path.abspath('../')
log_path = os.path.join(rootpath, 'logs/train/')
log_file = os.path.join(log_path, 'oneshot.log')

regex_mavg_pr = 'Avg Precision: [0-9]*\.[0-9]*'
regex_epoch = 'Epoch number [0-9]'


match_list = []
all_mavg_pr = []
with open(log_file, 'r') as log:
    n_mavg_pr = 0
    n_epochs = 0
    for line in log:
        for match in re.finditer(regex_mavg_pr, line, re.S):
            n_mavg_pr += 1
            match_text = match.group()
            # match_list.append(match_text)
            mavg_pr = match_text.split(' ')[-1]
            all_mavg_pr.append(float(mavg_pr))
            logging.info(f'Loss {n_mavg_pr}: {mavg_pr}')
        for match in re.finditer(regex_epoch, line, re.S):
            n_epochs += 1

# if n_epochs > len(all_mavg_pr):
#     n_epochs = n_epochs - 1

n_epoch_batches = len(all_mavg_pr)/n_epochs
# step = 1/n_epoch_batches
n_samples = n_epoch_batches * 32 # n_batch * batch_size

# Data for plotting
# loss_range = np.arange(0.0, max(all_mavg_pr), 0.001)
epoch_range = np.arange(5, n_epochs+1, 5)

print(n_epochs)

fig, ax = plt.subplots()
ax.plot(epoch_range, all_mavg_pr)

ax.set(xlabel='epoch', ylabel='mAP', title='Mean Average Precision')
ax.grid()

fig.savefig("mavg_pr.png")
plt.show()

