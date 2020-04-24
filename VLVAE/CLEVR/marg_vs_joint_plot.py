
# import sys, os
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('../../VLVAE'))
# sys.path.insert(0, os.path.abspath('../utils'))


from os.path import expanduser
home = expanduser("~")

import numpy as np
import math
import pickle
import random
import subprocess
import json
import random
import shutil
import time
import argparse
from collections import deque


import torch


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt






color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]












x = np.array(list(range(496))) * 500
# print (x)
# fsdafas


rows=3
cols=10
fig = plt.figure(figsize=(1+cols,1+rows), facecolor='white', dpi=150)
ax = plt.subplot2grid((1,3), (0,0), frameon=False, colspan=3, rowspan=1)

with open("/mnt/raisin/ccremer/VLVAE_exps/marg_inf_2/results.pkl", "rb") as f:
    dictname = pickle.load(f)


print (dictname.keys())

print ( len(dictname['all_y_recon_acc']))




ax.plot(x, dictname['all_y_recon_acc'][:496], c=color_defaults[0], label='q(z|x) trained, text accuracy')
ax.plot(x, dictname['all_x_inf_acc_entangle'][:496], c=color_defaults[0], linestyle='--', label='q(z|x) trained, image accuracy')





with open("/mnt/raisin/ccremer/VLVAE_exps/joint_inf_2/results.pkl", "rb") as f:
    dictname = pickle.load(f)


# print (dictname.keys())

print ( len(dictname['all_y_recon_acc']))


ax.plot(x, dictname['all_y_recon_acc'], c=color_defaults[2], label=r'$q(z|x,y) \, trained, \, z \sim q(z|x,y)$') #,\, y \sim p(y|z)
ax.plot(x, dictname['all_x_inf_acc_entangle'], c=color_defaults[2], linestyle='--', label=r'$q(z|x,y) \, trained, \, z \sim q(z|x)$')  #,\, y \sim p(y|z)


ax.legend(prop={'size':10})
ax.set_ylabel('Accuracy', family='serif', size=12)
ax.set_xlabel('Steps', family='serif', size=12)


ax.grid(True, alpha=.3)

# plt.tight_layout()
plt_path = home + '/Documents/VLVAE_exps/marg_vs_joint.png'
# plt_path = home + '/Documents/VLVAE_exps/marg_vs_joint.pdf'
plt.savefig(plt_path)
print ('saved viz',plt_path)
plt.close()












