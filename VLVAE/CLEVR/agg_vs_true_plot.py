
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











def smooth_list(x, window_len=75, window='flat'):
    if len(x) < window_len:
        return x
    w = np.ones(window_len,'d') 
    y = np.convolve(w/ w.sum(), x, mode='same')
    return y





# x = np.array(list(range(496))) * 500
# x = np.array(list(range(4000))) * 500
# xs = 2000
xs = 1500
x = np.array(list(range(xs))) * 100
# print (x)
# fsdafas


rows=3
cols=10
fig = plt.figure(figsize=(1+cols,1+rows), facecolor='white', dpi=150)
ax = plt.subplot2grid((1,2), (0,0), frameon=False, colspan=2, rowspan=1)

# with open("/home/ccremer/Documents/VLVAE_exps/qy_true_2flows_BN/results.pkl", "rb") as f:
with open("/home/ccremer/Documents/VLVAE_exps/qy_true_new/results.pkl", "rb") as f:
    dictname = pickle.load(f)
# print (dictname.keys())
print ( len(dictname['qy_real_image_gen_words_acc_val']))
y1 = smooth_list(dictname['qy_real_image_gen_words_acc_val'])
y2 = smooth_list(dictname['qy_gen_image_real_words_acc_val'])
ax.plot(x, y1[:xs], c=color_defaults[0], label=r'$z \, \sim \, q_{True}(z|y), \, \hat{y} \, \sim \, p(y|z)$')
ax.plot(x, y2[:xs], c=color_defaults[0], linestyle='--', label=r'$z \, \sim \, q_{True}(z|y), \, \hat{x} \, \sim \, p(x|z)$')



with open("/home/ccremer/Documents/VLVAE_exps/qy_clevr_true_seed2/results.pkl", "rb") as f:
    dictname = pickle.load(f)
# print (dictname.keys())
print ( len(dictname['qy_real_image_gen_words_acc_val']))
y1 = smooth_list(dictname['qy_real_image_gen_words_acc_val'])
y2 = smooth_list(dictname['qy_gen_image_real_words_acc_val'])
ax.plot(x, y1[:xs], c=color_defaults[0], label=r'$z \, \sim \, q_{True}(z|y), \, \hat{y} \, \sim \, p(y|z)$')
ax.plot(x, y2[:xs], c=color_defaults[0], linestyle='--', label=r'$z \, \sim \, q_{True}(z|y), \, \hat{x} \, \sim \, p(x|z)$')



with open("/home/ccremer/Documents/VLVAE_exps/qy_clevr_true_seed1/results.pkl", "rb") as f:
    dictname = pickle.load(f)
# print (dictname.keys())
print ( len(dictname['qy_real_image_gen_words_acc_val']))
y1 = smooth_list(dictname['qy_real_image_gen_words_acc_val'])
y2 = smooth_list(dictname['qy_gen_image_real_words_acc_val'])
ax.plot(x, y1[:xs], c=color_defaults[0], label=r'$z \, \sim \, q_{True}(z|y), \, \hat{y} \, \sim \, p(y|z)$')
ax.plot(x, y2[:xs], c=color_defaults[0], linestyle='--', label=r'$z \, \sim \, q_{True}(z|y), \, \hat{x} \, \sim \, p(x|z)$')

















# with open("/home/ccremer/Documents/VLVAE_exps/qy_agg_2flows_BN/results.pkl", "rb") as f:
with open("/home/ccremer/Documents/VLVAE_exps/qy_agg_new/results.pkl", "rb") as f:
    dictname = pickle.load(f)
# print (dictname.keys())
print ( len(dictname['qy_gen_image_real_words_acc_val']))
y1 = smooth_list(dictname['qy_real_image_gen_words_acc_val'])
y2 = smooth_list(dictname['qy_gen_image_real_words_acc_val'])
ax.plot(x, y1[:xs], c=color_defaults[2], label=r'$z \, \sim \, q_{Agg}(z|y), \, \hat{y} \, \sim \, p(y|z)$') #,\, y \sim p(y|z)
ax.plot(x, y2[:xs], c=color_defaults[2], linestyle='--', label=r'$z \, \sim \, q_{Agg}(z|y), \, \hat{x} \, \sim \, p(x|z)$')  #,\, y \sim p(y|z)





# with open("/home/ccremer/Documents/VLVAE_exps/qy_agg_2flows_BN/results.pkl", "rb") as f:
with open("/home/ccremer/Documents/VLVAE_exps/qy_clevr_agg_seed1/results.pkl", "rb") as f:
    dictname = pickle.load(f)
# print (dictname.keys())
print ( len(dictname['qy_gen_image_real_words_acc_val']))
y1 = smooth_list(dictname['qy_real_image_gen_words_acc_val'])
y2 = smooth_list(dictname['qy_gen_image_real_words_acc_val'])
ax.plot(x, y1[:xs], c=color_defaults[2], label=r'$z \, \sim \, q_{Agg}(z|y), \, \hat{y} \, \sim \, p(y|z)$') #,\, y \sim p(y|z)
ax.plot(x, y2[:xs], c=color_defaults[2], linestyle='--', label=r'$z \, \sim \, q_{Agg}(z|y), \, \hat{x} \, \sim \, p(x|z)$')  #,\, y \sim p(y|z)






# with open("/home/ccremer/Documents/VLVAE_exps/qy_agg_2flows_BN/results.pkl", "rb") as f:
with open("/home/ccremer/Documents/VLVAE_exps/qy_clevr_agg_seed2/results.pkl", "rb") as f:
    dictname = pickle.load(f)
# print (dictname.keys())
print ( len(dictname['qy_gen_image_real_words_acc_val']))
y1 = smooth_list(dictname['qy_real_image_gen_words_acc_val'])
y2 = smooth_list(dictname['qy_gen_image_real_words_acc_val'])
ax.plot(x, y1[:xs], c=color_defaults[2], label=r'$z \, \sim \, q_{Agg}(z|y), \, \hat{y} \, \sim \, p(y|z)$') #,\, y \sim p(y|z)
ax.plot(x, y2[:xs], c=color_defaults[2], linestyle='--', label=r'$z \, \sim \, q_{Agg}(z|y), \, \hat{x} \, \sim \, p(x|z)$')  #,\, y \sim p(y|z)














ax.legend(prop={'size':12})
ax.set_ylabel('Correctness', family='serif', size=13)
ax.set_xlabel('Steps', family='serif', size=13)


ax.grid(True, alpha=.3)

# plt.tight_layout()
plt_path = home + '/Documents/VLVAE_exps/true_vs_agg_plot.png'
# plt_path = home + '/Documents/VLVAE_exps/marg_vs_joint.pdf'
plt.savefig(plt_path)
print ('saved viz',plt_path)
plt.close()












