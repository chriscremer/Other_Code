


import sys, os
sys.path.insert(0, os.path.abspath('.'))

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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt






def f_vec(x, n_bins, bin_values, tail_bound, inverse=False):
    # x: [B,X]
    
    # Ignore the ones outside the interval
    inside_interval_mask = (x >= -tail_bound) & (x <= tail_bound)
    outside_interval_mask = ~inside_interval_mask
    outputs = np.zeros_like(x)
    outputs[outside_interval_mask] = x[outside_interval_mask]
    
    #Scale to [0,1]
    lower_tailbound = -tail_bound
    upper_tailbound = tail_bound
    inputs_inside_range = np.reshape(x[inside_interval_mask], [-1,1])
    x = (inputs_inside_range - lower_tailbound) / (upper_tailbound - lower_tailbound)
    B = x.shape[0]

    bin_intervals = np.reshape(np.linspace(0, 1, n_bins+1), [1,n_bins+1]) # [1,n_bins+1]
    cdf = np.reshape(np.cumsum(bin_values, -1),[1,n_bins+1])


    if inverse:

        #Figure out which bin x is in
        cdf_lower_than_x = (cdf < x).astype(float) # [B,n_bins+1]
        x_bin_idx = np.reshape((np.sum(cdf_lower_than_x, 1)-1).astype(int), [B,1]) #[B,1]

        # Reverse the output computation
        lower_bin_cdf = np.reshape(cdf[:,x_bin_idx], [B,1]) # [B,1]
        pos_in_bin = x - lower_bin_cdf
        pos_in_bin = pos_in_bin / bin_values[x_bin_idx+1]

        # Reverse the within bin computation
        bin_x_below = np.reshape(bin_intervals[:,x_bin_idx], [B,1]) # [B,1]
        bin_x_above = np.reshape(bin_intervals[:,x_bin_idx+1], [B,1])  # [B,1]
        output = pos_in_bin * (bin_x_above - bin_x_below) + bin_x_below



    else:

        #Figure out which bin x is in
        bins_lower_than_x = (bin_intervals < x).astype(float) # [B,n_bins+1]  
        x_bin_idx = np.reshape((np.sum(bins_lower_than_x, 1)-1).astype(int), [B,1]) #[B,1]

        # Get position in bin
        bin_x_below = np.reshape(bin_intervals[:,x_bin_idx], [B,1]) # [B,1]
        bin_x_above = np.reshape(bin_intervals[:,x_bin_idx+1], [B,1])  # [B,1]
        pos_in_bin = (x- bin_x_below) / (bin_x_above - bin_x_below) # [B,1]

        # Compute output
        lower_sum = np.reshape(cdf[:,x_bin_idx], [B,1]) # [B,1]
        within_bin_sum = pos_in_bin * bin_values[x_bin_idx+1]
        output = lower_sum + within_bin_sum



    # Rescale back to tail bound space
    output = output * (upper_tailbound - lower_tailbound) + lower_tailbound
    output  = np.reshape(output, [-1])
    outputs[inside_interval_mask] = output

    return outputs





# x = np.reshape(np.array([.2, .4]), [2,1])

# y = f_vec(x)

# x = f()

# fasdfa



# print (f(0.0001))
# # fasdfa


# print (f(.3333))
# # print ( .2)
# print()
# print (f(.5))
# print()
# print (f(.6666))
# # print ( .2 +  .5)
# print ()
# print (f(.9999))
# # print (.2 +  .5 + .3)
# fsda



if __name__ == "__main__":


    save_to_dir = home + "/Documents/Flow/"
    exp_name = 'spline_quad'


    exp_dir = save_to_dir + exp_name + '/'
    params_dir = exp_dir + 'params/'
    images_dir = exp_dir + 'images/'
    code_dir = exp_dir + 'code/'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print ('Made dir', exp_dir) 

    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
        print ('Made dir', params_dir) 

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print ('Made dir', images_dir) 

    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
        print ('Made dir', code_dir) 


    #Save args and code
    # json_path = exp_dir+'args_dict.json'
    # with open(json_path, 'w') as outfile:
    #     json.dump(args_dict, outfile, sort_keys=True, indent=4)
    subprocess.call("(rsync -r --exclude=__pycache__/ . "+code_dir+" )", shell=True)


    torch.manual_seed(999)





    # Transform
    tail_bound = 1
    n_bins = 3
    W = np.array([.2,.5,.3])
    # bin_values = np.array([0,.2,.5,.3])
    V = np.array([2.,5.,3.,1.])
    Area = np.sum(.5*( np.exp(V[:-1]) + np.exp(V[1:])) * W)
    V = np.exp(V) / Area
    # print (list(V))
    # fadsfa

    W_positions = np.cumsum(W)
    print (W_positions)
    np.interp(.5, W_positions, fp)

    x = np.linspace(-tail_bound - .5, tail_bound +.5, 99)
    x = np.reshape(x, [-1,1])


    ys = f_vec(x, n_bins=n_bins, bin_values=bin_values, tail_bound=tail_bound)

    y_inverse = f_vec(x, n_bins=n_bins, bin_values=bin_values, tail_bound=tail_bound, inverse=True)


    # dfasad



    #PLOT
    rows = 1  
    cols = 1
    fig = plt.figure(figsize=(8+cols,4+rows), facecolor='white') #, dpi=150)
    # xlimits=[-3, 3]
    # ylimits=[0, .43]

    fontsize = 9
    

    # p0_mean = torch.tensor([0,0]).float()
    # p0_logvar = torch.tensor([0.8,0.8]).float()


    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)


    ax.plot(x, x, alpha=.3, ls='--')


    ax.plot(x, ys, label='f')
    # ax.set_ylim(ylimits)
    # ax.set_title('Initial Distribution', fontsize=fontsize)

    # ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)
    ax.plot(x, y_inverse, label='f inv')

    plt.legend(fontsize=fontsize)
    plt.gca().set_aspect('equal', adjustable='box') 

    plt_path = images_dir+'first.png'
    plt.savefig(plt_path)
    print ('saved plot', plt_path)
    plt.close()








































