


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



def f(x):
    # x: [X]

    if x<0 or x>1:
        return x

    else:        

        n_bins = 3
        bins = np.linspace(0, 1, n_bins+1) #[1:]#[:-1]
        bin_values = np.array([0,.2,.5,.3])
        # bin_width = 1/ 3.

        bins_lower_than_x = (bins < x).astype(float)
        x_bin_idx = (np.sum(bins_lower_than_x) ).astype(int)
        bin_min = bins[x_bin_idx-1]
        bin_max = bins[x_bin_idx]
        pos_in_bin = (x- bin_min) / (bin_max - bin_min)

        # output = np.sum(bins_lower_than_x * bin_values * bin_width) + pos_in_bin*bin_values[x_bin_idx]* bin_width

        lower_sum = np.sum(bins_lower_than_x * bin_values )
        within_bin_sum = pos_in_bin * bin_values[x_bin_idx]
        output = lower_sum + within_bin_sum

        # print (bins)
        # print (bins_lower_than_x)
        # print (x_bin_idx)
        # print (bin_min)
        # print (bin_max)
        # print ('pos', pos_in_bin)
        # print (lower_sum)
        # print (within_bin_sum)
        # print (output)
        # fafds



        return output



def f_vec(x):
    # x: [B,X]


    # if x<0 or x>1:
    #     return x

    # else: 

    B = x.shape[0]     

    tail_bound = 1
    n_bins = 3
    # bins = np.reshape(np.linspace(-tail_bound, tail_bound, n_bins+1), [1,4]) # [1,n_bins+1]
    bins = np.reshape(np.linspace(0, 1, n_bins+1), [1,4]) # [1,n_bins+1]
    bin_values = np.array([0,.2,.5,.3])

    bins_lower_than_x = (bins < x).astype(float) # [B,n_bins+1]
    x_bin_idx = np.reshape((np.sum(bins_lower_than_x, 1)).astype(int), [B,1]) #[B,1]
    # x_bin_idx = x_bin_idx, [B])
    # print (bins_lower_than_x)
    # print (bins_lower_than_x.shape)
    # # print (x_bin_idx)
    # print (x_bin_idx.shape)
    # print (bins.shape)

    # print (bins[])
    # print (x_bin_idx-1)
    bin_min = np.reshape(bins[:,x_bin_idx-1], [B,1]) # [B,1]
    bin_max = np.reshape(bins[:,x_bin_idx], [B,1])  # [B,1]
    # print (x.shape)
    # print (bin_min.shape)
    # fds
    pos_in_bin = (x- bin_min) / (bin_max - bin_min) # [B,1]

    # print (pos_in_bin)
    # print (pos_in_bin.shape)

    # output = np.sum(bins_lower_than_x * bin_values * bin_width) + pos_in_bin*bin_values[x_bin_idx]* bin_width

    lower_sum = np.sum(bins_lower_than_x * bin_values ,1, keepdims=True)
    within_bin_sum = pos_in_bin * bin_values[x_bin_idx]
    output = lower_sum + within_bin_sum

    # print (bins)
    # print (bins_lower_than_x)
    # print (x_bin_idx)
    # print (bin_min)
    # print (bin_max)
    # print ('pos', pos_in_bin)
    # print (lower_sum)
    # print (within_bin_sum)
    # print (output)
    # fafds



    return output





# x = np.reshape(np.array([.2, .4]), [2,1])

# print (f_vec(x))

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
    exp_name = 'spline'


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

    x = np.linspace(-tail_bound - .5, tail_bound +.5, 99)
    x = np.reshape(x, [-1,1])



    inside_interval_mask = (x >= -tail_bound) & (x <= tail_bound)
    outside_interval_mask = ~inside_interval_mask


    outputs = np.zeros_like(x)
    outputs[outside_interval_mask] = x[outside_interval_mask]
    
    inputs_inside_range = np.reshape(x[inside_interval_mask], [-1,1])
    # print (inputs_inside_range.shape)
    inputs_ = (inputs_inside_range - -tail_bound) / (tail_bound - -tail_bound)
    ys = f_vec(inputs_)
    ys = ys * (tail_bound - -tail_bound) + -tail_bound

    # print (ys.shape)
    # print (outputs.shape)
    ys  = np.reshape(ys, [-1])
    outputs[inside_interval_mask] = ys

    ys = outputs
    # ys = []
    # for i in range(len(x)):
    #     y = f(x[i])
    #     ys.append(y)

    # print(ys)
    # fsdafa



    fontsize = 9

    rows = 1  
    cols = 1
    fig = plt.figure(figsize=(8+cols,4+rows), facecolor='white') #, dpi=150)
    # xlimits=[-3, 3]
    # ylimits=[0, .43]
    


    # p0_mean = torch.tensor([0,0]).float()
    # p0_logvar = torch.tensor([0.8,0.8]).float()


    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
    ax.plot(x, ys)
    # ax.set_ylim(ylimits)
    # ax.set_title('Initial Distribution', fontsize=fontsize)




    plt_path = images_dir+'first.png'
    plt.savefig(plt_path)
    print ('saved plot', plt_path)
    plt.close()








































