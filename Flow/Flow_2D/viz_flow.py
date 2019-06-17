

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


def tensor(x):
    return torch.tensor(x).float()
def numpy(x):
    return x.data.numpy()



def plot_isocontours(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)

    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)

    # print ('ff')
    # print (zs.shape)
    zs = numpy(zs)

    Z = zs.reshape(X.shape)


    # Z_sum = torch.sum(Z)
    # print (Z_sum)
    # infinitesimal = (xlimits[1] - xlimits[0]) / numticks
    # # print (infinitesimal)
    # integral = infinitesimal**2 * Z_sum
    # print (integral)
    # # fdsfa
    # # print (Z.shape)
    # # fsd
    # # Z = Z/Z_sum
    # # Z_sum = np.sum(Z)
    # # print (Z_sum)


    # levels = [ 0. ,   0.04 , 0.08 , 0.12 , 0.16 , 0.2  , 0.24 , 0.28 , 0.32]
    levels = [ 0.  ,   0.005 , 0.01 ,  0.015 , 0.02  , 0.025,  0.03 ,  0.035 , 0.04 ]
    # levels = [ 0.  ,   0.005 , 0.01 ,  0.015 , 0.02  , 0.025,  0.03 ,  0.035 , 0.09 ]
    # levels = [ 0.  ,  0.02 , 0.04 , 0.06 , 0.08 , 0.1  , 0.12 , 0.14 , 0.16]
    # c = ax.contourf(X, Y, Z, levels, cmap=cmap)

    c = ax.contourf(X, Y, Z, cmap=cmap)
    # print (c.levels)
    # # fsdfa


    ax.set_yticks([])
    ax.set_xticks([])

    plt.gca().set_aspect('equal', adjustable='box')





def log_normal(x, mean, log_var, D=2):
    '''
    x is [P, D]
    mean is [D]
    log_var is [D]
    '''

    # print (x.shape)

    # x = torch.tensor(x).float()
    # print (log_var.type())
    # print (mean.type())
    # fsad

    term1 = torch.tensor(D * np.log(2*math.pi)).float()
    term2 = torch.sum(log_var) #sum over dimensions, now its a scalar [1]
    # print (term1, term2)

    term3 = (x - mean)**2 / torch.exp(log_var)
    term3 = torch.sum(term3, 1) #sum over dimensions so now its [particles]
    # print (term3.shape)

    all_ = term1 + term2 + term3
    # all_ = torch.sum(all_, 1) 

    log_normal = -.5 * all_  

    return log_normal #.data.numpy()


def sample_normal(mean, logvar):

    e = torch.randn(2)
    z = e*torch.exp(.5*logvar) + mean

    return z





def log_targetdist(x, D=2):

    # return self.log_normal(x, tf.zeros(self.D), tf.ones(self.D))
    return torch.log(  
                (torch.exp(log_normal(x, tensor([1,3]), torch.ones(D)/100))
                # + tf.exp(self.log_normal(x, [0,3], tf.ones(self.D)/100))

                + torch.exp(log_normal(x, tensor([3,0]), torch.ones(D)/100))
                # + tf.exp(self.log_normal(x, [3,-1], tf.ones(self.D)/100))


                + torch.exp(log_normal(x, tensor([1,1]), torch.ones(D)/100))
                

                + torch.exp(log_normal(x, tensor([-3,-1]), torch.ones(D)/100))
                + torch.exp(log_normal(x, tensor([-1,-3]), torch.ones(D)/100))

                + torch.exp(log_normal(x, tensor([2,2]), torch.ones(D)))
                + torch.exp(log_normal(x, tensor([-2,-2]), torch.ones(D)))
                + torch.exp(log_normal(x, tensor([0,0]), torch.ones(D)))) / 7.8104


                )



        

def flow_logprob(zT, p0_mean, p0_logvar):
    # print (zT.shape)
    # print (zT[:,0].shape)
    B = zT.shape[0]


    transform = flow1(zT[:,0].view(B,1))
    # print (zT.shape, transform[:,0].shape)
    z_temp = (zT- transform[:,0].view(B,1))  /transform[:,1].view(B,1)
    z0 = zT*mask1 + z_temp*mask0

    logqz0 = log_normal(z0, p0_mean, p0_logvar)
    logabsdetjac = torch.log(torch.abs(transform[:,1].view(B,1)))

    # print (logqz0.shape, logabsdetjac.shape)
    return logqz0 + logabsdetjac.view(B)











if __name__ == "__main__":


    save_to_dir = home + "/Documents/Flow/"
    exp_name = 'viz_flow'


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



    # x = torch.tensor([[-5.4,1.3]]).float()
    # mean = torch.tensor([0,0]).float()
    # logvar = torch.tensor([0,0]).float()
    # print (log_normal(x, mean, logvar, D=2))

    # print (log_targetdist(x))
    # fasdfa


    # m = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    # print (m.log_prob(x))

    # fsdfa




    # Train flow
    layer1 = [nn.Linear(1, 2)]
    flow1 = nn.Sequential(*layer1)

    params = [list(flow1.parameters())]
    optimizer = optim.Adam(params[0], lr=.001, weight_decay=.0000001)

    p0_mean = torch.tensor([0,0]).float()
    p0_logvar = torch.tensor([0.8,0.8]).float()

    mask0 = torch.ones(2)
    mask0[0] = 0
    mask1 = torch.ones(2)
    mask1[1] = 0

    for i in range(2000):

        #sample gauss
        # m = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        z0 = sample_normal(p0_mean, p0_logvar).view(1,2)

        logqz0 = log_normal(z0, p0_mean, p0_logvar)

        # print(z0)
        transform = flow1(z0[:,0].view(1,1))
        # print (transform.shape)
        z_temp = z0*transform[:,1] + transform[:,0]
        z1 = z0*mask1 + z_temp*mask0


        # transform = flow1(z1[:,0])
        # z_temp = (z1- transform[0])  /transform[1] 
        # z0 = z1*mask1 + z_temp*mask0
        # print (z0)
        # fadsfsa



        absdetjac = torch.abs(transform[:,1])
        logflow = logqz0 - torch.log(absdetjac)

        logtarget = log_targetdist(z1)


        loss = logflow - logtarget


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%50 == 0:
            print (i,loss)




    rows = 2
    cols = 3
    fig = plt.figure(figsize=(8+cols,6+rows), facecolor='white') #, dpi=150)
    # xlimits=[-3, 3]
    # ylimits=[-3, 3]
    xlimits=[-6, 6]
    ylimits=[-6, 6]


    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
    p = lambda x: np.exp(log_normal(tensor(x), p0_mean, p0_logvar))
    plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)


    ax = plt.subplot2grid((rows,cols), (1,0), frameon=False)
    p = lambda x: np.exp(log_normal(tensor(x), p0_mean, p0_logvar))
    plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)





    ax = plt.subplot2grid((rows,cols), (1,1), frameon=False)
    p = lambda x: torch.exp(flow_logprob(tensor(x), p0_mean, p0_logvar))
    plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)




    ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)
    p = lambda x: torch.exp(log_targetdist(tensor(x)))
    plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)



    

    plt_path = images_dir+'plot.png'
    plt.savefig(plt_path)
    print ('saved plot', plt_path)
    plt.close()























































































