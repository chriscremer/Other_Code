

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
                     numticks=101, cmap=None, contour_type='f', alpha=1.):
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

    if contour_type != 'f':
        c = ax.contour(X, Y, Z, cmap=cmap, alpha=alpha)
    else:
        c = ax.contourf(X, Y, Z, cmap=cmap, alpha=alpha)
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




class flow1(nn.Module):

    def __init__(self, n_flows):
        super(flow1, self).__init__()


        self.n_flows = n_flows
        self.p0_mean = torch.tensor([0,0]).float()
        self.p0_logvar = torch.tensor([0.8,0.8]).float()


        layer1 = [nn.Linear(1, 20), nn.ReLU(), nn.Linear(20, 2)]
        self.seq = nn.Sequential(*layer1)

        params = [list(self.seq.parameters())]
        self.optimizer = optim.Adam(params[0], lr=.001, weight_decay=.0000001)

        self.mask0 = torch.ones(2)
        self.mask0[0] = 0
        self.mask1 = torch.ones(2)
        self.mask1[1] = 0


    def transform(self, z0, n_flows):

        transform = self.seq(z0[:,0].view(1,1))
        z_temp = z0*transform[:,1] + transform[:,0]
        z1 = z0*self.mask1 + z_temp*self.mask0
        logflow = torch.log(torch.abs(transform[:,1]))
        return z1, logflow


    def transform_reverse(self, zT, n_flows):
        B = zT.shape[0]
        transform = self.seq(zT[:,0].view(B,1))
        z_temp = (zT- transform[:,0].view(B,1))  /transform[:,1].view(B,1)
        z0 = zT*self.mask1 + z_temp*self.mask0
        logabsdetjac = torch.log(torch.abs(transform[:,1].view(B,1)))
        return z0, logabsdetjac



    def sample(self):

        z0 = sample_normal(self.p0_mean, self.p0_logvar).view(1,2)
        logqz0 = log_normal(z0, self.p0_mean, self.p0_logvar)
        zT, logdet = self.transform(z0, self.n_flows)
        logprob = logqz0 - logdet

        return zT, logprob


    def logprob(self, z, n_flows=-1):
        B = z.shape[0]
        if n_flows ==-1:
            n_flows = self.n_flows

        zT = z
        z0, logdet = self.transform_reverse(zT, n_flows=self.n_flows)
        logqz0 = log_normal(z0, self.p0_mean, self.p0_logvar)
        logprob = logqz0 + logdet.view(B)

        return logprob











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




    torch.manual_seed(999)

    flow = flow1(n_flows=1)

    for i in range(1000):

        z, logprob = flow.sample()
        logtarget = log_targetdist(z)

        loss = logprob - logtarget


        flow.optimizer.zero_grad()
        loss.backward()
        flow.optimizer.step()

        if i%50 == 0:
            print (i,numpy(loss))




    rows = 2
    cols = 3
    fig = plt.figure(figsize=(8+cols,6+rows), facecolor='white') #, dpi=150)
    # xlimits=[-3, 3]
    # ylimits=[-3, 3]
    xlimits=[-6, 6]
    ylimits=[-6, 6]

    # p0_mean = torch.tensor([0,0]).float()
    # p0_logvar = torch.tensor([0.8,0.8]).float()


    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
    p = lambda x: torch.exp(log_targetdist(tensor(x)))
    plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits, contour_type='')



    ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)
    
    p = lambda x: torch.exp(log_targetdist(tensor(x)))
    plot_isocontours(ax, p, cmap='Greys', xlimits=xlimits, ylimits=ylimits, contour_type='', alpha=.1)

    p = lambda x: np.exp(log_normal(tensor(x), flow.p0_mean, flow.p0_logvar))
    plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)



    ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)

    p = lambda x: torch.exp(log_targetdist(tensor(x)))
    plot_isocontours(ax, p, cmap='Greys', xlimits=xlimits, ylimits=ylimits, contour_type='', alpha=.1)

    p = lambda x: torch.exp(flow.logprob(tensor(x)))
    plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)






    ax = plt.subplot2grid((rows,cols), (1,1), frameon=False)

    p = lambda x: np.exp(log_normal(tensor(x), flow.p0_mean, flow.p0_logvar))
    plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)

    n_samps = 400
    samps = []
    for i in range(n_samps):
        z = sample_normal(flow.p0_mean, flow.p0_logvar)
        samps.append(numpy(z))
    samps = np.array(samps)
    ax.scatter(samps.T[0], samps.T[1], alpha=.1, c='Black')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box') 



    ax = plt.subplot2grid((rows,cols), (1,2), frameon=False)

    p = lambda x: torch.exp(flow.logprob(tensor(x)))
    plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)
   
    samps = []
    for i in range(n_samps):
        z, logprob = flow.sample()
        samps.append(numpy(z))
    samps = np.array(samps)
    ax.scatter(samps.T[0], samps.T[1], alpha=.1, c='Black')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box') 
    ax.set_xlim(left=xlimits[0], right=xlimits[1])
    ax.set_ylim(ylimits)


    plt_path = images_dir+'plot.png'
    plt.savefig(plt_path)
    print ('saved plot', plt_path)
    plt.close()























































































