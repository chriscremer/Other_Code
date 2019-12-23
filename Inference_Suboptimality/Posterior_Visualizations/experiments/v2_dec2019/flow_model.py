
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

    term1 = torch.tensor(D * np.log(2*math.pi)).float().cuda()
    term2 = torch.sum(log_var) #sum over dimensions, now its a scalar [1]
    # print (term1, term2)

    term3 = (x - mean)**2 / torch.exp(log_var)
    term3 = torch.sum(term3, 1) #sum over dimensions so now its [particles]
    # print (term3.shape)

    all_ = term1 + term2 + term3
    # all_ = torch.sum(all_, 1) 

    log_normal = -.5 * all_  

    return log_normal #.data.numpy()


def sample_normal(N, mean, logvar):

    e = torch.randn(N, 2).cuda()
    z = e*torch.exp(.5*logvar) + mean

    return z








class flow1(nn.Module):

    def __init__(self, n_flows):
        super(flow1, self).__init__()

        self.D = 2

        self.n_flows = n_flows
        self.p0_mean = torch.nn.Parameter(torch.tensor([0,0]).float().cuda())
        self.p0_logvar = torch.nn.Parameter(torch.tensor([0.8,0.8]).float().cuda())
        self.p0_mean.requires_grad = True
        self.p0_logvar.requires_grad = True


        self.masks = []
        self.nets = []
        params = []
        mask = torch.ones(2).cuda()
        mask[0] = 0
        for i in range(n_flows):
            self.masks.append(mask)
            mask = 1-mask
            L = 50 #30 #100 #30
            # layer = [nn.Linear(2, L), nn.ReLU(), nn.Linear(L, L), nn.ReLU(), nn.Linear(L, 2)]
            layer = [nn.Linear(2, L).cuda(), nn.ReLU(), nn.Linear(L, L).cuda(), nn.ReLU(), nn.Linear(L, 2).cuda()]
            # layer = [nn.Linear(2, 100), nn.ReLU(), nn.Linear(100, 2)]
            # layer = [nn.Linear(2, 2)]
            seq = nn.Sequential(*layer)
            self.nets.append(seq)

            params.extend(list(seq.parameters()))

        params.append(self.p0_mean)
        params.append(self.p0_logvar)

        # params = [list(self.seq.parameters())]
        self.optimizer = optim.Adam(params, lr=.0001, weight_decay=.0000001)




    def transform(self, z, mask, net, reverse=False):
        B = z.shape[0]

        z_constant = z * mask
        mu_sig = net(z_constant)
        mu = mu_sig[:,0].view(B,1)
        # sig = mu_sig[:,1].view(B,1)
        # sig = torch.tanh(mu_sig[:,1].view(B,1)) 
        # sig = torch.sigmoid(mu_sig[:,1].view(B,1)) 
        sig = torch.tanh(mu_sig[:,1].view(B,1))  + 1.01

        if reverse == True:
            z_temp = (z-mu) /sig
        else:
            z_temp = z*sig + mu

        z = z_constant + z_temp*(1-mask)

        if (z!=z).any():
            print (z)
            print (mu)
            print(sig)
            print (reverse)
            afdsaf

        logdet = torch.log(torch.abs(sig))
        return z, logdet



    def sample(self, N=1, n_flows=-1):

        # print (self.p0_mean)

        if n_flows==-1:
            n_flows = self.n_flows
        # else:
        #     print (n_flows)

        z0 = sample_normal(N, self.p0_mean, self.p0_logvar) #.cuda() #.view(1,2)
        logqz0 = log_normal(z0, self.p0_mean, self.p0_logvar).view(N,1)

        z = z0
        logdetsum = 0
        for i in range(n_flows):
            z, logdet = self.transform(z, mask=self.masks[i], net=self.nets[i])
            logdetsum += logdet
        zT = z

        logprob = logqz0 - logdetsum
        return zT, logprob.view(N)


    def logprob(self, z, n_flows=-1):

        B = z.shape[0]
        if n_flows==-1:
            n_flows = self.n_flows
        elif n_flows==0:
            return log_normal(z, self.p0_mean, self.p0_logvar)
        # else:
        #     print (n_flows)
        #     print (list(range(n_flows))[::-1])

        logdetsum = 0
        for i in list(range(n_flows))[::-1]:
            z, logdet = self.transform(z, mask=self.masks[i], net=self.nets[i], reverse=True)
            logdetsum += logdet
        z0 = z

        logqz0 = log_normal(z0, self.p0_mean, self.p0_logvar)
        logprob = logqz0 + logdetsum.view(B)

        return logprob


