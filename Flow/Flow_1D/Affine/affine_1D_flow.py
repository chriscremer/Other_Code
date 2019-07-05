


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



# def plot_isocontours(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
#                      numticks=101, cmap=None, contour_type='f', alpha=1.):
#     x = np.linspace(*xlimits, num=numticks)
#     y = np.linspace(*ylimits, num=numticks)
#     X, Y = np.meshgrid(x, y)

#     zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)

#     # print ('ff')
#     # print (zs.shape)
#     zs = numpy(zs)

#     Z = zs.reshape(X.shape)


#     # Z_sum = torch.sum(Z)
#     # print (Z_sum)
#     # infinitesimal = (xlimits[1] - xlimits[0]) / numticks
#     # # print (infinitesimal)
#     # integral = infinitesimal**2 * Z_sum
#     # print (integral)
#     # # fdsfa
#     # # print (Z.shape)
#     # # fsd
#     # # Z = Z/Z_sum
#     # # Z_sum = np.sum(Z)
#     # # print (Z_sum)


#     # levels = [ 0. ,   0.04 , 0.08 , 0.12 , 0.16 , 0.2  , 0.24 , 0.28 , 0.32]
#     levels = [ 0.  ,   0.005 , 0.01 ,  0.015 , 0.02  , 0.025,  0.03 ,  0.035 , 0.04 ]
#     # levels = [ 0.  ,   0.005 , 0.01 ,  0.015 , 0.02  , 0.025,  0.03 ,  0.035 , 0.09 ]
#     # levels = [ 0.  ,  0.02 , 0.04 , 0.06 , 0.08 , 0.1  , 0.12 , 0.14 , 0.16]
#     # c = ax.contourf(X, Y, Z, levels, cmap=cmap)

#     if contour_type != 'f':
#         c = ax.contour(X, Y, Z, cmap=cmap, alpha=alpha)
#     else:
#         c = ax.contourf(X, Y, Z, cmap=cmap, alpha=alpha)
#     # print (c.levels)
#     # # fsdfa


#     ax.set_yticks([])
#     ax.set_xticks([])

#     plt.gca().set_aspect('equal', adjustable='box')




def log_normal(x, mean, log_var):
    '''
    x is [P, D]
    mean is [D]
    log_var is [D]
    '''

    D = x.shape[1]

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


def sample_normal(N, mean, logvar):
    D = mean.shape[0]
    e = torch.randn(N, D)
    z = e*torch.exp(.5*logvar) + mean

    return z





def log_targetdist(x):

    # return self.log_normal(x, tf.zeros(self.D), tf.ones(self.D))
    # px = (torch.exp(log_normal(x, tensor([1,3]), torch.ones(D)/100))
    #             + torch.exp(log_normal(x, tensor([3,0]), torch.ones(D)/100))
    #             + torch.exp(log_normal(x, tensor([1,1]), torch.ones(D)/100))
    #             + torch.exp(log_normal(x, tensor([-3,-1]), torch.ones(D)/100))
    #             + torch.exp(log_normal(x, tensor([-1,-3]), torch.ones(D)/100))
    #             + torch.exp(log_normal(x, tensor([2,2]), torch.ones(D)))
    #             + torch.exp(log_normal(x, tensor([-2,-2]), torch.ones(D)))
    #             + torch.exp(log_normal(x, tensor([0,0]), torch.ones(D)))) / 7.8104

    px = .5*torch.exp(log_normal(x, torch.ones(1)*3., torch.ones(1)*.3)) + .5*torch.exp(log_normal(x, torch.ones(1)*-3., torch.ones(1)*.3)) 

    px = torch.clamp(px, min=1e-10, max=1-1e-10)

    return torch.log(px)










# def inv_sig(x):
#     return -torch.log(1./x - 1)

def sig_der(x):
    return torch.sigmoid(x) * (1-torch.sigmoid(x))


def tanh_der(x):
    return 1 - torch.tanh(x)**2



class flow1(nn.Module):

    def __init__(self, n_flows):
        super(flow1, self).__init__()

        self.D = 1

        self.n_flows = n_flows

        self.p0_mean = torch.tensor([0.]).float()
        self.p0_logvar = torch.tensor([0.]).float()

        self.sigma = nn.Parameter(torch.tensor([1.]).float())
        self.mu = nn.Parameter(torch.tensor([0.]).float())

        # self.flows = []
        # params = []
        # for i in range(n_flows):
        #     # self.masks.append(mask)
        #     # mask = 1-mask
        #     # L = 30 #100 #30
        #     # layer = [nn.Linear(2, L), nn.ReLU(), nn.Linear(L, L), nn.ReLU(), nn.Linear(L, 2)]
        #     # layer = [nn.Linear(2, 100), nn.ReLU(), nn.Linear(100, 2)]
        #     # layer = [nn.Linear(2, 2)]
        #     # seq = nn.Sequential(*layer)
        #     flow_n = [nn.Parameter(torch.tensor([.5]).float()), nn.Parameter(torch.tensor([.5]).float())]
        #     self.flows.append(flow_n)

        #     params.extend(flow_n)

        # self.a = nn.Parameter(torch.tensor([.5]).float())
        # self.b = nn.Parameter(torch.tensor([.5]).float())
        # self.c = nn.Parameter(torch.tensor([.5]).float())
        # self.d = nn.Parameter(torch.tensor([.5]).float())

        params = [self.sigma, self.mu]
        # self.optimizer = optim.Adam([self.a, self.b], lr=.001, weight_decay=.0000001)
        self.optimizer = optim.Adam(params, lr=.001, weight_decay=.0000001)


    # def linear_transform(self, a, b, reverse=False):
    #     flow_probfadsf



    def transform(self, z, a, b, reverse=False):

        if reverse == True:
            z1 = z
            z0 = (z1-b) / a
            z1 = z0
            # jac = 1 + sig_der(z1*a + b)* a
            # logdet = torch.log(torch.abs(jac))
        else:
            z0=z
            # z1 = z0 + torch.sigmoid(z0*a + b)
            z1 = z0 * a +b

        # jac = 1 + sig_der(z0*a + b)* a
        jac = torch.ones(z.shape[0]) *a #1 + tanh_der(z0*a + b)* a
        logdet = torch.log(torch.abs(jac))
        # print (logdet.shape)

        return z1, logdet



    def sample(self, N=1, n_flows=-1):

        if n_flows==-1:
            n_flows = self.n_flows

        z0 = sample_normal(N, self.p0_mean, self.p0_logvar) #[B,1]
        logqz0 = log_normal(z0, self.p0_mean, self.p0_logvar).view(N,1)

        z = z0
        logdetsum = 0
        # for i in range(n_flows):
        # print (z)
        z, logdet = self.transform(z, self.sigma, self.mu) #mask=self.masks[i], net=self.nets[i])
        # z, logdet = self.transform(z, self.a, self.b, reverse=True)
        # print (z)
        # fds
        logdetsum += logdet
        zT = z

        # print (logqz0.shape, logdetsum.shape)
        logprob = logqz0.view(N) - logdetsum.view(N)
        return zT, logprob.view(N)


    def newtons(self, z, w, b):

        z1 = z
        z0 = z
        trans = lambda x: x + torch.tanh(x*w + b) - z1
        der = lambda x: 1 + (1- torch.tanh(x*w + b)**2)*w


        for i in range(5):
            z0 = z0 - (trans(z0)/der(z0))
            print (z0)
        fasdf

        return z0



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
        # print ()

        # z = z*0
        # print (z)
        for i in list(range(n_flows))[::-1]:
            # z, logdet = self.transform(z, self.a, self.b, reverse=True)

            # original_z = z



            # z_js = []
            # for j in range(len(z)):

            #     z_j = z[j].view(1,1)

            #     #NEWTONS method
            #     z_j = self.newtons(z_j, self.flows[i][0], self.flows[i][1])

            #     z_js.append(z_j)

            #     # #confirm its right
            #     # z_test, logdet = self.transform(z, self.flows[i][0], self.flows[i][1])
            #     # print ()
            #     # print (z)
            #     # print (z_test)
            #     # fsfa
            # z = torch.stack(z_js).view(B,1)



            # z = self.newtons(z, self.sigma, self.mu)

            z, logdet = self.transform(z, self.sigma, self.mu, reverse=True)


            # forward_z, logdet = self.transform(z, self.sigma, self.mu)
            logdetsum += logdet

            # print (original_z, forward_z)
            # fadss

        z0 = z

        logqz0 = log_normal(z0, self.p0_mean, self.p0_logvar)

        # print (logqz0.shape, logdetsum.shape)
        logprob = logqz0 - logdetsum.view(B)

        # print (logqz0, logdetsum)
        # fasda

        return logprob




# # Inverting these functions!


# w = 3
# b = 2

# forward = lambda x: x + np.tanh(w*x + b) 
# derivative = lambda x: 1 + (1 - np.tanh(w*x + b)**2)*w

# x = 1
# y = forward(x)


# forward2 = lambda x: x + np.tanh(w*x + b) - y


# #NEWTONS METHOD
# print(y)
# x_i = y
# # x_i = 1.2
# for i in range(10):
#     x_i = x_i - (forward2(x_i)/derivative(x_i))
#     print (x_i)
# # IT WORKED






# print ()
# # Try gradient descent instead
# print(y)
# x_i = y
# for i in range(10):
#     x_i = x_i - .1*derivative(x_i)
#     print (x_i)
# #this could work to, I need to change the objective to beign a squard error 
# # the way I have it right now isnt correct. 

# fsdf









if __name__ == "__main__":


    save_to_dir = home + "/Documents/Flow/"
    exp_name = 'affine_1D_flow'


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








    n_flows = 1
    flow = flow1(n_flows=n_flows)
    # print (flow)
    # fsf
    batch_size = 128 #64 #32

    for i in range(2000):
    # for i in range(100):

        z, logprob = flow.sample(batch_size)

        logtarget = log_targetdist(z) #.detach()

        loss = logprob - logtarget
        loss = torch.mean(loss)

        if (loss!=loss).any() or loss > 999999:
            print (i)
            print ('z', z)
            print ('logprob', logprob)
            print ('logtarget', logtarget)
            fdsf

        flow.optimizer.zero_grad()
        loss.backward()
        flow.optimizer.step()

        if i%100 == 0:
            print (i,numpy(loss))










    n_ticks = 100 #3 # 100

    xlimits=[-10, 10]

    aa = np.linspace(xlimits[0],xlimits[1], n_ticks)
    # print (aa)

    aa_tensor = torch.tensor(aa).view(n_ticks,1).float()
    mean = flow.p0_mean# torch.zeros(1).float()
    logvar = flow.p0_logvar # torch.zeros(1).float()

    logprob = log_normal(aa_tensor, mean=mean, log_var=logvar)

    logprob = numpy(logprob)
    prob = np.exp(logprob)

    target_logprob = log_targetdist(aa_tensor)
    target_prob = np.exp(numpy(target_logprob))

    flow_logprob = flow.logprob(aa_tensor)
    flow_prob_density = np.exp(numpy(flow_logprob))
    # print (flow_prob)

    z, logprob = flow.sample(1000)
    flow_prob = np.exp(numpy(logprob.view(1000)))
    z = numpy(z)

    # print (z.shape, flow_prob.shape)
    # flow_prob = flow_prob
    # fasdf



    fontsize = 9

    rows = 3 #2 
    cols = 4 #n_flows
    fig = plt.figure(figsize=(8+cols,4+rows), facecolor='white') #, dpi=150)
    # xlimits=[-3, 3]
    ylimits=[0, .43]
    


    # p0_mean = torch.tensor([0,0]).float()
    # p0_logvar = torch.tensor([0.8,0.8]).float()


    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
    ax.plot(aa, prob)
    ax.set_ylim(ylimits)
    ax.set_title('Initial Distribution', fontsize=fontsize)
    # plt.gca().set_aspect('equal', adjustable='box')

    ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)
    ax.scatter(z, flow_prob, s=5, alpha=.05)
    ax.set_ylim(ylimits)
    ax.set_xlim(xlimits)
    ax.plot(aa, target_prob, alpha=.3, c='red')
    ax.set_title('Flow samples', fontsize=fontsize)
    # plt.gca().set_aspect('equal', adjustable='box')


    ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)
    ax.plot(aa, flow_prob_density)
    ax.set_ylim(ylimits)
    ax.set_title('Flow Density (using Newtons)', fontsize=fontsize)
    # plt.gca().set_aspect('equal', adjustable='box')


    ax = plt.subplot2grid((rows,cols), (0,3), frameon=False)
    ax.plot(aa, target_prob)
    ax.set_ylim(ylimits)
    ax.set_title('Target Distribution', fontsize=fontsize)
    # plt.gca().set_aspect('equal', adjustable='box')


    # p = lambda x: torch.exp(log_targetdist(tensor(x)))
    # plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits, contour_type='')


    # for i in range(n_flows):

    #     # ax = plt.subplot2grid((rows,cols), (0,1+i), frameon=False)
        
    #     # p = lambda x: torch.exp(log_targetdist(tensor(x)))
    #     # plot_isocontours(ax, p, cmap='Greys', xlimits=xlimits, ylimits=ylimits, contour_type='', alpha=.1)

    #     # p = lambda x: torch.exp(flow.logprob(tensor(x), n_flows=i))
    #     # plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)



    #     ax = plt.subplot2grid((rows,cols), (2,i), frameon=False)

    #     z, logprob = flow.sample(N=1000, n_flows=i+1)
    #     flow_prob = np.exp(numpy(logprob))
    #     z = numpy(z)
    #     ax.scatter(z, flow_prob, s=5, alpha=.05)
    #     ax.set_ylim(ylimits)
    #     ax.set_xlim(xlimits)
    #     ax.set_title('Flow '+str(i+1), fontsize=fontsize)


    #     # plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)
       
    #     # n_samps = 1000
    #     # samps = []
    #     # for n in range(n_samps):
    #     #     z, logprob = flow.sample(n_flows=i)
    #     #     samps.append(numpy(z))
    #     # samps = np.array(samps)
    #     # ax.scatter(samps.T[0], samps.T[1], alpha=.1, c='Black')
    #     # ax.set_yticks([])
    #     # ax.set_xticks([])
    #     # plt.gca().set_aspect('equal', adjustable='box') 
    #     # ax.set_xlim(left=xlimits[0], right=xlimits[1])
    #     # ax.set_ylim(ylimits)




    plt_path = images_dir+'1d_plot3.png'
    plt.savefig(plt_path)
    print ('saved plot', plt_path)
    plt.close()























































































