

#using bernoulli distribution and straight through estimator


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

from torch.distributions import Bernoulli
from torch.autograd import Function


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


def sample_normal(N, mean, logvar):

    e = torch.randn(N, 2)
    z = e*torch.exp(.5*logvar) + mean

    return z





def log_targetdist(x, D=2):

    # return self.log_normal(x, tf.zeros(self.D), tf.ones(self.D))
    px = (torch.exp(log_normal(x, tensor([1,3]), torch.ones(D)/100))
                + torch.exp(log_normal(x, tensor([3,0]), torch.ones(D)/100))
                + torch.exp(log_normal(x, tensor([1,1]), torch.ones(D)/100))
                + torch.exp(log_normal(x, tensor([-3,-1]), torch.ones(D)/100))
                + torch.exp(log_normal(x, tensor([-1,-3]), torch.ones(D)/100))
                + torch.exp(log_normal(x, tensor([2,2]), torch.ones(D)))
                + torch.exp(log_normal(x, tensor([-2,-2]), torch.ones(D)))
                + torch.exp(log_normal(x, tensor([0,0]), torch.ones(D)))) / 7.8104

    px = torch.clamp(px, min=1e-10, max=1-1e-10)

    return torch.log(px)










class flow1(nn.Module):

    def __init__(self, n_flows):
        super(flow1, self).__init__()

        self.D = 2

        self.n_flows = n_flows
        self.p0_mean = torch.tensor([0,0]).float()
        self.p0_logvar = torch.tensor([0.8,0.8]).float()

        #TODO: add params for the mask

        self.masks = []
        self.nets = []
        params = []
        # mask = torch.ones(2)
        # mask[0] = 0
        for i in range(n_flows):

            # mask = torch.randn(2, requires_grad=True)
            mask = torch.ones(2) * 0. #, requires_grad=True) * .5
            mask.requires_grad_()


            # print (mask)
            self.masks.append(mask)
            # mask = 1-mask
            



            L = 30 #100 #30
            layer = [nn.Linear(2, L), nn.ReLU(), nn.Linear(L, L), nn.ReLU(), nn.Linear(L, 2)]
            # layer = [nn.Linear(2, 100), nn.ReLU(), nn.Linear(100, 2)]
            # layer = [nn.Linear(2, 2)]
            seq = nn.Sequential(*layer)
            self.nets.append(seq)

            
            params.extend(list(seq.parameters()))

        params.extend(list(self.masks))
        # params = [list(self.seq.parameters())]
        self.optimizer = optim.Adam(params, lr=.001, weight_decay=.0000001)

        # print (params)
        # print (len(params))
        # fsda


    def transform(self, z, pre_mask, net, reverse=False, temp=1.):
        B = z.shape[0]

        #TODO sigmoid of mask and modify how other half is transformed
        # print (mask)

        # temp = 53.

        # print (pre_mask)
        # fafds

        
        # print (torch.sigmoid(pre_mask))
        # print 
        # print(m.sample())  # 30% chance 1; 70% chance 0
        # fasdf

        # m = Bernoulli(torch.sigmoid(pre_mask))
        # mask = m.sample()

        probs = torch.sigmoid(pre_mask)

        if temp != -1:
            mask = bern_sampler(probs)
        else:
            mask = (probs > .5).float()



        # mask = torch.sigmoid(pre_mask* temp) #, temp=temp)

        # if temp > 50:
        #     mask = (mask > .5).float()

        # print (mask)
        # fasda

        # print (pre_mask)
        # print (mask)
        # mask = torch.sigmoid(pre_mask *2) #, temp=temp)
        # print ()
        # print (pre_mask *2)
        # print (mask)
        # mask = torch.sigmoid(pre_mask *20) #, temp=temp)
        # print ()
        # print (pre_mask *20)
        # print (mask)
        # fsadf

        z_constant = z * mask
        mu_sig = net(z_constant)
        mu = mu_sig[:,0].view(B,1)
        # sig = mu_sig[:,1].view(B,1)
        # sig = torch.tanh(mu_sig[:,1].view(B,1)) 
        sig = torch.sigmoid(mu_sig[:,1].view(B,1)) 

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



    def sample(self, N=1, n_flows=-1, temp=100.):

        if n_flows==-1:
            n_flows = self.n_flows
        # else:
        #     print (n_flows)

        z0 = sample_normal(N, self.p0_mean, self.p0_logvar) #.view(1,2)
        logqz0 = log_normal(z0, self.p0_mean, self.p0_logvar).view(N,1)

        z = z0
        logdetsum = 0
        for i in range(n_flows):
            z, logdet = self.transform(z, pre_mask=self.masks[i], net=self.nets[i], temp=temp)
            logdetsum += logdet
        zT = z

        logprob = logqz0 - logdetsum
        return zT, logprob.view(N)


    def logprob(self, z, n_flows=-1, temp=100.):

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
            z, logdet = self.transform(z, pre_mask=self.masks[i], net=self.nets[i], reverse=True, temp=temp)
            logdetsum += logdet
        z0 = z

        logqz0 = log_normal(z0, self.p0_mean, self.p0_logvar)
        logprob = logqz0 + logdetsum.view(B)

        return logprob











if __name__ == "__main__":


    save_to_dir = home + "/Documents/Flow/"
    exp_name = 'learn_part'


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











    # bb = torch.tensor(2)
    # a = Bernoulli(.5)
    # a.sample(bb)

    # print( torch.bernoulli(torch.ones([2]).float()*.8))
    # print( torch.bernoulli(2))

    # fsdfa





    # class Exp(Function):

    #      @staticmethod
    #      def forward(ctx, i):
    #          result = i.exp()
    #          ctx.save_for_backward(result)
    #          return result

    #      @staticmethod
    #      def backward(ctx, grad_output):
    #          result, = ctx.saved_tensors
    #          return grad_output * result




    class Sample_Bern(Function):

         @staticmethod
         def forward(ctx, probs):
             # result = i.exp()

             u = torch.rand(probs.shape)
             result = (u<probs).float()

             ctx.save_for_backward(result)
             return result

         @staticmethod
         def backward(ctx, grad_output):
             result, = ctx.saved_tensors
             # print (result, grad_output)
             # STRAIGHT THROUGH
             return grad_output #grad_output * result


    bern_sampler = Sample_Bern.apply

    # bb=[]
    # for i in range(1000):
    #     mask = bern_sampler(torch.ones(1, requires_grad=True)  *.2)
    #     bb.append(mask.data.numpy())
    # print (np.mean(bb))
    # fasdf

    # # import pdb; pdb.set_trace()


    # # exp_func = Exp.apply
    # 
    # a = torch.ones(1, requires_grad=True) # *.5
    # c = a *.5
    # # b = exp_func(a)

    # make sure not to have in-place fucniton
        # so a basically cant be modified like a = a*.5 because you lose what a was.. 
    

    # print(c)
    # b = func(c)
    # # b = 3*a
    # print (b)


    # print (a.grad)
    # print (b.grad)
    # torch.sum(b).backward(retain_graph=True)
    # # torch.autograd.backward(torch.sum(b))
    # print (a.grad)
    # print (b.grad)
    # # print (b.grad_fn)


    # # print (torch.autograd.grad(torch.sum(b), a))


    # # b = func(a)
    # # print (b)
    # # b = func(a)
    # # print (b)
    # # b = func(a)
    # # print (b)
    # # b = func(a)
    # # print (b)
    # # b = func(a)
    # # print (b)
    # # print (a.grad)
    # # b.backward()
    # # print (a.grad)
    # # # print (a)
    # fads


























    n_flows = 3
    flow = flow1(n_flows=n_flows)
    # print (flow)
    # fsf
    batch_size = 128 #64 #32

    

    for i in range(2000):

        # print ()
        # fas
        # temp = np.minimum(i/5. + 1, 51.)

        # print (i, temp)


        z, logprob = flow.sample(batch_size) #, temp=temp)
        logtarget = log_targetdist(z) #.detach()

        loss = logprob - logtarget
        loss = torch.mean(loss)

        if (loss!=loss).any() or loss > 999999:
            print (i)
            # print ('temp', temp)
            print ('z', z)
            print ('logprob', logprob)
            print ('logtarget', logtarget)
            fdsf

        flow.optimizer.zero_grad()

        loss.backward()




        # print (flow.masks[0].grad)
        # print (flow.masks[0])
        # print ()

        # flow.masks[0].grad = 1.
        # print (flow.nets[0].grad)
        # fsdafdf
        

        

        if i%100 == 0:
            print (i,numpy(loss), torch.sigmoid(flow.masks[0]), torch.sigmoid(flow.masks[1]), torch.sigmoid(flow.masks[2])) #, 'temp:', temp)



        flow.optimizer.step()


        # if i %500 ==0:
        #     # print (torch.sigmoid(flow.masks[0]), torch.sigmoid(flow.masks[1]), torch.sigmoid(flow.masks[2]))
        #     print (flow.masks[0].grad, flow.masks[1].grad, flow.masks[2].grad)
        #     print (torch.sigmoid(flow.masks[0]), torch.sigmoid(flow.masks[1]), torch.sigmoid(flow.masks[2]))



    rows = 2 
    cols = 2 + n_flows
    fig = plt.figure(figsize=(8+cols,4+rows), facecolor='white') #, dpi=150)
    # xlimits=[-3, 3]
    # ylimits=[-3, 3]
    xlimits=[-6, 6]
    ylimits=[-6, 6]

    # p0_mean = torch.tensor([0,0]).float()
    # p0_logvar = torch.tensor([0.8,0.8]).float()


    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
    p = lambda x: torch.exp(log_targetdist(tensor(x)))
    plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits, contour_type='')


    for i in range(n_flows+1):


        ax = plt.subplot2grid((rows,cols), (0,1+i), frameon=False)
        
        p = lambda x: torch.exp(log_targetdist(tensor(x)))
        plot_isocontours(ax, p, cmap='Greys', xlimits=xlimits, ylimits=ylimits, contour_type='', alpha=.1)

        p = lambda x: torch.exp(flow.logprob(tensor(x), n_flows=i, temp=-1))
        plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)



        ax = plt.subplot2grid((rows,cols), (1,i+1), frameon=False)

        plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)
       
        n_samps = 400
        samps = []
        for n in range(n_samps):
            z, logprob = flow.sample(n_flows=i, temp=-1)
            samps.append(numpy(z))
        samps = np.array(samps)
        ax.scatter(samps.T[0], samps.T[1], alpha=.1, c='Black')
        ax.set_yticks([])
        ax.set_xticks([])
        plt.gca().set_aspect('equal', adjustable='box') 
        ax.set_xlim(left=xlimits[0], right=xlimits[1])
        ax.set_ylim(ylimits)





    # ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)

    # p = lambda x: torch.exp(log_targetdist(tensor(x)))
    # plot_isocontours(ax, p, cmap='Greys', xlimits=xlimits, ylimits=ylimits, contour_type='', alpha=.1)

    # p = lambda x: torch.exp(flow.logprob(tensor(x)))
    # plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)


        # ax = plt.subplot2grid((rows,cols), (1,1+i), frameon=False)

        # p = lambda x: np.exp(log_normal(tensor(x), flow.p0_mean, flow.p0_logvar))
        # plot_isocontours(ax, p, cmap='Blues', xlimits=xlimits, ylimits=ylimits)

        # n_samps = 400
        # samps = []
        # for i in range(n_samps):
        #     z = sample_normal(1, flow.p0_mean, flow.p0_logvar)
        #     samps.append(numpy(z))
        # samps = np.array(samps)
        # ax.scatter(samps.T[0], samps.T[1], alpha=.05, c='Black')
        # ax.set_yticks([])
        # ax.set_xticks([])
        # plt.gca().set_aspect('equal', adjustable='box') 



    plt_path = images_dir+'plot.png'
    plt.savefig(plt_path)
    print ('saved plot', plt_path)
    plt.close()























































































