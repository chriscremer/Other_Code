


# I should combine the code from flow3.py here.


import numpy as np
# import pickle
# import cPickle as pickle
import os
from os.path import expanduser
home = expanduser("~")
import time
import sys
# sys.path.insert(0, 'utils')

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn


import math






def lognormal(x, mean, logvar):
    # return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).mean(1))

    assert len(x.shape) == len(mean.shape)
    
    return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(1))

    # return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(2))



    
def log_bernoulli(pred_no_sig, target):
    '''
    pred_no_sig is [P, B, X] 
    t is [B, X]
    output is [P, B]
    '''
    B = target.shape[0]
    X = pred_no_sig.shape[1]
    pred_no_sig = pred_no_sig.view(1,B,X)

    assert len(pred_no_sig.size()) == 3
    assert len(target.size()) == 2
    assert pred_no_sig.size()[1] == target.size()[0]

    aaa= -(torch.clamp(pred_no_sig, min=0)
                        - pred_no_sig * target
                        + torch.log(1. + torch.exp(-torch.abs(pred_no_sig)))).sum(2) #sum over dimensions

    return aaa.view(B)









class Gauss(nn.Module):

    def __init__(self, z_size, mu=None, logvar=None):
        super(Gauss, self).__init__()

        # self.__dict__.update(kwargs)
        self.z_size = z_size

        # count=0
        if mu is None:
            self.mu = nn.Parameter(torch.zeros(1, self.z_size), requires_grad=True)
            self.logvar = nn.Parameter(torch.zeros(1, self.z_size), requires_grad=True)
        else:
            self.mu = nn.Parameter(mu, requires_grad=True)
            self.logvar = nn.Parameter(logvar, requires_grad=True)

        # self.add_module(str(count), self.mu)
        # count+=1
        # self.add_module(str(count), self.logvar)
        



    def sample(self, mu=None, logvar=None):

        if mu is None:
            mu = self.mu #torch.zeros(B, self.z_size).cuda()
            logvar = self.logvar #torch.zeros(B, self.z_size).cuda()
            B = 1
        else:
            B = mu.shape[0]

        eps = torch.FloatTensor(B, self.z_size).normal_().cuda() #[B,Z]

        z = eps.mul(torch.exp(.5*logvar)) + mu  #[B,Z]

        #maybe mu and logvar should be deatched..
        logprob = lognormal(z, mu, logvar)

        return z, logprob



    def logprob(self, z, mu=None, logvar=None):

        B = z.shape[0]

        if mu is None:
            mu = self.mu #torch.zeros(B, self.z_size).cuda()
            logvar = self.logvar #torch.zeros(B, self.z_size).cuda()
            # mu = torch.zeros(B, self.z_size).cuda()
            # logvar = torch.zeros(B, self.z_size).cuda()

        return lognormal(z, mu, logvar)









class Flow1(nn.Module):

    def __init__(self, kwargs, mu=None, logvar=None):#, mean, logvar):

        super(Flow1, self).__init__()

        self.__dict__.update(kwargs)

        # self.z_size = z_size #hyper_config['z_size']
        

        count =1

        if mu is not None:
            self.mu = nn.Parameter(mu, requires_grad=True)
            self.logvar = nn.Parameter(logvar, requires_grad=True)

        # n_flows = 2
        n_flows = kwargs['n_flows']
        self.n_flows = n_flows
        h_s = 50

        self.z_half_size = int(self.z_size / 2)

        
        self.flow_params = []
        for i in range(n_flows):
            #first is for v, second is for z
            self.flow_params.append([
                                [nn.Linear(self.z_half_size, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)],
                                [nn.Linear(self.z_half_size, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)]
                                ])
        
        for i in range(n_flows):

            self.add_module(str(count), self.flow_params[i][0][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][2])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][2])
            count+=1
    



    def norm_flow(self, params, z1, z2):
        # print (z.size())
        h = torch.tanh(params[0][0](z1))
        mew_ = params[0][1](h)
        # sig_ = F.sigmoid(params[0][2](h)+5.) #[PB,Z]
        sig_ = torch.sigmoid(params[0][2](h)) #[PB,Z]

        z2 = z2*sig_ + mew_
        logdet = torch.sum(torch.log(sig_), 1)


        h = torch.tanh(params[1][0](z2))
        mew_ = params[1][1](h)
        # sig_ = F.sigmoid(params[1][2](h)+5.) #[PB,Z]
        sig_ = torch.sigmoid(params[1][2](h)) #[PB,Z]
        z1 = z1*sig_ + mew_
        logdet2 = torch.sum(torch.log(sig_), 1)

        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet





    def sample(self, mean=None, logvar=None, k=1):

        if mean is None:
            mean = self.mu #torch.zeros(B, self.z_size).cuda()
            logvar = self.logvar #torch.zeros(B, self.z_size).cuda()
            B = 1
        else:
            B = mean.shape[0]

        # self.B = mean.size()[0]
        # gaus = Gaussian(self.z_size)

        # q(z0)
        # z, logqz0 = gaus.sample(mean, logvar, k)

        eps = torch.FloatTensor(B, self.z_size).normal_().cuda() #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz0 = lognormal(z, mean, logvar) #[P,B]



        #[PB,Z]
        # z = z.view(-1,self.z_size)
        # v = v.view(-1,self.z_size)

        #Split z  [PB,Z/2]
        z1 = z.narrow(1, 0, self.z_half_size)
        z2 = z.narrow(1, self.z_half_size, self.z_half_size) 

        #Transform
        logdetsum = 0.
        for i in range(self.n_flows):

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z1, z2, logdet = self.norm_flow(params,z1,z2)
            logdetsum += logdet

        logdetsum = logdetsum.view(B)

        #Put z back together  [PB,Z]
        z = torch.cat([z1,z2],1)

        # z = z.view(self.B, self.z_size)


        logqz = logqz0-logdetsum

        # logpz = lognormal(z, torch.zeros(self.B, self.z_size).cuda(), 
        #                     torch.zeros(self.B, self.z_size).cuda())

        return z, logqz, #logpz, logqz





    def norm_flow_reverse(self, params, z1, z2):

        h = torch.tanh(params[1][0](z2))
        mew_ = params[1][1](h)
        sig_ = torch.sigmoid(params[1][2](h)) #[PB,Z]
        z1 = (z1 - mew_) / sig_
        logdet2 = torch.sum(torch.log(sig_), 1)

        h = torch.tanh(params[0][0](z1))
        mew_ = params[0][1](h)
        sig_ = torch.sigmoid(params[0][2](h)) #[PB,Z]
        z2 = (z2 - mew_) / sig_
        logdet = torch.sum(torch.log(sig_), 1)
        
        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet




    def logprob(self, z, mu, logvar):

        #reverse z_T to z_0 to get q0(z0)

        B = z.shape[0]

        #Split z  [B,Z/2]
        z1 = z.narrow(1, 0, self.z_half_size)
        z2 = z.narrow(1, self.z_half_size, self.z_half_size) 


        #Reverse Transform
        logdetsum = 0.
        reverse_ = list(range(self.n_flows))[::-1]
        for i in reverse_:

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z1, z2, logdet = self.norm_flow_reverse(params,z1,z2)
            logdetsum += logdet

        logdetsum = logdetsum.view(B)


        z0 = torch.cat([z1,z2],1)
        logqz0 = lognormal(z0, mu, logvar)

        logqz = logqz0-logdetsum

        return logqz 










class Flow1(nn.Module):

    def __init__(self, kwargs, mu=None, logvar=None):#, mean, logvar):

        super(Flow1, self).__init__()

        self.__dict__.update(kwargs)

        # self.z_size = z_size #hyper_config['z_size']
        

        count =1

        if mu is not None:
            self.mu = nn.Parameter(mu, requires_grad=True)
            self.logvar = nn.Parameter(logvar, requires_grad=True)

        # n_flows = 2
        n_flows = kwargs['n_flows']
        self.n_flows = n_flows
        h_s = 50

        self.z_half_size = int(self.z_size / 2)

        
        self.flow_params = []
        for i in range(n_flows):
            #first is for v, second is for z
            self.flow_params.append([
                                [nn.Linear(self.z_half_size, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)],
                                [nn.Linear(self.z_half_size, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)]
                                ])
        
        for i in range(n_flows):

            self.add_module(str(count), self.flow_params[i][0][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][2])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][2])
            count+=1
    



    def norm_flow(self, params, z1, z2):
        # print (z.size())
        h = torch.tanh(params[0][0](z1))
        mew_ = params[0][1](h)
        # sig_ = F.sigmoid(params[0][2](h)+5.) #[PB,Z]
        sig_ = torch.sigmoid(params[0][2](h)) #[PB,Z]

        z2 = z2*sig_ + mew_
        logdet = torch.sum(torch.log(sig_), 1)


        h = torch.tanh(params[1][0](z2))
        mew_ = params[1][1](h)
        # sig_ = F.sigmoid(params[1][2](h)+5.) #[PB,Z]
        sig_ = torch.sigmoid(params[1][2](h)) #[PB,Z]
        z1 = z1*sig_ + mew_
        logdet2 = torch.sum(torch.log(sig_), 1)

        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet





    def sample(self, mean=None, logvar=None, k=1):

        if mean is None:
            mean = self.mu #torch.zeros(B, self.z_size).cuda()
            logvar = self.logvar #torch.zeros(B, self.z_size).cuda()
            B = 1
        else:
            B = mean.shape[0]

        # self.B = mean.size()[0]
        # gaus = Gaussian(self.z_size)

        # q(z0)
        # z, logqz0 = gaus.sample(mean, logvar, k)

        eps = torch.FloatTensor(B, self.z_size).normal_().cuda() #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz0 = lognormal(z, mean, logvar) #[P,B]



        #[PB,Z]
        # z = z.view(-1,self.z_size)
        # v = v.view(-1,self.z_size)

        #Split z  [PB,Z/2]
        z1 = z.narrow(1, 0, self.z_half_size)
        z2 = z.narrow(1, self.z_half_size, self.z_half_size) 

        #Transform
        logdetsum = 0.
        for i in range(self.n_flows):

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z1, z2, logdet = self.norm_flow(params,z1,z2)
            logdetsum += logdet

        logdetsum = logdetsum.view(B)

        #Put z back together  [PB,Z]
        z = torch.cat([z1,z2],1)

        # z = z.view(self.B, self.z_size)


        logqz = logqz0-logdetsum

        # logpz = lognormal(z, torch.zeros(self.B, self.z_size).cuda(), 
        #                     torch.zeros(self.B, self.z_size).cuda())

        return z, logqz, #logpz, logqz





    def norm_flow_reverse(self, params, z1, z2):

        h = torch.tanh(params[1][0](z2))
        mew_ = params[1][1](h)
        sig_ = torch.sigmoid(params[1][2](h)) #[PB,Z]
        z1 = (z1 - mew_) / sig_
        logdet2 = torch.sum(torch.log(sig_), 1)

        h = torch.tanh(params[0][0](z1))
        mew_ = params[0][1](h)
        sig_ = torch.sigmoid(params[0][2](h)) #[PB,Z]
        z2 = (z2 - mew_) / sig_
        logdet = torch.sum(torch.log(sig_), 1)
        
        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet




    def logprob(self, z, mu, logvar):

        #reverse z_T to z_0 to get q0(z0)

        B = z.shape[0]

        #Split z  [B,Z/2]
        z1 = z.narrow(1, 0, self.z_half_size)
        z2 = z.narrow(1, self.z_half_size, self.z_half_size) 


        #Reverse Transform
        logdetsum = 0.
        reverse_ = list(range(self.n_flows))[::-1]
        for i in reverse_:

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z1, z2, logdet = self.norm_flow_reverse(params,z1,z2)
            logdetsum += logdet

        logdetsum = logdetsum.view(B)


        z0 = torch.cat([z1,z2],1)
        logqz0 = lognormal(z0, mu, logvar)

        logqz = logqz0-logdetsum

        return logqz 

















class Flow_Cond(nn.Module):

    def __init__(self, kwargs, mu=None, logvar=None):#, mean, logvar):

        super(Flow_Cond, self).__init__()

        self.__dict__.update(kwargs)

        # self.z_size = z_size #hyper_config['z_size']
        

        count =1

        if mu is not None:
            self.mu = nn.Parameter(mu, requires_grad=True)
            self.logvar = nn.Parameter(logvar, requires_grad=True)

        # n_flows = 2
        n_flows = kwargs['n_flows']
        self.n_flows = n_flows
        h_s = 50

        self.z_half_size = int(self.z_size / 2)

        
        self.flow_params = []
        for i in range(n_flows):
            #first is for v, second is for z
            self.flow_params.append([
                                [nn.Linear(self.z_half_size + self.z_size, h_s), nn.Linear(h_s, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)],
                                [nn.Linear(self.z_half_size + self.z_size, h_s), nn.Linear(h_s, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)]
                                ])
        
        for i in range(n_flows):

            self.add_module(str(count), self.flow_params[i][0][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][2])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][2])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][3])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][3])
            count+=1    



    def norm_flow(self, params, z1, z2, mean):

        h = torch.cat((z1,mean), dim=1)
        h = torch.tanh(params[0][0](h))
        h = torch.tanh(params[0][1](h))

        mew_ = params[0][2](h)
        # sig_ = F.sigmoid(params[0][2](h)+5.) #[PB,Z]
        sig_ = torch.sigmoid(params[0][3](h)) #[PB,Z]

        z2 = z2*sig_ + mew_
        logdet = torch.sum(torch.log(sig_), 1)

        h = torch.cat((z2,mean), dim=1)
        h = torch.tanh(params[1][0](h))
        h = torch.tanh(params[1][1](h))
        mew_ = params[1][2](h)
        # sig_ = F.sigmoid(params[1][2](h)+5.) #[PB,Z]
        sig_ = torch.sigmoid(params[1][3](h)) #[PB,Z]
        z1 = z1*sig_ + mew_
        logdet2 = torch.sum(torch.log(sig_), 1)

        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet





    def sample(self, mean=None, logvar=None, k=1):

        if mean is None:
            mean = self.mu #torch.zeros(B, self.z_size).cuda()
            logvar = self.logvar #torch.zeros(B, self.z_size).cuda()
            B = 1
        else:
            B = mean.shape[0]

        # self.B = mean.size()[0]
        # gaus = Gaussian(self.z_size)

        # q(z0)
        # z, logqz0 = gaus.sample(mean, logvar, k)

        eps = torch.FloatTensor(B, self.z_size).normal_().cuda() #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz0 = lognormal(z, mean, logvar) #[P,B]



        #[PB,Z]
        # z = z.view(-1,self.z_size)
        # v = v.view(-1,self.z_size)

        #Split z  [PB,Z/2]
        z1 = z.narrow(1, 0, self.z_half_size)
        z2 = z.narrow(1, self.z_half_size, self.z_half_size) 

        #Transform
        logdetsum = 0.
        for i in range(self.n_flows):

            params = self.flow_params[i]

            # z1 = torch.cat((z1,mean), dim=1)
            # z2 = torch.cat((z2,mean), dim=1)

            z1, z2, logdet = self.norm_flow(params,z1,z2,mean)
            logdetsum += logdet

        logdetsum = logdetsum.view(B)

        #Put z back together  [PB,Z]
        z = torch.cat([z1,z2],1)

        # z = z.view(self.B, self.z_size)


        logqz = logqz0-logdetsum

        # logpz = lognormal(z, torch.zeros(self.B, self.z_size).cuda(), 
        #                     torch.zeros(self.B, self.z_size).cuda())

        return z, logqz, #logpz, logqz





    def norm_flow_reverse(self, params, z1, z2):

        h = torch.tanh(params[1][0](z2))
        mew_ = params[1][1](h)
        sig_ = torch.sigmoid(params[1][2](h)) #[PB,Z]
        z1 = (z1 - mew_) / sig_
        logdet2 = torch.sum(torch.log(sig_), 1)

        h = torch.tanh(params[0][0](z1))
        mew_ = params[0][1](h)
        sig_ = torch.sigmoid(params[0][2](h)) #[PB,Z]
        z2 = (z2 - mew_) / sig_
        logdet = torch.sum(torch.log(sig_), 1)
        
        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet




    def logprob(self, z, mu, logvar):

        #reverse z_T to z_0 to get q0(z0)

        B = z.shape[0]

        #Split z  [B,Z/2]
        z1 = z.narrow(1, 0, self.z_half_size)
        z2 = z.narrow(1, self.z_half_size, self.z_half_size) 


        #Reverse Transform
        logdetsum = 0.
        reverse_ = list(range(self.n_flows))[::-1]
        for i in reverse_:

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z1, z2, logdet = self.norm_flow_reverse(params,z1,z2)
            logdetsum += logdet

        logdetsum = logdetsum.view(B)


        z0 = torch.cat([z1,z2],1)
        logqz0 = lognormal(z0, mu, logvar)

        logqz = logqz0-logdetsum

        return logqz 







































