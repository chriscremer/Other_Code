


import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import math

# from IAF import *

# Gaussian

def lognormal(x, mean, logvar):
    # return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).mean(1))

    assert len(x.shape) == len(mean.shape)
    
    return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(1))

    # return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(2))



class Flow1(nn.Module):

    def __init__(self, z_size):#, mean, logvar):
        #mean,logvar: [B,Z]
        super(Flow1, self).__init__()

        # if torch.cuda.is_available():
        #     self.dtype = torch.cuda.FloatTensor
        # else:
        #     self.dtype = torch.FloatTensor

        # self.hyper_config = hyper_config
        # self.B = mean.size()[0]
        self.z_size = z_size #hyper_config['z_size']
        # self.x_size = hyper_config['x_size']

        # self.act_func = hyper_config['act_func']
        

        count =1


        n_flows = 2
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





    def sample(self, mean, logvar, k=1):

        self.B = mean.size()[0]
        # gaus = Gaussian(self.z_size)

        # q(z0)
        # z, logqz0 = gaus.sample(mean, logvar, k)

        eps = torch.FloatTensor(self.B, self.z_size).normal_().cuda() #[P,B,Z]
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

        logdetsum = logdetsum.view(self.B)

        #Put z back together  [PB,Z]
        z = torch.cat([z1,z2],1)

        # z = z.view(self.B, self.z_size)


        logqz = logqz0-logdetsum

        logpz = lognormal(z, torch.zeros(self.B, self.z_size).cuda(), 
                            torch.zeros(self.B, self.z_size).cuda())

        return z, logpz, logqz





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











