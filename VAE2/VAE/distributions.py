


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









































class Flow1_Cond(nn.Module):

    def __init__(self, z_size, h_size, n_flows=2): ##, mean, logvar):
        #mean,logvar: [B,Z]
        super(Flow1_Cond, self).__init__()

        self.z_size = z_size 
        self.h_size = h_size   #this is the datapoint encoding
        h_size_enc = 50


        self.n_flows = n_flows
        h_s = 50

        self.z_half_size = int(self.z_size / 2)


        self.h_encoder = nn.Linear(h_size, h_size_enc)

        
        self.flow_params = []
        for i in range(n_flows):
            #first is for v, second is for z
            self.flow_params.append([
                                [nn.Linear(self.z_half_size + h_size_enc, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)],
                                [nn.Linear(self.z_half_size + h_size_enc, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)]
                                ])
        
        count =1
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
    



    def norm_flow(self, params, z1, z2, x):


        # print (z.size())
        h = torch.tanh(params[0][0](torch.cat([z1,x],1)))
        mew_ = params[0][1](h)
        # sig_ = F.sigmoid(params[0][2](h)+5.) #[PB,Z]
        sig_ = torch.sigmoid(params[0][2](h)) #[PB,Z]

        z2 = z2*sig_ + mew_
        logdet = torch.sum(torch.log(sig_), 1)


        h = torch.tanh(params[1][0](torch.cat([z2,x],1)))
        mew_ = params[1][1](h)
        # sig_ = F.sigmoid(params[1][2](h)+5.) #[PB,Z]
        sig_ = torch.sigmoid(params[1][2](h)) #[PB,Z]
        z1 = z1*sig_ + mew_
        logdet2 = torch.sum(torch.log(sig_), 1)

        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet





    def sample(self, mean, logvar, h):

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


        h = self.h_encoder(h)

        #Transform
        logdetsum = 0.
        for i in range(self.n_flows):

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z1, z2, logdet = self.norm_flow(params,z1,z2,h)
            logdetsum += logdet

        logdetsum = logdetsum.view(self.B)

        #Put z back together  [PB,Z]
        z = torch.cat([z1,z2],1)

        # z = z.view(self.B, self.z_size)


        logqz = logqz0-logdetsum

        logpz = lognormal(z, torch.zeros(self.B, self.z_size).cuda(), 
                            torch.zeros(self.B, self.z_size).cuda())

        return z, logpz, logqz





    def norm_flow_reverse(self, params, z1, z2, x):

        # print (torch.cat([z2,x],1).shape)


        h = torch.tanh(params[1][0](torch.cat([z2,x],1)))
        mew_ = params[1][1](h)
        sig_ = torch.sigmoid(params[1][2](h)) #[PB,Z]
        z1 = (z1 - mew_) / sig_
        logdet2 = torch.sum(torch.log(sig_), 1)

        h = torch.tanh(params[0][0](torch.cat([z1,x],1)))
        mew_ = params[0][1](h)
        sig_ = torch.sigmoid(params[0][2](h)) #[PB,Z]
        z2 = (z2 - mew_) / sig_
        logdet = torch.sum(torch.log(sig_), 1)
        
        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet




    def logprob(self, z, mu, logvar, h):

        #reverse z_T to z_0 to get q0(z0)

        B = z.shape[0]

        #Split z  [B,Z/2]
        z1 = z.narrow(1, 0, self.z_half_size)
        z2 = z.narrow(1, self.z_half_size, self.z_half_size) 

        h = self.h_encoder(h)

        #Reverse Transform
        logdetsum = 0.
        reverse_ = list(range(self.n_flows))[::-1]
        for i in reverse_:

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z1, z2, logdet = self.norm_flow_reverse(params,z1,z2,h)
            logdetsum += logdet

        logdetsum = logdetsum.view(B)


        z0 = torch.cat([z1,z2],1)
        logqz0 = lognormal(z0, mu, logvar)

        logqz = logqz0-logdetsum

        return logqz 


































































class ConvBlock(nn.Module):
    def __init__(self, in_features):
        super(ConvBlock, self).__init__()

        self.f = nn.Sequential(nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3))

    def forward(self, x):
        return self.f(x)



class Flow1_grid(nn.Module):

    def __init__(self, z_shape, n_flows=2):#, mean, logvar):
        #mean,logvar: [B,Z]
        super(Flow1_grid, self).__init__()

        # z_shape = z_shape #[C,H,W]
        n_channels = z_shape[0]
        half_channels = n_channels //2
        self.n_flows = n_flows

        self.flows = {}
        count = 0
        for i in range(n_flows):

            self.flows[str(i)] = {}

            perm = np.random.permutation(n_channels)
            f1_mu = ConvBlock(half_channels)
            f1_sig = ConvBlock(half_channels)
            f2_mu = ConvBlock(half_channels)
            f2_sig = ConvBlock(half_channels)
            
            self.flows[str(i)]['perm'] = perm
            self.flows[str(i)]['inv_perm'] = np.argsort(perm)
            self.flows[str(i)]['f1_mu'] = f1_mu
            self.flows[str(i)]['f1_sig'] = f1_sig
            self.flows[str(i)]['f2_mu'] = f2_mu
            self.flows[str(i)]['f2_sig'] = f2_sig

            self.add_module(str(count), f1_mu)
            count+=1
            self.add_module(str(count), f1_sig)
            count+=1
            self.add_module(str(count), f2_mu)
            count+=1
            self.add_module(str(count), f2_sig)
            count+=1




    def sample(self, shape, eps=None):

        C = shape[1]
        if eps is None:
            eps = torch.FloatTensor(shape).normal_().cuda() #[B,C,H,W]

        f = self.flows

        logdet = 0.
        z = eps
        for i in range(self.n_flows):
            # z = z[:,f[str(i)]['perm']]
            z1 = eps[:,:C//2]
            z2 = eps[:,C//2:]

            sig2 = torch.sigmoid(f[str(i)]['f1_sig'](z2))
            mu2 = f[str(i)]['f1_mu'](z2)

            z1 = z1*sig2 + mu2

            mu1 = f[str(i)]['f2_mu'](z1)
            sig1 = torch.sigmoid(f[str(i)]['f2_sig'](z1))

            z2 = z2*sig1 + mu1
            z = torch.cat([z1,z2],1)

            # sig1 = sig1.view(B, -1)
            # sig2 = sig2.view(B, -1)
            # logdet += torch.sum(torch.log(sig1), 1)
            # logdet += torch.sum(torch.log(sig2), 1)

        return z





    def logprob(self, z):

        B = z.shape[0]
        C = z.shape[1]

        f = self.flows

        logdet = 0.
        reverse_ = list(range(self.n_flows))[::-1]
        for i in reverse_:
            z1 = z[:,:C//2]
            z2 = z[:,C//2:]
            sig1 = torch.sigmoid(f[str(i)]['f2_sig'](z1))
            mu1 = f[str(i)]['f2_mu'](z1)

            z2 = (z2 - mu1) / sig1

            sig2 = torch.sigmoid(f[str(i)]['f1_sig'](z2))
            mu2 = f[str(i)]['f1_mu'](z2)

            z1 = (z1 - mu2) / sig2
            
            z = torch.cat([z1,z2],1)
            # z = z[:,f[str(i)]['inv_perm']]

            sig1 = sig1.view(B, -1)
            sig2 = sig2.view(B, -1)
            logdet += torch.sum(torch.log(sig1), 1)
            logdet += torch.sum(torch.log(sig2), 1)

        

        flat_z = z.view(B, -1)
        logpz0 = lognormal(flat_z, torch.zeros(B, 384).cuda(), torch.zeros(B, 384).cuda()) #[B]

        logpz = logpz0 - logdet


        return logpz 

























class Gauss(nn.Module):

    def __init__(self, z_shape):#, mean, logvar):
        #mean,logvar: [B,Z]
        super(Gauss, self).__init__()

        # z_shape = z_shape #[C,H,W]
        self.n_channels = z_shape[0]


    def sample(self, mu, logvar):

        B = mu.shape[0]
        eps = torch.FloatTensor(B,6,8,8).normal_().cuda() #[B,Z]

        z = eps.mul(torch.exp(.5*logvar)) + mu  #[B,Z]

        flat_z = z.view(B, -1)
        logqz = lognormal(flat_z, mu.view(B, -1).detach(), logvar.view(B, -1).detach())

        return z, logqz

















class ConvBlock(nn.Module):
    def __init__(self, in_features):
        super(ConvBlock, self).__init__()

        self.f = nn.Sequential(nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3))

    def forward(self, x):
        return self.f(x)



class Flow1_grid_cond(nn.Module):

    def __init__(self, z_shape, n_flows=2):#, mean, logvar):
        #mean,logvar: [B,Z]
        super(Flow1_grid_cond, self).__init__()

        # z_shape = z_shape #[C,H,W]
        self.n_channels = z_shape[0]
        half_channels = self.n_channels //2
        self.n_flows = n_flows

        self.flows = {}
        count = 0
        for i in range(n_flows):

            self.flows[str(i)] = {}

            perm = np.random.permutation(self.n_channels)
            f1_mu = ConvBlock(half_channels)
            f1_sig = ConvBlock(half_channels)
            f2_mu = ConvBlock(half_channels)
            f2_sig = ConvBlock(half_channels)
            
            self.flows[str(i)]['perm'] = perm
            self.flows[str(i)]['inv_perm'] = np.argsort(perm)
            self.flows[str(i)]['f1_mu'] = f1_mu
            self.flows[str(i)]['f1_sig'] = f1_sig
            self.flows[str(i)]['f2_mu'] = f2_mu
            self.flows[str(i)]['f2_sig'] = f2_sig

            self.add_module(str(count), f1_mu)
            count+=1
            self.add_module(str(count), f1_sig)
            count+=1
            self.add_module(str(count), f2_mu)
            count+=1
            self.add_module(str(count), f2_sig)
            count+=1




    def sample(self, mu, logvar):

        B = mu.shape[0]
        eps = torch.FloatTensor(B,6,8,8).normal_().cuda() #[B,Z]

        z = eps.mul(torch.exp(.5*logvar)) + mu  #[B,Z]

        flat_z = z.view(B, -1)
        logqz = lognormal(flat_z, mu.view(B, -1).detach(), logvar.view(B, -1).detach())

        # C = shape[1]
        # if eps is None:
        #     eps = torch.FloatTensor(shape).normal_().cuda() #[B,C,H,W]

        f = self.flows
        C = self.n_channels

        logdet = 0.
        # z = eps
        for i in range(self.n_flows):
            # z = z[:,f[str(i)]['perm']]
            z1 = eps[:,:C//2]
            z2 = eps[:,C//2:]
            sig1 = torch.sigmoid(f[str(i)]['f2_sig'](z1))
            sig2 = torch.sigmoid(f[str(i)]['f1_sig'](z2))
            mu1 = f[str(i)]['f2_mu'](z1)
            mu2 = f[str(i)]['f1_mu'](z2)
            z1 = z1*sig2 + mu2
            z2 = z2*sig1 + mu1
            z = torch.cat([z1,z2],1)

            sig1 = sig1.view(B, -1)
            sig2 = sig2.view(B, -1)
            logdet += torch.sum(torch.log(sig1), 1)
            logdet += torch.sum(torch.log(sig2), 1)


        return z, logqz - logdet





    # def logprob(self, z):
 
    #     B = z.shape[0]
    #     C = z.shape[1]

    #     f = self.flows

    #     logdet = 0.
    #     reverse_ = list(range(self.n_flows))[::-1]
    #     for i in reverse_:
    #         z1 = z[:,:C//2]
    #         z2 = z[:,C//2:]
    #         sig1 = torch.sigmoid(f[str(i)]['f2_sig'](z1))
    #         sig2 = torch.sigmoid(f[str(i)]['f1_sig'](z2))
    #         mu1 = f[str(i)]['f2_mu'](z1)
    #         mu2 = f[str(i)]['f1_mu'](z2)
    #         z2 = (z2 - mu1) / sig1
    #         z1 = (z1 - mu2) / sig2
    #         z = torch.cat([z1,z2],1)
    #         z = z[:,f[str(i)]['inv_perm']]

    #         sig1 = sig1.view(B, -1)
    #         sig2 = sig2.view(B, -1)
    #         logdet += torch.sum(torch.log(sig1), 1)
    #         logdet += torch.sum(torch.log(sig2), 1)

        

    #     flat_z = z.view(B, -1)
    #     logpz = lognormal(flat_z, torch.zeros(B, 384).cuda(), 
    #                         torch.zeros(B, 384).cuda()) #[B]

    #     logpz = logpz - logdet


    #     return logpz 















