

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

# from utils import lognormal2 as lognormal
# from utils import log_bernoulli


# from generator import Generator

# from distributions import Gauss
# from inference_net import inf_net

from torch.distributions import Beta




class VAE(nn.Module):
    def __init__(self, kwargs, seed=1):
        super(VAE, self).__init__()

        # self.__dict__.update(kwargs)
        # torch.manual_seed(seed)

        #q(z|x)
        self.encoder = kwargs['encoder'] #Inference_Q_flow(kwargs)
        # p(z)
        self.prior = kwargs['prior'] #Flow1_grid(z_shape=[6,8,8], n_flows=self.n_prior_flows)
        # p(x|z)
        self.generator = kwargs['generator'] #Generator(hyper_config=hyper_config)


        self.beta_scale = 100.


    def forward(self, x, warmup=1.):

        B = x.shape[0]
        outputs = {}

        # if inf_net is None:
        z, logqz = self.encoder.sample(x) 
        # else:
        #     z, logqz = inf_net.sample(x) 
        # print (logqz.shape)

        logpz = self.prior.logprob(z)
        # print (logpz.shape)


        x_hat = self.generator.decode(x,z)
        alpha = torch.sigmoid(x_hat)
        beta = Beta(alpha*self.beta_scale, (1.-alpha)*self.beta_scale)
        logpxz = beta.log_prob(x) #[120,3,112,112]
        logpxz = torch.sum(logpxz.view(B, -1),1) # [B]  * self.w_logpx
        # print (logpxz.shape)
        # logpxz = logpxz * .02 #self.w_logpx

        # #Compute elbo
        # elbo = logpxz - (warmup*logqz) #[P,B]
        # if k>1:
        #     max_ = torch.max(elbo, 0)[0] #[B]
        #     elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
        elbo = logpxz + logpz - logqz
        welbo = logpxz + warmup*(logpz - logqz)
            
        # elbo = torch.mean(elbo) #[1]
        outputs['logpxz'] = torch.mean(logpxz) #[1]
        outputs['logqz'] = torch.mean(logqz)
        outputs['logpz'] = torch.mean(logpz)
        outputs['elbo'] = torch.mean(elbo)
        outputs['welbo'] = torch.mean(welbo)
        outputs['x_hat'] = alpha
        # outputs['elbo_B'] = elbo

        return outputs







    def load_params_v3(self, save_dir, step, name):
        save_to=os.path.join(save_dir, name + str(step)+".pt")
        state_dict = torch.load(save_to)
        # # # print (state_dict)
        # for key, val in state_dict.items():
        #     print (key)
        # fddsf
        self.load_state_dict(state_dict)
        print ('loaded params', save_to)


    def save_params_v3(self, save_dir, step, name):
        save_to=os.path.join(save_dir, name + str(step)+".pt")
        torch.save(self.state_dict(), save_to)
        print ('saved params', save_to)
        



















































