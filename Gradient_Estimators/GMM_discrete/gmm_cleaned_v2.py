


#reinfoce can do inference here.

from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

import numpy as np

import torch

import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.normal import Normal

import pickle
import subprocess

# import sys, os
# # sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('./new_discrete_estimators'))

from NN import NN3

from plot_functions import *


def to_print1(x):
    return torch.mean(x).data.cpu().numpy()
def to_print2(x):
    return x.data.cpu().numpy()

def sample_gmm(batch_size, mixture_weights):
    cat = Categorical(probs=mixture_weights)
    cluster = cat.sample([batch_size]) # [B]
    mean = (cluster*10.).float().cuda()
    std = torch.ones([batch_size]).cuda() *5.
    norm = Normal(mean, std)
    samp = norm.sample()
    samp = samp.view(batch_size, 1)
    return samp


def logprob_undercomponent(x, component):
    B = x.shape[0]
    mean = (component.float()*10.).view(B,1)
    std = (torch.ones([B]) *5.).view(B,1)
    m = Normal(mean.cuda(), std.cuda())
    logpx_given_z = m.log_prob(x)
    return logpx_given_z


def true_posterior(x, mixture_weights):
    B = x.shape[0]
    probs_ = []
    for c in range(n_components):
        cluster = torch.ones(B,1).cuda() * c
        logpx_given_z = logprob_undercomponent(x, component=cluster)
        pz = mixture_weights[c] 
        pxz = torch.exp(logpx_given_z) * pz #[B,1]
        probs_.append(pxz)
    probs_ = torch.stack(probs_, dim=1)  #[B,C,1]
    sum_ = torch.sum(probs_, dim=1, keepdim=True) #[B,1,1]
    probs_ = probs_ / sum_
    return probs_.view(B,n_components)





def reinforce(x, logits, mixtureweights, k=1):
    B = logits.shape[0]
    probs = torch.softmax(logits, dim=1)

    cat = Categorical(probs=probs)

    net_loss = 0
    for jj in range(k):

        cluster_H = cat.sample()
        logq = cat.log_prob(cluster_H).view(B,1)

        logpx_given_z = logprob_undercomponent(x, component=cluster_H)
        logpz = torch.log(mixtureweights[cluster_H]).view(B,1)
        logpxz = logpx_given_z + logpz #[B,1]
        f = logpxz - logq
        net_loss += - torch.mean((f.detach() - 1.) * logq)
        # net_loss += - torch.mean( -logq.detach()*logq)

    net_loss = net_loss/ k

    return net_loss, f, logpx_given_z, logpz, logq




def train(n_components, needsoftmax_mixtureweight=None):

    if needsoftmax_mixtureweight is None:
        needsoftmax_mixtureweight = torch.randn(n_components, requires_grad=True, device="cuda")
    else:
        needsoftmax_mixtureweight = torch.tensor(needsoftmax_mixtureweight, 
                                            requires_grad=True, device="cuda")

    mixtureweights = torch.softmax(needsoftmax_mixtureweight, dim=0).float() #[C]
    
    encoder = NN3(input_size=1, output_size=n_components, n_residual_blocks=2).cuda()
    # optim_net = torch.optim.SGD(encoder.parameters(), lr=1e-4, weight_decay=1e-7)
    optim_net = torch.optim.Adam(encoder.parameters(), lr=1e-5, weight_decay=1e-7)


    batch_size = 10
    n_steps = 300000
    k = 1

    data_dict = {}
    data_dict['steps'] = []
    data_dict['losses'] = []

    for step in range(n_steps):

        x = sample_gmm(batch_size, mixture_weights=mixtureweights)
        logits = encoder.net(x)

        net_loss, f, logpx_given_z, logpz, logq = reinforce(x, logits, mixtureweights, k=1)

        optim_net.zero_grad()
        net_loss.backward(retain_graph=True)
        optim_net.step()



        if step%200==0:
            # print (step, to_print(net_loss), to_print(logpxz - logq), to_print(logpx_given_z), to_print(logpz), to_print(logq))

            print( 
                'S:{:5d}'.format(step),
                # 'T:{:.2f}'.format(time.time() - start_time),
                'Loss:{:.4f}'.format(to_print1(net_loss)),
                'ELBO:{:.4f}'.format(to_print1(f)),
                'lpx|z:{:.4f}'.format(to_print1(logpx_given_z)),
                'lpz:{:.4f}'.format(to_print1(logpz)),
                'lqz:{:.4f}'.format(to_print1(logq)),
                )

            pz_give_x = true_posterior(x, mixture_weights=mixtureweights)
            probs = torch.softmax(logits, dim=1)

            if step%1000==0:
                print (to_print2(pz_give_x[0]))
                print (to_print2(probs[0]))


    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    with open( exp_dir+"data_simplax.p", "wb" ) as f:
        pickle.dump(data_dict, f)
    print ('saved data')























if __name__ == "__main__":


    exp_name = 'fewercats_2cats_bernoulli'
    save_dir = home+'/Documents/Grad_Estimators/GMM/'


    exp_dir = save_dir + exp_name + '/'
    # params_dir = exp_dir + 'params/'
    # images_dir = exp_dir + 'images/'
    code_dir = exp_dir + 'code/'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print ('Made dir', exp_dir) 

    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
        print ('Made dir', code_dir) 

    subprocess.call("(rsync -r --exclude=__pycache__/ . "+code_dir+" )", shell=True)



    seed=0
    torch.manual_seed(seed)


    n_components = 3


    denom = 0.
    for c in range(n_components):
        denom += c+5.
    # print (denom)
    true_mixture_weights = []
    for c in range(n_components):
        true_mixture_weights.append((c+5.) / denom)
    true_mixture_weights = np.array(true_mixture_weights)
    print ('Mixture Weights', true_mixture_weights)

    needsoftmax_mixtureweight = np.log(true_mixture_weights)

    train(n_components=n_components, 
                needsoftmax_mixtureweight=needsoftmax_mixtureweight)


    fsdfasd


























