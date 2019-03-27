

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
import time 

# import sys, os
# # sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('./new_discrete_estimators'))

from NN import NN3
from NN import NN4

from plot_functions import *


def to_print_mean(x):
    return torch.mean(x).data.cpu().numpy()
def to_print(x):
    return x.data.cpu().numpy()

def H(soft):
    return torch.argmax(soft, dim=1)

def L2_error(w1,w2):
    dif = w1-w2
    return np.sqrt(np.sum(dif**2))

def L2_batch(w1,w2):
    dif = w1-w2
    return torch.mean(torch.sqrt(torch.sum(dif**2, dim=1)))

def check_nan(x):
    if (x != x).any():
        print (x)
        fsdfsd

def logsumexp(x):

    max_ = torch.max(x, dim=1, keepdim=True)[0]
    lse = torch.log(torch.sum(torch.exp(x - max_), dim=1, keepdim=True)) + max_
    return lse

def myclamp(x):
    min_ = 1e-7 
    max_ = 1.-1e-7
    return (min_ - max_) *x + max_


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












def reinforce(x, logits, mixtureweights, k=1, get_grad=False):
    B = logits.shape[0]
    probs = torch.softmax(logits, dim=1)
    outputs = {}

    cat = Categorical(probs=probs)

    grads =[]
    # net_loss = 0
    for jj in range(k):

        cluster_H = cat.sample()
        outputs['logq'] = logq = cat.log_prob(cluster_H).view(B,1)
        outputs['logpx_given_z'] = logpx_given_z = logprob_undercomponent(x, component=cluster_H)
        outputs['logpz'] = logpz = torch.log(mixtureweights[cluster_H]).view(B,1)
        logpxz = logpx_given_z + logpz #[B,1]
        outputs['f'] = f = logpxz - logq - 1.
        # outputs['net_loss'] = net_loss = net_loss - torch.mean((f.detach() ) * logq)
        outputs['net_loss'] = net_loss = - torch.mean((f.detach() ) * logq)
        # net_loss += - torch.mean( -logq.detach()*logq)

        # print (f)
        # afsda

        if get_grad:
            grad = torch.autograd.grad([net_loss], [logits], create_graph=True, retain_graph=True)[0]
            grads.append(grad)

    # net_loss = net_loss/ k

    if get_grad:
        grads = torch.stack(grads)
        # print (grads.shape)
        # outputs['grad_avg'] = torch.mean(torch.mean(grads, dim=0),dim=0)
        outputs['grad_avg'] = torch.mean(grads, dim=0)
        outputs['grad_std'] = torch.std(grads, dim=0)[0]

    # return net_loss, f, logpx_given_z, logpz, logq
    return outputs




def reinforce_baseline(surrogate, x, logits, mixtureweights, k=1, get_grad=False):
    B = logits.shape[0]
    probs = torch.softmax(logits, dim=1)
    outputs = {}

    cat = Categorical(probs=probs)

    grads =[]
    # net_loss = 0
    for jj in range(k):

        cluster_H = cat.sample()
        outputs['logq'] = logq = cat.log_prob(cluster_H).view(B,1)
        outputs['logpx_given_z'] = logpx_given_z = logprob_undercomponent(x, component=cluster_H)
        outputs['logpz'] = logpz = torch.log(mixtureweights[cluster_H]).view(B,1)
        logpxz = logpx_given_z + logpz #[B,1]

        surr_pred = surrogate.net(x)

        outputs['f'] = f = logpxz - logq - 1. 
        # outputs['net_loss'] = net_loss = net_loss - torch.mean((f.detach() ) * logq)
        outputs['net_loss'] = net_loss = - torch.mean((f.detach() - surr_pred.detach()) * logq)
        # net_loss += - torch.mean( -logq.detach()*logq)

        # surr_loss = torch.mean(torch.abs(f.detach() - surr_pred))

        grad_logq =  torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0]
        surr_loss = torch.mean(((f.detach() - surr_pred) * grad_logq )**2)

        if get_grad:
            grad = torch.autograd.grad([net_loss], [logits], create_graph=True, retain_graph=True)[0]
            grads.append(grad)

    # net_loss = net_loss/ k

    if get_grad:
        grads = torch.stack(grads)
        # print (grads.shape)
        outputs['grad_avg'] = torch.mean(torch.mean(grads, dim=0),dim=0)
        outputs['grad_std'] = torch.std(grads, dim=0)[0]

    outputs['surr_loss'] = surr_loss
    # return net_loss, f, logpx_given_z, logpz, logq
    return outputs




def reinforce_pz(x, logits, mixtureweights, k=1):
    B = logits.shape[0]
    probs = torch.softmax(logits, dim=1)

    cat = RelaxedOneHotCategorical(probs=probs, temperature=torch.tensor([1.]).cuda())

    net_loss = 0
    for jj in range(k):

        cluster_S = cat.rsample()
        cluster_H = H(cluster_S)
        logq = cat.log_prob(cluster_S.detach()).view(B,1)

        logpx_given_z = logprob_undercomponent(x, component=cluster_H)
        logpz = torch.log(mixtureweights[cluster_H]).view(B,1)
        logpxz = logpx_given_z + logpz #[B,1]
        f = logpxz - logq - 1.
        net_loss += - torch.mean((f.detach()) * logq)
        # net_loss += - torch.mean( -logq.detach()*logq)

    net_loss = net_loss/ k

    return net_loss, f, logpx_given_z, logpz, logq






def simplax(surrogate, x, logits, mixtureweights, k=1):

    B = logits.shape[0]
    probs = torch.softmax(logits, dim=1)

    cat = RelaxedOneHotCategorical(probs=probs, temperature=torch.tensor([1.]).cuda())

    outputs = {}
    net_loss = 0
    surr_loss = 0
    for jj in range(k):

        cluster_S = cat.rsample()
        cluster_H = H(cluster_S)

        logq = cat.log_prob(cluster_S.detach()).view(B,1)
        logpx_given_z = logprob_undercomponent(x, component=cluster_H)
        logpz = torch.log(mixtureweights[cluster_H]).view(B,1)
        logpxz = logpx_given_z + logpz #[B,1]
        f = logpxz - logq - 1.

        surr_input = torch.cat([cluster_S, x, logits], dim=1) #[B,21]
        surr_pred = surrogate.net(surr_input)

        net_loss += - torch.mean((f.detach() - surr_pred.detach()) * logq  + surr_pred)


        # surr_loss += torch.mean(torch.abs(f.detach()-1.-surr_pred))
        # grad_logq =  torch.mean( torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)
        # grad_surr = torch.mean( torch.autograd.grad([torch.mean(surr_pred)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)

        # grad_logq =  torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0]
        # grad_surr =  torch.autograd.grad([torch.mean(surr_pred)], [logits], create_graph=True, retain_graph=True)[0]
        # surr_loss = torch.mean(((f.detach() - surr_pred) * grad_logq + grad_surr)**2)

        net_loss2 = torch.mean((f.detach() - surr_pred) * logq  + surr_pred)
        grad = torch.autograd.grad([net_loss2], [logits], create_graph=True, retain_graph=True)[0]
        surr_loss = torch.mean(grad**2)
        # surr_loss = torch.mean(torch.abs(f.detach() - surr_pred))

        surr_dif = torch.mean(torch.abs(f.detach() - surr_pred))
        surr_dif2 = torch.mean(f.detach() - surr_pred)
        

        grad_path = torch.autograd.grad([torch.mean(surr_pred)], [logits], create_graph=True, retain_graph=True)[0]
        grad_score = torch.autograd.grad([torch.mean((f.detach() - surr_pred.detach()) * logq)], [logits], create_graph=True, retain_graph=True)[0]
        grad_path = torch.mean(torch.abs(grad_path))
        grad_score = torch.mean(torch.abs(grad_score))
   
    net_loss = net_loss/ k
    surr_loss = surr_loss/ k

    outputs['net_loss'] = net_loss
    outputs['f'] = f
    outputs['logpx_given_z'] = logpx_given_z
    outputs['logpz'] = logpz
    outputs['logq'] = logq
    outputs['surr_loss'] = surr_loss
    outputs['surr_dif'] = surr_dif   
    outputs['surr_dif2'] = surr_dif2 
    outputs['grad_path'] = grad_path   
    outputs['grad_score'] = grad_score   

    return outputs #net_loss, f, logpx_given_z, logpz, logq, surr_loss, surr_dif, grad_path, grad_score



























def sample_relax(x, logits, surrogate):
    B = logits.shape[0]
    C = logits.shape[1]

    cat = Categorical(logits=logits)
    # u = torch.rand(B,C).clamp(1e-10, 1.-1e-10).cuda()
    u = myclamp(torch.rand(B,C).cuda())
    gumbels = -torch.log(-torch.log(u))
    z = logits + gumbels
    b = torch.argmax(z, dim=1) #.view(B,1)
    logprob = cat.log_prob(b).view(B,1)


    # czs = []
    # for j in range(1):
    #     z = sample_relax_z(logits)
    #     surr_input = torch.cat([z, x, logits.detach()], dim=1)
    #     cz = surrogate.net(surr_input)
    #     czs.append(cz)
    # czs = torch.stack(czs)
    # cz = torch.mean(czs, dim=0)#.view(1,1)
    surr_input = torch.cat([z, x, logits.detach()], dim=1)
    cz = surrogate.net(surr_input)


    # cz_tildes = []
    # for j in range(1):
    #     z_tilde = sample_relax_given_b(logits, b)
    #     surr_input = torch.cat([z_tilde, x, logits.detach()], dim=1)
    #     cz_tilde = surrogate.net(surr_input)
    #     cz_tildes.append(cz_tilde)
    # cz_tildes = torch.stack(cz_tildes)
    # cz_tilde = torch.mean(cz_tildes, dim=0) #.view(B,1)
    z_tilde = sample_relax_given_b(logits, b)
    surr_input = torch.cat([z_tilde, x, logits.detach()], dim=1)
    cz_tilde = surrogate.net(surr_input)


    return b, logprob, cz, cz_tilde, z, z_tilde, gumbels, u


def sample_relax_z(logits):
    B = logits.shape[0]
    C = logits.shape[1]

    u = torch.rand(B,C).clamp(1e-10, 1.-1e-10).cuda()
    gumbels = -torch.log(-torch.log(u))
    z = logits + gumbels
    return z


def sample_relax_given_b(logits, b):
    B = logits.shape[0]
    C = logits.shape[1]

    # u_b = torch.rand(B,1).clamp(1e-10, 1.-1e-10).cuda()
    u_b = myclamp(torch.rand(B,1).cuda())
    # print (u_b)
    # fafas
    z_tilde_b = -torch.log(-torch.log(u_b))

    u = myclamp(torch.rand(B,C).cuda())
    # z_tilde = -torch.log((- torch.log(u) / torch.softmax(logits,dim=1)) - torch.log(u_b))
    z_tilde = -torch.log((- torch.log(u) / torch.exp(logits)) - torch.log(u_b)) #since I parameterize logits with probs
    z_tilde.scatter_(dim=1, index=b.view(B,1), src=z_tilde_b)

    return z_tilde



def relax(step, surrogate, x, logits, mixtureweights, k=1, get_grad=False):




    outputs = {}
    B = logits.shape[0]
    C = logits.shape[1]

    grads =[]
    for jj in range(k):

        b, logq, cz, cz_tilde, z, z_tilde, gumbels, u = sample_relax(x, logits, surrogate)
        # print (b)
        # print (b.shape)
        # fsadfa
        surr_pred_z = cz
        surr_pred_z_tilde = cz_tilde

        logpx_given_z = logprob_undercomponent(x, component=b)
        logpz = torch.log(mixtureweights[b]).view(B,1)
        logpxz = logpx_given_z + logpz #[B,1]
        # print(logpxz.shape, logpz.shape)
        # fsdf


        
        

        #Encoder loss
        # warmup = np.minimum( (step+1) / 50000., 1.)
        # warmup = .0001
        warmup = 1.

        # f = logpxz - logq - 1.
        # net_loss = - torch.mean(   warmup*((f.detach() - surr_pred_z_tilde.detach()) * logq)  +  surr_pred_z - surr_pred_z_tilde )

        f = logpxz - logq 
        net_loss = - torch.mean( (f.detach() - surr_pred_z_tilde.detach()) * logq - logq +  surr_pred_z - surr_pred_z_tilde )

        if (net_loss != net_loss).any():
            print ('net_loss', net_loss)
            print ('f', f)
            print ('logpxz', logpxz)
            print ('logq', logq)
            print ('surr_pred_z_tilde', surr_pred_z_tilde)
            print ('surr_pred_z', surr_pred_z)
            print ('logits', logits)
            print ((logits != logits).any())
            print ((1/logits != 1/logits).any())
            print ('gumbels', gumbels)
            print ((gumbels != gumbels).any())
            print ((1/gumbels != 1/gumbels).any())
            print ('z', z)
            print ((z != z).any())
            print ((1./z != 1./z).any())
            print ('u', u)

            # print (z)
            # print (probs)
            # print (gumbels)
            fasdfas

        if get_grad:
            grad = torch.autograd.grad([net_loss], [logits], create_graph=True, retain_graph=True)[0]
            grads.append(grad)
            surr_dif = torch.mean(torch.abs(f.detach() - surr_pred_z_tilde))
            surr_dif2 = torch.mean(f.detach() - surr_pred_z_tilde)


        else:
            # #Surrogate loss
            # grad_logq =  torch.mean( torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)
            # grad_surr_z = torch.mean( torch.autograd.grad([torch.mean(surr_pred_z)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)
            # grad_surr_z_tilde = torch.mean( torch.autograd.grad([torch.mean(surr_pred_z_tilde)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)

            # print (f.shape, surr_pred_z_tilde.shape, grad_logq.shape, grad_surr_z.shape, grad_surr_z_tilde.shape)
            # fasdfdas
            # print (grad_surr_z_tilde)
            # fsfa

            grad_logq =  torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0]
            grad_surr_z =  torch.autograd.grad([torch.mean(surr_pred_z)], [logits], create_graph=True, retain_graph=True)[0]
            grad_surr_z_tilde = torch.autograd.grad([torch.mean(surr_pred_z_tilde)], [logits], create_graph=True, retain_graph=True)[0]
            grad_path = torch.autograd.grad([torch.mean(surr_pred_z - surr_pred_z_tilde)], [logits], create_graph=True, retain_graph=True)[0]
            # surr_loss = torch.mean(((f.detach() - surr_pred_z_tilde) * grad_logq - grad_logq  + grad_surr_z - grad_surr_z_tilde)**2)  

            # net_loss2 = torch.mean( (f.detach() - surr_pred_z_tilde) * logq - logq +  surr_pred_z - surr_pred_z_tilde )
            # grad = torch.autograd.grad([net_loss2], [logits], create_graph=True, retain_graph=True)[0]
            # surr_loss = torch.mean(grad**2)

            surr_loss = torch.mean(torch.abs(f.detach() - surr_pred_z_tilde))

            surr_dif = torch.mean(torch.abs(f.detach() - surr_pred_z_tilde))
            surr_dif2 = torch.mean(f.detach() - surr_pred_z_tilde)
            # surr_loss = surr_dif

            if (surr_loss != surr_loss).any():
                # print ('net_loss', net_loss)
                # print ('surr_loss', surr_loss)
                # print ('f', f)
                # print ('logpxz', logpxz)
                # print ('logq', logq)
                # print ('surr_pred_z_tilde', surr_pred_z_tilde)
                # print ('surr_pred_z', surr_pred_z)

                # print (z)
                # print (probs)
                # print (gumbels)
                print ('grad_logq', grad_logq)
                print ((grad_logq != grad_logq).any())
                print ('grad_surr_z', grad_surr_z)
                print ((grad_surr_z != grad_surr_z).any())
                print ('grad_surr_z_tilde', grad_surr_z_tilde)
                print ((grad_surr_z_tilde != grad_surr_z_tilde).any())
                print (logits)
                print (z_tilde)
                print ((z_tilde != z_tilde).any())
                print (torch.max(z_tilde))
                print (torch.min(z_tilde))
                aaa = torch.autograd.grad([torch.mean(z_tilde)], [logits], create_graph=True, retain_graph=True)[0]
                print ((aaa != aaa).any())
                print (torch.min(torch.exp(logits)))
                print (torch.max(torch.exp(logits)))


                fasdfas



    
    outputs['net_loss'] = net_loss
    outputs['f'] = f
    outputs['logpx_given_z'] = logpx_given_z
    outputs['logpz'] = logpz
    outputs['logq'] = logq


    if get_grad:
        grads = torch.stack(grads)
        # print (grads.shape)
        outputs['grad_avg'] = torch.mean(grads, dim=0)
        outputs['grad_std'] = torch.std(grads, dim=0)[0]
        outputs['surr_dif'] = surr_dif   
        outputs['surr_dif2'] = surr_dif2   
    else:
        outputs['surr_loss'] = surr_loss
        outputs['surr_dif'] = surr_dif   
        outputs['surr_dif2'] = surr_dif2   
        outputs['grad_logq'] = torch.abs(grad_logq)  
        outputs['grad_surr_z'] = torch.abs(grad_surr_z  ) 
        outputs['grad_surr_z_tilde'] = torch.abs(grad_surr_z_tilde )  
        outputs['grad_path'] = torch.abs(grad_path )  
        outputs['grad_score'] = torch.abs(grad_logq*(f.detach() - surr_pred_z_tilde.detach()))  

    # return net_loss, f, logpx_given_z, logpz, logq, surr_loss, surr_dif
    return outputs




def relax_grad(x, logits, b, surrogate, mixtureweights):
    B = logits.shape[0]
    C = logits.shape[1]

    cat = Categorical(logits=logits)
    # u = torch.rand(B,C).clamp(1e-10, 1.-1e-10).cuda()
    u = myclamp(torch.rand(B,C).cuda())
    gumbels = -torch.log(-torch.log(u))
    z = logits + gumbels
    # b = torch.argmax(z, dim=1) #.view(B,1)
    logq = cat.log_prob(b).view(B,1)

    surr_input = torch.cat([z, x, logits.detach()], dim=1)
    cz = surrogate.net(surr_input)

    z_tilde = sample_relax_given_b(logits, b)
    surr_input = torch.cat([z_tilde, x, logits.detach()], dim=1)
    cz_tilde = surrogate.net(surr_input)

    logpx_given_z = logprob_undercomponent(x, component=b)
    logpz = torch.log(mixtureweights[b]).view(B,1)
    logpxz = logpx_given_z + logpz #[B,1]

    f = logpxz - logq 

    grad_logq =  torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0]
    grad_surr_z =  torch.autograd.grad([torch.mean(cz)], [logits], create_graph=True, retain_graph=True)[0]
    grad_surr_z_tilde = torch.autograd.grad([torch.mean(cz_tilde)], [logits], create_graph=True, retain_graph=True)[0]
    # surr_loss = torch.mean(((f.detach() - cz_tilde) * grad_logq - grad_logq + grad_surr_z - grad_surr_z_tilde)**2, dim=1, keepdim=True)
    surr_loss = ((f.detach() - cz_tilde) * grad_logq - grad_logq + grad_surr_z - grad_surr_z_tilde)**2

    # print (surr_loss.shape)
    # print (logq.shape)
    # fasda

    # print (surr_loss,  torch.exp(logq))
    return surr_loss, torch.exp(logq)







def relax_grad2(x, logits, b, surrogate, mixtureweights):
    B = logits.shape[0]
    C = logits.shape[1]

    cat = Categorical(logits=logits)
    # u = torch.rand(B,C).clamp(1e-10, 1.-1e-10).cuda()
    u = myclamp(torch.rand(B,C).cuda())
    gumbels = -torch.log(-torch.log(u))
    z = logits + gumbels
    # b = torch.argmax(z, dim=1) #.view(B,1)
    logq = cat.log_prob(b).view(B,1)

    surr_input = torch.cat([z, x, logits.detach()], dim=1)
    cz = surrogate.net(surr_input)

    z_tilde = sample_relax_given_b(logits, b)
    surr_input = torch.cat([z_tilde, x, logits.detach()], dim=1)
    cz_tilde = surrogate.net(surr_input)

    logpx_given_z = logprob_undercomponent(x, component=b)
    logpz = torch.log(mixtureweights[b]).view(B,1)
    logpxz = logpx_given_z + logpz #[B,1]

    f = logpxz - logq 
    net_loss = - torch.mean( (f.detach() - cz_tilde.detach()) * logq - logq +  cz - cz_tilde )

    grad = torch.autograd.grad([net_loss], [logits], create_graph=True, retain_graph=True)[0] #[B,C]
    pb = torch.exp(logq)

    return grad, pb














































def train(method, n_components, true_mixture_weights, exp_dir, needsoftmax_mixtureweight=None):

    print('Method:', method)
    C = n_components

    true_mixture_weights = torch.tensor(true_mixture_weights, 
                                            requires_grad=True, device="cuda")

    if needsoftmax_mixtureweight is None:
        needsoftmax_mixtureweight = torch.randn(n_components, requires_grad=True, device="cuda")
    else:
        needsoftmax_mixtureweight = torch.tensor(needsoftmax_mixtureweight, 
                                            requires_grad=True, device="cuda")
    
    # lr = 1e-3
    load_step = 0 # 95000

    optim = torch.optim.Adam([needsoftmax_mixtureweight], lr=1e-4, weight_decay=1e-7)

    encoder = NN3(input_size=1, output_size=n_components, n_residual_blocks=3).cuda()
    # encoder.load_params_v3(save_dir=save_dir+'relax_C20_fixed2_test' +'/params/', name='encoder', step=load_step)
    # optim_net = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-7)
    optim_net = torch.optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=1e-7)

    if method in ['simplax', 'relax']:
        surrogate = NN3(input_size=1+n_components+n_components, output_size=1, n_residual_blocks=4).cuda()
        # surrogate = NN3(input_size=1+n_components+n_components, output_size=1, n_residual_blocks=10).cuda()
        # optim_surr = torch.optim.Adam(surrogate.parameters(), lr=5e-3)
        optim_surr = torch.optim.Adam(surrogate.parameters(), lr=1e-3)

        # surrogate.load_params_v3(save_dir=save_dir+'relax_C20_fixed2_test' +'/params/', name='surrogate', step=load_step)

    if method in ['reinforce_baseline']:
        surrogate = NN3(input_size=1, output_size=1, n_residual_blocks=4).cuda()
        optim_surr = torch.optim.Adam(surrogate.parameters(), lr=lr)        

    if method == 'HLAX':
        # surrogate = NN4(input_size=1+n_components, output_size=1, n_residual_blocks=4).cuda()
        surrogate = NN3(input_size=1+n_components, output_size=1, n_residual_blocks=4).cuda()
        surrogate2 = NN3(input_size=1, output_size=1, n_residual_blocks=2).cuda()
        optim_surr = torch.optim.Adam(surrogate.parameters(), lr=lr)
        optim_surr2 = torch.optim.Adam(surrogate2.parameters(), lr=lr)

    data_dict = {}
    data_dict['steps'] = []
    data_dict['theta_losses'] = []
    data_dict['inference_L2'] = []
    data_dict['x'] = {}
    data_dict['x']['f'] = []
    data_dict['x']['lpx_given_z'] = []
    data_dict['z'] = {}
    data_dict['z']['lpz'] = []
    data_dict['z']['lqz'] = []
    
    # data_dict['grad_var'] = []
    # data_dict['grad_avg'] = []

    if method in ['simplax', 'HLAX', 'relax']:
        data_dict['surr_loss']= {}
        data_dict['surr_loss']['single_sample'] = []
        data_dict['surr_dif'] = []
        data_dict['surr_dif2'] = []
    if method in ['simplax', 'HLAX']:
        data_dict['grad_split'] = {}
        data_dict['grad_split']['score'] = []
        data_dict['grad_split']['path'] = []
    if method=='HLAX':
        data_dict['alpha'] = []
    if method=='relax':
        data_dict['surr_loss']['actual_var'] = []
        data_dict['grad_logq'] = [] 

        data_dict['surr_grads'] = {}
        data_dict['surr_grads']['grad_surr_z'] = [] 
        data_dict['surr_grads']['grad_surr_z_tilde'] = [] 

        data_dict['grad_split'] = {}
        data_dict['grad_split']['score'] = []
        data_dict['grad_split']['path'] = []

        data_dict['var'] = {}
        data_dict['var']['reinforce'] = []
        data_dict['var']['relax'] = []
        # data_dict['var']['actual'] = []
        data_dict['grad_abs'] = [] 

        data_dict['bias'] = {}
        data_dict['bias']['reinforce_k1'] = []
        data_dict['bias']['relax_k1'] = []
        data_dict['bias']['reinforce_k100'] = []
        data_dict['bias']['relax_k100'] = []
        
        data_dict['SNR'] = {}
        data_dict['SNR']['reinforce'] = []
        data_dict['SNR']['relax'] = []

    if method=='reinforce_baseline':
        data_dict['surr_loss'] = []
    # if method=='simplax':
    #     data_dict['surr_loss']['actual_var'] = []
    #     data_dict['grad_logq'] = [] 
    #     data_dict['surr_grads'] = {}
    #     # data_dict['surr_grads']['grad_surr_z'] = [] 
    #     # data_dict['surr_grads']['grad_surr_z_tilde'] = [] 
    #     data_dict['grad_split'] = {}
    #     data_dict['grad_split']['score'] = []
    #     data_dict['grad_split']['path'] = []
    #     data_dict['var'] = {}
    #     data_dict['var']['reinforce'] = []
    #     data_dict['var']['simplax'] = []
    #     # data_dict['var']['actual'] = []
    #     data_dict['grad_abs'] = [] 
    #     data_dict['bias'] = {}
    #     data_dict['bias']['relax_k1'] = []
    #     data_dict['bias']['reinforce_k1'] = []
    #     data_dict['bias']['relax_k100'] = []
    #     data_dict['bias']['reinforce_k100'] = []
    #     data_dict['SNR'] = {}
    #     data_dict['SNR']['reinforce'] = []
    #     data_dict['SNR']['simplax'] = []




    cur_time = time.time()
    for step in range(load_step,n_steps+1):

        mixtureweights = torch.softmax(needsoftmax_mixtureweight, dim=0).float() #[C]
        # x = sample_gmm(batch_size, mixture_weights=true_mixture_weights)
        x = sample_gmm(batch_size, mixture_weights=torch.ones(C).cuda()/C)
        prelogits = encoder.net(x)
        prelogits = prelogits * .001 # pre-condition so that it starts out with mass on most classes
        # prelogits = torch.tanh(prelogits) * 20. # avoid large grads/nans, min -20, max 20
        logits = prelogits - logsumexp(prelogits) #log of probs, probs = softmax of prelogits

        # logits = logits.clamp(-10, 10)
        logits = logits.clamp(-5, 1)

        # print (torch.softmax(logits, dim=1)[0])
        # print (torch.exp(logits)[0])



        #RUN
        if method == 'reinforce':
            # net_loss, f, logpx_given_z, logpz, logq = reinforce(x, logits, mixtureweights, k=1)
            outputs = reinforce(x, logits, mixtureweights, k=1)
        elif method == 'reinforce_pz':
            net_loss, f, logpx_given_z, logpz, logq = reinforce_pz(x, logits, mixtureweights, k=1)
        elif method == 'simplax':
            # net_loss, f, logpx_given_z, logpz, logq, surr_loss, surr_dif, grad_path, grad_score = simplax(surrogate, x, logits, mixtureweights, k=1)
            outputs = simplax(surrogate, x, logits, mixtureweights, k=1)
        elif method == 'HLAX':
            net_loss, f, logpx_given_z, logpz, logq, surr_loss, surr_dif, grad_path, grad_score, alpha = HLAX(surrogate, surrogate2, x, logits, mixtureweights, k=1)
        elif method=='relax':
            outputs = relax(step, surrogate, x, logits, mixtureweights, k=1)
        elif method=='reinforce_baseline':
            outputs = reinforce_baseline(surrogate, x, logits, mixtureweights, k=1)



        #UPDATES
        if step > 300000:

            # Update generator
            loss = - torch.mean(outputs['f'])
            optim.zero_grad()
            loss.backward(retain_graph=True)  
            optim.step()

        if step > 300000:
        # if step > 3000: # and step %2==0: #10000:
            # Update encoder
            optim_net.zero_grad()
            outputs['net_loss'].backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), .25)
            optim_net.step()

        # Update surrogate
        if method in ['simplax', 'HLAX', 'relax', 'reinforce_baseline']:
            optim_surr.zero_grad()
            outputs['surr_loss'].backward(retain_graph=True)
            optim_surr.step()

        if method == 'HLAX':
            optim_surr2.zero_grad()
            surr_loss.backward(retain_graph=True)
            optim_surr2.step()



        with torch.no_grad():



            prelogits2 = encoder.net(x)

            if (prelogits2 != prelogits2).any():
                print (prelogits)
                print (logits)
                print (torch.max(logits))
                print (torch.min(logits))
                print (prelogits2)

                prelogits2 = prelogits2 * .001 # pre-condition so that it starts out with mass on most classes
                print (prelogits2)

                logits2 = prelogits2 - logsumexp(prelogits2) #log of probs, probs = softmax of prelogits
                
                print (logits2)
                fdsafdasdf






        if step%print_steps==0:
            # print (step, to_print(net_loss), to_print(logpxz - logq), to_print(logpx_given_z), to_print(logpz), to_print(logq))

            current_theta = torch.softmax(needsoftmax_mixtureweight, dim=0).float()
            theta_loss = L2_error(to_print(true_mixture_weights),
                                    to_print(current_theta))

            pz_give_x = true_posterior(x, mixture_weights=mixtureweights)
            probs = torch.softmax(logits, dim=1)
            inference_L2 = L2_batch(pz_give_x, probs)


            if method =='relax':
                #Compute actual variance
                var = torch.zeros(batch_size, C).cuda()
                for b_i in range(C):
                    b = torch.tensor([b_i]*batch_size).cuda()
                    grad_squared, pb = relax_grad(x, logits, b, surrogate, mixtureweights)
                    var += grad_squared * pb


                #Compute actual grad
                actual_grad = torch.zeros(batch_size, C).cuda()
                for b_i in range(C):
                    b = torch.tensor([b_i]*batch_size).cuda()
                    grad, pb = relax_grad2(x, logits, b, surrogate, mixtureweights)
                    actual_grad += grad * pb

                    # if step% (plot_steps*2) ==0:
                    #     print (b_i, -grad, pb)
                    #     print ()

                # print (actual_grad.shape)
                grad_abs = to_print_mean( torch.abs(actual_grad ))

                # RELAX GRAD k=100
                # outputs22 = relax(step=step, surrogate=surrogate, x=x[0].view(1,1), 
                #                 logits=logits[0].view(1,C), mixtureweights=mixtureweights, k=100, get_grad=True)
                outputs22 = relax(step=step, surrogate=surrogate, x=x, 
                                logits=logits, mixtureweights=mixtureweights, k=100, get_grad=True)

                bias_relax_k100 = to_print_mean( torch.abs(actual_grad - outputs22['grad_avg']))

                # RELAX GRAD k=1
                outputs_k1 = relax(step=step, surrogate=surrogate, x=x, 
                                logits=logits, mixtureweights=mixtureweights, k=1, get_grad=True)
                bias_relax_k1 = to_print_mean( torch.abs(actual_grad - outputs_k1['grad_avg']))

                # reinforce k100
                outputs11 = reinforce(x, logits, mixtureweights, k=100, get_grad=True)
                bias_reinforce_k100 = to_print_mean( torch.abs(actual_grad - outputs11['grad_avg']))

                outputs111 = reinforce(x, logits, mixtureweights, k=1, get_grad=True)
                bias_reinforce_k1 = to_print_mean( torch.abs(actual_grad - outputs111['grad_avg']))

            # # RELAX GRAD k=1
            # outputs_1 = relax(step=step, surrogate=surrogate, x=x[0].view(1,1), 
            #                 logits=logits[0].view(1,C), mixtureweights=mixtureweights, k=1, get_grad=True)

            # #CHECK GRADS
            # outputs11 = reinforce(x[0].view(1,1), logits[0].view(1,C), mixtureweights, k=100, get_grad=True)


            # print ('reinforce')
            # print (outputs['grad_avg'])
            # print (outputs['grad_std'])

            
            # print ('relax')
            # print (outputs['grad_avg'])
            # print (outputs['grad_std'])
            # print ()
            # print (to_print(outputs['surr_dif']))
            # print ()


            # #CHECK GRADS
            # outputs = reinforce(x[0].view(1,1), logits[0].view(1,C), mixtureweights, k=1000, get_grad=True)
            # print ('reinforce')
            # print (outputs['grad_avg'])
            # print (outputs['grad_std'])

            # outputs = relax(step=step, surrogate=surrogate, x=x[0].view(1,1), 
            #                 logits=logits[0].view(1,C), mixtureweights=mixtureweights, k=1000, get_grad=True)
            # print ('relax')
            # print (outputs['grad_avg'])
            # print (outputs['grad_std'])
            # print ()
            # print (to_print(outputs['surr_dif']))
            # fadsaf



            # #Grad variance 
            # grad = torch.autograd.grad([outputs['net_loss']], [logits], create_graph=True, retain_graph=True)[0]
            # # grad_var = torch.mean(torch.std(grad, dim=0))
            # grad_avg = torch.mean(torch.abs(grad))

            # plot_posteriors2(n_components, trueposteriors=to_print(pz_give_x), qs=to_print(probs), exp_dir=exp_dir, name=str(step))
            # fasfs





            print( 
                'S:{:5d}'.format(step),
                'T:{:.2f}'.format(time.time() - cur_time),
                'Theta_loss:{:.3f}'.format(theta_loss),
                'Loss:{:.3f}'.format(to_print_mean(outputs['net_loss'])),
                'ELBO:{:.3f}'.format(to_print_mean(outputs['f'])),
                'lpx|z:{:.3f}'.format(to_print_mean(outputs['logpx_given_z'])),
                'lpz:{:.3f}'.format(to_print_mean(outputs['logpz'])),
                'lqz:{:.3f}'.format(to_print_mean(outputs['logq'])),
                )
            cur_time = time.time()

            if step> 0:
                data_dict['steps'].append(step)
                data_dict['theta_losses'].append(theta_loss)
                data_dict['x']['f'].append(to_print_mean(outputs['f']))
                data_dict['x']['lpx_given_z'].append(to_print_mean(outputs['logpx_given_z']))
                data_dict['z']['lpz'].append(to_print_mean(outputs['logpz']))
                data_dict['z']['lqz'].append(to_print_mean(outputs['logq']))
                data_dict['inference_L2'].append(to_print(inference_L2))
                # data_dict['grad_var'].append(to_print(grad_var))
                # data_dict['grad_avg'].append(to_print(grad_avg))
                if method in ['simplax', 'HLAX', 'relax']:
                    data_dict['surr_loss']['single_sample'].append(to_print(outputs['surr_loss']))
                    data_dict['surr_dif'].append(to_print(outputs['surr_dif']))
                    data_dict['surr_dif2'].append(to_print(outputs['surr_dif2']))
                if method in ['simplax', 'HLAX']:
                    data_dict['grad_split']['score'].append(to_print(outputs['grad_score']))
                    data_dict['grad_split']['path'].append(to_print(outputs['grad_path']))
                if method == 'HLAX':
                    data_dict['alpha'].append(to_print(alpha))
                if method == 'relax':
                    data_dict['surr_loss']['actual_var'].append(to_print_mean(var))
                    data_dict['grad_logq'].append(to_print_mean(outputs['grad_logq']))
                    data_dict['surr_grads']['grad_surr_z'].append(to_print_mean(outputs['grad_surr_z']))
                    data_dict['surr_grads']['grad_surr_z_tilde'].append(to_print_mean(outputs['grad_surr_z_tilde']))
                    data_dict['grad_split']['score'].append(to_print_mean(outputs['grad_score']))
                    data_dict['grad_split']['path'].append(to_print_mean(outputs['grad_path']))
                    data_dict['var']['reinforce'].append(to_print_mean(outputs11['grad_std']))
                    data_dict['var']['relax'].append(to_print_mean(outputs22['grad_std']))

                    data_dict['grad_abs'].append(grad_abs)

                    data_dict['bias']['relax_k1'].append( bias_relax_k1)
                    data_dict['bias']['relax_k100'].append( bias_relax_k100)
                    data_dict['bias']['reinforce_k100'].append( bias_reinforce_k100)
                    data_dict['bias']['reinforce_k1'].append( bias_reinforce_k1)
                    # data_dict['bias'].append(to_print_mean( torch.abs(outputs11['grad_avg'] - outputs22['grad_avg'])))

                    data_dict['SNR']['reinforce'].append(to_print_mean( torch.abs(outputs11['grad_avg'] / outputs11['grad_std'] )))
                    data_dict['SNR']['relax'].append(to_print_mean(  torch.abs(outputs22['grad_avg'] / outputs22['grad_std'] )))

                # if method == 'simplax':
                #     data_dict['surr_loss']['actual_var'].append(to_print_mean(var))
                #     data_dict['grad_logq'].append(to_print_mean(outputs['grad_logq']))
                #     # data_dict['surr_grads']['grad_surr_z'].append(to_print_mean(outputs['grad_surr_z']))
                #     # data_dict['surr_grads']['grad_surr_z_tilde'].append(to_print_mean(outputs['grad_surr_z_tilde']))
                #     data_dict['grad_split']['score'].append(to_print_mean(outputs['grad_score']))
                #     data_dict['grad_split']['path'].append(to_print_mean(outputs['grad_path']))
                #     data_dict['var']['reinforce'].append(to_print_mean(outputs11['grad_std']))
                #     data_dict['var']['simplax'].append(to_print_mean(outputs22['grad_std']))

                #     data_dict['grad_abs'].append(grad_abs)

                #     data_dict['bias']['relax_k1'].append( bias_relax_k1)
                #     data_dict['bias']['relax_k100'].append( bias_relax_k100)
                #     data_dict['bias']['reinforce_k100'].append( bias_reinforce_k100)
                #     data_dict['bias']['reinforce_k1'].append( bias_reinforce_k1)
                #     # data_dict['bias'].append(to_print_mean( torch.abs(outputs11['grad_avg'] - outputs22['grad_avg'])))

                #     data_dict['SNR']['reinforce'].append(to_print_mean( torch.abs(outputs11['grad_avg'] / outputs11['grad_std'] )))
                #     data_dict['SNR']['simplax'].append(to_print_mean(  torch.abs(outputs22['grad_avg'] / outputs22['grad_std'] )))


                if method in ['reinforce_baseline']:
                    data_dict['surr_loss']['single_sample'].append(to_print(outputs['surr_loss']))

            check_nan(outputs['net_loss'])



        if step%plot_steps==0 and step!=0:

            # fsdfasd

            plot_curve2(data_dict, exp_dir)

            # list_of_posteriors = []
            # for ii in range(5):
            #     pz_give_x = true_posterior(x, mixture_weights=mixtureweights)
            #     probs = torch.softmax(logits, dim=1)
            #     inference_L2 = L2_batch(pz_give_x, probs)    
            #     list_of_posteriors.append([to_print(pz_give_x), to_print(probs)])      

            if step% (plot_steps*2) ==0:
            # if step% plot_steps ==0:
                plot_posteriors2(n_components, trueposteriors=to_print(pz_give_x), qs=to_print(probs), exp_dir=images_dir, name=str(step))
            
                plot_dist2(n_components, mixture_weights=to_print(current_theta), true_mixture_weights=to_print(true_mixture_weights), exp_dir=images_dir, name=str(step))

                # #CHECK GRADS
                # outputs = reinforce(x[0].view(1,1), logits[0].view(1,C), mixtureweights, k=1000, get_grad=True)
                # print ('reinforce')
                # print (outputs['grad_avg'])
                # print (outputs['grad_std'])

                # outputs = relax(step=step, surrogate=surrogate, x=x[0].view(1,1), 
                #                 logits=logits[0].view(1,C), mixtureweights=mixtureweights, k=1000, get_grad=True)
                # print ('relax')
                # print (outputs['grad_avg'])
                # print (outputs['grad_std'])
                # print ()
                # print (to_print(outputs['surr_dif']))
                # print ()

        if step % params_step==0 and step>0:

            # save_dir = home+'/Documents/Grad_Estimators/GMM/'
            with open( exp_dir+"data.p", "wb" ) as f:
                pickle.dump(data_dict, f)
            print ('saved data')

            surrogate.save_params_v3(save_dir=params_dir, name='surrogate', step=step)
            encoder.save_params_v3(save_dir=params_dir, name='encoder', step=step)


            # if step%1000==0:
            #     # pz_give_x = true_posterior(x, mixture_weights=mixtureweights)
            #     # probs = torch.softmax(logits, dim=1)
            #     # print (to_print(pz_give_x[0]))
            #     # print (to_print(probs[0]))
            #     print (to_print(true_mixture_weights))
            #     print (to_print(current_theta))

























if __name__ == "__main__":


    # exp_name = 'reinforce_pz_3cats'
    # exp_name = 'reinforce_3cats_2'
    # exp_name = 'simplax_3cats_2_temp4'
    # exp_name = 'simplax_3cats_withsplit'
    # exp_name = 'HLAX_3cats_withalphaplot'
    # exp_name = 'reinforce_3cats_b64'
    # exp_name = 'relax_3cats'
    # exp_name = 'relax_C20'
    # exp_name = 'relax_C6_testu_surrloss2'
    # exp_name = 'simplax_C6_new'
    # exp_name = 'relax_C6_new'
    # exp_name = 'relax_C6_surrgetslogits_u5_v5'
    # exp_name = 'relax_C6_surrgetslogits_u1_v1'
    # exp_name = 'relax_C20_u7_v3_logitdetached'
    # exp_name = 'relax_C20_u7_v3_trainsurr100k'
    # exp_name = 'relax_C20_u3_v1_surr10'
    # exp_name = 'relax_C6_u1v1'
    # exp_name = 'relax_C3_u1v1'
    # exp_name = 'relax_C20_u7v3'
    # exp_name = 'relax_C20_u1v1_surrloss1'
    # exp_name = 'relax_C20_u3v1_surrloss2_traininference1'
    # exp_name = 'relax_C20_u1v1_surrloss2_traininference1_inc_lr'
    # exp_name = 'relax_C6_u1v1_surrloss2_traininference1_inc_lr'
    # exp_name = 'relax_C6_u1v1_seeSNR'
    # exp_name = 'relax_C6_u1v1_oldzway'
    # exp_name = 'relax_C6_fixed_surronly10k'
    # exp_name = 'relax_C6_fixed2'
    # exp_name = 'relax_C20_fixed2_lookingatgrads'
    # exp_name = 'relax_C6_fixed2_lookingatgrads'
    # exp_name = 'relax_C20_fixed2_lookingatgrads_moreclamp'
    # exp_name = 'relax_C20_difsurrloss_clipgrad'
    # exp_name = 'relax_C20_difsurrloss_clipgrad_inclr'
    # exp_name = 'relax_C20_oldsurrloss'
    # exp_name = 'simplax_C20_fixed2'
    # exp_name = 'relax_C20_onlysurr'
    exp_name = 'relax_C20_onlysurr_mindif'
    # exp_name = 'relax_C20_u1v1_surrloss2'
    # exp_name = 'relax_C6_u1v1_surrloss2'
    # exp_name = 'relax_C20_u1v1_oldwayforz'
    # exp_name = 'relax_C6_u1v1'
    # exp_name = 'simplax_C20_balancedsampling'
    # exp_name = 'reinforce_C20'
    # exp_name = 'reinforce_C20_balancedsampling'
    # exp_name = 'reinforce_C6_balancedsampling'
    # exp_name = 'reinforcebaseline_C6_balancedsampling'
    # exp_name = 'reinforcebaseline_C20_balancedsampling'
    # exp_name = 'relax_C3'
    # exp_name = 'simplax_3cats_newsurrobjective'
    # exp_name = 'reinforce_3cats_findingproblem'
    

    print ('Exp:', exp_name)


    save_dir = home+'/Documents/Grad_Estimators/GMM/'
    exp_dir = save_dir + exp_name + '/'
    params_dir = exp_dir + 'params/'
    images_dir = exp_dir + 'images/'
    code_dir = exp_dir + 'code/'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print ('Made dir', exp_dir) 

    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
        print ('Made dir', code_dir) 

    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
        print ('Made dir', params_dir) 

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print ('Made dir', images_dir) 

    #Save code
    subprocess.call("(rsync -r --exclude=__pycache__/ . "+code_dir+" )", shell=True)



    seed=1
    torch.manual_seed(seed)


    n_components = 20
    method='relax'
    # method='simplax'


    denom = 0.
    for c in range(n_components):
        denom += c+5.
    # print (denom)
    true_mixture_weights = []
    for c in range(n_components):
        true_mixture_weights.append((c+5.) / denom)
    true_mixture_weights = np.array(true_mixture_weights)
    print ('Mixture Weights', true_mixture_weights)

    # needsoftmax_mixtureweight = np.log(true_mixture_weights)

    # train(n_components=n_components, 
    #             needsoftmax_mixtureweight=needsoftmax_mixtureweight)

    batch_size = 50 #6 #4
    n_steps = 300000
    k = 1

    print_steps = 500
    plot_steps = 2000
    params_step = 50000

    train(method=method, n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)

    # train(method='reinforce', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)
    # train(method='reinforce_baseline', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)
    # train(method='relax', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)
    # train(method='simplax', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)
    # train(method='HLAX', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)
    # train(method='reinforce_pz', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)

    print('Done.')



























