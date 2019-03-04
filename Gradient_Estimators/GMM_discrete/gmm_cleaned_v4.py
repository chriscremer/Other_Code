

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
from NN import NN4

from plot_functions import *


def to_print1(x):
    return torch.mean(x).data.cpu().numpy()
def to_print2(x):
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

        if get_grad:
            grad = torch.autograd.grad([net_loss], [logits], create_graph=True, retain_graph=True)[0]
            grads.append(grad)

    # net_loss = net_loss/ k

    if get_grad:
        grads = torch.stack(grads)
        # print (grads.shape)
        outputs['grad_avg'] = torch.mean(torch.mean(grads, dim=0),dim=0)
        outputs['grad_std'] = torch.std(grads, dim=0)[0]

    # return net_loss, f, logpx_given_z, logpz, logq
    return outputs









def relax(step, surrogate, x, logits, mixtureweights, k=1, get_grad=False):

    def sample_relax(probs):
        cat = Categorical(probs=probs)
        #Sample z
        u = torch.rand(B,C).cuda()
        u = u.clamp(1e-8, 1.-1e-8)
        gumbels = -torch.log(-torch.log(u))
        z = torch.log(probs) + gumbels

        b = torch.argmax(z, dim=1)
        logprob = cat.log_prob(b).view(B,1)

        #Sample z_tilde
        u_b = torch.rand(B,1).cuda()
        u_b = u_b.clamp(1e-8, 1.-1e-8)
        z_tilde_b = -torch.log(-torch.log(u_b))
        u = torch.rand(B,C).cuda()
        u = u.clamp(1e-8, 1.-1e-8)
        z_tilde = -torch.log((- torch.log(u) / probs) - torch.log(u_b))
        z_tilde.scatter_(dim=1, index=b.view(B,1), src=z_tilde_b)
        return z, b, logprob, z_tilde, gumbels

    outputs = {}
    B = logits.shape[0]
    C = logits.shape[1]
    probs = torch.softmax(logits, dim=1)

    #seems to get nans without this, but also sucks with it, so maybe this is the cause...
    probs = probs.clamp(1e-12, 1.-1e-12)

    grads =[]
    for jj in range(k):

        z, b, logq, z_tilde, gumbels = sample_relax(probs)

        logpx_given_z = logprob_undercomponent(x, component=b)
        logpz = torch.log(mixtureweights[b]).view(B,1)
        logpxz = logpx_given_z + logpz #[B,1]
        # print(logpxz.shape, logq.shape)
        # fsdf
        f = logpxz - logq - 1.

        # surr_input = torch.cat([z, x, logits], dim=1)
        surr_input = torch.cat([z, x], dim=1)
        surr_pred_z = surrogate.net(surr_input)
        surr_pred_z_tilde = surrogate.net(surr_input)

        #Encoder loss
        # warmup = np.minimum( (step+1) / 50000., 1.)
        # warmup = .0001
        warmup = 1.
        # print (warmup)
        # fsaf
        net_loss = - torch.mean(   warmup*((f.detach() - surr_pred_z_tilde.detach()) * logq)  +  surr_pred_z - surr_pred_z_tilde )
        # if get_grad:
        #     net_loss = - torch.mean((f.detach()) * logq )

        if (net_loss != net_loss).any():
            print (net_loss)
            print (f)
            print (logpxz)
            print (logq)
            print (surr_pred_z_tilde)
            print (surr_pred_z)
            print (z)
            print (probs)
            print (gumbels)
            fasdfas

        if get_grad:
            grad = torch.autograd.grad([net_loss], [logits], create_graph=True, retain_graph=True)[0]
            grads.append(grad)

        else:
            # #Surrogate loss
            # grad_logq =  torch.mean( torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)
            # grad_surr_z = torch.mean( torch.autograd.grad([torch.mean(surr_pred_z)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)
            # grad_surr_z_tilde = torch.mean( torch.autograd.grad([torch.mean(surr_pred_z_tilde)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)

            grad_logq =  torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0]
            grad_surr_z =  torch.autograd.grad([torch.mean(surr_pred_z)], [logits], create_graph=True, retain_graph=True)[0]
            grad_surr_z_tilde = torch.autograd.grad([torch.mean(surr_pred_z_tilde)], [logits], create_graph=True, retain_graph=True)[0]

            # print (f.shape, surr_pred_z_tilde.shape, grad_logq.shape, grad_surr_z.shape, grad_surr_z_tilde.shape)
            # fasdfdas
            # print (grad_surr_z_tilde)
            # fsfa

            # surr_loss = torch.mean(((f.detach() - surr_pred_z_tilde) * grad_logq + grad_surr_z - grad_surr_z_tilde)**2)
            surr_loss = torch.mean(torch.abs(f.detach() - surr_pred_z_tilde))

            if (surr_loss != surr_loss).any():
                print (net_loss)
                print (f)
                print (logpxz)
                print (logq)
                print (surr_pred_z_tilde)
                print (surr_pred_z)
                print (z)
                print (probs)
                print (gumbels)
                print (grad_logq)
                print (grad_surr_z)
                print (grad_surr_z_tilde)
                fasdfas

            surr_dif = torch.mean(torch.abs(f.detach() - surr_pred_z_tilde))
            # surr_loss = surr_dif

    
    outputs['net_loss'] = net_loss
    outputs['f'] = f
    outputs['logpx_given_z'] = logpx_given_z
    outputs['logpz'] = logpz
    outputs['logq'] = logq


    if get_grad:
        grads = torch.stack(grads)
        # print (grads.shape)
        outputs['grad_avg'] = torch.mean(torch.mean(grads, dim=0),dim=0)
        outputs['grad_std'] = torch.std(grads, dim=0)[0]
    else:
        outputs['surr_loss'] = surr_loss
        outputs['surr_dif'] = surr_dif   
        outputs['grad_logq'] = torch.abs(grad_logq )  
        outputs['grad_surr_z'] = torch.abs(grad_surr_z  ) 
        outputs['grad_surr_z_tilde'] = torch.abs(grad_surr_z_tilde )  

    # return net_loss, f, logpx_given_z, logpz, logq, surr_loss, surr_dif
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

        surr_input = torch.cat([cluster_S, x], dim=1) #[B,21]
        surr_pred = surrogate.net(surr_input)

        net_loss += - torch.mean((f.detach() - surr_pred.detach()) * logq  + surr_pred)


        # surr_loss += torch.mean(torch.abs(f.detach()-1.-surr_pred))
        # grad_logq =  torch.mean( torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)
        # grad_surr = torch.mean( torch.autograd.grad([torch.mean(surr_pred)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)

        # grad_logq =  torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0]
        # grad_surr =  torch.autograd.grad([torch.mean(surr_pred)], [logits], create_graph=True, retain_graph=True)[0]
        # surr_loss += torch.mean(((f.detach() - surr_pred) * grad_logq + grad_surr)**2)

        surr_dif = torch.mean(torch.abs(f.detach() - surr_pred))
        surr_loss = torch.mean(torch.abs(f.detach() - surr_pred))

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
    outputs['grad_path'] = grad_path   
    outputs['grad_score'] = grad_score   

    return outputs #net_loss, f, logpx_given_z, logpz, logq, surr_loss, surr_dif, grad_path, grad_score






def HLAX(surrogate, surrogate2, x, logits, mixtureweights, k=1):
    B = logits.shape[0]
    probs = torch.softmax(logits, dim=1)

    cat = RelaxedOneHotCategorical(probs=probs, temperature=torch.tensor([1.]).cuda())
    cat_bernoulli = Categorical(probs=probs)

    net_loss = 0
    surr_loss = 0
    for jj in range(k):

        cluster_S = cat.rsample()
        cluster_H = H(cluster_S)

        logq_z = cat.log_prob(cluster_S.detach()).view(B,1)
        logq_b = cat_bernoulli.log_prob(cluster_H.detach()).view(B,1)


        logpx_given_z = logprob_undercomponent(x, component=cluster_H)
        logpz = torch.log(mixtureweights[cluster_H]).view(B,1)
        logpxz = logpx_given_z + logpz #[B,1]
        f_z = logpxz - logq_z - 1.
        f_b = logpxz - logq_b - 1.

        surr_input = torch.cat([cluster_S, x], dim=1) #[B,21]
        # surr_pred, alpha = surrogate.net(surr_input)
        surr_pred = surrogate.net(surr_input)
        alpha = torch.sigmoid(surrogate2.net(x))

        net_loss += - torch.mean(     alpha.detach()*(f_z.detach()  - surr_pred.detach()) * logq_z  
                                    + alpha.detach()*surr_pred 
                                    + (1-alpha.detach())*(f_b.detach()  ) * logq_b)

        # surr_loss += torch.mean(torch.abs(f_z.detach() - surr_pred))

        grad_logq_z = torch.mean( torch.autograd.grad([torch.mean(logq_z)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)
        grad_logq_b =  torch.mean( torch.autograd.grad([torch.mean(logq_b)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)
        grad_surr = torch.mean( torch.autograd.grad([torch.mean(surr_pred)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)
        # print (alpha.shape, f_z.shape, surr_pred.shape, grad_logq_z.shape, grad_surr.shape)
        # fsdfa
        # grad_surr = torch.autograd.grad([surr_pred[0]], [logits], create_graph=True, retain_graph=True)[0]
        # print (grad_surr)
        # fsdfasd
        surr_loss += torch.mean(
                                    (alpha*(f_z.detach() - surr_pred) * grad_logq_z 
                                    + alpha*grad_surr
                                    + (1-alpha)*(f_b.detach()) * grad_logq_b )**2
                                    )

        surr_dif = torch.mean(torch.abs(f_z.detach() - surr_pred))
        # gradd = torch.autograd.grad([surr_loss], [alpha], create_graph=True, retain_graph=True)[0]
        # print (gradd)
        # fdsf
        grad_path = torch.autograd.grad([torch.mean(surr_pred)], [logits], create_graph=True, retain_graph=True)[0]
        grad_score = torch.autograd.grad([torch.mean((f_z.detach() - surr_pred.detach()) * logq_z)], [logits], create_graph=True, retain_graph=True)[0]
        grad_path = torch.mean(torch.abs(grad_path))
        grad_score = torch.mean(torch.abs(grad_score))


    net_loss = net_loss/ k
    surr_loss = surr_loss/ k

    return net_loss, f_b, logpx_given_z, logpz, logq_b, surr_loss, surr_dif, grad_path, grad_score, torch.mean(alpha)














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
    
    lr = 1e-3
    optim = torch.optim.Adam([needsoftmax_mixtureweight], lr=1e-5, weight_decay=1e-7)

    encoder = NN3(input_size=1, output_size=n_components, n_residual_blocks=3).cuda()
    optim_net = torch.optim.Adam(encoder.parameters(), lr=1e-5, weight_decay=1e-7)

    if method in ['simplax', 'relax']:
        surrogate = NN3(input_size=1+n_components, output_size=1, n_residual_blocks=4).cuda()
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
    data_dict['f'] = []
    data_dict['lpx_given_z'] = []
    data_dict['lpz'] = []
    data_dict['lqz'] = []
    data_dict['inference_L2'] = []
    # data_dict['grad_var'] = []
    # data_dict['grad_avg'] = []

    if method in ['simplax', 'HLAX', 'relax']:
        data_dict['surr_loss'] = []
        data_dict['surr_dif'] = []
    if method in ['simplax', 'HLAX']:
        data_dict['grad_split'] = {}
        data_dict['grad_split']['score'] = []
        data_dict['grad_split']['path'] = []
    if method=='HLAX':
        data_dict['alpha'] = []
    if method=='relax':
        data_dict['grad_logq'] = [] 
        data_dict['grad_surr_z'] = [] 
        data_dict['grad_surr_z_tilde'] = [] 

    for step in range(0,n_steps+1):

        mixtureweights = torch.softmax(needsoftmax_mixtureweight, dim=0).float() #[C]

        x = sample_gmm(batch_size, mixture_weights=true_mixture_weights)
        logits = encoder.net(x)
        logits = logits/100.
        # logits = logits-10.

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




        if step > 200000:

            # Update generator
            loss = - torch.mean(outputs['f'])
            optim.zero_grad()
            loss.backward(retain_graph=True)  
            optim.step()

        if step > 80000:
            # Update encoder
            optim_net.zero_grad()
            outputs['net_loss'].backward(retain_graph=True)
            optim_net.step()

        # Update surrogate
        if method in ['simplax', 'HLAX', 'relax']:
            optim_surr.zero_grad()
            outputs['surr_loss'].backward(retain_graph=True)
            optim_surr.step()

        if method == 'HLAX':
            optim_surr2.zero_grad()
            surr_loss.backward(retain_graph=True)
            optim_surr2.step()

        if step%print_steps==0:
            # print (step, to_print(net_loss), to_print(logpxz - logq), to_print(logpx_given_z), to_print(logpz), to_print(logq))

            current_theta = torch.softmax(needsoftmax_mixtureweight, dim=0).float()
            theta_loss = L2_error(to_print2(true_mixture_weights),
                                    to_print2(current_theta))

            pz_give_x = true_posterior(x, mixture_weights=mixtureweights)
            probs = torch.softmax(logits, dim=1)
            inference_L2 = L2_batch(pz_give_x, probs)


            # outputs = reinforce(x[0].view(1,1), logits[0].view(1,C), mixtureweights, k=5000, get_grad=True)
            # print ('reinforce')
            # print (outputs['grad_avg'])
            # print (outputs['grad_std'])

            # outputs = relax(step=step, surrogate=surrogate, x=x[0].view(1,1), 
            #                 logits=logits[0].view(1,C), mixtureweights=mixtureweights, k=5000, get_grad=True)
            # print ('relax')
            # print (outputs['grad_avg'])
            # print (outputs['grad_std'])
            # print ()
            # fadsaf


            # #Grad variance 
            # grad = torch.autograd.grad([outputs['net_loss']], [logits], create_graph=True, retain_graph=True)[0]
            # # grad_var = torch.mean(torch.std(grad, dim=0))
            # grad_avg = torch.mean(torch.abs(grad))

            # plot_posteriors2(n_components, trueposteriors=to_print2(pz_give_x), qs=to_print2(probs), exp_dir=exp_dir, name=str(step))
            # fasfs

            print( 
                'S:{:5d}'.format(step),
                'Theta_loss:{:.3f}'.format(theta_loss),
                'Loss:{:.3f}'.format(to_print1(outputs['net_loss'])),
                'ELBO:{:.3f}'.format(to_print1(outputs['f'])),
                'lpx|z:{:.3f}'.format(to_print1(outputs['logpx_given_z'])),
                'lpz:{:.3f}'.format(to_print1(outputs['logpz'])),
                'lqz:{:.3f}'.format(to_print1(outputs['logq'])),
                )

            if step> 0:
                data_dict['steps'].append(step)
                data_dict['theta_losses'].append(theta_loss)
                data_dict['f'].append(to_print1(outputs['f']))
                data_dict['lpx_given_z'].append(to_print1(outputs['logpx_given_z']))
                data_dict['lpz'].append(to_print1(outputs['logpz']))
                data_dict['lqz'].append(to_print1(outputs['logq']))
                data_dict['inference_L2'].append(to_print2(inference_L2))
                # data_dict['grad_var'].append(to_print2(grad_var))
                # data_dict['grad_avg'].append(to_print2(grad_avg))
                if method in ['simplax', 'HLAX', 'relax']:
                    data_dict['surr_loss'].append(to_print2(outputs['surr_loss']))
                    data_dict['surr_dif'].append(to_print2(outputs['surr_dif']))
                if method in ['simplax', 'HLAX']:
                    data_dict['grad_split']['score'].append(to_print2(outputs['grad_score']))
                    data_dict['grad_split']['path'].append(to_print2(outputs['grad_path']))
                if method == 'HLAX':
                    data_dict['alpha'].append(to_print2(alpha))
                if method == 'relax':
                    data_dict['grad_logq'].append(to_print1(outputs['grad_logq']))
                    data_dict['grad_surr_z'].append(to_print1(outputs['grad_surr_z']))
                    data_dict['grad_surr_z_tilde'].append(to_print1(outputs['grad_surr_z_tilde']))


            check_nan(outputs['net_loss'])



        if step%plot_steps==0 and step!=0:

            # fsdfasd

            plot_curve2(data_dict, exp_dir)

            # list_of_posteriors = []
            # for ii in range(5):
            #     pz_give_x = true_posterior(x, mixture_weights=mixtureweights)
            #     probs = torch.softmax(logits, dim=1)
            #     inference_L2 = L2_batch(pz_give_x, probs)    
            #     list_of_posteriors.append([to_print2(pz_give_x), to_print2(probs)])      

            if step% (plot_steps*2) ==0:
            # if step% plot_steps ==0:
                plot_posteriors2(n_components, trueposteriors=to_print2(pz_give_x), qs=to_print2(probs), exp_dir=exp_dir, name=str(step))
            
                plot_dist2(n_components, mixture_weights=to_print2(current_theta), true_mixture_weights=to_print2(true_mixture_weights), exp_dir=exp_dir, name=str(step))

        if step % params_step==0 and step>0:

            # save_dir = home+'/Documents/Grad_Estimators/GMM/'
            with open( exp_dir+"data.p", "wb" ) as f:
                pickle.dump(data_dict, f)
            print ('saved data')


            # if step%1000==0:
            #     # pz_give_x = true_posterior(x, mixture_weights=mixtureweights)
            #     # probs = torch.softmax(logits, dim=1)
            #     # print (to_print2(pz_give_x[0]))
            #     # print (to_print2(probs[0]))
            #     print (to_print2(true_mixture_weights))
            #     print (to_print2(current_theta))
























if __name__ == "__main__":


    # exp_name = 'reinforce_pz_3cats'
    # exp_name = 'reinforce_3cats_2'
    # exp_name = 'simplax_3cats_2_temp4'
    # exp_name = 'simplax_3cats_withsplit'
    # exp_name = 'HLAX_3cats_withalphaplot'
    # exp_name = 'reinforce_3cats_b64'
    # exp_name = 'relax_3cats'
    # exp_name = 'relax_C20'
    exp_name = 'simplax_C20'
    # exp_name = 'relax_C3'
    # exp_name = 'simplax_3cats_newsurrobjective'
    # exp_name = 'reinforce_3cats_findingproblem'
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


    n_components = 20


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

    batch_size = 64
    n_steps = 300000
    k = 1

    print_steps = 500
    plot_steps = 5000
    params_step = 100000

    # train(method='reinforce', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)
    # train(method='relax', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)
    train(method='simplax', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)
    # train(method='HLAX', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)
    # train(method='reinforce_pz', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)

    print('Done.')


























