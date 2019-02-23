

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

def H(soft):
    return torch.argmax(soft, dim=1)

def L2_error(w1,w2):
    dif = w1-w2
    return np.sqrt(np.sum(dif**2))

def L2_batch(w1,w2):
    dif = w1-w2
    return torch.mean(torch.sqrt(torch.sum(dif**2, dim=1)))


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
        f = logpxz - logq
        net_loss += - torch.mean((f.detach() - 1.) * logq)
        # net_loss += - torch.mean( -logq.detach()*logq)

    net_loss = net_loss/ k

    return net_loss, f, logpx_given_z, logpz, logq



def simplax(surrogate, x, logits, mixtureweights, k=1):
    B = logits.shape[0]
    probs = torch.softmax(logits, dim=1)

    cat = RelaxedOneHotCategorical(probs=probs, temperature=torch.tensor([1.]).cuda())

    net_loss = 0
    surr_loss = 0
    for jj in range(k):

        cluster_S = cat.rsample()
        cluster_H = H(cluster_S)

        logq = cat.log_prob(cluster_S.detach()).view(B,1)
        logpx_given_z = logprob_undercomponent(x, component=cluster_H)
        logpz = torch.log(mixtureweights[cluster_H]).view(B,1)
        logpxz = logpx_given_z + logpz #[B,1]
        f = logpxz - logq

        surr_input = torch.cat([cluster_S, x], dim=1) #[B,21]
        surr_pred = surrogate.net(surr_input)

        net_loss += - torch.mean((f.detach() - 1. - surr_pred.detach()) * logq  + surr_pred)


        # surr_loss += torch.mean(torch.abs(f.detach()-1.-surr_pred))
        grad_logq = torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0]
        grad_surr = torch.autograd.grad([torch.mean(surr_pred)], [logits], create_graph=True, retain_graph=True)[0]
        surr_loss += torch.mean(((f.detach() - 1. - surr_pred) * grad_logq + grad_surr)**2)

        # grad_path = torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0]
        # grad_score = torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0]


    net_loss = net_loss/ k
    surr_loss = surr_loss/ k

    return net_loss, f, logpx_given_z, logpz, logq, surr_loss












def train(method, n_components, true_mixture_weights, exp_dir, needsoftmax_mixtureweight=None):

    print('Method:', method)

    true_mixture_weights = torch.tensor(true_mixture_weights, 
                                            requires_grad=True, device="cuda")

    if needsoftmax_mixtureweight is None:
        needsoftmax_mixtureweight = torch.randn(n_components, requires_grad=True, device="cuda")
    else:
        needsoftmax_mixtureweight = torch.tensor(needsoftmax_mixtureweight, 
                                            requires_grad=True, device="cuda")
    
    optim = torch.optim.Adam([needsoftmax_mixtureweight], lr=1e-5, weight_decay=1e-7)

    encoder = NN3(input_size=1, output_size=n_components, n_residual_blocks=3).cuda()
    optim_net = torch.optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=1e-7)

    if method == 'simplax':
        surrogate = NN3(input_size=1+n_components, output_size=1, n_residual_blocks=4).cuda()
        optim_surr = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
    

    data_dict = {}
    data_dict['steps'] = []
    data_dict['theta_losses'] = []
    data_dict['f'] = []
    data_dict['lpx_given_z'] = []
    data_dict['lpz'] = []
    data_dict['lqz'] = []
    data_dict['inference_L2'] = []
    data_dict['grad_var'] = []
    data_dict['grad_avg'] = []

    if method =='simplax':
        data_dict['surr_loss'] = []


    for step in range(0,n_steps+1):

        mixtureweights = torch.softmax(needsoftmax_mixtureweight, dim=0).float() #[C]

        x = sample_gmm(batch_size, mixture_weights=true_mixture_weights)
        logits = encoder.net(x)

        if method == 'reinforce':
            net_loss, f, logpx_given_z, logpz, logq = reinforce(x, logits, mixtureweights, k=1)
        elif method == 'simplax':
            net_loss, f, logpx_given_z, logpz, logq, surr_loss = simplax(surrogate, x, logits, mixtureweights, k=1)
        elif method == 'reinforce_pz':
            net_loss, f, logpx_given_z, logpz, logq = reinforce_pz(x, logits, mixtureweights, k=1)


        #Grad variance 
        grad = torch.autograd.grad([net_loss], [logits], create_graph=True, retain_graph=True)[0]
        grad_var = torch.mean(torch.std(grad, dim=0))
        grad_avg = torch.mean(torch.abs(grad))


        # Update encoder
        optim_net.zero_grad()
        net_loss.backward(retain_graph=True)
        optim_net.step()

        # Update generator
        loss = - torch.mean(f)
        optim.zero_grad()
        loss.backward(retain_graph=True)  
        optim.step()

        # Update surrogate
        if method == 'simplax':
            optim_surr.zero_grad()
            surr_loss.backward(retain_graph=True)
            optim_surr.step()


        if step%print_steps==0:
            # print (step, to_print(net_loss), to_print(logpxz - logq), to_print(logpx_given_z), to_print(logpz), to_print(logq))

            current_theta = torch.softmax(needsoftmax_mixtureweight, dim=0).float()
            theta_loss = L2_error(to_print2(true_mixture_weights),
                                    to_print2(current_theta))

            pz_give_x = true_posterior(x, mixture_weights=mixtureweights)
            probs = torch.softmax(logits, dim=1)
            inference_L2 = L2_batch(pz_give_x, probs)


            print( 
                'S:{:5d}'.format(step),
                'Theta_loss:{:.3f}'.format(theta_loss),
                'Loss:{:.3f}'.format(to_print1(net_loss)),
                'ELBO:{:.3f}'.format(to_print1(f)),
                'lpx|z:{:.3f}'.format(to_print1(logpx_given_z)),
                'lpz:{:.3f}'.format(to_print1(logpz)),
                'lqz:{:.3f}'.format(to_print1(logq)),
                )

            if step> 0:
                data_dict['steps'].append(step)
                data_dict['theta_losses'].append(theta_loss)
                data_dict['f'].append(to_print1(f))
                data_dict['lpx_given_z'].append(to_print1(logpx_given_z))
                data_dict['lpz'].append(to_print1(logpz))
                data_dict['lqz'].append(to_print1(logq))
                data_dict['inference_L2'].append(to_print2(inference_L2))
                data_dict['grad_var'].append(to_print2(grad_var))
                data_dict['grad_avg'].append(to_print2(grad_avg))
                if method == 'simplax':
                    data_dict['surr_loss'].append(to_print2(surr_loss))




        if step%plot_steps==0 and step!=0:

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
    exp_name = 'simplax_3cats_2'
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

    # needsoftmax_mixtureweight = np.log(true_mixture_weights)

    # train(n_components=n_components, 
    #             needsoftmax_mixtureweight=needsoftmax_mixtureweight)

    batch_size = 10
    n_steps = 300000
    k = 1

    print_steps = 500
    plot_steps = 5000
    params_step = 100000

    # train(method='reinforce', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)
    train(method='simplax', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)
    # train(method='reinforce_pz', n_components=n_components, exp_dir=exp_dir, true_mixture_weights=true_mixture_weights)

    print('Done.')


























