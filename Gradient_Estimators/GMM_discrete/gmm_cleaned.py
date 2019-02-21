

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




def reinforce(n_components, needsoftmax_mixtureweight=None):

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


    # needsoftmax_qprobs = torch.randn((1,n_components), requires_grad=True, device="cuda")
    # optim_net = torch.optim.SGD([needsoftmax_qprobs], lr=1e-3, weight_decay=1e-7)

    # probs = torch.softmax(needsoftmax_qprobs, dim=1)
    # print ('probs:', to_print2(probs))

    # x = sample_gmm(batch_size, mixture_weights=mixtureweights)

    # count = np.zeros(3)

    for step in range(n_steps):

        x = sample_gmm(batch_size, mixture_weights=mixtureweights)
        logits = encoder.net(x)
        # logits = needsoftmax_qprobs
        # print (logits.shape)
        # fdsfd
        probs = torch.softmax(logits, dim=1)
        # print (probs)
        # print (torch.log(probs))
        # print (torch.softmax(torch.log(probs), dim=1))
        
        # print (probs.shape)
        # print (probs)
        # probs = probs.repeat(batch_size, 1)
        cat = Categorical(probs=probs)

        net_loss = 0
        for jj in range(k):

            cluster_H = cat.sample()

            # c_ = cluster_H.data.cpu().numpy()[0]
            # count[c_]+=1

            # print (cluster_H.shape)
            # print (cluster_H)

            # print(logits)
            # print (cluster_H)
            # print (cluster_H.shape)
            # cluster_H = torch.tensor([0,1,2]).cuda()
            # print (cluster_H.shape)

            # fsfsad
            

            # print(logits.shape)
            # print (cluster_H.shape)
            # print ()

            # tt = torch.tensor([0]).cuda() #.view(1,1)

            # print (tt.shape)


            # print (logits[tt])
            # print (logits[tt].shape)

            # aa = torch.index_select(logits, 1, tt)
            # print (aa.shape)
            # print (aa)
            # print ()

            # aa = torch.index_select(logits, 1, cluster_H)
            # print (aa.shape)
            # print (aa)


            # sfad


            # print (logits[0])
            # fsdfas
            # print (logits[cluster_H])
            # print (logits[cluster_H].shape)
            # dsfasd



            logq = cat.log_prob(cluster_H).view(batch_size,1)
            # print (logq1.shape)
            # print (logq)
            # # fsd
            # # print (torch.log(probs))
            # # fasfd
            # # logq2 = torch.index_select(logits, 1, cluster_H) #.view(batch_size,1)
            # # logq3 = torch.log(torch.index_select(probs, 1, cluster_H) )#.view(batch_size,1)

            # grad0 = torch.autograd.grad(outputs=logq[0], inputs=(probs), retain_graph=True)[0]
            # grad1 = torch.autograd.grad(outputs=logq[1], inputs=(probs), retain_graph=True)[0]
            # grad2 = torch.autograd.grad(outputs=logq[2], inputs=(probs), retain_graph=True)[0]

            # print (grad0)
            # print (grad1)
            # print (grad2)
            # print ()
            # print (grad0*probs[0][0])
            # print (grad1*probs[0][1])
            # print (grad2*probs[0][2])
            # print ()

            # grad0 = torch.autograd.grad(outputs=logq[0], inputs=(needsoftmax_qprobs), retain_graph=True)[0]
            # grad1 = torch.autograd.grad(outputs=logq[1], inputs=(needsoftmax_qprobs), retain_graph=True)[0]
            # grad2 = torch.autograd.grad(outputs=logq[2], inputs=(needsoftmax_qprobs), retain_graph=True)[0]
            # print (grad0)
            # print (grad1)
            # print (grad2)
            # print ()
            # print (grad0*probs[0][0])
            # print (grad1*probs[0][1])
            # print (grad2*probs[0][2])
            # print ()
            # print (grad0*probs[0][0] + grad1*probs[0][1] + grad2*probs[0][2])
            # print ()
            # print ()

            # grad0 = torch.autograd.grad(outputs=logq[0].detach()*logq[0], inputs=(needsoftmax_qprobs), retain_graph=True)[0]
            # grad1 = torch.autograd.grad(outputs=logq[1].detach()*logq[1], inputs=(needsoftmax_qprobs), retain_graph=True)[0]
            # grad2 = torch.autograd.grad(outputs=logq[2].detach()*logq[2], inputs=(needsoftmax_qprobs), retain_graph=True)[0]
            # print (grad0)
            # print (grad1)
            # print (grad2)
            # print ()
            # print (grad0*probs[0][0])
            # print (grad1*probs[0][1])
            # print (grad2*probs[0][2])
            # print ()
            # print (grad0*probs[0][0] + grad1*probs[0][1] + grad2*probs[0][2])
            # print ()



            # fsfad



            # print(logq1, logq2)
            # print(logq3)
            # fsdaf

            # print (logq.shape)
            # print (logq)
            # fads
            logpx_given_z = logprob_undercomponent(x, component=cluster_H)
            logpz = torch.log(mixtureweights[cluster_H]).view(batch_size,1)
            logpxz = logpx_given_z + logpz #[B,1]
            # net_loss += - torch.mean((logpxz.detach() - 100.) * logq)
            net_loss += - torch.mean( -logq.detach()*logq)

        net_loss = net_loss/ k


        optim_net.zero_grad()
        net_loss.backward(retain_graph=True)
        optim_net.step()


        # if step%10==0:
        #     print (count/np.sum(count), probs.data.cpu().numpy())




        if step%100==0:
            # print (step, to_print(net_loss), to_print(logpxz - logq), to_print(logpx_given_z), to_print(logpz), to_print(logq))

            
            print ()
            print( 
                'S:{:5d}'.format(step),
                # 'T:{:.2f}'.format(time.time() - start_time),
                'Loss:{:.4f}'.format(to_print1(net_loss)),
                'ELBO:{:.4f}'.format(to_print1(logpxz - logq)),
                'lpx|z:{:.4f}'.format(to_print1(logpx_given_z)),
                'lpz:{:.4f}'.format(to_print1(logpz)),
                'lqz:{:.4f}'.format(to_print1(logq)),
                )

            pz_give_x = true_posterior(x, mixture_weights=mixtureweights)
            # print (pz_give_x.shape)
            # print (to_print2(x[0]), to_print2(cluster_H[0]))
            print (to_print2(probs[0]))
            # print (to_print2(torch.exp(logq[0])))
            # before = to_print2(torch.exp(logq[0]))
            # firstH = cluster_H[0]











            # logits = encoder.net(x)
            # # logits = needsoftmax_qprobs
            # probs = torch.softmax(logits, dim=1)
            # cat = Categorical(probs=probs)

            # net_loss = 0
            # for jj in range(k):

            #     cluster_H = cat.sample()

            #     logq = cat.log_prob(cluster_H).view(batch_size,1)
            #     # logq = logits[cluster_H].view(batch_size,1)

            #     logpx_given_z = logprob_undercomponent(x, component=cluster_H)
            #     logpz = torch.log(mixtureweights[cluster_H]).view(batch_size,1)
            #     logpxz = logpx_given_z + logpz #[B,1]
            #     # net_loss += - torch.mean((logpxz.detach() - 100.) * logq)
            #     net_loss += - torch.mean( -logq.detach() * logq)

            # net_loss = net_loss/ k


            # # print ()
            # print( 
            #     'S:{:5d}'.format(step),
            #     # 'T:{:.2f}'.format(time.time() - start_time),
            #     'Loss:{:.4f}'.format(to_print1(net_loss)),
            #     'ELBO:{:.4f}'.format(to_print1(logpxz - logq)),
            #     'lpx|z:{:.4f}'.format(to_print1(logpx_given_z)),
            #     'lpz:{:.4f}'.format(to_print1(logpz)),
            #     'lqz:{:.4f}'.format(to_print1(logq)),
            #     )

            # pz_give_x = true_posterior(x, mixture_weights=mixtureweights)
            # # print (pz_give_x.shape)
            # print (to_print2(x[0]), to_print2(cluster_H[0]))
            # print (to_print2(probs[0]))

            # print (to_print2(torch.exp(cat.log_prob(firstH)[0])))
            # after = to_print2(torch.exp(cat.log_prob(firstH)[0]))
            # # logq = logits[cluster_H].view(batch_size,1)

            # dif = before - after 

            # print ('Dif:', dif, 'positive is good')

            # if dif < 0:
            #     print ('howww')
            #     fafsd




            # print (to_print2(torch.exp(logq[cluster_H[0]])))
            # print (to_print2(torch.exp(cat.log_prob(torch.tensor([0]).cuda())[0]))) #, to_print2(torch.exp(logq[1])), to_print2(torch.exp(logq[2])))
            # print (to_print2(torch.exp(cat.log_prob(torch.tensor([1]).cuda())[0]))) #, to_print2(torch.exp(logq[1])), to_print2(torch.exp(logq[2])))
            # print (to_print2(torch.exp(cat.log_prob(torch.tensor([2]).cuda())[0]))) #, to_print2(torch.exp(logq[1])), to_print2(torch.exp(logq[2])))
            # print (to_print2(torch.exp(logq[0])), to_print2(torch.exp(logq[1])), to_print2(torch.exp(logq[2])))
            # print (to_print2(pz_give_x[0]))



            # print (step, 'f:', torch.mean(f).cpu().data.numpy(), 'surr_loss:', surr_loss.cpu().data.detach().numpy(), 
            #                 'theta dif:', L2_mixtureweights(true_mixture_weights,torch.softmax(
            #                             needsoftmax_mixtureweight, dim=0).cpu().data.detach().numpy()))
            # # if step %5000==0:
            # #     print (torch.softmax(needsoftmax_mixtureweight, dim=0).cpu().data.detach().numpy()) 
            # #     # test_samp, test_cluster = sample_true2() 
            # #     # print (test_cluster.cpu().data.numpy(), test_samp.cpu().data.numpy(), torch.softmax(encoder.net(test_samp.cuda().view(1,1)), dim=1))           
            # #     print ()

            # if step > 0:
            #     L2_losses.append(L2_mixtureweights(true_mixture_weights,torch.softmax(
            #                                 needsoftmax_mixtureweight, dim=0).cpu().data.detach().numpy()))
            #     data_dict['steps'].append(step)
            #     surr_losses.append(surr_loss.cpu().data.detach().numpy())

            #     inf_error, kl_error = inference_error(needsoftmax_mixtureweight)
            #     inf_losses.append(inf_error)
            #     inf_losses_kl.append(kl_error)

            #     kl_batch = compute_kl_batch(x,probs,needsoftmax_mixtureweight)
            #     kl_losses_2.append(kl_batch)

            #     logpx = copmute_logpx(x, needsoftmax_mixtureweight)
            #     logpx_list.append(logpx)

            #     f_list.append(torch.mean(f).cpu().data.detach().numpy())
            #     logpxz_list.append(torch.mean(logpxz).cpu().data.detach().numpy())
            #     logprob_cluster_list.append(torch.mean(logprob_cluster).cpu().data.detach().numpy())




            #     # i_feel_like_it = 1
            #     # if i_feel_like_it:

            #     if len(inf_losses) > 0:
            #         print ('probs', probs[0])
            #         print('logpxz', logpxz[0])
            #         print('pred', surr_pred[0])
            #         print ('dif', logpxz.detach()[0]-surr_pred.detach()[0])
            #         print ('logq', logprob_cluster[0])
            #         print ('dif*logq', (logpxz.detach()[0]-surr_pred.detach()[0])*logprob_cluster[0])
                    
                    


            #         output= torch.mean((logpxz.detach()-surr_pred.detach()) * logprob_cluster, dim=0)[0] 
            #         output2 = torch.mean(surr_pred, dim=0)[0]
            #         output3 = torch.mean(logprob_cluster, dim=0)[0]
            #         # input_ = torch.mean(probs, dim=0) #[0]
            #         # print (probs.shape)
            #         # print (output.shape)
            #         # print (input_.shape)
            #         grad_reinforce = torch.autograd.grad(outputs=output, inputs=(probs), retain_graph=True)[0]
            #         grad_reparam = torch.autograd.grad(outputs=output2, inputs=(probs), retain_graph=True)[0]
            #         grad3 = torch.autograd.grad(outputs=output3, inputs=(probs), retain_graph=True)[0]
            #         # print (grad)
            #         # print (grad_reinforce.shape)
            #         # print (grad_reparam.shape)
            #         grad_reinforce = torch.mean(torch.abs(grad_reinforce))
            #         grad_reparam = torch.mean(torch.abs(grad_reparam))
            #         grad3 = torch.mean(torch.abs(grad3))
            #         # print (grad_reinforce)
            #         # print (grad_reparam)
            #         # dfsfda
            #         grad_reparam_list.append(grad_reparam.cpu().data.detach().numpy())
            #         grad_reinforce_list.append(grad_reinforce.cpu().data.detach().numpy())
            #         # grad_reinforce_list.append(grad_reinforce.cpu().data.detach().numpy())

            #         print ('reparam:', grad_reparam.cpu().data.detach().numpy())
            #         print ('reinforce:', grad_reinforce.cpu().data.detach().numpy())
            #         print ('logqz grad:', grad3.cpu().data.detach().numpy())

            #         print ('current mixuture weights')
            #         print (torch.softmax(needsoftmax_mixtureweight, dim=0))
            #         print()

            #         # print ()
            #     else:
            #         grad_reparam_list.append(0.)
            #         grad_reinforce_list.append(0.)                    



            # if len(surr_losses) > 3  and step %1000==0:
            #     plot_curve(steps=steps_list,  thetaloss=L2_losses, 
            #                 infloss=inf_losses, surrloss=surr_losses,
            #                 grad_reinforce_list=grad_reinforce_list, 
            #                 grad_reparam_list=grad_reparam_list,
            #                 f_list=f_list, logpxz_list=logpxz_list,
            #                 logprob_cluster_list=logprob_cluster_list,
            #                 inf_losses_kl=inf_losses_kl,
            #                 kl_losses_2=kl_losses_2,
            #                 logpx_list=logpx_list)


            #     plot_posteriors(needsoftmax_mixtureweight)
            #     plot_dist()
            #     show_surr_preds()
                

            # print (f)
            # print (surr_pred)

            #Understand surr preds
            # if step %5000==0:

            # if step ==0:
                
                # fasdf
                


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

    reinforce(n_components=n_components, 
                needsoftmax_mixtureweight=needsoftmax_mixtureweight)


    fsdfasd


























