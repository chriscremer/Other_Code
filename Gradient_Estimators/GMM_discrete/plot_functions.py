


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






def sample_true2():
    cat = Categorical(probs= torch.tensor(true_mixture_weights))
    cluster = cat.sample()
    # print (cluster)
    # fsd
    norm = Normal(torch.tensor([cluster*10.]).float(), torch.tensor([5.0]).float())
    samp = norm.sample()
    # print (samp)
    return samp,cluster


def copmute_logpx(x,needsoftmax_mixtureweight):

    probs_sum = 0
    for i in range(len(x)):
        x_i = x[i].view(1,1)
        logpx = logprob_givenmixtureeweights(x_i, needsoftmax_mixtureweight)
        probs_sum += logpx
    return logpx / len(x)



def logprob_givenmixtureeweights(x, needsoftmax_mixtureweight):

    mixture_weights = torch.softmax(needsoftmax_mixtureweight, dim=0)
    probs_sum = 0# = []
    for c in range(n_components):
        m = Normal(torch.tensor([c*10.]).float().cuda(), torch.tensor([5.0]).float().cuda())
        # for x in xs:
        component_i = torch.exp(m.log_prob(x))* mixture_weights[c] #.numpy()
        # probs.append(probs)
        probs_sum+=component_i
    logprob = torch.log(probs_sum)
    return logprob

def L2_mixtureweights(w1,w2):
    dif = w1-w2
    # print (w1)
    # print (w2)
    # print (dif)
    # ffasd
    return np.sqrt(np.sum(dif**2))


def KL_mixutreweights(p,q):
    p += .0000001
    kl = np.sum(q*np.log(q/p))
    return kl



def inference_error(needsoftmax_mixtureweight):

    error_sum = 0
    kl_sum = 0
    n=10
    for i in range(n):

        # if x is None:
        x = sample_true(1).cuda() 
        trueposterior = true_posterior(x, needsoftmax_mixtureweight).view(n_components)

        logits = encoder.net(x)
        probs = torch.softmax(logits, dim=1).view(n_components)

        error = L2_mixtureweights(trueposterior.data.cpu().numpy(),probs.data.cpu().numpy())
        kl = KL_mixutreweights(trueposterior.data.cpu().numpy(), probs.data.cpu().numpy())

        error_sum+=error
        kl_sum += kl
    
    return error_sum/n, kl_sum/n
    # fsdfa






def compute_kl_batch(x,probs,needsoftmax_mixtureweight):

    # print (x.shape) #[B,1]
    # print (probs.shape) #[B,C]

    #get true posterior 
    true_posts = []
    for i in range(len(x)):
        x_i = x[i].view(1,1)
        trueposterior = true_posterior(x_i, needsoftmax_mixtureweight).view(n_components)
        # print (trueposterior.shape) #[C]
        # print (trueposterior)
        # fsdf
        true_posts.append(trueposterior)
    true_posts = torch.stack(true_posts)
    # print (true_posts.shape)

    true_posts += .0000001
    kl = torch.sum(probs*torch.log(probs/true_posts), dim=1)
    kl = torch.mean(kl)
    # print (kl.shape)
    
    # fasdfa
    return kl





# def logprob_undercomponent(x, component, needsoftmax_mixtureweight, cuda=False):
#     # c= component
#     # C = c.
#     B = x.shape[0]
#     # print()
#     # print (needsoftmax_mixtureweight.shape)
#     mixture_weights = torch.softmax(needsoftmax_mixtureweight, dim=0)
#     # print (mixture_weights.shape)
#     # fdsfa
#     # probs_sum = 0# = []
#     # for c in range(n_components):
#     # m = Normal(torch.tensor([c*10.]).float().cuda(), torch.tensor([5.0]).float() )#.cuda())
#     mean = (component.float()*10.).view(B,1)
#     std = (torch.ones([B]) *5.).view(B,1)
#     # print (mean.shape) #[B]
#     if not cuda:
#         m = Normal(mean, std)#.cuda())
#     else:
#         m = Normal(mean.cuda(), std.cuda())
#     # for x in xs:
#     # component_i = torch.exp(m.log_prob(x))* mixture_weights[c] #.numpy()
#     # print (m.log_prob(x))
#     # print (torch.log(mixture_weights[c]))
#     # print(x.shape)
#     logpx_given_z = m.log_prob(x)
#     logpz = torch.log(mixture_weights[component]).view(B,1)
#     # print (px_given_z.shape)
#     # print (component)
#     # print (mixture_weights)
#     # print (mixture_weights[component])
#     # print (torch.log(mixture_weights[component]).shape)
#     # fdsasa
#     # print (logpx_given_z.shape)
#     # print (logpz.shape)
#     # fsdfas
#     logprob = logpx_given_z + logpz
#     # print (logprob.shape)
#     # fsfd
#     # probs.append(probs)
#     # probs_sum+=component_i
#     # logprob = torch.log(component_i)
#     return logprob







#heavy side function 
def H(soft):
    return torch.argmax(soft, dim=1)


def check_nan(x):
    if (x != x).any():
        print ('nan')
        fsda

# needsoftmax_mixtureweight = torch.tensor(np.random.randn(n_components), requires_grad=True).float() #.view(n_components,1)
# print (needsoftmax_mixtureweight)

#Compute log prob given mixture weights
# x = sample_true()
# print (logprob_givenmixtureeweights(x, needsoftmax_mixtureweight))
# fadfs





def inference_error(needsoftmax_mixtureweight):

    error_sum = 0
    kl_sum = 0
    n=10
    for i in range(n):

        # if x is None:
        x = sample_true(1).cuda() 
        trueposterior = true_posterior(x, needsoftmax_mixtureweight).view(n_components)

        logits = encoder.net(x)
        probs = torch.softmax(logits, dim=1).view(n_components)

        error = L2_mixtureweights(trueposterior.data.cpu().numpy(),probs.data.cpu().numpy())
        kl = KL_mixutreweights(trueposterior.data.cpu().numpy(), probs.data.cpu().numpy())

        error_sum+=error
        kl_sum += kl
    
    return error_sum/n, kl_sum/n
    # fsdfa

















def plot_curve(steps, thetaloss, infloss, surrloss, grad_reinforce_list, grad_reparam_list,
                f_list,
                logpxz_list, logprob_cluster_list,
                inf_losses_kl, kl_losses_2, logpx_list):

    rows = 10
    cols = 1
    fig = plt.figure(figsize=(8+cols,8+rows), facecolor='white') #, dpi=150)

    col =0
    row = 0
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,thetaloss, label='theta loss')
    # ax.legend()
    ax.grid(True, alpha=.3)
    ax.set_ylabel('theta loss')

    col =0
    row = 1
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,infloss, label='inference loss')
    # ax.legend()  
    ax.grid(True, alpha=.3) 
    ax.set_ylabel('inference loss')


    col =0
    row = 2
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,surrloss, label='surrogate loss')   
    # ax.legend()
    ax.grid(True, alpha=.3)
    ax.set_ylabel('surrogate loss')


    col =0
    row = 3
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,grad_reinforce_list, label='reinforce')   
    ax.plot(steps,grad_reparam_list, label='reparam')   
    ax.legend()
    ax.grid(True, alpha=.3)
    ax.set_ylim(0., 40.)


    col =0
    row = 4
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,f_list)   
    # ax.legend()
    ax.grid(True, alpha=.3)
    ax.set_ylabel('f/ELBO')


    col =0
    row = 5
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,logpxz_list) 
    # ax.legend()
    ax.grid(True, alpha=.3)
    ax.set_ylabel('logpxz')


    col =0
    row = 6
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,logprob_cluster_list)   
    # ax.legend()
    ax.grid(True, alpha=.3)
    ax.set_ylabel('logprob_cluster')


    col =0
    row = 7
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,inf_losses_kl, label='inference loss KL')
    # ax.legend()  
    ax.grid(True, alpha=.3) 
    ax.set_ylabel('KL')

    col =0
    row = 8
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,kl_losses_2, label='inference loss KL')
    # ax.legend()  
    ax.grid(True, alpha=.3) 
    ax.set_ylabel('KL 2')

    col =0
    row = 9
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,logpx_list, label='inference loss KL')
    # ax.legend()  
    ax.grid(True, alpha=.3) 
    ax.set_ylabel('logpx_list')




    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = exp_dir+'gmm_curveplot.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()






def show_surr_preds():

    batch_size = 1

    rows = 3
    cols = 1
    fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

    for i in range(rows):

        x = sample_true(1).cuda() #.view(1,1)
        logits = encoder.net(x)
        probs = torch.softmax(logits, dim=1)
        cat = RelaxedOneHotCategorical(probs=probs, temperature=torch.tensor([1.]).cuda())
        cluster_S = cat.rsample()
        cluster_H = H(cluster_S)
        logprob_cluster = cat.log_prob(cluster_S.detach()).view(batch_size,1)
        check_nan(logprob_cluster)

        z = cluster_S

        n_evals = 40
        x1 = np.linspace(-9,205, n_evals)
        x = torch.from_numpy(x1).view(n_evals,1).float().cuda()
        z = z.repeat(n_evals,1)
        cluster_H = cluster_H.repeat(n_evals,1)
        xz = torch.cat([z,x], dim=1) 

        logpxz = logprob_undercomponent(x, component=cluster_H, needsoftmax_mixtureweight=needsoftmax_mixtureweight, cuda=True)
        f = logpxz #- logprob_cluster

        surr_pred = surrogate.net(xz)
        surr_pred = surr_pred.data.cpu().numpy()
        f = f.data.cpu().numpy()

        col =0
        row = i
        # print (row)
        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

        ax.plot(x1,surr_pred, label='Surr')
        ax.plot(x1,f, label='f')
        ax.set_title(str(cluster_H[0]))
        ax.legend()


    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = exp_dir+'gmm_surr.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()























def plot_posteriors(needsoftmax_mixtureweight, name=''):

    x = sample_true(1).cuda() 
    trueposterior = true_posterior(x, needsoftmax_mixtureweight).view(n_components)

    logits = encoder.net(x)
    probs = torch.softmax(logits, dim=1).view(n_components)

    trueposterior = trueposterior.data.cpu().numpy()
    qz = probs.data.cpu().numpy()

    error = L2_mixtureweights(trueposterior,qz)
    kl = KL_mixutreweights(p=trueposterior, q=qz)


    rows = 1
    cols = 1
    fig = plt.figure(figsize=(8+cols,8+rows), facecolor='white') #, dpi=150)

    col =0
    row = 0
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    width = .3
    ax.bar(range(len(qz)), trueposterior, width=width, label='True')
    ax.bar(np.array(range(len(qz)))+width, qz, width=width, label='q')
    # ax.bar(np.array(range(len(q_b)))+width+width, q_b, width=width)
    ax.legend()
    ax.grid(True, alpha=.3)
    ax.set_title(str(error) + ' kl:' + str(kl))
    ax.set_ylim(0.,1.)

    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = exp_dir+'posteriors'+name+'.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()
    

















def plot_dist():


    mixture_weights = torch.softmax(needsoftmax_mixtureweight, dim=0)

    rows = 1
    cols = 1
    fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

    col =0
    row = 0
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)


    xs = np.linspace(-9,205, 300)
    sum_ = np.zeros(len(xs))

    # C = 20
    for c in range(n_components):
        m = Normal(torch.tensor([c*10.]).float(), torch.tensor([5.0]).float())
        ys = []
        for x in xs:
            # component_i = (torch.exp(m.log_prob(x) )* ((c+5.) / 290.)).numpy()
            component_i = (torch.exp(m.log_prob(x) )* mixture_weights[c]).detach().cpu().numpy()


            ys.append(component_i)

        ys = np.reshape(np.array(ys), [-1])
        sum_ += ys
        ax.plot(xs, ys, label='')

    ax.plot(xs, sum_, label='')

    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = exp_dir+'gmm_plot_dist.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()
    





















# if simplax or reinforce or marginal:

def plot_both_dists():

    # needsoftmax_mixtureweight = needsoftmax_mixtureweight.cpu()

    #MAKE PLOT OF DISTRIBUTION
    rows = 1
    cols = 1
    fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

    col =0
    row = 0
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)



    xs = np.linspace(-9,205, 300)
    sum_ = np.zeros(len(xs))
    # C = 20
    for c in range(n_components):
        m = Normal(torch.tensor([c*10.]).float(), torch.tensor([5.0]).float())
        # xs = torch.tensor(xs)
        # print (m.log_prob(lin))
        ys = []
        for x in xs:
            # print (m.log_prob(x))
            # component_i = (torch.exp(m.log_prob(x) )* ((c+5.) / denom)).numpy()
            component_i = (torch.exp(m.log_prob(x) )* true_mixture_weights[c]).numpy()
            ys.append(component_i)
        ys = np.reshape(np.array(ys), [-1])
        sum_ += ys
        ax.plot(xs, ys, label='', c='c')
    ax.plot(xs, sum_, label='')



    # mixture_weights = torch.softmax(needsoftmax_mixtureweight, dim=0)
    # xs = np.linspace(-9,205, 300)
    # sum_ = np.zeros(len(xs))
    # C = 20
    # for c in range(C):
    #     m = Normal(torch.tensor([c*10.]).float(), torch.tensor([5.0]).float())
    #     # xs = torch.tensor(xs)
    #     # print (m.log_prob(lin))
    #     ys = []
    #     for x in xs:
    #         # print (m.log_prob(x))
    #         component_i = (torch.exp(m.log_prob(x) )* mixture_weights[c]).detach().numpy()
    #         ys.append(component_i)
    #     ys = np.reshape(np.array(ys), [-1])
    #     sum_ += ys
    #     ax.plot(xs, ys, label='', c='r')
    # ax.plot(xs, sum_, label='')


    # #HISTOGRAM
    # xs = []
    # for i in range(10000):
    #     x = sample_true().item()
    #     xs.append(x)
    # ax.hist(xs, bins=200, density=True)



    # # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    # if simplax:
    #     plt_path = exp_dir+'gmm_pdf_plot_simplax.png'
    # elif reinforce:
    #     plt_path = exp_dir+'gmm_pdf_plot_reinforce.png'
    # elif marginal:
    #     plt_path = exp_dir+'gmm_pdf_plot_marginal.png'

    # plt.savefig(plt_path)
    # print ('saved training plot', plt_path)
    # plt.close()




    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = exp_dir+'gmm_distplot.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()






# plot_distribution=0
# if plot_distribution:



#     rows = 1
#     cols = 1
#     fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

#     col =0
#     row = 0
#     ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

#     clusters =[]
#     logits= torch.randn(n_components)
#     cat = RelaxedOneHotCategorical(probs= torch.softmax(logits, dim=0), temperature=torch.tensor([1.]))#.cuda())

#     for i in range(1000):
#         cluster_S = cat.rsample()
#         cluster_H = H(cluster_S)
#         clusters.append(cluster_H)

#     ax.hist(clusters, density=True, bins=n_components)
#     ax.plot(range(n_components),torch.softmax(logits, dim=0).numpy() )


#     # save_dir = home+'/Documents/Grad_Estimators/GMM/'
#     plt_path = exp_dir+'gmm_plot_cat.png'
#     plt.savefig(plt_path)
#     print ('saved training plot', plt_path)
#     plt.close()



#     fdsfasd

#     # sum_ = 0
#     # for i in range(20):
#     #     sum_+= i+5
#     # print (sum_)
#     # fds 290


#     # print (m.log_prob(2.))
#     xs = np.linspace(-9,205, 300)

#     sum_ = np.zeros(len(xs))

#     C = 20
#     for c in range(C):


#         m = Normal(torch.tensor([c*10.]).float(), torch.tensor([5.0]).float())

        
#         # xs = torch.tensor(xs)
#         # print (m.log_prob(lin))

#         ys = []
#         for x in xs:
#             # print (m.log_prob(x))
#             component_i = (torch.exp(m.log_prob(x) )* ((c+5.) / 290.)).numpy()
#             ys.append(component_i)

#         ys = np.reshape(np.array(ys), [-1])

#         # print (sum_.shape)
#         # print (ys.shape)

#         sum_ += ys

#         # print (sum_)
#         # fsdfa


#         ax.plot(xs, ys, label='')


#         # ax.grid(True, alpha=.3)
#         # ax.set_title(r'$f=(b-.4)^2$', size=6, family='serif')
#         # ax.tick_params(labelsize=6)
#         # ax.set_ylabel(ylabel, size=6, family='serif')
#         # ax.set_xlabel(xlabel, size=6, family='serif')
#         # ax.legend(prop={'size':5}) #, loc=2)  #upper left


#     ax.plot(xs, sum_, label='')




#     # save_dir = home+'/Documents/Grad_Estimators/GMM/'
#     plt_path = exp_dir+'gmm_plot.png'
#     plt.savefig(plt_path)
#     print ('saved training plot', plt_path)
#     plt.close()




















