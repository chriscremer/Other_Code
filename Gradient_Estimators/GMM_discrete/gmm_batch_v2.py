

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

from NN import NN
from NN import NN2
from NN import NN3



exp_name = 'optimize_single_datapoint'
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



#METHOD
marginal = 0
reinforce = 0
simplax = 1








seed=0
torch.manual_seed(seed)


n_components = 20


#Sample from true distribution
# sample component from categorical
# sample from the gaussian of that component

true_mixture_weights = []
for c in range(n_components):
    true_mixture_weights.append((c+5.) / 290.)
# print(true_mixture_weights)
true_mixture_weights = np.array(true_mixture_weights)


def sample_true(batch_size):
    # print (true_mixture_weights.shape)
    cat = Categorical(probs=torch.tensor(true_mixture_weights))
    cluster = cat.sample([batch_size]) # [B]
    mean = (cluster*10.).float()
    std = torch.ones([batch_size]) *5.
    # print (cluster.shape)
    # fsd
    # norm = Normal(torch.tensor([cluster*10.]).float(), torch.tensor([5.0]).float())
    norm = Normal(mean, std)
    samp = norm.sample()
    # print (samp.shape)
    # fadsf
    samp = samp.view(batch_size, 1)
    return samp

def sample_true2():
    cat = Categorical(probs= torch.tensor(true_mixture_weights))
    cluster = cat.sample()
    # print (cluster)
    # fsd
    norm = Normal(torch.tensor([cluster*10.]).float(), torch.tensor([5.0]).float())
    samp = norm.sample()
    # print (samp)
    return samp,cluster

def logprob_givenmixtureeweights(x, needsoftmax_mixtureweight):

    mixture_weights = torch.softmax(needsoftmax_mixtureweight, dim=0)
    probs_sum = 0# = []
    for c in range(n_components):
        m = Normal(torch.tensor([c*10.]).float(), torch.tensor([5.0]).float())
        # for x in xs:
        component_i = torch.exp(m.log_prob(x))* mixture_weights[c] #.numpy()
        # probs.append(probs)
        probs_sum+=component_i
    logprob = torch.log(probs_sum)
    return logprob

def L2_mixtureweights(w1,w2):
    dif = w1-w2
    return np.sqrt(np.sum(dif**2))

def inference_error():

    x = sample_true(1).cuda() 
    trueposterior = true_posterior(x, needsoftmax_mixtureweight).view(n_components)



    logits = encoder.net(x)
    probs = torch.softmax(logits, dim=1).view(n_components)


    # print(trueposterior)
    # print (probs)
    # print ((trueposterior-probs)**2)
    # print()

    # print (trueposterior.shape)
    # print (probs.shape)
    # print (L2_mixtureweights(trueposterior.data.cpu().numpy(),probs.data.cpu().numpy()))
    return L2_mixtureweights(trueposterior.data.cpu().numpy(),probs.data.cpu().numpy())
    # fsdfa


def plot_posteriors(x=None, name=''):

    if x is None:
        x = sample_true(1).cuda() 
    else:
        x = x[0].view(1,1)


    trueposterior = true_posterior(x, needsoftmax_mixtureweight).view(n_components)

    logits = encoder.net(x)
    probs = torch.softmax(logits, dim=1).view(n_components)

    trueposterior = trueposterior.data.cpu().numpy()
    qz = probs.data.cpu().numpy()

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
    ax.set_title(str(x))

    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = exp_dir+'posteriors' + name+'.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()
    




def logprob_undercomponent(x, component, needsoftmax_mixtureweight, cuda=False):
    # c= component
    # C = c.
    B = x.shape[0]
    # print()
    # print (needsoftmax_mixtureweight.shape)
    mixture_weights = torch.softmax(needsoftmax_mixtureweight, dim=0)
    # print (mixture_weights.shape)
    # fdsfa
    # probs_sum = 0# = []
    # for c in range(n_components):
    # m = Normal(torch.tensor([c*10.]).float().cuda(), torch.tensor([5.0]).float() )#.cuda())
    mean = (component.float()*10.).view(B,1)
    std = (torch.ones([B]) *5.).view(B,1)
    # print (mean.shape) #[B]
    if not cuda:
        m = Normal(mean, std)#.cuda())
    else:
        m = Normal(mean.cuda(), std.cuda())
    # for x in xs:
    # component_i = torch.exp(m.log_prob(x))* mixture_weights[c] #.numpy()
    # print (m.log_prob(x))
    # print (torch.log(mixture_weights[c]))
    # print(x.shape)
    logpx_given_z = m.log_prob(x)
    logpz = torch.log(mixture_weights[component]).view(B,1)
    # print (px_given_z.shape)
    # print (component)
    # print (mixture_weights)
    # print (mixture_weights[component])
    # print (torch.log(mixture_weights[component]).shape)
    # fdsasa
    logprob = logpx_given_z + logpz
    # print (logprob.shape)
    # fsfd
    # probs.append(probs)
    # probs_sum+=component_i
    # logprob = torch.log(component_i)
    return logprob



def true_posterior(x, needsoftmax_mixtureweight):

    mixture_weights = torch.softmax(needsoftmax_mixtureweight, dim=0)
    probs_ = []
    for c in range(n_components):
        m = Normal(torch.tensor([c*10.]).float().cuda(), torch.tensor([5.0]).float().cuda())
        component_i = torch.exp(m.log_prob(x))* mixture_weights[c] #.numpy()
        # print(component_i.shape)
        # fsdf
        probs_.append(component_i[0])
    probs_ = torch.stack(probs_)
    probs_ = probs_ / torch.sum(probs_)
    # print (probs_.shape)
    # fdssdfd
    # logprob = torch.log(probs_sum)
    return probs_




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



def plot_curve(steps, thetaloss, infloss, surrloss, grad_reinforce_list, grad_reparam_list,
                f_list,
                logpxz_list, logprob_cluster_list):

    rows = 7
    cols = 1
    fig = plt.figure(figsize=(8+cols,8+rows), facecolor='white') #, dpi=150)

    col =0
    row = 0
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,thetaloss, label='theta loss')
    ax.legend()
    ax.grid(True, alpha=.3)

    col =0
    row = 1
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,infloss, label='inference loss')
    ax.legend()  
    ax.grid(True, alpha=.3) 

    col =0
    row = 2
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(steps,surrloss, label='surrogate loss')   
    ax.legend()
    ax.grid(True, alpha=.3)


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
        f = logpxz - logprob_cluster

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



def plot_dist(x=None):

    if x is None:
        x1 = sample_true(1).cuda() 
    else:
        x1 = x[0].cpu().numpy()#.view(1,1)
        # print (x)

    mixture_weights = torch.softmax(needsoftmax_mixtureweight, dim=0)

    rows = 1
    cols = 1
    fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

    col =0
    row = 0
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)


    xs = np.linspace(-9,205, 300)
    sum_ = np.zeros(len(xs))

    C = 20
    for c in range(C):
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

    # print (x)
    ax.plot([x1,x1+.001],[0.,.002])
    # fasda

    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = exp_dir+'gmm_plot_dist.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()
    



















if simplax:


    def get_loss():

        x = sample_true(batch_size).cuda() #.view(1,1)
        logits = encoder.net(x)
        probs = torch.softmax(logits, dim=1)
        cat = RelaxedOneHotCategorical(probs=probs, temperature=torch.tensor([temp]).cuda())
        cluster_S = cat.rsample()
        cluster_H = H(cluster_S)
        # cluster_onehot = torch.zeros(n_components)
        # cluster_onehot[cluster_H] = 1.
        logprob_cluster = cat.log_prob(cluster_S.detach()).view(batch_size,1)
        check_nan(logprob_cluster)

        logpxz = logprob_undercomponent(x, component=cluster_H, needsoftmax_mixtureweight=needsoftmax_mixtureweight, cuda=True)
        f = logpxz - logprob_cluster

        surr_input = torch.cat([cluster_S, x], dim=1) #[B,21]
        surr_pred = surrogate.net(surr_input)
        
        # net_loss = - torch.mean((f.detach()-surr_pred.detach()) * logprob_cluster + surr_pred)
        # loss = - torch.mean(f)
        surr_loss = torch.mean(torch.abs(f.detach()-surr_pred))

        return surr_loss


    #SIMPLAX
    needsoftmax_mixtureweight = torch.randn(n_components, requires_grad=True, device="cuda")#.cuda()
    encoder = NN3(input_size=1, output_size=n_components).cuda()
    surrogate = NN3(input_size=1+n_components, output_size=1).cuda()
    # optim = torch.optim.Adam([needsoftmax_mixtureweight], lr=.00004)
    # optim_net = torch.optim.Adam(encoder.parameters(), lr=.0004)
    # optim_surr = torch.optim.Adam(surrogate.parameters(), lr=.004)
    optim = torch.optim.Adam([needsoftmax_mixtureweight], lr=.0001)
    optim_net = torch.optim.Adam(encoder.parameters(), lr=.001)
    optim_surr = torch.optim.Adam(surrogate.parameters(), lr=.005)
    temp = 1.
    batch_size = 1 #00
    n_steps = 300000
    surrugate_steps = 1#00
    k = 10000
    L2_losses = []
    inf_losses = []
    surr_losses = []
    steps_list =[]
    grad_reparam_list =[]
    grad_reinforce_list =[]
    f_list = []
    logpxz_list = []
    logprob_cluster_list = []
    # logprob_cluster_list = []

    x = sample_true(batch_size).cuda() #.view(1,1)

    counts = np.zeros(n_components)
    for step in range(n_steps):

        for ii in range(surrugate_steps):
            surr_loss = get_loss()
            optim_surr.zero_grad()
            surr_loss.backward()
            optim_surr.step()

        # x = sample_true(batch_size).cuda() #.view(1,1)
        logits = encoder.net(x)
        probs = torch.softmax(logits, dim=1)

        probs2 = probs.cpu().data.numpy()[0]
        print()
        for iii in range(len(probs2)):
            print(str(iii)+':'+str(probs2[iii]), end =" ")
        print ()

        # print (probs.shape)
        print (probs)
        # fsdf
        cat = RelaxedOneHotCategorical(probs=probs, temperature=torch.tensor([temp]).cuda())

        net_loss = 0
        loss = 0
        surr_loss = 0
        grads = 0
        for jj in range(k):

            cluster_S = cat.rsample()
            # print (cluster_S.shape)
            # dfsafd

            # cluster_onehot = torch.zeros(n_components).cuda()
            # cluster_onehot[jj%20] =1.
            # cluster_S = torch.softmax(cluster_onehot, dim=0).view(1,20)

            # print (cluster_onehot)
            # logprob_cluster = cat.log_prob(cluster_onehot.detach()).view(batch_size,1)
            # print (logprob_cluster)
            # fsdfasd


            cluster_H = H(cluster_S)
            
            comp = cluster_H.cpu().numpy()[0]
            # print (comp)
            counts[comp] +=1
            # fsfsa
            # print (x, 'samp', cluster_H)
            # cluster_onehot = torch.zeros(n_components)
            # cluster_onehot[cluster_H] = 1.




            logprob_cluster = cat.log_prob(cluster_S.detach()).view(batch_size,1)
            check_nan(logprob_cluster)

            logpxz = logprob_undercomponent(x, component=cluster_H, needsoftmax_mixtureweight=needsoftmax_mixtureweight, cuda=True)
            f = logpxz #- logprob_cluster
            # print (jj%20, f, logprob_cluster)
            print ()
            print ('H:', comp, 'f', f, )
            print ('logq', logprob_cluster)

            surr_input = torch.cat([cluster_S, x], dim=1) #[B,21]
            surr_pred = surrogate.net(surr_input)
            
            # print (f.shape)
            # print (surr_pred.shape)
            # print (logprob_cluster.shape)
            # fsadfsa
            # net_loss = - torch.mean((f.detach()-surr_pred.detach()) * logprob_cluster + surr_pred)
            net_loss = - torch.mean((f.detach()) * logprob_cluster)
            loss = - torch.mean(f)
            surr_loss = torch.mean(torch.abs(f.detach()-surr_pred))

            grad = torch.autograd.grad(outputs=net_loss, inputs=(probs), retain_graph=True)[0]
            grad = grad.cpu().numpy()[0]


            optim_net.zero_grad()
            net_loss.backward(retain_graph=True)
            optim_net.step()

            logits11 = encoder.net(x)
            probs11 = torch.softmax(logits11, dim=1)
            print ('probs', probs11)
            # print('cluster_S', cluster_S)
            print ('grad:', grad)
            # print ()
            # print()



            cat22 = RelaxedOneHotCategorical(probs=probs11, temperature=torch.tensor([temp]).cuda())
            print ('logprob post', cat22.log_prob(cluster_S.detach()).view(batch_size,1))
            print ()
            print()

            if jj ==2:
                ffasd

            grads += grad


            # if jj%100==0:

            #     print ()
            #     print (jj)
            #     print (counts)
            #     for iii in range(len(grads)):
            #         print(str(iii)+':'+str(grads[iii]//jj), end =" ")
            #     # print ()

            

        net_loss = net_loss/ k
        loss = loss / k
        surr_loss = surr_loss/ k


        plot_posteriors(x=x,name=str(step))


        grad = torch.autograd.grad(outputs=net_loss, inputs=(probs), retain_graph=True)[0]
        grad = grad.cpu().numpy()[0]
        for iii in range(len(grad)):
            print(str(iii)+':'+str(grad[iii]), end =" ")
        print ()
        plot_posteriors(x=x,name=str(step))


        fdadsd

        # if step %2==0:
        #     optim.zero_grad()
        #     loss.backward(retain_graph=True)  
        #     optim.step()

        optim_net.zero_grad()
        net_loss.backward(retain_graph=True)
        optim_net.step()

        optim_surr.zero_grad()
        surr_loss.backward(retain_graph=True)
        optim_surr.step()


        if step%500==0:
            print (step, 'f:', torch.mean(f).cpu().data.numpy(), 'surr_loss:', surr_loss.cpu().data.detach().numpy(), 
                            'theta dif:', L2_mixtureweights(true_mixture_weights,torch.softmax(
                                        needsoftmax_mixtureweight, dim=0).cpu().data.detach().numpy()))
            
            # print (x[0])
            # fsf
            # plot_posteriors(x=x,name=str(step))
            plot_dist(x=x)


            if step %5000==0:
                print (torch.softmax(needsoftmax_mixtureweight, dim=0).cpu().data.detach().numpy()) 
                # test_samp, test_cluster = sample_true2() 
                # print (test_cluster.cpu().data.numpy(), test_samp.cpu().data.numpy(), torch.softmax(encoder.net(test_samp.cuda().view(1,1)), dim=1))           
                print ()

            if step > 0:
                L2_losses.append(L2_mixtureweights(true_mixture_weights,torch.softmax(
                                            needsoftmax_mixtureweight, dim=0).cpu().data.detach().numpy()))
                steps_list.append(step)
                surr_losses.append(surr_loss.cpu().data.detach().numpy())

                inf_error = inference_error()
                inf_losses.append(inf_error)



                f_list.append(torch.mean(f).cpu().data.detach().numpy())
                logpxz_list.append(torch.mean(logpxz).cpu().data.detach().numpy())
                logprob_cluster_list.append(torch.mean(logprob_cluster).cpu().data.detach().numpy())




                # i_feel_like_it = 1
                # if i_feel_like_it:

                if len(inf_losses) > 1:
                    output= torch.mean((f.detach()-surr_pred.detach()) * logprob_cluster, dim=0)[0] 
                    output2 = torch.mean(surr_pred, dim=0)[0]
                    # input_ = torch.mean(probs, dim=0) #[0]
                    # print (probs.shape)
                    # print (output.shape)
                    # print (input_.shape)
                    grad_reinforce = torch.autograd.grad(outputs=output, inputs=(probs), retain_graph=True)[0]
                    grad_reparam = torch.autograd.grad(outputs=output2, inputs=(probs), retain_graph=True)[0]
                    # print (grad)
                    # print (grad_reinforce.shape)
                    # print (grad_reparam.shape)
                    grad_reinforce = torch.mean(torch.abs(grad_reinforce))
                    grad_reparam = torch.mean(torch.abs(grad_reparam))
                    # print (grad_reinforce)
                    # print (grad_reparam)
                    # dfsfda
                    grad_reparam_list.append(grad_reparam.cpu().data.detach().numpy())
                    grad_reinforce_list.append(grad_reinforce.cpu().data.detach().numpy())
                    print (grad_reparam.cpu().data.detach().numpy(), grad_reinforce.cpu().data.detach().numpy())
                else:
                    grad_reparam_list.append(0.)
                    grad_reinforce_list.append(0.)                    



            if len(surr_losses) > 3  and step %5000==0:
                plot_curve(steps=steps_list,  thetaloss=L2_losses, 
                            infloss=inf_losses, surrloss=surr_losses,
                            grad_reinforce_list=grad_reinforce_list, 
                            grad_reparam_list=grad_reparam_list,
                            f_list=f_list, logpxz_list=logpxz_list,
                            logprob_cluster_list=logprob_cluster_list)


                # plot_posteriors(x)
                # plot_dist()
                show_surr_preds()
                

            # print (f)
            # print (surr_pred)

            #Understand surr preds
            # if step %5000==0:

            # if step ==0:
                
                # fasdf
                





    data_dict = {}

    data_dict['steps'] = steps_list
    data_dict['losses'] = L2_losses

    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    with open( exp_dir+"data_simplax.p", "wb" ) as f:
        pickle.dump(data_dict, f)
    print ('saved data')



























if marginal:

    needsoftmax_mixtureweight = torch.randn(n_components, requires_grad=True)

    #INTEGRATING OUT /Marginalize
    # optim = torch.optim.Adam([needsoftmax_mixtureweight], lr=.004)
    optim = torch.optim.Adam([needsoftmax_mixtureweight], lr=.0001)
    batch_size = 10
    n_steps = 100000
    L2_losses = []
    steps_list = []
    for step in range(n_steps):

        optim.zero_grad()

        # batch = []
        # for i in range(batch_size):
        #     batch.append(sample_true())
        loss = 0
        for i in range(batch_size):
            x = sample_true()
            logprob = logprob_givenmixtureeweights(x, needsoftmax_mixtureweight)
            loss += -logprob
        loss = loss / batch_size

        loss.backward()  
        optim.step()

        if step%500==0:
            print (step, loss.item(), L2_mixtureweights(true_mixture_weights,
                                            torch.softmax(needsoftmax_mixtureweight, dim=0).detach().numpy()))
            steps_list.append(step)
            L2_losses.append(L2_mixtureweights(true_mixture_weights,
                                            torch.softmax(needsoftmax_mixtureweight, dim=0).detach().numpy()))


    data_dict = {}
    data_dict['steps'] = steps_list
    data_dict['losses'] = L2_losses

    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    with open( exp_dir+"data_marginal.p", "wb" ) as f:
        pickle.dump(data_dict, f)
    print ('saved data')









if reinforce:

    needsoftmax_mixtureweight = torch.randn(n_components, requires_grad=True)

    #REINFORCE
    encoder = NN(input_size=1, output_size=n_components)
    # optim = torch.optim.Adam([needsoftmax_mixtureweight], lr=.004)
    # optim_net = torch.optim.Adam(net.parameters(), lr=.004)
    optim = torch.optim.Adam([needsoftmax_mixtureweight], lr=.0001)
    optim_net = torch.optim.Adam(encoder.parameters(), lr=.001)
    batch_size = 10
    n_steps = 100000
    L2_losses = []
    steps_list = []
    for step in range(n_steps):

        optim.zero_grad()

        loss = 0
        net_loss = 0
        for i in range(batch_size):
            x = sample_true()
            logits = encoder.net(x)
            # print (logits.shape)
            # print (torch.softmax(logits, dim=0))
            # fsfd
            cat = Categorical(probs= torch.softmax(logits, dim=0))
            cluster = cat.sample()
            logprob_cluster = cat.log_prob(cluster.detach())
            # print (logprob_cluster)
            pxz = logprob_undercomponent(x, component=cluster, needsoftmax_mixtureweight=needsoftmax_mixtureweight, cuda=False)
            f = pxz - logprob_cluster
            # print (f)
            # logprob = logprob_givenmixtureeweights(x, needsoftmax_mixtureweight)
            net_loss += -f.detach() * logprob_cluster
            loss += -f
        loss = loss / batch_size
        net_loss = net_loss / batch_size

        # print (loss, net_loss)

        loss.backward(retain_graph=True)  
        optim.step()

        optim_net.zero_grad()
        net_loss.backward()  
        optim_net.step()

        if step%500==0:
            print (step, L2_mixtureweights(true_mixture_weights,
                                        torch.softmax(needsoftmax_mixtureweight, dim=0).detach().numpy()))
            steps_list.append(step)
            L2_losses.append(L2_mixtureweights(true_mixture_weights,
                                            torch.softmax(needsoftmax_mixtureweight, dim=0).detach().numpy()))



    data_dict = {}
    data_dict['steps'] = steps_list
    data_dict['losses'] = L2_losses

    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    with open( exp_dir+"data_reinforce.p", "wb" ) as f:
        pickle.dump(data_dict, f)
    print ('saved data')

















if simplax or reinforce or marginal:


    needsoftmax_mixtureweight = needsoftmax_mixtureweight.cpu()




    #MAKE PLOT OF DISTRIBUTION
    rows = 1
    cols = 1
    fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

    col =0
    row = 0
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)




    xs = np.linspace(-9,205, 300)
    sum_ = np.zeros(len(xs))
    C = 20
    for c in range(C):
        m = Normal(torch.tensor([c*10.]).float(), torch.tensor([5.0]).float())
        # xs = torch.tensor(xs)
        # print (m.log_prob(lin))
        ys = []
        for x in xs:
            # print (m.log_prob(x))
            component_i = (torch.exp(m.log_prob(x) )* ((c+5.) / 290.)).numpy()
            ys.append(component_i)
        ys = np.reshape(np.array(ys), [-1])
        sum_ += ys
        ax.plot(xs, ys, label='', c='c')
    ax.plot(xs, sum_, label='')



    mixture_weights = torch.softmax(needsoftmax_mixtureweight, dim=0)
    xs = np.linspace(-9,205, 300)
    sum_ = np.zeros(len(xs))
    C = 20
    for c in range(C):
        m = Normal(torch.tensor([c*10.]).float(), torch.tensor([5.0]).float())
        # xs = torch.tensor(xs)
        # print (m.log_prob(lin))
        ys = []
        for x in xs:
            # print (m.log_prob(x))
            component_i = (torch.exp(m.log_prob(x) )* mixture_weights[c]).detach().numpy()
            ys.append(component_i)
        ys = np.reshape(np.array(ys), [-1])
        sum_ += ys
        ax.plot(xs, ys, label='', c='r')
    ax.plot(xs, sum_, label='')






    # #HISTOGRAM
    # xs = []
    # for i in range(10000):
    #     x = sample_true().item()
    #     xs.append(x)
    # ax.hist(xs, bins=200, density=True)



    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    if simplax:
        plt_path = exp_dir+'gmm_pdf_plot_simplax.png'
    elif reinforce:
        plt_path = exp_dir+'gmm_pdf_plot_reinforce.png'
    elif marginal:
        plt_path = exp_dir+'gmm_pdf_plot_marginal.png'

    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()







plot_distribution=0
if plot_distribution:



    rows = 1
    cols = 1
    fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

    col =0
    row = 0
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    clusters =[]
    logits= torch.randn(n_components)
    cat = RelaxedOneHotCategorical(probs= torch.softmax(logits, dim=0), temperature=torch.tensor([1.]))#.cuda())

    for i in range(1000):
        cluster_S = cat.rsample()
        cluster_H = H(cluster_S)
        clusters.append(cluster_H)

    ax.hist(clusters, density=True, bins=n_components)
    ax.plot(range(n_components),torch.softmax(logits, dim=0).numpy() )


    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = exp_dir+'gmm_plot_cat.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()



    fdsfasd

    # sum_ = 0
    # for i in range(20):
    #     sum_+= i+5
    # print (sum_)
    # fds 290


    # print (m.log_prob(2.))
    xs = np.linspace(-9,205, 300)

    sum_ = np.zeros(len(xs))

    C = 20
    for c in range(C):


        m = Normal(torch.tensor([c*10.]).float(), torch.tensor([5.0]).float())

        
        # xs = torch.tensor(xs)
        # print (m.log_prob(lin))

        ys = []
        for x in xs:
            # print (m.log_prob(x))
            component_i = (torch.exp(m.log_prob(x) )* ((c+5.) / 290.)).numpy()
            ys.append(component_i)

        ys = np.reshape(np.array(ys), [-1])

        # print (sum_.shape)
        # print (ys.shape)

        sum_ += ys

        # print (sum_)
        # fsdfa


        ax.plot(xs, ys, label='')


        # ax.grid(True, alpha=.3)
        # ax.set_title(r'$f=(b-.4)^2$', size=6, family='serif')
        # ax.tick_params(labelsize=6)
        # ax.set_ylabel(ylabel, size=6, family='serif')
        # ax.set_xlabel(xlabel, size=6, family='serif')
        # ax.legend(prop={'size':5}) #, loc=2)  #upper left


    ax.plot(xs, sum_, label='')




    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = exp_dir+'gmm_plot.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()


