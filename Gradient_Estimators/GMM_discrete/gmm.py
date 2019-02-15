

from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



import numpy as np

import torch

import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.normal import Normal

import pickle

# import sys, os
# # sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('./new_discrete_estimators'))

from NN import NN
from NN import NN2
from NN import NN3




#METHOD
marginal = 0
reinforce = 1
simplax = 0








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


def sample_true():
    cat = Categorical(probs= torch.tensor(true_mixture_weights))
    cluster = cat.sample()
    # print (cluster)
    # fsd
    norm = Normal(torch.tensor([cluster*10.]).float(), torch.tensor([5.0]).float())
    samp = norm.sample()
    # print (samp)
    return samp

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

def logprob_undercomponent(x, component, needsoftmax_mixtureweight, cuda=False):
    c= component
    mixture_weights = torch.softmax(needsoftmax_mixtureweight, dim=0)
    # probs_sum = 0# = []
    # for c in range(n_components):
    # m = Normal(torch.tensor([c*10.]).float().cuda(), torch.tensor([5.0]).float() )#.cuda())
    if not cuda:
        m = Normal(torch.tensor([c*10.]).float(), torch.tensor([5.0]).float() )#.cuda())
    else:
        m = Normal(torch.tensor([c*10.]).float().cuda(), torch.tensor([5.0]).float().cuda())
    # for x in xs:
    # component_i = torch.exp(m.log_prob(x))* mixture_weights[c] #.numpy()
    logprob = m.log_prob(x) + torch.log(mixture_weights[c])
    # probs.append(probs)
    # probs_sum+=component_i
    # logprob = torch.log(component_i)
    return logprob

def H(soft):

    return torch.argmax(soft)


# needsoftmax_mixtureweight = torch.tensor(np.random.randn(n_components), requires_grad=True).float() #.view(n_components,1)
# print (needsoftmax_mixtureweight)

#Compute log prob given mixture weights
# x = sample_true()
# print (logprob_givenmixtureeweights(x, needsoftmax_mixtureweight))
# fadfs









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

    save_dir = home+'/Documents/Grad_Estimators/GMM/'
    with open( save_dir+"data_marginal.p", "wb" ) as f:
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

    save_dir = home+'/Documents/Grad_Estimators/GMM/'
    with open( save_dir+"data_reinforce.p", "wb" ) as f:
        pickle.dump(data_dict, f)
    print ('saved data')












if simplax:

    #SIMPLAX
    needsoftmax_mixtureweight = torch.randn(n_components, requires_grad=True, device="cuda")#.cuda()
    encoder = NN(input_size=1, output_size=n_components).cuda()
    surrogate = NN3(input_size=1+n_components, output_size=1).cuda()
    # optim = torch.optim.Adam([needsoftmax_mixtureweight], lr=.00004)
    # optim_net = torch.optim.Adam(encoder.parameters(), lr=.0004)
    # optim_surr = torch.optim.Adam(surrogate.parameters(), lr=.004)
    optim = torch.optim.Adam([needsoftmax_mixtureweight], lr=.0001)
    optim_net = torch.optim.Adam(encoder.parameters(), lr=.001)
    optim_surr = torch.optim.Adam(surrogate.parameters(), lr=.005)
    batch_size = 10
    n_steps = 100000
    L2_losses = []
    steps_list =[]
    for step in range(n_steps):

        optim.zero_grad()

        loss = 0
        net_loss = 0
        surr_loss = 0
        for i in range(batch_size):
            x = sample_true().cuda().view(1,1)
            logits = encoder.net(x)
            # cat = Categorical(probs= torch.softmax(logits, dim=0))
            cat = RelaxedOneHotCategorical(probs= torch.softmax(logits, dim=0), temperature=torch.tensor([1.]).cuda())
            cluster_S = cat.rsample()
            cluster_H = H(cluster_S)
            # cluster_onehot = torch.zeros(n_components)
            # cluster_onehot[cluster_H] = 1.
            # print (cluster_onehot)
            # print (cluster_H)
            # print (cluster_S)
            # fdsa
            logprob_cluster = cat.log_prob(cluster_S.detach())
            if logprob_cluster != logprob_cluster:
                print ('nan')
            # print (logprob_cluster)
            pxz = logprob_undercomponent(x, component=cluster_H, needsoftmax_mixtureweight=needsoftmax_mixtureweight, cuda=True)
            f = pxz - logprob_cluster
            # print (f)
            # logprob = logprob_givenmixtureeweights(x, needsoftmax_mixtureweight)
            surr_pred = surrogate.net(torch.cat([cluster_S.view(1,n_components), x], dim=1))
            # print (surr_pred)
            # print (surr_pred.shape)
            net_loss += - ((f.detach()-surr_pred) * logprob_cluster + surr_pred)
            loss += -f
            surr_loss += torch.abs(f.detach()-surr_pred)
        loss = loss / batch_size
        net_loss = net_loss / batch_size
        surr_loss = surr_loss / batch_size

        # print (loss, net_loss)

        loss.backward(retain_graph=True)  
        optim.step()

        optim_net.zero_grad()
        net_loss.backward(retain_graph=True)
        optim_net.step()

        optim_surr.zero_grad()
        surr_loss.backward()
        optim_surr.step()

        if step%500==0:
            print (step, loss.cpu().data.numpy(), surr_loss.cpu().data.detach().numpy(), 
                            L2_mixtureweights(true_mixture_weights,torch.softmax(
                                        needsoftmax_mixtureweight, dim=0).cpu().data.detach().numpy()))
            print (torch.softmax(needsoftmax_mixtureweight, dim=0).cpu().data.detach().numpy())             
            print ()

            L2_losses.append(L2_mixtureweights(true_mixture_weights,torch.softmax(
                                        needsoftmax_mixtureweight, dim=0).cpu().data.detach().numpy()))
            steps_list.append(step)


    data_dict = {}

    data_dict['steps'] = steps_list
    data_dict['losses'] = L2_losses

    save_dir = home+'/Documents/Grad_Estimators/GMM/'
    with open( save_dir+"data_simplax.p", "wb" ) as f:
        pickle.dump(data_dict, f)
    print ('saved data')










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



save_dir = home+'/Documents/Grad_Estimators/GMM/'
if simplax:
    plt_path = save_dir+'gmm_pdf_plot_simplax.png'
elif reinforce:
    plt_path = save_dir+'gmm_pdf_plot_reinforce.png'
elif marginal:
    plt_path = save_dir+'gmm_pdf_plot_marginal.png'

plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()

fasdf






rows = 1
cols = 1
fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

col =0
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

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




save_dir = home+'/Documents/Grad_Estimators/GMM/'
plt_path = save_dir+'gmm_plot.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()


