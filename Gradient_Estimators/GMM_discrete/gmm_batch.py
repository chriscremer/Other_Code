

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

    # print (trueposterior.shape)
    # print (probs.shape)
    # print (L2_mixtureweights(trueposterior.data.cpu().numpy(),probs.data.cpu().numpy()))
    return L2_mixtureweights(trueposterior.data.cpu().numpy(),probs.data.cpu().numpy())
    # fsdfa

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



def plot_curve(steps, thetaloss, infloss, surrloss):

    rows = 3
    cols = 1
    fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

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


    save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = save_dir+'gmm_curveplot.png'
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


    save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = save_dir+'gmm_surr.png'
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

    save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = save_dir+'gmm_plot_dist.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()
    


































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
    batch_size = 50
    n_steps = 100000
    L2_losses = []
    inf_losses = []
    surr_losses = []
    steps_list =[]
    for step in range(n_steps):


        # loss = 0
        # net_loss = 0
        # surr_loss = 0
        # for i in range(batch_size):

        x = sample_true(batch_size).cuda() #.view(1,1)
        # print (x)
        logits = encoder.net(x)
        # print (logits.shape)
        probs = torch.softmax(logits, dim=1)
        # print (probs.shape)
        # cat = Categorical(probs= torch.softmax(logits, dim=0))
        cat = RelaxedOneHotCategorical(probs=probs, temperature=torch.tensor([1.]).cuda())
        cluster_S = cat.rsample()
        cluster_H = H(cluster_S)
        # print (cluster_S)
        # print (cluster_H)
        # cluster_onehot = torch.zeros(n_components)
        # cluster_onehot[cluster_H] = 1.
        # print (cluster_onehot)
        # print (cluster_H.shape) #[B]
        # print (cluster_S.shape) #[B,20]
        logprob_cluster = cat.log_prob(cluster_S.detach()).view(batch_size,1)
        # print (logprob_cluster.shape) #[B]
        check_nan(logprob_cluster)

        logpxz = logprob_undercomponent(x, component=cluster_H, needsoftmax_mixtureweight=needsoftmax_mixtureweight, cuda=True)
        # print (logpxz.shape, logprob_cluster.shape)
        # fafd
        f = logpxz - logprob_cluster
        # print (f)
        # logprob = logprob_givenmixtureeweights(x, needsoftmax_mixtureweight)
        # print (cluster_S.shape)
        # fsdfad
        surr_input = torch.cat([cluster_S, x], dim=1) #[B,21]
        surr_pred = surrogate.net(surr_input)
        # print (surr_pred)
        # print (cluster_S)
        # print (cluster_S.view(1,n_components).shape)
        # print(x.shape)
        # print (torch.cat([cluster_S.view(1,n_components), x], dim=1).shape)
        # fsdfasfd
        net_loss = - torch.mean((f.detach()-surr_pred.detach()) * logprob_cluster + surr_pred)
        loss = - torch.mean(f)
        # print (f.shape)
        # print (surr_pred.shape)
        # ffds
        surr_loss = torch.mean(torch.abs(f.detach()-surr_pred))


        # loss = loss / batch_size
        # net_loss = net_loss / batch_size
        # surr_loss = surr_loss / batch_size

        # print (loss, net_loss)

        # optim.zero_grad()
        # loss.backward(retain_graph=True)  
        # optim.step()

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

            if len(surr_losses) > 3  and step %5000==0:
                plot_curve(steps=steps_list,  thetaloss=L2_losses, infloss=inf_losses, surrloss=surr_losses)

            # print (f)
            # print (surr_pred)

            #Understand surr preds
            if step %5000==0:

                show_surr_preds()


            if step ==0:
                plot_dist()
                # fasdf
                





    data_dict = {}

    data_dict['steps'] = steps_list
    data_dict['losses'] = L2_losses

    save_dir = home+'/Documents/Grad_Estimators/GMM/'
    with open( save_dir+"data_simplax.p", "wb" ) as f:
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


    save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = save_dir+'gmm_plot_cat.png'
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




    save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = save_dir+'gmm_plot.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()


