





import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import math

#Plot 2 distributions
#Compute log p/q
#Get grad of params of q





class Gaus_1D(nn.Module):
    def __init__(self, mean, logvar, seed=1):
        super(Gaus_1D, self).__init__()

        torch.manual_seed(seed)

        self.x_size = 1

        self.mean = Variable(torch.FloatTensor(mean), requires_grad = True) #[1]
        self.logvar = Variable(torch.FloatTensor(logvar), requires_grad = True) #[1]


    def log_prob(self, x):
        '''
        x: [B,X]
        mean,logvar: [X]
        output: [B]
        '''

        assert len(x.size()) == 2
        assert x.size()[1] == self.mean.size()[0]

        D = x.size()[1]
        term1 = Variable(D * torch.log(torch.FloatTensor([2.*math.pi]))) #[1]
        aaa = -.5 * (term1 + self.logvar.sum(0) + ((x - self.mean).pow(2)/torch.exp(self.logvar)).sum(1))
        return aaa



    def sample(self, k):
        '''
        k: # of samples
        output: [k,X]
        '''

        eps = Variable(torch.FloatTensor(k, self.x_size).normal_()) #.type(self.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*self.logvar)) + self.mean  #[P,B,Z]
        return z

        





def return_1d_distribution(distribution, xlimits, numticks):

    x = np.expand_dims(np.linspace(*xlimits, num=numticks),1) #[B,1]

    x_pytorch = Variable(torch.FloatTensor(x))
    log_probs_pytorch = distribution.log_prob(x_pytorch)

    log_probs = log_probs_pytorch.data.numpy()

    px = np.exp(log_probs)

    return x, px








if __name__ == "__main__":

    # Define 2 distribution p and q
    #PLot them

    p_mean = [-2.5]
    p_logvar = [1.]

    q_mean = [4.5]
    q_logvar = [.5]

    p_x = Gaus_1D(p_mean, p_logvar)
    q_x = Gaus_1D(q_mean, q_logvar)


    # Get distribution of gradients
    n_grads = 500
    n_samps = 1
    mean_grads = []
    logvar_grads = []
    for i in range(n_grads):
        #Sample
        samps = q_x.sample(n_samps)
        log_qx = q_x.log_prob(samps)
        log_px = p_x.log_prob(samps)

        #Compute KL
        log_qp = log_qx - log_px
        # print (log_qp)

        #Compute grad
        log_qp_avg = torch.mean(log_qp)
        log_qp_avg.backward()
        # print(q_x.mean.grad, q_x.logvar.grad)
        mean_grads.append(q_x.mean.grad.data.numpy()[0])
        logvar_grads.append(q_x.logvar.grad.data.numpy()[0])

        q_x.mean.grad.data.zero_()
        q_x.logvar.grad.data.zero_()

    # print (mean_grads)


    # Get distribution of gradients, differnet number of MC samples
    n_grads = 500
    n_samps = 10
    mean_grads_10MC = []
    logvar_grads_10MC = []
    for i in range(n_grads):
        #Sample
        samps = q_x.sample(n_samps)
        log_qx = q_x.log_prob(samps)
        log_px = p_x.log_prob(samps)

        #Compute KL
        log_qp = log_qx - log_px
        # print (log_qp)

        #Compute grad
        log_qp_avg = torch.mean(log_qp)
        log_qp_avg.backward()
        # print(q_x.mean.grad, q_x.logvar.grad)
        mean_grads_10MC.append(q_x.mean.grad.data.numpy()[0])
        logvar_grads_10MC.append(q_x.logvar.grad.data.numpy()[0])

        q_x.mean.grad.data.zero_()
        q_x.logvar.grad.data.zero_()


    # # Get distribution of gradients, SF samples
    # n_grads = 500
    # n_samps = 10
    # mean_grads_10MC_SF = []
    # logvar_grads_10MC_SF = []
    # for i in range(n_grads):
    #     #Sample
    #     samps = q_x.sample(n_samps)
    #     log_qx = q_x.log_prob(samps)
    #     log_px = p_x.log_prob(samps)

    #     #Compute KL
    #     log_qp = log_qx - log_px
    #     # print (log_qp)

    #     #Compute grad
    #     log_qp_avg = torch.mean(log_qp)
    #     # log_qp_avg.backward()

    #     log_q_avg = torch.mean(log_qx)
    #     log_q_avg.backward()

    #     # print(q_x.mean.grad, q_x.logvar.grad)
    #     mean_grads_10MC_SF.append(q_x.mean.grad.data.numpy()[0]*log_qp_avg)
    #     logvar_grads_10MC_SF.append(q_x.logvar.grad.data.numpy()[0]*log_qp_avg)

    #     q_x.mean.grad.data.zero_()
    #     q_x.logvar.grad.data.zero_()



    #Plot distributions
    rows = 3
    cols = 2
    fig = plt.figure(figsize=(4+cols,5+rows), facecolor='white')
    viz_range = [-10,10]
    numticks = 200

    # # samps = samps.data.numpy()
    # #Plot samples
    # for i in range(len(samps)):
    #     ax.plot([samps[i],samps[i]], [0,.1], linewidth=2, label=r'$z_q$')



    #Plot q and p
    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False, colspan=2)
    ax.axis('off')
    ax.set_yticks([])
    ax.set_xticks([])

    x, y = return_1d_distribution(distribution=p_x, xlimits=viz_range, numticks=numticks)
    ax.plot(x, y, linewidth=2, label=r'$p(z)$')

    x, y = return_1d_distribution(distribution=q_x, xlimits=viz_range, numticks=numticks)
    ax.plot(x, y, linewidth=2, label=r'$q(z)$')

    ax.legend(fontsize=9, loc=2)




    bins = 50
    # weights = np.empty_like(mean_grads)
    # weights.fill(bins / 7 / len(mean_grads))

    #Plot mean grads histogram
    ax = plt.subplot2grid((rows,cols), (1,0), frameon=False, colspan=1)
    ax.hist(mean_grads, bins=bins, normed=True, range=[-1,6])
    ax.set_xlabel(r'$\nabla \mu$')
    ax.set_ylim(top=3.)
    ax.set_ylabel('1 MC Sample', family='serif')

    #Plot logvar grads histogram
    ax = plt.subplot2grid((rows,cols), (1,1), frameon=False, colspan=1)
    ax.hist(logvar_grads, bins=bins, normed=True, range=[-4,6])
    ax.set_xlabel(r'$\nabla \log \sigma$')
    ax.set_ylim(top=1.)


    #Plot mean grads histogram
    ax = plt.subplot2grid((rows,cols), (2,0), frameon=False, colspan=1)
    ax.hist(mean_grads_10MC, bins=bins, normed=True, range=[-1,6])
    ax.set_xlabel(r'$\nabla \mu$')
    ax.set_ylim(top=3.)
    ax.set_ylabel('10 MC Sample', family='serif')

    #Plot logvar grads histogram
    ax = plt.subplot2grid((rows,cols), (2,1), frameon=False, colspan=1)
    ax.hist(logvar_grads_10MC, bins=bins, normed=True, range=[-4,6])
    ax.set_xlabel(r'$\nabla \log \sigma$')
    ax.set_ylim(top=1.)




    # plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=.5)
    # plt.tight_layout()


    # plt.show()


    # name_file = home+'/Documents/tmp/plot.pdf'
    name_file = home+'/Downloads/grads_estimators.pdf'
    plt.savefig(name_file)
    print ('Saved fig', name_file)


    print ('Done.')

















