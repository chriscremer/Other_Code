



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



    #Sample
    n_samps = 1
    samps = q_x.sample(n_samps)
    log_qx = q_x.log_prob(samps)
    log_px = p_x.log_prob(samps)

    #Compute KL
    log_qp = log_qx - log_px
    print (log_qp)

    #Compute grad
    log_qp_avg = torch.mean(log_qp)
    log_qp_avg.backward()
    print(q_x.mean.grad, q_x.logvar.grad)



    #Plot distribution and samples
    samps = samps.data.numpy()

    viz_range = [-10,10]
    numticks = 200

    fig, ax = plt.subplots(1, 1, facecolor='white')
    ax.axis('off')
    ax.set_yticks([])
    ax.set_xticks([])

    x, y = return_1d_distribution(distribution=p_x, xlimits=viz_range, numticks=numticks)
    ax.plot(x, y, linewidth=2, label=r'$p(z)$')

    x, y = return_1d_distribution(distribution=q_x, xlimits=viz_range, numticks=numticks)
    ax.plot(x, y, linewidth=2, label=r'$q(z)$')

    #Plot samples
    for i in range(len(samps)):
        ax.plot([samps[i],samps[i]], [0,.1], linewidth=2, label=r'$z_q$')



    plt.legend(fontsize=9, loc=2)
    plt.show()










    print ('Done.')

















