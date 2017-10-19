


import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# from utils import lognormal2 as lognormal
# from utils import lognormal333

# from utils import log_bernoulli

import time

import pickle







# print (torch.exp(torch.from_numpy(np.array([-1000.]))))
# ffdas


elbo = Variable(torch.FloatTensor(100, 1).normal_())*10. - 100.
elbo[99] = 10.

print (elbo)

# elbo = logpx-logq #[P,B]
vae = torch.mean(elbo)

max_ = torch.max(elbo, 0)[0] #[B]

# print (torch.max(elbo, 0))
print (max_)



elbo_ = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
print (elbo_)
iwae = torch.mean(elbo_)

print (vae)
print (iwae)

faf


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 0)
    return to_scalar(idx)



# vec = Variable(torch.FloatTensor(100, 1).normal_())*10. - 100.
# vec = Variable(torch.FloatTensor(100,1).normal_())*10. - 100.

vec = elbo


max_score = vec[0, argmax(vec)]
max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
aaa =  max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

print (aaa)










