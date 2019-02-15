

from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



import numpy as np

import torch

from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.normal import Normal


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def my_logprob_v4(y, theta):
    return np.log((theta - theta**2)/ (-2*theta*y + theta + y)**2)

# are these logistic or gumbel??
def my_sample_nosigmoid(theta, temp=1.):
    u = np.random.rand()
    y = np.log(u) - np.log(1-u) + np.log(theta) - np.log(1-theta)
    y = y/ temp
    return y
def my_logprob_nosigmoid(theta, y, temp=1.):
    logit = np.log(theta / (1-theta))
    diff = logit - (y * temp)
    logprob = np.log(temp) + diff - 2 * np.log(np.exp(diff)+1) 
    return logprob

samps = []
z_samps = []
sig_samps = []
theta = .99
temp = 5. #doesnt work for my sigmoid logprob since I didnt derive with temp, needs to be 1
for i in range(400):
    y = my_sample_nosigmoid(theta, temp=temp)
    z_samps.append(y)
    sig_samps.append(sigmoid(y))
    samps.append(float(y>0))
print (np.mean(samps))
print (np.mean(z_samps))




rows = 1
cols = 3
fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

col =0
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.hist(z_samps, density=True, bins=30)


probs = []
xs = np.linspace(-6,6, 40)
for x in xs:
    prob = np.exp(my_logprob_nosigmoid(theta, x, temp=temp))
    probs.append(prob)

ax.plot(xs, probs)

ax.set_title('No sigmoid')


col =1
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(sig_samps, density=True, bins=30)

probs = []
xs = np.linspace(.0001,.9999, 40)
for x in xs:
    prob = np.exp(my_logprob_v4(x, theta))
    probs.append(prob)

ax.plot(xs, probs)

ax.set_title('With sigmoid')

save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'my_relaxedbernoulli_plot.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()


fasdfa













# sum_ = 0
# for i in range(20):
#     sum_+= i+5
# print (sum_)
# fds 290


# print (m.log_prob(2.))
xs = np.linspace(.001,.999, 30)

# dist = RelaxedBernoulli(temperature=torch.Tensor([1.]), logits=torch.tensor([0.]))
dist = RelaxedBernoulli(temperature=torch.Tensor([.2]), probs=torch.tensor([0.3]))

ys = []
samps = []
for x in xs:
    # samp = dist.sample()
    # samps.append(samp.data.numpy()[0])
    # print (samp)
    # print (torch.exp(dist.log_prob(samp)))
    # print ()
    # print (m.log_prob(x))

    prob = torch.exp(dist.log_prob(torch.tensor([x]))).numpy()[0]
    # print (x)
    # print (prob)
    # component_i = ().numpy()[0]
    ys.append(prob)

samps = []
b_samps = []
for i in range(1000):
    samp = dist.sample()
    samps.append(samp.data.numpy()[0]) 
    b_samps.append(samp.data.numpy()[0] > .5)
ax.hist(samps, density=True, bins=30)

ax.plot(xs, ys)
ax.set_title('.2')
print (np.mean(samps))
print (np.mean(b_samps))
print ()





col =1
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

dist = RelaxedBernoulli(temperature=torch.Tensor([1.]), probs=torch.tensor([0.3]))

ys = []
samps = []
for x in xs:
    prob = torch.exp(dist.log_prob(torch.tensor([x]))).numpy()[0]
    ys.append(prob)

samps = []
b_samps = []
for i in range(1000):
    samp = dist.sample()
    samps.append(samp.data.numpy()[0]) 
    b_samps.append(samp.data.numpy()[0] > .5)
ax.hist(samps, density=True, bins=30)

ax.plot(xs, ys)
ax.set_title('1.')
print (np.mean(samps))
print (np.mean(b_samps))
print ()




col =2
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

dist = RelaxedBernoulli(temperature=torch.Tensor([5.]), probs=torch.tensor([0.3]))

ys = []
samps = []
for x in xs:
    prob = torch.exp(dist.log_prob(torch.tensor([x]))).numpy()[0]
    ys.append(prob)

samps = []
b_samps = []
for i in range(1000):
    samp = dist.sample()
    samps.append(samp.data.numpy()[0]) 
    b_samps.append(samp.data.numpy()[0] > .5)
ax.hist(samps, density=True, bins=30)

ax.plot(xs, ys)
ax.set_title('5.')
print (np.mean(samps))
print (np.mean(b_samps))







# # sum_ = np.zeros(len(xs))

# C = 20
# for c in range(C):


#     m = Normal(torch.tensor([c*10.]).float(), torch.tensor([5.0]).float())

    
#     # xs = torch.tensor(xs)
#     # print (m.log_prob(lin))

#     ys = []
#     for x in xs:
#         # print (m.log_prob(x))
#         component_i = (torch.exp(m.log_prob(x) )* ((c+5.) / 290.)).numpy()
#         ys.append(component_i)

#     ys = np.reshape(np.array(ys), [-1])

#     # print (sum_.shape)
#     # print (ys.shape)

#     sum_ += ys

#     # print (sum_)
#     # fsdfa


#     ax.plot(xs, ys, label='')


#     # ax.grid(True, alpha=.3)
#     # ax.set_title(r'$f=(b-.4)^2$', size=6, family='serif')
#     # ax.tick_params(labelsize=6)
#     # ax.set_ylabel(ylabel, size=6, family='serif')
#     # ax.set_xlabel(xlabel, size=6, family='serif')
#     # ax.legend(prop={'size':5}) #, loc=2)  #upper left


# ax.plot(xs, sum_, label='')




save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'relaxedbernoulli_plot.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()


