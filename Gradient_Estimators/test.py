

from os.path import expanduser
home = expanduser("~")


# import sys, os
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('./VAE'))

import numpy as np

import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from NN import NN
from NN_forrelax import NN as NN2


def H(x):
    if x > .5:
        return 1
    else:
        return 0

def Hpy(x):
    if x > .5:
        return torch.tensor([1])
    else:
        return torch.tensor([0])



n= 10000
theta = .3
bern_param = torch.tensor([theta], requires_grad=True)
dist = RelaxedBernoulli(torch.Tensor([1.]), bern_param)
val=.499
f = lambda x: (x-val)**2

print ('Value:', val)
print ('n:', n)
print ('theta:', theta)
print()




print ('SimpLAX NN')
net = NN()

train_ = 1
if train_:
    net.train(func=f, dist=dist, save_dir=home+'/Downloads/tmmpp/')
    print ('Done training\n')
else:
    net.load_params_v3(save_dir=home+'/Downloads/tmmpp/', step=17285, name='') #.499
    # net.load_params_v3(save_dir=home+'/Downloads/tmmpp/', step=102710, name='') #.4
print()



print ('Relax NN')
net_relax = NN2()

train_ = 1
if train_:
    net_relax.train(func=f, dist=dist, save_dir=home+'/Downloads/tmmpp/', bern_param=bern_param)
    print ('Done training\n')
# fada
else:
    # net_relax.load_params_v3(save_dir=home+'/Downloads/tmmpp/', step=30551, name='') #.499
    net_relax.load_params_v3(save_dir=home+'/Downloads/tmmpp/', step=125423, name='') #.4
print()










dist_bern = Bernoulli(bern_param)
dist = RelaxedBernoulli(torch.Tensor([1.]), bern_param)

samps = []
grads = []
eta = 1.
difs = []
for i in range(n):
    z = dist.rsample()
    # print ('z',z)

    b = Hpy(z)
    #p(z|b)
    if b ==0:
        v= np.random.rand()*(1-bern_param)
    else:
        v = np.random.rand()*bern_param+(1-bern_param)
    z_tilde = torch.log(bern_param/(1-bern_param)) + torch.log(v/(1-v))
    z_tilde = torch.sigmoid(z_tilde)
    # print ('z\'',z_tilde)
    # print ('exp z\'',torch.sigmoid(z_tilde))
    # print ()

    logprob = dist_bern.log_prob(b)
    # logprob = dist_bern.log_prob(samp)
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
    

    pred = net_relax.net(z_tilde)
    # print (samp.detach().numpy(), H(samp), f(H(samp)), pred.detach().numpy())
    logprobgrad_2 = torch.autograd.grad(outputs=pred, inputs=(bern_param), retain_graph=True)[0]
    # print (logprobgrad_2)

    pred3 = net_relax.net(z)
    logprobgrad_3 = torch.autograd.grad(outputs=pred3, inputs=(bern_param), retain_graph=True)[0]
    # print (logprobgrad_3)

    grad = (f(b) - eta*pred) * logprobgrad - eta*logprobgrad_2 + logprobgrad_3

    # g1 = f(b)*logprobgrad
    # g2 = pred*logprobgrad
    # g3 = logprobgrad_2
    # g4 = logprobgrad_3

    # print (g1)
    # print (g2)
    # print (g3)
    # print (g4)

    # fada

    # print (grad)
    # fdfsf

    dif = (f(b) - eta*pred).detach().numpy()

    difs.append((dif)**2)
    samps.append(b)
    grads.append(grad.detach().numpy())

print ('Grad Estimator: RELAX')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ('Avg dif', np.mean(difs))

print ()


# fsadfds















samps = []
grads = []
eta = 1.
difs = []
for i in range(n):
    samp = dist.rsample()

    logprob = dist.log_prob(samp.detach())
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
    

    pred = net.net(samp)
    # print (samp.detach().numpy(), H(samp), f(H(samp)), pred.detach().numpy())

    logprobgrad_2 = torch.autograd.grad(outputs=pred, inputs=(bern_param), retain_graph=True)[0]

    Hsamp = H(samp)

    grad = (f(Hsamp) - eta*pred) * logprobgrad + eta*logprobgrad_2

    dif = (f(Hsamp) - eta*pred).detach().numpy()

    difs.append((dif)**2)
    samps.append(Hsamp)
    grads.append(grad.detach().numpy())

print ('Grad Estimator: SimpLAX')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ('Avg dif', np.mean(difs))

print ()
















dist = Bernoulli(bern_param)


samps = []
grads = []
for i in range(n):
    samp = dist.sample()

    logprob = dist.log_prob(samp)
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

    samps.append(samp.numpy())
    grads.append(f(samp.numpy()) * logprobgrad.numpy())

print ('Grad Estimator: REINFORCE')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print()








dist = RelaxedBernoulli(torch.Tensor([.5]), bern_param)


samps = []
grads = []
for i in range(n):
    samp = dist.sample()

    logprob = dist.log_prob(samp)
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

    samp = samp.numpy()
    samp = H(samp)

    samps.append(samp)
    grads.append(f(samp) * logprobgrad.numpy())

print ('Grad Estimator: REINFORCE H(z), temp=.5')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ()






dist = RelaxedBernoulli(torch.Tensor([1.]), bern_param)


samps = []
grads = []
for i in range(n):
    samp = dist.sample()

    logprob = dist.log_prob(samp)
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

    samp = samp.numpy()
    samp = H(samp)

    samps.append(samp)
    grads.append(f(samp) * logprobgrad.numpy())

print ('Grad Estimator: REINFORCE H(z), temp=1')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ()





dist = RelaxedBernoulli(torch.Tensor([2.]), bern_param)


samps = []
grads = []
for i in range(n):
    samp = dist.sample()

    logprob = dist.log_prob(samp)
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

    samp = samp.numpy()
    samp = H(samp)

    samps.append(samp)
    grads.append(f(samp) * logprobgrad.numpy())

print ('Grad Estimator: REINFORCE H(z), temp=2')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ()







































fsdfsad




# m = RelaxedBernoulli(torch.Tensor([1.]), torch.Tensor([0.1, 0.2, 0.3, 0.99]))
m = RelaxedBernoulli(torch.Tensor([1.]), torch.Tensor([0.3]))
print (m.sample())
print (m.sample())

print (m.log_prob(m.sample()))



bern_param = torch.tensor([.7], requires_grad=True)
m = RelaxedBernoulli(torch.Tensor([1.]), bern_param)
samp = m.sample()
# print (samp)
logprob = m.log_prob(samp)

logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

print (logprobgrad)
fdadsf



theta = .3
for i in range(10):
    u = np.random.rand()
    z = np.log(theta) - np.log(1-theta) + np.log(u) - np.log(1-u)
    gumb = lambda x:  np.exp(-x + theta) * np.exp(- np.exp(-x + theta))
    print (gumb(z))


fsdfas

# for i in range(10):

#   print (np.random.rand())


theta = .3
n= 100000

# E[b]
bs= []
# E[H(z)]
Hzs= []
for i in range(n):

    u = np.random.rand()

    if u > theta:
        b = 0
    else:
        b = 1

    bs.append(b)


    z = np.log(theta) - np.log(1-theta) + np.log(u) - np.log(1-u)

    if z >= 0: 
        Hz = 1
    else:
        Hz = 0

    Hzs.append(Hz)

print (np.mean(bs))
print (np.mean(Hzs))


#Get p(b) and p(z) 

bern = lambda x: theta**x * (1-theta)**(1-x)

print (bern(1))
print (bern(0))



#Get grad of p(b) and p(z) 






















