

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
        return torch.tensor([1]).float()
    else:
        return torch.tensor([0]).float()


def prob_to_logit(prob):
    return torch.log(prob) - torch.log(1-prob)

def logit_to_prob(logit):
    return torch.sigmoid(logit)


n= 10000 #10000
theta = .01 #.99 #.1 #95 #.3 #.9 #.05 #.3
bern_param = torch.tensor([theta], requires_grad=True)
val= .4
f = lambda x: (x-val)**2
# f = lambda x: x
dif = f(1)-f(0)

print ()
print ('Value:', val)
print (f(0))
print (f(1))
print ('so dif in f(1) and f(0) is', dif)
print()
print ('n:', n)
print ('theta:', theta)
print ()


temp = 1.
def my_logprob(y):
    logit = np.log(theta / (1-theta))
    logdet =  - np.log((1/y) + (1/(1 - y)))
    x = np.log(y) - np.log(1-y)
    diff = logit - (x * temp)
    return np.log(temp) + diff - 2 * np.log(np.exp(diff)+1)  - logdet


def my_logprob_v2(y):
    return np.log(theta / (1-theta)) + np.log(1/y**2) - 2*np.log( (theta/(1.-theta)) * ((1-y)/y) + 1 )

def my_logprob_v3(y):
    return np.log(theta * (1-theta)) - 2*np.log( -2*theta*y + theta + y)

def my_logprob_v4(y):
    return np.log((theta - theta**2)/ (-2*theta*y + theta + y)**2)
def my_logprob_v4_grad(y):
    return (y-theta) / ((theta-1)*theta*((2*theta-1)*y-theta))

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


# samps = []
# theta = .3
# for i in range(100):
#     y = my_sample_nosigmoid(theta)
#     samps.append(y>0)
# print (np.mean(samps))
# fdasdf






# print (-(y.reciprocal() + (1 - y).reciprocal()).log())




# y=.5
# print (- np.log( (1/y) + (1/(1-y))))
# fsfsa


# dist = RelaxedBernoulli(torch.Tensor([.2]), bern_param)
# samp = dist.sample()
# logprob = dist.log_prob(samp)
# print (samp.data.numpy(), Hpy(samp).data.numpy(), logprob.data.numpy())
# # fsdfa



# # TEST OF MY FORMULAS FOR THE DENSITY
# dist = RelaxedBernoulli(temperature=torch.Tensor([1.]), probs=bern_param)
# samp = dist.sample()
# logprob = dist.log_prob(samp)
# samp_np = samp.data.numpy()[0]
# print ('samp', samp.data.numpy()) 
# print ('logprob', logprob.data.numpy())
# print ('prob', np.exp(logprob.data.numpy()))
# print ('my prob', np.exp(my_logprob(samp_np)))
# print ('my prob v2', np.exp(my_logprob_v2(samp_np)))
# print ('my prob v3', np.exp(my_logprob_v3(samp_np)))
# print ('my prob v4', np.exp(my_logprob_v4(samp_np)))
# fsdfda





# print ('SimpLAX NN')
# net = NN()
# dist = RelaxedBernoulli(torch.Tensor([.2]), bern_param)
# train_ = 1
# if train_:
#     # net.train(func=f, dist=dist, save_dir=home+'/Downloads/tmmpp/')
#     net.train(func=f, dist=dist, save_dir=home+'/Documents/Grad_Estimators/new/',
#                       early_stop=4)
#     print ('Done training\n')
# else:
#     net.load_params_v3(save_dir=home+'/Downloads/tmmpp/', step=17285, name='') #.499
#     # net.load_params_v3(save_dir=home+'/Downloads/tmmpp/', step=102710, name='') #.4
# print()



# print ('Relax NN')
# net_relax = NN2()

# train_ = 0
# if train_:
#     # net_relax.train(func=f, dist=dist, save_dir=home+'/Downloads/tmmpp/', bern_param=bern_param,
#     #                     early_stop=5)
#     net_relax.train(func=f, dist=dist, save_dir=home+'/Documents/Grad_Estimators/new/', bern_param=bern_param,
#                         early_stop=5)
#     print ('Done training\n')
# # fada
# else:
#     # net_relax.load_params_v3(save_dir=home+'/Downloads/tmmpp/', step=30551, name='') #.499
#     net_relax.load_params_v3(save_dir=home+'/Documents/Grad_Estimators/new/', step=1607, name='') #.4
# print()








dist = Bernoulli(bern_param)
samps = []
grads = []
logprobgrads = []
for i in range(n):
    samp = dist.sample()

    logprob = dist.log_prob(samp.detach())
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
    # print (samp.data.numpy(), logprob.data.numpy(), logprobgrad.data.numpy())
    # fsdfa

    samps.append(samp.numpy())
    grads.append( (f(samp.numpy()) - 0.) * logprobgrad.numpy())
    logprobgrads.append(logprobgrad.numpy())


# print (grads[:10])

print ('Grad Estimator: REINFORCE')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ('Avg logprobgrad', np.mean(logprobgrads))
print ('Std logprobgrad', np.std(logprobgrads))
print()






# n=1000
# print (n)
dist = RelaxedBernoulli(torch.Tensor([5.]), bern_param)


samps = []
grads = []
logprobgrads = []
for i in range(n):
    samp = dist.sample()

    samp = torch.clamp(samp, min=.0000001, max=.99999999999999)
    # print (smp)

    logprob = dist.log_prob(samp)

    if logprob != logprob:
        print (samp, logprob)
    
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

    # Hpy(samp).data.numpy(), 
    

    
    # print (samp.data.numpy(), logprob.data.numpy(), logprobgrad.data.numpy())
    # npsamp = samp.data.numpy()
    # print (npsamp, my_logprob_v4(npsamp), my_logprob_v4_grad(npsamp))
    # print ()

    samp = samp.numpy()
    samp = H(samp)

    samps.append(samp)
    grads.append(  (f(samp)- 0.) * logprobgrad.numpy())
    logprobgrads.append(logprobgrad.numpy())

print ('Grad Estimator: REINFORCE H(z), temp=.2')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ('Avg logprobgrad', np.mean(logprobgrads))
print ('Std logprobgrad', np.std(logprobgrads))
print ()

fdfasdfda





























# n=1000
print (n)
dist = RelaxedBernoulli(torch.Tensor([.2]), bern_param)


samps = []
grads = []
logprobgrads = []
for i in range(n):
    samp = dist.sample()

    samp = torch.clamp(samp, min=.0000001, max=.999999)

    logprob = dist.log_prob(samp)
    

    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

    # print (samp.data.numpy(), Hpy(samp).data.numpy(), logprob.data.numpy(), logprobgrad.data.numpy())
    # fsdfa

    samp = samp.numpy()
    samp = H(samp)

    samps.append(samp)
    grads.append(f(samp) * logprobgrad.numpy())
    logprobgrads.append(logprobgrad.numpy())

print ('Grad Estimator: REINFORCE H(z), temp=.2')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ('Avg logprobgrad', np.mean(logprobgrads))
print ('Std logprobgrad', np.std(logprobgrads))
print ()

# fsfasdfd

n=10000
print (n)
dist = RelaxedBernoulli(torch.Tensor([.2]), bern_param)


samps = []
grads = []
logprobgrads = []
for i in range(n):
    samp = dist.sample()

    samp = torch.clamp(samp, min=.0000001, max=.999999)

    logprob = dist.log_prob(samp)
    

    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

    # print (samp.data.numpy(), Hpy(samp).data.numpy(), logprob.data.numpy(), logprobgrad.data.numpy())
    # fsdfa

    samp = samp.numpy()
    samp = H(samp)

    samps.append(samp)
    grads.append(f(samp) * logprobgrad.numpy())
    logprobgrads.append(logprobgrad.numpy())

print ('Grad Estimator: REINFORCE H(z), temp=.2')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ('Avg logprobgrad', np.mean(logprobgrads))
print ('Std logprobgrad', np.std(logprobgrads))
print ()




n=100000
print (n)
dist = RelaxedBernoulli(torch.Tensor([.2]), bern_param)


samps = []
grads = []
logprobgrads = []
for i in range(n):
    samp = dist.sample()

    samp = torch.clamp(samp, min=.0000001, max=.999999)

    logprob = dist.log_prob(samp)
    

    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

    # print (samp.data.numpy(), Hpy(samp).data.numpy(), logprob.data.numpy(), logprobgrad.data.numpy())
    # fsdfa

    samp = samp.numpy()
    samp = H(samp)

    samps.append(samp)
    grads.append(f(samp) * logprobgrad.numpy())
    logprobgrads.append(logprobgrad.numpy())

print ('Grad Estimator: REINFORCE H(z), temp=.2')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ('Avg logprobgrad', np.mean(logprobgrads))
print ('Std logprobgrad', np.std(logprobgrads))
print ()











n=10000
dist = RelaxedBernoulli(torch.Tensor([.2]), bern_param)

samps = []
grads = []
eta = 1.
difs = []
for i in range(n):
    samp = dist.rsample()

    samp = torch.clamp(samp, min=.0000001, max=.999999)

    logprob = dist.log_prob(samp.detach())
    # print(logprob)


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









fsfa











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





dist = RelaxedBernoulli(torch.Tensor([10.]), bern_param)


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

print ('Grad Estimator: REINFORCE H(z), temp=10')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ()







# REINFORCE but sampling p(z)
dist_relaxedbern = RelaxedBernoulli(torch.Tensor([1.]), bern_param)
dist_bern = Bernoulli(bern_param)

samps = []
grads = []
for i in range(n):
    samp = dist_relaxedbern.sample()
    Hsamp = Hpy(samp)

    logprob = dist_bern.log_prob(Hsamp)
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

    samps.append(Hsamp.numpy())
    grads.append(f(Hsamp.numpy()) * logprobgrad.numpy())

print ('Grad Estimator: REINFORCE but sampling p(z)')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print()
















fasfda
fasda









#RELAX v2 - parameterized by logits. Closer to code used for optimization
logits = torch.log(bern_param / (1.-bern_param))
dist_bern = Bernoulli(logits=logits)
dist = RelaxedBernoulli(torch.Tensor([1.]), logits=logits)

samps = []
grads = []
eta = 1.
difs = []
for i in range(n):
    z = dist.rsample()
    # print (z.shape)
    # fsf
    b = Hpy(z)
    # print (b.shape)
    # fasdf

    #p(z|b)
    theta = bern_param #logit_to_prob(bern_param)
    v = torch.rand(z.shape[0]) 
    v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
    z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
    # z_tilde = logits.detach() + torch.log(v_prime) - torch.log1p(-v_prime) #detachign biases it..I used detach initally 
    z_tilde = torch.sigmoid(z_tilde)


    # #p(z|b)
    # if b ==0:
    #     v= np.random.rand()*(1-bern_param)
    # else:
    #     v = np.random.rand()*bern_param+(1-bern_param)
    # z_tilde = torch.log(bern_param/(1-bern_param)) + torch.log(v/(1-v))
    # z_tilde = torch.sigmoid(z_tilde)

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

    dif = (f(b) - eta*pred).detach().numpy()

    difs.append((dif)**2)
    samps.append(b)
    grads.append(grad.detach().numpy())

print ('Grad Estimator: RELAX v2')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ('Avg dif', np.mean(difs))

print ()

fasdf




















#RELAX
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
    # #p(z|b)
    # if b ==0:
    #     v= np.random.rand()*(1-bern_param)
    # else:
    #     v = np.random.rand()*bern_param+(1-bern_param)
    v = (1-b)*(np.random.rand()*(1-bern_param)) + b*(np.random.rand()*bern_param+(1-bern_param))
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


fdsfas
















#IM ingoring DLAX because I dont know how to train c, since its not a control variate 
# dist_bern = Bernoulli(bern_param)
# dist_relaxedbern = RelaxedBernoulli(torch.Tensor([1.]), bern_param)

# samps = []
# grads = []
# eta = 1.
# difs = []
# for i in range(n):

#     z = dist.rsample()
#     b = Hpy(z)

#     logprob = dist_relaxedbern.log_prob(z.detach())
#     logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

#     logprob = dist_bern.log_prob(b.detach())
#     logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

#     pred = net.net(z)
#     # print (samp.detach().numpy(), H(samp), f(H(samp)), pred.detach().numpy())
#     logprobgrad_2 = torch.autograd.grad(outputs=pred, inputs=(bern_param), retain_graph=True)[0]



#     grad = (f(Hsamp) - eta*pred) * logprobgrad + eta*logprobgrad_2

#     dif = (f(Hsamp) - eta*pred).detach().numpy()

#     difs.append((dif)**2)
#     samps.append(Hsamp)
#     grads.append(grad.detach().numpy())

# print ('Grad Estimator: DLAX')
# print ('Avg samp', np.mean(samps))
# print ('Grad mean', np.mean(grads))
# print ('Grad std', np.std(grads))
# print ('Avg dif', np.mean(difs))

# print ()
















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

    logprob = dist_bern.log_prob(b)
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

    # print (z_tilde)
    # print (z)
    # fdsf
    
    pred = f(z_tilde).float()
    logprobgrad_2 = torch.autograd.grad(outputs=pred, inputs=(bern_param), retain_graph=True)[0]

    pred3 = f(z).float()
    logprobgrad_3 = torch.autograd.grad(outputs=pred3, inputs=(bern_param), retain_graph=True)[0]

    grad = (f(b) - eta*pred) * logprobgrad - eta*logprobgrad_2 + logprobgrad_3

    dif = (f(b) - eta*pred).detach().numpy()

    difs.append((dif)**2)
    samps.append(b)
    grads.append(grad.detach().numpy())

print ('Grad Estimator: REBAR')
print ('Avg samp', np.mean(samps))
print ('Grad mean', np.mean(grads))
print ('Grad std', np.std(grads))
print ('Avg dif', np.mean(difs))

print ()


fsadfds












































dist = RelaxedBernoulli(torch.Tensor([1.]), bern_param)

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






















