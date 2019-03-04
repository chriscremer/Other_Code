



from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import sys, os
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('./VAE'))

import numpy as np

import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical




from NN import NN
from NN_forrelax import NN as NN2


# def H(x):
#     if x > .5:
#         return 1
#     else:
#         return 0

# def Hpy(x):
#     if x > .5:
#         return torch.tensor([1]).float()
#     else:
#         return torch.tensor([0]).float()


def H(x):
    if x > 0.:
        return 1
    else:
        return 0

def Hpy(x):
    if x > 0.:
        return torch.tensor([1]).float()
    else:
        return torch.tensor([0]).float()



# def H(soft):
#     return torch.argmax(soft, dim=0)

def prob_to_logit(prob):
    return torch.log(prob) - torch.log(1-prob)

def logit_to_prob(logit):
    return torch.sigmoid(logit)



# val= .4
# f = lambda x: (x-val)**2
def f(x):
    if x==0:
        return 0.
    else:
        return 1.
# f = lambda x: x
dif = f(1)-f(0)

print ()
# print ('Value:', val)
print (f(0))
print (f(1))
print ('so dif in f(1) and f(0) is', dif)
print()



temp = 1.
# def my_logprob(y):
#     logit = np.log(theta / (1-theta))
#     logdet =  - np.log((1/y) + (1/(1 - y)))
#     x = np.log(y) - np.log(1-y)
#     diff = logit - (x * temp)
#     return np.log(temp) + diff - 2 * np.log(np.exp(diff)+1)  - logdet


# def my_logprob_v2(y):
#     return np.log(theta / (1-theta)) + np.log(1/y**2) - 2*np.log( (theta/(1.-theta)) * ((1-y)/y) + 1 )

# def my_logprob_v3(y):
#     return np.log(theta * (1-theta)) - 2*np.log( -2*theta*y + theta + y)

# def my_logprob_v4(y):
#     return np.log((theta - theta**2)/ (-2*theta*y + theta + y)**2)
    
# def my_logprob_v4_grad(y):
#     return (y-theta) / ((theta-1)*theta*((2*theta-1)*y-theta))

# def my_sample_nosigmoid(theta, temp=1.):
#     u = np.random.rand()
#     y = np.log(u) - np.log(1-u) + np.log(theta) - np.log(1-theta)
#     y = y/ temp
#     return y
# def my_logprob_nosigmoid(theta, y, temp=1.):
#     logit = np.log(theta / (1-theta))
#     diff = logit - (y * temp)
#     logprob = np.log(temp) + diff - 2 * np.log(np.exp(diff)+1) 
#     return logprob

n= 1000 #10000
print ('n:', n)
print ()

thetas = np.linspace(.001,.999, 20)
# thetas = np.linspace(.001,.03, 20)
# thetas = np.linspace(.97,.999, 12)

reinforce_grad_means = []
reinforce_grad_stds = []
pz_grad_means = []
pz_grad_stds = []
for theta in thetas:
    

#     print ()
    print ('theta:', theta)
#     # theta = .01 #.99 #.1 #95 #.3 #.9 #.05 #.3
    bern_param = torch.tensor([theta], requires_grad=True)


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
    # print ('Avg samp', np.mean(samps))
    print ('Grad mean', np.mean(grads))
    print ('Grad std', np.std(grads))
    # print ('Avg logprobgrad', np.mean(logprobgrads))
    # print ('Std logprobgrad', np.std(logprobgrads))
    print()

    reinforce_grad_means.append(np.mean(grads))
    reinforce_grad_stds.append(np.std(grads))












    # n=1000
    # print (n)
    # dist = RelaxedBernoulli(torch.Tensor([1.]), bern_param)
    dist = LogitRelaxedBernoulli(torch.Tensor([1.]), bern_param)

    

    samps = []
    grads = []
    logprobgrads = []
    for i in range(n):
        samp = dist.sample()

        # samp = torch.clamp(samp, min=.0000001, max=.99999999999999)
        # print (smp)
        # print (samp)
        # fsd

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

    print ('Grad Estimator: REINFORCE H(z)')
    # print ('Avg samp', np.mean(samps))
    print ('Grad mean', np.mean(grads))
    print ('Grad std', np.std(grads))
    # print ('Avg logprobgrad', np.mean(logprobgrads))
    # print ('Std logprobgrad', np.std(logprobgrads))
    print ()

    pz_grad_means.append(np.mean(grads))
    pz_grad_stds.append(np.std(grads))






















    # # n=1000
    # # print (n)
    # # dist = RelaxedBernoulli(torch.Tensor([1.]), bern_param)
    # # dist = LogitRelaxedBernoulli(torch.Tensor([1.]), bern_param)

    # # needsoftmax_mixtureweight = torch.tensor([theta,1.-theta], requires_grad=True)
    # # probs = torch.softmax(needsoftmax_mixtureweight, dim=0).float()

    # bern_param = torch.tensor([theta], requires_grad=True)
    # probs = (torch.tensor([0.,1.]) - bern_param*torch.tensor([0.,1.]))  + bern_param* torch.tensor([1.,0.])
    # # print (probs)
    # # fdsf
    # dist = ExpRelaxedCategorical(torch.Tensor([1.]), probs=probs)

    # # fdsafs
    

    # samps = []
    # grads = []
    # logprobgrads = []
    # for i in range(n):
    #     samp = dist.sample()

    #     # samp = torch.clamp(samp, min=.0000001, max=.99999999999999)
    #     # print (smp)
    #     print (samp)
    #     # fsd

    #     logprob = dist.log_prob(samp)

    #     if logprob != logprob:
    #         print (samp, logprob)
        
    #     # print (probs[1])
    #     logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
    #     # logprobgrad = logprobgrad[1]

    #     # Hpy(samp).data.numpy(), 
    
    #     # print (samp.data.numpy(), logprob.data.numpy(), logprobgrad.data.numpy())
    #     # npsamp = samp.data.numpy()
    #     # print (npsamp, my_logprob_v4(npsamp), my_logprob_v4_grad(npsamp))
    #     # print ()

    #     # samp = samp.numpy()
    #     samp = H(samp)

    #     # print (samp, logprobgrad)
    #     # fsaf

    #     samps.append(samp)
    #     grads.append(  (f(samp)) * logprobgrad.numpy())
    #     logprobgrads.append(logprobgrad.numpy())

    # print ('Grad Estimator: REINFORCE H(z)')
    # # print ('Avg samp', np.mean(samps))
    # print ('Grad mean', np.mean(grads))
    # print ('Grad std', np.std(grads))
    # # print ('Avg logprobgrad', np.mean(logprobgrads))
    # # print ('Std logprobgrad', np.std(logprobgrads))
    # print ()

    # pz_grad_means.append(np.mean(grads))
    # pz_grad_stds.append(np.std(grads))


print (thetas)
print (reinforce_grad_means)
print (pz_grad_means)


reinforce_grad_means = np.array(reinforce_grad_means)
pz_grad_means = np.array(pz_grad_means)
reinforce_grad_stds = np.array(reinforce_grad_stds)
pz_grad_stds = np.array(pz_grad_stds)


rows = 2
cols = 2
fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

col =0
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(thetas, reinforce_grad_means, label='reinforce')
plt.gca().fill_between(thetas.flat, 
        reinforce_grad_means-reinforce_grad_stds, reinforce_grad_means+reinforce_grad_stds, color="#dddddd")
ax.set_ylim(bottom=-5, top=5)
ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')
ax.legend()


col =0
row = 1
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(thetas, pz_grad_means, label='reinforce p(z)')
plt.gca().fill_between(thetas.flat, 
        pz_grad_means-pz_grad_stds, pz_grad_means+pz_grad_stds, color="#dddddd")

ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')
ax.set_ylim(bottom=-5, top=5)
ax.legend()



col =1
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(thetas, reinforce_grad_means, label='reinforce')
plt.gca().fill_between(thetas.flat, 
        reinforce_grad_means-reinforce_grad_stds, reinforce_grad_means+reinforce_grad_stds, color="#dddddd")
ax.set_ylim(bottom=-70, top=70)
ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')
ax.legend()


col =1
row = 1
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(thetas, pz_grad_means, label='reinforce p(z)')
plt.gca().fill_between(thetas.flat, 
        pz_grad_means-pz_grad_stds, pz_grad_means+pz_grad_stds, color="#dddddd")

ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')
ax.set_ylim(bottom=-70, top=70)
ax.legend()











save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'is_grad_dependent_2.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()
























