



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
from torch.distributions.categorical import Categorical

from NN import NN
from NN_forrelax import NN as NN2

from NN2 import NN3


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

n= 300 #10000
print ('n:', n)
print ()

thetas = np.linspace(.01,.99, 10)
# thetas = np.linspace(.001,.999, 20)
# thetas = np.linspace(.001,.03, 20)
# thetas = np.linspace(.97,.999, 12)


# ### TRAIN RELAX SURROGATE
# print ('training relax surrogate')
# B=1
# C=2
# n_components=C
# probs = torch.ones(B,C)
# bern_param = torch.tensor([.03], requires_grad=True)
# bern_param = bern_param.view(B,1)
# aa = 1 - bern_param
# probs = torch.cat([aa, bern_param], dim=1)
# cat = Categorical(probs= probs)
# surrogate = NN3(input_size=n_components, output_size=1, n_residual_blocks=2)#.cuda()
# optim_surr = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
# for i in range(1000):
#     #Sample z
#     u = torch.rand(B,C)
#     gumbels = -torch.log(-torch.log(u))
#     z = torch.log(probs) + gumbels

#     b = torch.argmax(z, dim=1)
#     logprob = cat.log_prob(b)

#     #Sample z_tilde
#     u_b = torch.rand(B,1)
#     z_tilde_b = -torch.log(-torch.log(u_b))
#     u = torch.rand(B,C)
#     z_tilde = -torch.log((- torch.log(u) / probs) - torch.log(u))
#     z_tilde[:,b] = z_tilde_b

#     surr_pred_z_tilde = surrogate.net(z_tilde)

#     surr_dif = torch.mean(torch.abs(f(b)-surr_pred_z_tilde))

#     optim_surr.zero_grad()
#     surr_dif.backward()
#     optim_surr.step()

#     if i %50==0:
#         print (i, surr_dif.data.numpy())










reinforce_grad_means = []
reinforce_grad_stds = []
pz_grad_means = []
pz_grad_stds = []
hlax_grad_means = []
hlax_grad_stds = []
relax_grad_means = []
relax_grad_stds = []
reinforce_cat_grad_means = []
reinforce_cat_grad_stds = []
for theta in thetas:
    

    print ()
    print ('theta:', theta)
    # theta = .01 #.99 #.1 #95 #.3 #.9 #.05 #.3
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
        logprob = dist.log_prob(samp)

        if logprob != logprob:
            print (samp, logprob)
        
        logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

        samp = samp.numpy()
        samp = H(samp)

        samps.append(samp)
        grads.append(  (f(samp)) * logprobgrad.numpy())
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















    dist = LogitRelaxedBernoulli(torch.Tensor([1.]), bern_param)
    dist_bernoulli = Bernoulli(bern_param)

    # samps = []
    grads = []
    # logprobgrads = []
    for i in range(n):
        z = dist.sample()
        b = Hpy(z)

        logprob_z = dist.log_prob(z)
        logprob_b = dist_bernoulli.log_prob(b)

        # if logprob != logprob:
        #     print (samp, logprob)
        #     fadsads
        
        logprobgrad_z = torch.autograd.grad(outputs=logprob_z, inputs=(bern_param), retain_graph=True)[0]
        logprobgrad_b = torch.autograd.grad(outputs=logprob_b, inputs=(bern_param), retain_graph=True)[0]

        grad = .5 * f(b) * (logprobgrad_z + logprobgrad_b)

        # samp = samp.numpy()
        # samp = H(samp)

        # samps.append(samp)
        grads.append(grad.numpy())
        # logprobgrads.append(logprobgrad.numpy())

    print ('Grad Estimator: HLAX')
    print ('Grad mean', np.mean(grads))
    print ('Grad std', np.std(grads))
    print ()

    hlax_grad_means.append(np.mean(grads))
    hlax_grad_stds.append(np.std(grads))

















    # dist = LogitRelaxedBernoulli(torch.Tensor([1.]), bern_param)
    # dist_bernoulli = Bernoulli(bern_param)
    C= 2
    n_components = C
    B=1
    probs = torch.ones(B,C)
    bern_param = bern_param.view(B,1)
    aa = 1 - bern_param
    probs = torch.cat([aa, bern_param], dim=1)
    # probs[:,0] = probs[:,0]* bern_param
    # probs[:,1] = probs[:,1] - bern_param

    cat = Categorical(probs= probs)
    surrogate = NN3(input_size=n_components, output_size=1, n_residual_blocks=2)#.cuda()


    ### TRAIN RELAX SURROGATE
    print ('training relax surrogate')
    # B=1
    # C=2
    # n_components=C
    # probs = torch.ones(B,C)
    # bern_param = torch.tensor([.03], requires_grad=True)
    # bern_param = bern_param.view(B,1)
    # aa = 1 - bern_param
    # probs = torch.cat([aa, bern_param], dim=1)
    # cat = Categorical(probs= probs)
    # surrogate = NN3(input_size=n_components, output_size=1, n_residual_blocks=2)#.cuda()
    optim_surr = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
    for i in range(500):
        #Sample z
        u = torch.rand(B,C)
        gumbels = -torch.log(-torch.log(u))
        z = torch.log(probs) + gumbels

        b = torch.argmax(z, dim=1)
        logprob = cat.log_prob(b)

        #Sample z_tilde
        u_b = torch.rand(B,1)
        z_tilde_b = -torch.log(-torch.log(u_b))
        u = torch.rand(B,C)
        z_tilde = -torch.log((- torch.log(u) / probs) - torch.log(u))
        z_tilde[:,b] = z_tilde_b

        surr_pred_z_tilde = surrogate.net(z_tilde)

        surr_dif = torch.mean(torch.abs(f(b)-surr_pred_z_tilde))

        optim_surr.zero_grad()
        surr_dif.backward()
        optim_surr.step()

        if i %100==0:
            print (i, surr_dif.data.numpy())



    grads = []
    for i in range(n):
        # z = dist.sample()
        #Sample z
        u = torch.rand(B,C)
        gumbels = -torch.log(-torch.log(u))
        z = torch.log(probs) + gumbels

        b = torch.argmax(z, dim=1)
        logprob = cat.log_prob(b)
        # one_hot = torch.zeros(n_cats)
        # one_hot[b] = 1.

        #Sample z_tilde
        u_b = torch.rand(B,1)
        z_tilde_b = -torch.log(-torch.log(u_b))
        u = torch.rand(B,C)
        z_tilde = -torch.log((- torch.log(u) / probs) - torch.log(u))
        z_tilde[:,b] = z_tilde_b

        surr_pred_z = surrogate.net(z)
        surr_pred_z_tilde = surrogate.net(z_tilde)

        logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
        grad = (f(b)-surr_pred_z_tilde.detach()) * logprobgrad + surr_pred_z - surr_pred_z_tilde

        grads.append(grad[0][0].data.numpy())

    print ('Grad Estimator: RELAX')
    print ('Grad mean', np.mean(grads))
    print ('Grad std', np.std(grads))
    print ()

    relax_grad_means.append(np.mean(grads))
    relax_grad_stds.append(np.std(grads))









    # dist = LogitRelaxedBernoulli(torch.Tensor([1.]), bern_param)
    # dist_bernoulli = Bernoulli(bern_param)
    C= 2
    n_components = C
    B=1
    probs = torch.ones(B,C)
    bern_param = bern_param.view(B,1)
    aa = 1 - bern_param
    probs = torch.cat([aa, bern_param], dim=1)

    cat = Categorical(probs= probs)

    grads = []
    for i in range(n):
        b = cat.sample()
        logprob = cat.log_prob(b.detach())
        # b_ = torch.argmax(z, dim=1)

        logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
        grad = f(b) * logprobgrad

        grads.append(grad[0][0].data.numpy())

    print ('Grad Estimator: Reinfoce categorical')
    print ('Grad mean', np.mean(grads))
    print ('Grad std', np.std(grads))
    print ()

    reinforce_cat_grad_means.append(np.mean(grads))
    reinforce_cat_grad_stds.append(np.std(grads))















print (thetas)
print (reinforce_grad_means)
print (pz_grad_means)


reinforce_grad_means = np.array(reinforce_grad_means)
reinforce_grad_stds = np.array(reinforce_grad_stds)
pz_grad_means = np.array(pz_grad_means)
pz_grad_stds = np.array(pz_grad_stds)
hlax_grad_means = np.array(hlax_grad_means)
hlax_grad_stds = np.array(hlax_grad_stds)
relax_grad_means = np.array(relax_grad_means)
relax_grad_stds = np.array(relax_grad_stds)
reinforce_cat_grad_means = np.array(reinforce_cat_grad_means)
reinforce_cat_grad_stds = np.array(reinforce_cat_grad_stds)


rows = 5
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

col =1
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(thetas, reinforce_grad_means, label='reinforce')
plt.gca().fill_between(thetas.flat, 
        reinforce_grad_means-reinforce_grad_stds, reinforce_grad_means+reinforce_grad_stds, color="#dddddd")
ax.set_ylim(bottom=-70, top=70)
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
row = 1
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(thetas, pz_grad_means, label='reinforce p(z)')
plt.gca().fill_between(thetas.flat, 
        pz_grad_means-pz_grad_stds, pz_grad_means+pz_grad_stds, color="#dddddd")

ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')
ax.set_ylim(bottom=-70, top=70)
ax.legend()






col =0
row = 2
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(thetas, hlax_grad_means, label='hlax')
plt.gca().fill_between(thetas.flat, 
        hlax_grad_means-hlax_grad_stds, hlax_grad_means+hlax_grad_stds, color="#dddddd")
ax.set_ylim(bottom=-5, top=5)
ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')
ax.legend()

col =1
row = 2
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(thetas, hlax_grad_means, label='hlax')
plt.gca().fill_between(thetas.flat, 
        hlax_grad_means-hlax_grad_stds, hlax_grad_means+hlax_grad_stds, color="#dddddd")
ax.set_ylim(bottom=-70, top=70)
ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')
ax.legend()




col =0
row = 3
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(thetas, relax_grad_means, label='relax')
plt.gca().fill_between(thetas.flat, 
        relax_grad_means-relax_grad_stds, relax_grad_means+relax_grad_stds, color="#dddddd")
ax.set_ylim(bottom=-5, top=5)
ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')
ax.legend()

col =1
row = 3
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(thetas, relax_grad_means, label='relax')
plt.gca().fill_between(thetas.flat, 
        relax_grad_means-relax_grad_stds, relax_grad_means+relax_grad_stds, color="#dddddd")
ax.set_ylim(bottom=-70, top=70)
ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')
ax.legend()





col =0
row = 4
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(thetas, reinforce_cat_grad_means, label='reinforce cat')
plt.gca().fill_between(thetas.flat, 
        reinforce_cat_grad_means-reinforce_cat_grad_stds, reinforce_cat_grad_means+reinforce_cat_grad_stds, color="#dddddd")
ax.set_ylim(bottom=-5, top=5)
ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')
ax.legend()

col =1
row = 4
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(thetas, reinforce_cat_grad_means, label='reinforce cat')
plt.gca().fill_between(thetas.flat, 
        reinforce_cat_grad_means-reinforce_cat_grad_stds, reinforce_cat_grad_means+reinforce_cat_grad_stds, color="#dddddd")
ax.set_ylim(bottom=-70, top=70)
ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')
ax.legend()




save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'is_grad_dependent_4.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()
























