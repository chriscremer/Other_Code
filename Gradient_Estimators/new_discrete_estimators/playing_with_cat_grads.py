

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
# from torch.distributions.bernoulli import Bernoulli
# from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.categorical import Categorical

from NN2 import NN3



def to_print1(x):
    return torch.mean(x).data.cpu().numpy()
def to_print2(x):
    return x.data.cpu().numpy()


B=1
C=3
logits =  torch.ones((B,C), requires_grad=True)
# probs = torch.ones((B,C), requires_grad=True) / C
# rewards = torch.tensor([-1., 0., 1., 4.])
rewards = torch.tensor([-1., 1., 4.])
def f(ind):
    return rewards[ind]

# dist = Categorical(probs=probs)


# def my_logprob(x, probs):
#     # prob = probs[x]
#     # print (probs.shape)
#     # print (x.shape)
#     prob = torch.gather(input=probs, dim=1, index=x)
#     return torch.log(prob)

# mylogprob = my_logprob(samp, probs)
# mylogprobgrad = torch.autograd.grad(outputs=mylogprob, inputs=(probs), retain_graph=True)[0]



print ('rewards', rewards)
# print ('probs', probs)
print ('logits', logits)







#Plot p(z)
zs = []
z_tildes=[]
n_samps = 1000
count=0
while count < n_samps:
    u = torch.rand(B,C).clamp(1e-8, 1.-1e-8)
    gumbels = -torch.log(-torch.log(u))
    z = logits + gumbels
    b = torch.argmax(z, dim=1)
    if b==0:
        zs.append(z)
        count+=1

        u_b = torch.gather(input=u, dim=1, index=b.view(B,1))
        z_tilde_b = -torch.log(-torch.log(u_b))
        z_tilde = -torch.log((- torch.log(u) / torch.softmax(logits,dim=1)) - torch.log(u_b))
        z_tilde.scatter_(dim=1, index=b.view(B,1), src=z_tilde_b)
        z_tildes.append(z_tilde)


zs = torch.stack(zs).view(n_samps,C)
z_tildes = torch.stack(z_tildes).view(n_samps,C)
# print (zs.shape)
# fsdfasd

n_bins = 80

rows = C
cols = 2
fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

x_lim_left=-5

col =0
row =0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(zs[:,0]), bins=n_bins, density=True)
ax.set_xlim(x_lim_left, 10)
ax.set_ylim(0., .6)
ax.set_ylabel('x0')

col =0
row =1
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(zs[:,1]), bins=n_bins, density=True)
ax.set_xlim(x_lim_left, 10)
ax.set_ylim(0., .6)
ax.set_ylabel('x1')


col =0
row =2
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(zs[:,2]), bins=n_bins, density=True)
ax.set_xlim(x_lim_left, 10)
ax.set_ylim(0., .6)
ax.set_ylabel('x2')


col =1
row =0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(z_tildes[:,0]), bins=n_bins, density=True)
ax.set_xlim(x_lim_left, 10)
ax.set_ylim(0., .6)


col =1
row =1
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(z_tildes[:,1]), bins=n_bins, density=True)
ax.set_xlim(x_lim_left, 10)
ax.set_ylim(0., .6)



col =1
row =2
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(z_tildes[:,2]), bins=n_bins, density=True)
ax.set_xlim(x_lim_left, 10)
ax.set_ylim(0., .6)


save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'view_pz.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()



fsfasd















def sample_relax(logits):
    cat = Categorical(logits=logits)

    u = torch.rand(B,C).cuda()
    u = u.clamp(1e-8, 1.-1e-8)
    gumbels = -torch.log(-torch.log(u))
    z = logits + gumbels

    b = torch.argmax(z, dim=1)
    logprob = cat.log_prob(b).view(B,1)

    u_b = torch.gather(input=u, dim=1, index=b.view(B,1))
    z_tilde_b = -torch.log(-torch.log(u_b))
    
    z_tilde = -torch.log((- torch.log(u) / torch.softmax(logits, dim=1)) - torch.log(u_b))
    z_tilde.scatter_(dim=1, index=b.view(B,1), src=z_tilde_b)
    return z, b, logprob, z_tilde



def sample_reinforce_given_class(logits, samp):
    dist = Categorical(logits=logits)
    logprob = dist.log_prob(samp)
    return logprob


def sample_relax_given_class(logits, samp):

    cat = Categorical(logits=logits)

    u = torch.rand(B,C).clamp(1e-8, 1.-1e-8)
    gumbels = -torch.log(-torch.log(u))
    z = logits + gumbels

    b = samp #torch.argmax(z, dim=1)
    logprob = cat.log_prob(b).view(B,1)


    u_b = torch.gather(input=u, dim=1, index=b.view(B,1))
    z_tilde_b = -torch.log(-torch.log(u_b))
    
    z_tilde = -torch.log((- torch.log(u) / torch.softmax(logits, dim=1)) - torch.log(u_b))
    z_tilde.scatter_(dim=1, index=b.view(B,1), src=z_tilde_b)


    z = z_tilde

    u_b = torch.gather(input=u, dim=1, index=b.view(B,1))
    z_tilde_b = -torch.log(-torch.log(u_b))
    
    u = torch.rand(B,C).clamp(1e-8, 1.-1e-8)
    z_tilde = -torch.log((- torch.log(u) / torch.softmax(logits, dim=1)) - torch.log(u_b))
    z_tilde.scatter_(dim=1, index=b.view(B,1), src=z_tilde_b)

    return z, z_tilde, logprob


#integrate out z
def sample_relax_given_class_k(logits, samp, k):

    cat = Categorical(logits=logits)
    b = samp #torch.argmax(z, dim=1)
    logprob = cat.log_prob(b).view(B,1)

    zs = []
    z_tildes = []
    for i in range(k):

        u = torch.rand(B,C).clamp(1e-8, 1.-1e-8)
        gumbels = -torch.log(-torch.log(u))
        z = logits + gumbels

        u_b = torch.gather(input=u, dim=1, index=b.view(B,1))
        z_tilde_b = -torch.log(-torch.log(u_b))
        
        z_tilde = -torch.log((- torch.log(u) / torch.softmax(logits, dim=1)) - torch.log(u_b))
        z_tilde.scatter_(dim=1, index=b.view(B,1), src=z_tilde_b)

        z = z_tilde

        u_b = torch.gather(input=u, dim=1, index=b.view(B,1))
        z_tilde_b = -torch.log(-torch.log(u_b))
        
        u = torch.rand(B,C).clamp(1e-8, 1.-1e-8)
        z_tilde = -torch.log((- torch.log(u) / torch.softmax(logits, dim=1)) - torch.log(u_b))
        z_tilde.scatter_(dim=1, index=b.view(B,1), src=z_tilde_b)

        zs.append(z)
        z_tildes.append(z_tilde)

    zs= torch.stack(zs)
    z_tildes= torch.stack(z_tildes)
    
    z = torch.mean(zs, dim=0)
    z_tilde = torch.mean(z_tildes, dim=0)

    return z, z_tilde, logprob



surrogate = NN3(input_size=C, output_size=1, n_residual_blocks=2)

train_ = 1
n_steps = 300
if train_:
    optim = torch.optim.Adam(surrogate.parameters(), lr=1e-4, weight_decay=1e-7)
    #Train surrogate
    for i in range(n_steps):
        for c in range(C):
            samp = torch.tensor([c]).view(B,1)
            z, z_tilde, logprob = sample_relax_given_class(logits, samp)

            cz_tilde = surrogate.net(z_tilde)
            reward = f(samp)

            loss = (cz_tilde - reward)**2 
            optim.zero_grad()
            loss.backward()
            optim.step()

        if i % (n_steps/10)==0:
            print(i, loss)


else:
    #Load
    net.load_params_v3(save_dir=home+'/Downloads/tmmpp/', step=17285, name='') 







#REINFORCE
print ('REINFORCE')
grads = []
for c in range(C):
    # samp = dist.sample()
    samp = torch.tensor([c]).view(B,1)

    logprob = sample_reinforce_given_class(logits, samp)
    # z, z_tilde, logprob/ = sample_relax_given_class(logits, samp)

    reward = f(samp) #-1.
    gradlogprob = torch.autograd.grad(outputs=logprob, inputs=(logits), retain_graph=True)[0]

    print()
    print ('samp', samp)
    print ('logprob', logprob)
    print ('grad logprob', gradlogprob)
    print ('f', reward)
    print ('f * grad logprob', reward*gradlogprob)
    grads.append(reward*gradlogprob)
    
print ()
grads = torch.stack(grads).view(C,C)
# print (grads.shape)
grad_mean_reinforce = torch.mean(grads,dim=0)
grad_std_reinforce = torch.std(grads,dim=0)

print ('REINFORCE')
print ('mean:', grad_mean_reinforce)
print ('std:', grad_std_reinforce)
print ()
print ()









#RELAX
print ('RELAX')
grads = []
for c in range(C):

    samp = torch.tensor([c]).view(B,1)
    # z, z_tilde, logprob = sample_relax_given_class(logits, samp)
    z, z_tilde, logprob = sample_relax_given_class_k(logits, samp, k=50)

    cz = surrogate.net(z)
    cz_tilde = surrogate.net(z_tilde)

    reward = f(samp) #-1.
    gradlogprob = torch.autograd.grad(outputs=logprob, inputs=(logits), retain_graph=True)[0]
    gradcz = torch.autograd.grad(outputs=cz, inputs=(logits), retain_graph=True)[0]
    gradcz_tilde = torch.autograd.grad(outputs=cz_tilde, inputs=(logits), retain_graph=True)[0]
    grad = (reward-cz_tilde).detach() *gradlogprob + gradcz - gradcz_tilde

    print()
    print ('samp', samp)
    print ('f', reward)
    print ('cz_tilde', cz_tilde)
    print ('logprob', logprob)
    print ('grad logprob', gradlogprob)
    print ('grad logprob * dif', gradlogprob*(reward-cz_tilde).detach())
    print ('gradcz', gradcz)
    print ('gradcz_tilde', gradcz_tilde)
    print ('grad', grad)

    grads.append(grad)
    
print ()
grads = torch.stack(grads).view(C,C)
grad_mean = torch.mean(grads,dim=0)
grad_std = torch.std(grads,dim=0)


print ('REINFORCE')
print ('mean:', grad_mean_reinforce)
print ('std:', grad_std_reinforce)
print ()
print ('RELAX')
print ('mean:', grad_mean)
print ('std:', grad_std)
print ()












































