













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


seed=1
torch.manual_seed(seed)




def to_print1(x):
    return torch.mean(x).data.cpu().numpy()
def to_print2(x):
    return x.data.cpu().numpy()


B=1
C=3
N = 5000

# theta = .5
# bern_param = torch.tensor([theta], requires_grad=True).view(B,1)
# aa = 1 - bern_param
# probs = torch.cat([aa, bern_param], dim=1)
# logits = torch.log(probs)

# vs

# logits =  torch.ones((B,C), requires_grad=True) # this is what I had before, not sure how the B even worked * -0.6931

# logits =  torch.ones((1,C), requires_grad=True) #* -0.6931
# logits =  torch.zeros((1,C), requires_grad=True) #* -0.6931
probs = torch.softmax(torch.ones((1,C)), dim=1)
logits = torch.log(probs)
logits = torch.tensor(logits, requires_grad=True)


# probs = torch.ones((B,C), requires_grad=True) / C
# rewards = torch.tensor([-1., 0., 1., 4.])
rewards = torch.tensor([-1., 1., 2.]) #*.1 #* 100.
# rewards = torch.tensor([-1., 2.])

true = np.array([-.5478, .1122, .4422])
# true = np.array([0,0])

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
print ('probs', probs)
print ('logits', logits)













#REINFORCE
print ('REINFORCE')

# def sample_reinforce_given_class(logits, samp):    
#     return logprob

grads = []
for i in range (N):

    dist = Categorical(logits=logits)
    samp = dist.sample()
    logprob = dist.log_prob(samp)
    reward = f(samp) 
    gradlogprob = torch.autograd.grad(outputs=logprob, inputs=(logits), retain_graph=True)[0]
    grads.append(reward*gradlogprob)
    
print ()
grads = torch.stack(grads).view(N,C)
# print (grads.shape)
grad_mean_reinforce = torch.mean(grads,dim=0)
grad_std_reinforce = torch.std(grads,dim=0)

print ('REINFORCE')
print ('mean:', grad_mean_reinforce)
print ('std:', grad_std_reinforce)
print ()
# print ('True')
# print ('[-.5478, .1122, .4422]')
# print ('dif:', np.abs(grad_mean_reinforce.numpy() -  true))
# print ()









#RELAX
print ('RELAX')

def sample_relax(logits): #, k=1):
    

    # u = torch.rand(B,C).clamp(1e-8, 1.-1e-8) #.cuda()
    u = torch.rand(B,C).clamp(1e-12, 1.-1e-12) #.cuda()
    gumbels = -torch.log(-torch.log(u))
    z = logits + gumbels
    b = torch.argmax(z, dim=1)

    cat = Categorical(logits=logits)
    logprob = cat.log_prob(b).view(B,1)

    v_k = torch.rand(B,1).clamp(1e-12, 1.-1e-12)
    z_tilde_b = -torch.log(-torch.log(v_k))
    
    # # v = torch.rand(B,C) #.clamp(1e-12, 1.-1e-12) #.cuda()
    # v_k = torch.gather(input=u, dim=1, index=b.view(B,1))
    # # z_tilde_b = -torch.log(-torch.log(v_k))
    # z_tilde_b = torch.gather(input=z, dim=1, index=b.view(B,1))
    # # print (z_tilde_b)

    v = torch.rand(B,C).clamp(1e-12, 1.-1e-12) #.cuda()
    probs = torch.softmax(logits,dim=1).repeat(B,1)
    # print (probs.shape, torch.log(v_k).shape, torch.log(v).shape)
    # fasdfa

    # print (v.shape)
    # print (v.shape)
    z_tilde = -torch.log((- torch.log(v) / probs) - torch.log(v_k))

    # print (z_tilde)
    # print (z_tilde_b)
    z_tilde.scatter_(dim=1, index=b.view(B,1), src=z_tilde_b)
    # print (z_tilde)
    # fasdfs

    return z, b, logprob, z_tilde


def sample_relax_z(logits):

    u = torch.rand(B,C).clamp(1e-10, 1.-1e-10) #.cuda()
    gumbels = -torch.log(-torch.log(u))
    z = logits + gumbels
    return z


def sample_relax_given_b(logits, b):

    u_b = torch.rand(B,1).clamp(1e-10, 1.-1e-10)
    z_tilde_b = -torch.log(-torch.log(u_b))

    u = torch.rand(B,C).clamp(1e-10, 1.-1e-10) #.cuda()
    z_tilde = -torch.log((- torch.log(u) / torch.softmax(logits,dim=1)) - torch.log(u_b))
    z_tilde.scatter_(dim=1, index=b.view(B,1), src=z_tilde_b)

    return z_tilde








surrogate = NN3(input_size=C, output_size=1, n_residual_blocks=1)
train_ = 1
n_steps = 1000#0 #0 #1000 #50000 #
B = 32 #0
k=3
if train_:
    optim = torch.optim.Adam(surrogate.parameters(), lr=1e-4, weight_decay=1e-7)
    #Train surrogate
    for i in range(n_steps+1):

        # warmup = np.minimum( (i+1) / 1000., 1.)
        warmup = 1.

        # zs = []
        # z_tildes = []
        # logprobs = []
        # rewards1 = []
        # for j in range(B):

        # logits = torch.log(probs)
        # logits =  torch.ones((1,C), requires_grad=True) #* -0.6931
        # logits =  torch.zeros((1,C), requires_grad=True) #* -0.6931
            # print (z.shape)

            # zs.append(z)
            # z_tildes.append(z_tilde)
            # logprobs.append(logprob)
            # rewards1.append(reward)

        # z = torch.stack(zs)
        # z_tilde = torch.stack(z_tildes)
        # logprob = torch.stack(logprobs)
        # reward = torch.stack(rewards1)

        # print (z.shape)
        # print (b.shape)
        
        # print (logprob.shape)
        # # fsaad

        z, b, logprob, z_tilde = sample_relax(logits)
        reward = f(b)
        cz_tilde = surrogate.net(z_tilde)
        cz = surrogate.net(z)




        # czs = []
        # cz_tildes = []
        # # logprobs = []
        # # rewards1 = []
        # for j in range(k):
        #     z = sample_relax_z(logits)
        #     z_tilde = sample_relax_given_b(logits, b)
            
            
        #     cz_tilde = surrogate.net(z_tilde)
        #     cz = surrogate.net(z)

        #     czs.append(cz)
        #     cz_tildes.append(cz_tilde)
        #     # logprobs.append(logprob)
        #     # rewards1.append(reward)

        # czs = torch.stack(czs)
        # cz_tildes = torch.stack(cz_tildes)
        # # logprobs = torch.stack(logprobs)
        # # rewards1 = torch.stack(rewards1)

        # cz = torch.mean(czs, dim=0).view(B,1)
        # cz_tilde = torch.mean(cz_tildes, dim=0).view(B,1)
        # # logprob = torch.mean(logprobs, dim=0).view(B,1)
        # # reward = torch.mean(rewards1, dim=0).view(B,1)


        # print (logprob)

        # logits_repeat = logits.repeat(B,1)
        # print (logits_repeat.shape)
        grad_logq =  torch.autograd.grad([torch.mean(logprob)], [logits], create_graph=True, retain_graph=True)[0]
        grad_surr_z =  torch.autograd.grad([torch.mean(cz)], [logits], create_graph=True, retain_graph=True)[0]
        grad_surr_z_tilde = torch.autograd.grad([torch.mean(cz_tilde)], [logits], create_graph=True, retain_graph=True)[0]

        # print()
        # print (reward.view(B,1).shape)
        # print (cz_tilde.shape)
        # print (grad_logq.repeat(B,1).shape)
        # print (grad_surr_z.repeat(B,1).shape)
        # print (grad_surr_z_tilde.repeat(B,1).shape)
        
        # loss = ((reward.view(B,1) - cz_tilde) * grad_logq.repeat(B,1) + grad_surr_z.repeat(B,1) - grad_surr_z_tilde.repeat(B,1))
        # print (loss.shape)
        # fdsf        
        # loss = torch.mean(((reward.view(B,1) - cz_tilde) * grad_logq.repeat(B,1) +  (grad_surr_z.repeat(B,1) - grad_surr_z_tilde.repeat(B,1))*warmup   )**2)
        # loss = torch.mean(((reward.view(B,1) - cz_tilde) * grad_logq.repeat(B,1) +  (grad_surr_z.repeat(B,1) - grad_surr_z_tilde.repeat(B,1))*.1   )**2)
        loss = torch.mean(((reward.view(B,1) - cz_tilde) * grad_logq.repeat(B,1) +  (grad_surr_z.repeat(B,1) - grad_surr_z_tilde.repeat(B,1))   )**2)
        # loss = torch.mean((reward.view(B,1) - cz_tilde)**2)
        # fdsaf


        optim.zero_grad()
        loss.backward()
        optim.step()

        cz_tilde_after = surrogate.net(z_tilde)


        # if i % (n_steps/10)==0:
        if i % (100)==0:
            print(i, 'loss:', loss.data.numpy(), ' dif:', torch.mean(torch.abs(reward.view(B,1) - cz_tilde)).data.numpy(), 'pred:', cz_tilde[0].data.numpy(), 'R:', reward[0].data.numpy(), '  z:', z[0].data.numpy(), '  b:', b[0].data.numpy(), '  z_t:', z_tilde[0].data.numpy(), '  pred2:', cz_tilde_after[0].data.numpy(),)
else:
    #Load
    net.load_params_v3(save_dir=home+'/Downloads/tmmpp/', step=17285, name='') 


grads_relax = []
grads_relax_score = []
grads_relax_path = []
grads_relax_score_b1 = []
grads_relax_path_b1 = []
grads_relax_score_b2 = []
grads_relax_path_b2 = []
B=1
for i in range (N):
    if i%1000==0:
        print(i,N)
    z, b, logprob, z_tilde = sample_relax(logits)

    # cz = surrogate.net(z)
    czs = []
    for j in range(1):
        z = sample_relax_z(logits)
        cz = surrogate.net(z)
        czs.append(cz)
    czs = torch.stack(czs)
    cz = torch.mean(czs).view(1,1)


    # cz_tilde = surrogate.net(z_tilde)
    cz_tildes = []
    for j in range(1):
        z_tilde = sample_relax_given_b(logits, b)
        cz_tilde = surrogate.net(z_tilde)
        cz_tildes.append(cz_tilde)
    cz_tildes = torch.stack(cz_tildes)
    cz_tilde = torch.mean(cz_tildes).view(1,1)


    # gg = torch.autograd.grad(outputs=z[0][0], inputs=(logits), retain_graph=True)[0]
    # gg2 = torch.autograd.grad(outputs=z[0][1], inputs=(logits), retain_graph=True)[0]
    # print (b)
    # print (gg)
    # print (gg2)
    # fsaf

    
    reward = f(b).view(B,1) 
    gradlogprob = torch.autograd.grad(outputs=logprob, inputs=(logits), retain_graph=True)[0]
    gradcz = torch.autograd.grad(outputs=cz, inputs=(logits), retain_graph=True)[0]
    gradcz_tilde = torch.autograd.grad(outputs=cz_tilde, inputs=(logits), retain_graph=True)[0]
    # print (reward.shape)
    # print (cz_tilde.shape)
    # print (gradlogprob.shape)
    # print (gradcz.shape)
    # print (gradcz_tilde.shape)

    grad = (reward-cz_tilde).detach() *gradlogprob + gradcz - gradcz_tilde
    # grad = reward *gradlogprob #+ gradcz - gradcz_tilde

    # print (b)
    # print (reward-cz_tilde, 'reward-cz_tilde')
    # print (gradlogprob, 'gradlogprob')
    # print (gradcz, 'gradcz')
    # print (gradcz_tilde, 'gradcz_tilde')
    # print (grad, 'grad')
    # print()


    if (grad != grad).any():
        print ('nan')
        fsfsa

    grads_relax.append(grad)
    grads_relax_score.append((reward-cz_tilde).detach() *gradlogprob)
    grads_relax_path.append(gradcz - gradcz_tilde)

    if b==1:
        grads_relax_score_b1.append((reward-cz_tilde).detach() *gradlogprob)
        grads_relax_path_b1.append(gradcz - gradcz_tilde)       
    
    if b==2:
        grads_relax_score_b2.append((reward-cz_tilde).detach() *gradlogprob)
        grads_relax_path_b2.append(gradcz - gradcz_tilde) 

print ()
grads_relax = torch.stack(grads_relax).view(N,C)
grads_relax_score = torch.stack(grads_relax_score).view(N,C)
grads_relax_path = torch.stack(grads_relax_path).view(N,C)
grads_relax_score_b1 = torch.stack(grads_relax_score_b1).view(-1,C)
grads_relax_path_b1 = torch.stack(grads_relax_path_b1).view(-1,C)
grads_relax_score_b2 = torch.stack(grads_relax_score_b2).view(-1,C)
grads_relax_path_b2 = torch.stack(grads_relax_path_b2).view(-1,C)

grad_mean_relax = torch.mean(grads_relax,dim=0)
grad_std_relax = torch.std(grads_relax,dim=0)


print ('RELAX')
print ('mean:', grad_mean_relax)
print ('std:', grad_std_relax)
print ()



print ('True')
print ('[-.5478, .1122, .4422]')
print ('dif:', np.abs(grad_mean_relax.numpy() - true ))
print ()





# fasfa





















n_bins = 80

rows = 5
cols = C
fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

x_lim_left=-2
x_lim_right=3
y_lim_top =3

col =0
row =0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads[:,0]), bins=n_bins, density=True)
ax.plot([grad_mean_reinforce[0],grad_mean_reinforce[0]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
ax.set_ylabel('reinforce')
ax.set_title('R=-1')


col =1
row =0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads[:,1]), bins=n_bins, density=True)
ax.plot([grad_mean_reinforce[1],grad_mean_reinforce[1]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
ax.set_title('R=1')


col =2
row =0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads[:,2]), bins=n_bins, density=True)
ax.plot([grad_mean_reinforce[2],grad_mean_reinforce[2]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
ax.set_title('R=2')


col =0
row =1
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads_relax[:,0]), bins=n_bins, density=True)
ax.plot([grad_mean_relax[0],grad_mean_relax[0]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
# ax.set_xlabel('x0')
ax.set_ylabel('relax')


col =1
row =1
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads_relax[:,1]), bins=n_bins, density=True)
ax.plot([grad_mean_relax[1],grad_mean_relax[1]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
# ax.set_xlabel('x1')



col =2
row =1
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads_relax[:,2]), bins=n_bins, density=True)
ax.plot([grad_mean_relax[2],grad_mean_relax[2]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
# ax.set_xlabel('x2')




col =0
row =2
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads_relax_score[:,0]), bins=n_bins, density=True, label='score')
ax.hist(to_print2(grads_relax_path[:,0]), bins=n_bins, density=True, alpha=.5, label='path')
# ax.plot([grad_mean_relax[0],grad_mean_relax[0]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
# ax.set_xlabel('x0')
ax.set_ylabel('relax\nscore vs path grad')
ax.legend()

col =1
row =2
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads_relax_score[:,1]), bins=n_bins, density=True, label='score')
ax.hist(to_print2(grads_relax_path[:,1]), bins=n_bins, density=True, alpha=.5, label='path')
# ax.plot([grad_mean_relax[1],grad_mean_relax[1]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
# ax.set_xlabel('x1')
# ax.legend()

col =2
row =2
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads_relax_score[:,2]), bins=n_bins, density=True, label='score')
ax.hist(to_print2(grads_relax_path[:,2]), bins=n_bins, density=True, alpha=.5, label='path')
# ax.plot([grad_mean_relax[2],grad_mean_relax[2]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
# ax.set_xlabel('x2')
# ax.legend()





col =0
row =3
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads_relax_score_b1[:,0]), bins=n_bins, density=True, label='score')
ax.hist(to_print2(grads_relax_path_b1[:,0]), bins=n_bins, density=True, alpha=.5, label='path')
# ax.plot([grad_mean_relax[0],grad_mean_relax[0]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
# ax.set_xlabel('x0')
ax.set_ylabel('relax\nscore vs path grad\nwhen b=1')
ax.legend()

col =1
row =3
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads_relax_score_b1[:,1]), bins=n_bins, density=True, label='score')
ax.hist(to_print2(grads_relax_path_b1[:,1]), bins=n_bins, density=True, alpha=.5, label='path')
# ax.plot([grad_mean_relax[1],grad_mean_relax[1]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
# ax.set_xlabel('x1')
# ax.legend()

col =2
row =3
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads_relax_score_b1[:,2]), bins=n_bins, density=True, label='score')
ax.hist(to_print2(grads_relax_path_b1[:,2]), bins=n_bins, density=True, alpha=.5, label='path')
# ax.plot([grad_mean_relax[2],grad_mean_relax[2]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
# ax.set_xlabel('x2')
# ax.legend()





col =0
row =4
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads_relax_score_b2[:,0]), bins=n_bins, density=True, label='score')
ax.hist(to_print2(grads_relax_path_b2[:,0]), bins=n_bins, density=True, alpha=.5, label='path')
# ax.plot([grad_mean_relax[0],grad_mean_relax[0]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
ax.set_xlabel('x0')
ax.set_ylabel('relax\nscore vs path grad\nwhen b=2')
ax.legend()

col =1
row =4
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads_relax_score_b2[:,1]), bins=n_bins, density=True, label='score')
ax.hist(to_print2(grads_relax_path_b2[:,1]), bins=n_bins, density=True, alpha=.5, label='path')
# ax.plot([grad_mean_relax[1],grad_mean_relax[1]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
ax.set_xlabel('x1')
# ax.legend()

col =2
row =4
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
ax.hist(to_print2(grads_relax_score_b2[:,2]), bins=n_bins, density=True, label='score')
ax.hist(to_print2(grads_relax_path_b2[:,2]), bins=n_bins, density=True, alpha=.5, label='path')
# ax.plot([grad_mean_relax[2],grad_mean_relax[2]], [0,y_lim_top])
ax.set_xlim(x_lim_left, x_lim_right)
ax.set_ylim(0., y_lim_top)
ax.set_xlabel('x2')
# ax.legend()




save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'view_grads.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()































































fafds










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

    u = torch.rand(B,C).clamp(1e-8, 1.-1e-8)#.cuda()
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














































