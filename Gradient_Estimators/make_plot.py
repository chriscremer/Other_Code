


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

from NN import NN
from NN_forrelax import NN as NN2


def H(x):
    if x > .5:
        return 1
    else:
        return 0

def Hpy(x):
    # if x > .5:
    #     return torch.tensor([1])
    # else:
    #     return torch.tensor([0])

    return (x > .5).float()


def prob_to_logit(prob):
    return torch.log(prob) - torch.log(1-prob)

def logit_to_prob(logit):
    return torch.sigmoid(logit)



# n= 10000
# theta = .5
f = lambda x: (x-val)**2

total_steps = 10000














logits = 0
bern_param = torch.tensor([logits], requires_grad=True)
val=.49


print()
print ('RELAX')
print ('Value:', val)
print()

net = NN()


# print (len(net.parameters()))

# optim = torch.optim.Adam([bern_param] + list(net.parameters()), lr=.004)
optim = torch.optim.Adam([bern_param], lr=.004)
optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


steps = []
losses8 = []
for step in range(total_steps):

    dist = RelaxedBernoulli(torch.Tensor([1.]), logits=bern_param)
    dist_bern = Bernoulli(bern_param)

    optim.zero_grad()

    zs = []
    for i in range(20):
        z = dist.rsample()
        zs.append(z)
    zs = torch.stack(zs)

    b = Hpy(zs)

    #p(z|b)
    theta = logit_to_prob(bern_param)
    v = torch.rand(zs.shape[0], zs.shape[1]) 
    v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
    z_tilde = bern_param.detach() + torch.log(v_prime) - torch.log1p(-v_prime)
    z_tilde = torch.sigmoid(z_tilde)

    EHRERE, I think the bug is above, see grad of stuff 

    logprob = dist_bern.log_prob(b)
    
    pred = net.net(z_tilde)
    pred2 = net.net(zs)
    f_b = f(b)

    loss = torch.mean((f_b-pred.detach()) * logprob - pred + pred2 )

    # logprobgrad_2 = torch.autograd.grad(outputs=torch.mean(pred), inputs=(bern_param), retain_graph=True)[0]
    # print (logprobgrad_2)

    # logprobgrad_2 = torch.autograd.grad(outputs=torch.mean(pred2), inputs=(bern_param), retain_graph=True)[0]
    # print (logprobgrad_2)

    # fdfa

    loss.backward(retain_graph=True)  
    optim.step()

    optim_NN.zero_grad()
    NN_loss = torch.mean((f_b - pred)**2) 
    NN_loss.backward()  
    optim_NN.step()

    if step%50 ==0:
        if step %500==0:
            print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
        losses8.append(torch.mean(f_b).numpy())
        steps.append(step)




















logits = 0
bern_param = torch.tensor([logits], requires_grad=True)
val=.4

print()
print('REINFORCE')
print ('Value:', val)
# print ('n:', n)
# print ('theta:', theta)
print()


optim = torch.optim.Adam([bern_param], lr=.004)

steps = []
losses= []
for step in range(total_steps):

    dist = Bernoulli(logits=bern_param)

    optim.zero_grad()

    bs = []
    for i in range(20):
        samps = dist.sample()
        bs.append(H(samps))
    bs = torch.FloatTensor(bs).unsqueeze(1)

    logprob = dist.log_prob(bs)
    # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

    loss = torch.mean(f(bs) * logprob)

    #review the pytorch_toy and the RL code to see how PG was done 

    loss.backward()  
    optim.step()

    if step%50 ==0:
        if step %500==0:
            print (step, torch.mean(f(bs)).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy())
        losses.append(torch.mean(f(bs)).numpy())
        steps.append(step)








# logits = 0
# bern_param = torch.tensor([logits], requires_grad=True)
# val=.49

# print()
# print('REINFORCE')
# print ('Value:', val)
# # print ('n:', n)
# # print ('theta:', theta)
# print()

# optim = torch.optim.Adam([bern_param], lr=.004)

# steps = []
# losses2= []
# for step in range(total_steps):

#     dist = Bernoulli(logits=bern_param)

#     optim.zero_grad()

#     bs = []
#     for i in range(20):
#         samps = dist.sample()
#         bs.append(H(samps))
#     bs = torch.FloatTensor(bs).unsqueeze(1)

#     logprob = dist.log_prob(bs)
#     # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

#     loss = torch.mean(f(bs) * logprob)

#     #review the pytorch_toy and the RL code to see how PG was done 

#     loss.backward()  
#     optim.step()

#     if step%50 ==0:
#         if step %500==0:
#             print (step, torch.mean(f(bs)).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy())
#         losses2.append(torch.mean(f(bs)).numpy())
#         steps.append(step)







# logits = 0
# bern_param = torch.tensor([logits], requires_grad=True)
# val=.499

# print()
# print('REINFORCE')
# print ('Value:', val)
# # print ('n:', n)
# # print ('theta:', theta)
# print()

# optim = torch.optim.Adam([bern_param], lr=.004)

# steps = []
# losses3= []
# for step in range(total_steps):

#     dist = Bernoulli(logits=bern_param)

#     optim.zero_grad()

#     bs = []
#     for i in range(20):
#         samps = dist.sample()
#         bs.append(H(samps))
#     bs = torch.FloatTensor(bs).unsqueeze(1)

#     logprob = dist.log_prob(bs)
#     # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

#     loss = torch.mean(f(bs) * logprob)

#     #review the pytorch_toy and the RL code to see how PG was done 

#     loss.backward()  
#     optim.step()

#     if step%50 ==0:
#         if step %500==0:
#             print (step, torch.mean(f(bs)).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy())
#         losses3.append(torch.mean(f(bs)).numpy())
#         steps.append(step)

























# logits = 0
# bern_param = torch.tensor([logits], requires_grad=True)
# val=.4


# print()
# print ('REINFORCE H(z)')
# print ('Value:', val)
# print()

# # net = NN()

# optim = torch.optim.Adam([bern_param], lr=.004)
# # optim_NN = torch.optim.Adam([net.parameters()], lr=.0004)


# steps = []
# losses4= []
# for step in range(total_steps):

#     dist = RelaxedBernoulli(torch.Tensor([1.]), logits=bern_param)

#     optim.zero_grad()

#     zs = []
#     for i in range(20):
#         z = dist.rsample()
#         zs.append(z)
#     zs = torch.FloatTensor(zs).unsqueeze(1)

#     logprob = dist.log_prob(zs.detach())
#     # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

#     H_z = Hpy(zs)

#     # print (H_z)
#     # fdsafd
#     f_b = f(H_z)

#     # print (f_b.shape)
#     # print (logprob.shape)
#     # fad

#     loss = torch.mean(f_b * logprob)

#     #review the pytorch_toy and the RL code to see how PG was done 

#     loss.backward()  
#     optim.step()

#     if step%50 ==0:
#         if step %500==0:
#             print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy())
#         losses4.append(torch.mean(f_b).numpy())
#         steps.append(step)




    






















# logits = 0
# bern_param = torch.tensor([logits], requires_grad=True)
# val=.4


# print()
# print ('SimpLAX')
# print ('Value:', val)
# print()

# net = NN()


# # print (len(net.parameters()))

# # optim = torch.optim.Adam([bern_param] + list(net.parameters()), lr=.004)
# optim = torch.optim.Adam([bern_param], lr=.004)
# optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


# steps = []
# losses5= []
# for step in range(total_steps):

#     dist = RelaxedBernoulli(torch.Tensor([1.]), logits=bern_param)

#     optim.zero_grad()

#     zs = []
#     for i in range(20):
#         z = dist.rsample()
#         zs.append(z)
#     # zs = torch.FloatTensor(zs).unsqueeze(1)
#     zs = torch.stack(zs)

#     logprob = dist.log_prob(zs.detach())
    
#     pred = net.net(zs)
#     H_z = Hpy(zs)
#     f_b = f(H_z)

#     loss = torch.mean((f_b-pred.detach()) * logprob + pred )

#     loss.backward(retain_graph=True)  
#     # loss.backward()  
#     optim.step()


#     optim_NN.zero_grad()
#     # pred = net.net(zs)
#     NN_loss = torch.mean((f_b - pred)**2)
#     NN_loss.backward()  
#     optim_NN.step()


#     if step%50 ==0:
#         if step %500==0:
#             print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
#         losses5.append(torch.mean(f_b).numpy())
#         steps.append(step)








# logits = 0
# bern_param = torch.tensor([logits], requires_grad=True)
# val=.49


# print()
# print ('SimpLAX')
# print ('Value:', val)
# print()

# net = NN()


# # print (len(net.parameters()))

# # optim = torch.optim.Adam([bern_param] + list(net.parameters()), lr=.004)
# optim = torch.optim.Adam([bern_param], lr=.004)
# optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


# steps = []
# losses6= []
# for step in range(total_steps):

#     dist = RelaxedBernoulli(torch.Tensor([1.]), logits=bern_param)

#     optim.zero_grad()

#     zs = []
#     for i in range(20):
#         z = dist.rsample()
#         zs.append(z)
#     # zs = torch.FloatTensor(zs).unsqueeze(1)
#     zs = torch.stack(zs)

#     logprob = dist.log_prob(zs.detach())
    
#     pred = net.net(zs)
#     H_z = Hpy(zs)
#     f_b = f(H_z)

#     loss = torch.mean((f_b-pred.detach()) * logprob + pred )

#     loss.backward(retain_graph=True)  
#     # loss.backward()  
#     optim.step()


#     optim_NN.zero_grad()
#     # pred = net.net(zs)
#     NN_loss = torch.mean((f_b - pred)**2)
#     NN_loss.backward()  
#     optim_NN.step()


#     if step%50 ==0:
#         if step %500==0:
#             print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
#         losses6.append(torch.mean(f_b).numpy())
#         steps.append(step)












# logits = 0
# bern_param = torch.tensor([logits], requires_grad=True)
# val=.499


# print()
# print ('SimpLAX')
# print ('Value:', val)
# print()

# net = NN()


# # print (len(net.parameters()))

# # optim = torch.optim.Adam([bern_param] + list(net.parameters()), lr=.004)
# optim = torch.optim.Adam([bern_param], lr=.004)
# optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


# steps = []
# losses7= []
# for step in range(total_steps):

#     dist = RelaxedBernoulli(torch.Tensor([1.]), logits=bern_param)

#     optim.zero_grad()

#     zs = []
#     for i in range(20):
#         z = dist.rsample()
#         zs.append(z)
#     # zs = torch.FloatTensor(zs).unsqueeze(1)
#     zs = torch.stack(zs)

#     logprob = dist.log_prob(zs.detach())
    
#     pred = net.net(zs)
#     H_z = Hpy(zs)
#     f_b = f(H_z)

#     loss = torch.mean((f_b-pred.detach()) * logprob + pred )

#     loss.backward(retain_graph=True)  
#     # loss.backward()  
#     optim.step()


#     optim_NN.zero_grad()
#     # pred = net.net(zs)
#     NN_loss = torch.mean((f_b - pred)**2)
#     NN_loss.backward()  
#     optim_NN.step()


#     if step%50 ==0:
#         if step %500==0:
#             print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
#         losses7.append(torch.mean(f_b).numpy())
#         steps.append(step)











































# print (len(steps))
# print (len(losses))
# print (len(losses2))
# print (len(losses3))

ylabel = 'f(b)'
xlabel = 'Steps'



rows = 1
cols = 3
# text_col_width = cols
fig = plt.figure(figsize=(4+cols,2+rows), facecolor='white', dpi=150)

col =0
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(steps, losses, label='REINFORCE', alpha=.8)
# ax.plot(steps, losses4, label='REINFORCE H(z)', alpha=.8)
# ax.plot(steps, losses5, label='SimpLAX', alpha=.8)
ax.plot(steps, losses8, label='RELAX', alpha=.8)

ax.grid(True, alpha=.3)
ax.set_title(r'$f=(b-.4)^2$', size=6, family='serif')
ax.tick_params(labelsize=6)
ax.set_ylabel(ylabel, size=6, family='serif')
ax.set_xlabel(xlabel, size=6, family='serif')
ax.legend(prop={'size':5}) #, loc=2)  #upper left


# col =1
# row = 0
# ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

# ax.plot(steps, losses2, label='REINFORCE')
# ax.plot(steps, losses6, label='SimpLAX')

# ax.grid(True, alpha=.3)
# ax.set_title(r'$f=(b-.49)^2$', size=6, family='serif')
# ax.tick_params(labelsize=6)
# # ax.set_ylabel(ylabel, size=6, family='serif')
# ax.set_xlabel(xlabel, size=6, family='serif')
# ax.legend(prop={'size':5})

# col =2
# row = 0
# ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

# ax.plot(steps, losses3, label='REINFORCE')
# ax.plot(steps, losses7, label='SimpLAX')

# ax.grid(True, alpha=.3)
# ax.set_title(r'$f=(b-.499)^2$', size=6, family='serif')
# ax.tick_params(labelsize=6)
# # ax.set_ylabel(ylabel, size=6, family='serif')
# ax.set_xlabel(xlabel, size=6, family='serif')
# ax.legend(prop={'size':5})









save_dir = home+'/Downloads/tmmpp/'
plt_path = save_dir+'curves_plot.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()













































fasfa


    # samps = []
    # grads = []
    # for i in range(n):
    #     samp = dist.sample()

    #     logprob = dist.log_prob(samp)
    #     logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

    #     samps.append(samp.numpy())
    #     grads.append(f(samp.numpy()) * logprobgrad.numpy())

    # print ('Grad Estimator: REINFORCE')
    # print ('Avg samp', np.mean(samps))
    # print ('Grad mean', np.mean(grads))
    # print ('Grad std', np.std(grads))
    # print()

































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

