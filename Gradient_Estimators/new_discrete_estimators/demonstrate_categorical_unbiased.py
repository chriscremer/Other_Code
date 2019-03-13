

from os.path import expanduser
home = expanduser("~")


# import sys, os
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('./VAE'))

import numpy as np

import torch
# from torch.distributions.bernoulli import Bernoulli
# from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.categorical import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

# from NN import NN
# from NN_forrelax import NN as NN2
from NN2 import NN3





#quick check

f = np.array([0,1,2])
p = np.array([.3, .2, .4])
E= np.dot(f,p)
print (f*p)
print (E)
grad_E_dim0 = fsdaf
fdsaf








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

def H(soft):
    return torch.argmax(soft, dim=0)

# def prob_to_logit(prob):
#     return torch.log(prob) - torch.log(1-prob)

# def logit_to_prob(logit):
#     return torch.sigmoid(logit)

def f(x):
    rewards = np.zeros(n_cats)
    rewards[0] = 1.
    rewards = torch.tensor(rewards).float()
    # print (x, rewards)
    return torch.dot(x,rewards)


n= 5000 #10000
n_cats = 3 #20
n_components = n_cats
C = n_cats
# theta = .01 #.99 #.1 #95 #.3 #.9 #.05 #.3
# bern_param = torch.tensor([theta], requires_grad=True)
# val= .4
# f = lambda x: (x-val)**2
# f = lambda x: x
# dif = f(1)-f(0)

# print ()
# print ('Value:', val)
# print (f(0))
# print (f(1))
# print ('so dif in f(1) and f(0) is', dif)
print()
print ('n:', n)
# print ('theta:', theta)
print ()


temp = 1.












#REINFORCE


# needsoftmax_mixtureweight = torch.randn(n_cats, requires_grad=True)
needsoftmax_mixtureweight = torch.tensor(np.ones(n_cats), requires_grad=True)
# needsoftmax_mixtureweight = torch.randn(n_cats, requires_grad=True)
weights = torch.softmax(needsoftmax_mixtureweight, dim=0)

cat = Categorical(probs= weights)



# dist = Bernoulli(bern_param)
# samps = []
avg_samp = torch.zeros(n_cats)
grads = []
logprobgrads = []
for i in range(n):
    # samp = dist.sample()
    cluster = cat.sample()
    logprob = cat.log_prob(cluster.detach())
    one_hot = torch.zeros(n_cats)
    one_hot[cluster] = 1.

    # logprob = dist.log_prob(samp.detach())
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(needsoftmax_mixtureweight), retain_graph=True)[0]
    # print (samp.data.numpy(), logprob.data.numpy(), logprobgrad.data.numpy())
    # fsdfas
    # print (logprobgrad.shape)
    # fdfas
    avg_samp += one_hot
    # samps.append(one_hot.numpy())
    grads.append( (f(one_hot) * logprobgrad).numpy())
    logprobgrads.append(logprobgrad.numpy())

grads = np.array(grads)
logprobgrads = np.array(logprobgrads)
# print (grads.shape, logprobgrads.shape)

# print (grads[:10])

print ('Grad Estimator: REINFORCE')
print ()
print ('Avg samp', avg_samp/n)
print ('Grad mean', np.mean(grads, axis=0))
print ('Grad std', np.std(grads, axis=0))
print ('Avg logprobgrad', np.mean(logprobgrads, axis=0))
print ('Std logprobgrad', np.std(logprobgrads, axis=0))
print()













#REINFORCE P(Z)

# needsoftmax_mixtureweight = torch.randn(n_cats, requires_grad=True)
needsoftmax_mixtureweight = torch.tensor(np.ones(n_cats), requires_grad=True).float()
# needsoftmax_mixtureweight = torch.randn(n_cats, requires_grad=True)
weights = torch.softmax(needsoftmax_mixtureweight, dim=0)

# cat = Categorical(probs= weights)
cat = RelaxedOneHotCategorical(probs=weights, temperature=torch.tensor([1.]))#.cuda())


# dist = Bernoulli(bern_param)
# samps = []
avg_samp = torch.zeros(n_cats)
gradspz = []
logprobgrads = []
for i in range(n):
    # samp = dist.sample()
    cluster_S = cat.sample()
    logprob = cat.log_prob(cluster_S.detach())
    cluster_H = H(cluster_S) #
    one_hot = torch.zeros(n_cats)
    one_hot[cluster_H] = 1.

    # logprob = dist.log_prob(samp.detach())
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(needsoftmax_mixtureweight), retain_graph=True)[0]
    # print (samp.data.numpy(), logprob.data.numpy(), logprobgrad.data.numpy())
    # fsdfas
    # print (logprobgrad.shape)
    # fdfas
    avg_samp += one_hot
    # samps.append(one_hot.numpy())
    gradspz.append( (f(one_hot) * logprobgrad).numpy())
    logprobgrads.append(logprobgrad.numpy())

gradspz = np.array(gradspz)
logprobgrads = np.array(logprobgrads)
# print (grads.shape, logprobgrads.shape)

# print (grads[:10])

print ('Grad Estimator: REINFORCE p(z)')
print ()
print ('Avg samp', avg_samp/n)
print ('Grad mean', np.mean(gradspz, axis=0))
print ('Grad std', np.std(gradspz, axis=0))
print ('Avg logprobgrad', np.mean(logprobgrads, axis=0))
print ('Std logprobgrad', np.std(logprobgrads, axis=0))
print()

print ('Dif')
# dif = np.mean(grads, axis=0) - np.mean(grads, axis=0)
# for i in range(len(dif)):
#     print (i, dif[i])
print (np.mean(gradspz, axis=0) - np.mean(grads, axis=0))


















#RELAX
B  =1
# needsoftmax_mixtureweight = torch.randn(n_cats, requires_grad=True)
needsoftmax_mixtureweight = torch.tensor(np.ones(n_cats), requires_grad=True).float()
weights = torch.softmax(needsoftmax_mixtureweight, dim=0)
probs = weights.view(B,C)

cat = Categorical(probs= probs)
surrogate = NN3(input_size=n_components, output_size=1, n_residual_blocks=2)#.cuda()
# optim_surr = torch.optim.Adam(surrogate.parameters(), lr=1e-3)


avg_samp = torch.zeros(n_cats)
gradspz = []
logprobgrads = []
for i in range(n):

    #Sample z
    u = torch.rand(B,C)
    gumbels = -torch.log(-torch.log(u))
    z = torch.log(probs) + gumbels

    b = torch.argmax(z, dim=1)
    logprob = cat.log_prob(b)
    one_hot = torch.zeros(n_cats)
    one_hot[b] = 1.

    #Sample z_tilde
    u_b = torch.rand(B,1)
    z_tilde_b = -torch.log(-torch.log(u_b))
    u = torch.rand(B,C)
    z_tilde = -torch.log((- torch.log(u) / probs) - torch.log(u))
    z_tilde[:,b] = z_tilde_b

    surr_pred_z = torch.sigmoid( surrogate.net(z) )
    surr_pred_z_tilde = torch.sigmoid( surrogate.net(z_tilde) )

    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(needsoftmax_mixtureweight), retain_graph=True)[0]

    grad = (f(one_hot)-surr_pred_z_tilde.detach()) * logprobgrad + surr_pred_z - surr_pred_z_tilde

    avg_samp += one_hot
    # samps.append(one_hot.numpy())
    gradspz.append( grad.data.numpy())
    # logprobgrads.append(logprobgrad.numpy())




gradspz = np.array(gradspz)
logprobgrads = np.array(logprobgrads)
# print (grads.shape, logprobgrads.shape)

# print (grads[:10])

print()
print ('Grad Estimator: RELAX')
print ()
print ('Avg samp', avg_samp/n)
print ('Grad mean', np.mean(gradspz, axis=0))
print ('Grad std', np.std(gradspz, axis=0))
# print ('Avg logprobgrad', np.mean(logprobgrads, axis=0))
# print ('Std logprobgrad', np.std(logprobgrads, axis=0))
print()

print ('Dif')
# dif = np.mean(grads, axis=0) - np.mean(grads, axis=0)
# for i in range(len(dif)):
#     print (i, dif[i])
print (np.mean(gradspz, axis=0) - np.mean(grads, axis=0))








# def relax(surrogate, x, logits, mixtureweights, k=1):
#     B = logits.shape[0]
#     C = logits.shape[1]
#     probs = torch.softmax(logits, dim=1)

#     cat = Categorical(probs=probs)

#     #Sample z
#     u = torch.rand(C).cuda()
#     gumbels = -torch.log(-torch.log(u))
#     z = torch.log(probs) + gumbels

#     b = torch.argmax(z, dim=1)
#     logq = cat.log_prob(b)
#     logpx_given_z = logprob_undercomponent(x, component=b)
#     logpz = torch.log(mixtureweights[b]).view(B,1)
#     logpxz = logpx_given_z + logpz #[B,1]
#     f = logpxz - logq - 1.

#     #Sample z_tilde
#     u_b = torch.rand(1).cuda()
#     z_tilde_b = -torch.log(-torch.log(u_b))
#     u = torch.rand(C).cuda()
#     z_tilde = -torch.log((- torch.log(u) / probs) - torch.log(u))
#     z_tilde[b] = z_tilde_b

#     surr_pred_z = surrogate.net(torch.cat([z, x], dim=1))
#     surr_pred_z_tilde = surrogate.net(torch.cat([z_tilde, x], dim=1))

#     #Encoder loss
#     net_loss = - torch.mean((f.detach() - surr_pred_z_tilde.detach()) * logq  + surr_pred_z - surr_pred_z_tilde)

#     #Surrogate loss
#     grad_logq =  torch.mean( torch.autograd.grad([torch.mean(logq)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)
#     grad_surr_z = torch.mean( torch.autograd.grad([torch.mean(surr_pred_z)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)
#     grad_surr_z_tilde = torch.mean( torch.autograd.grad([torch.mean(surr_pred_z_tilde)], [logits], create_graph=True, retain_graph=True)[0], dim=1, keepdim=True)
#     surr_loss = torch.mean(((f.detach() - surr_pred_z_tilde) * grad_logq + grad_surr_z - grad_surr_z_tilde)**2)

#     surr_dif = torch.mean(torch.abs(f.detach() - surr_pred_z_tilde))











fadsfd





























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






















