




import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import math



from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli




#Plot distributions
rows = 1
cols = 2
# fig = plt.figure(figsize=(4+cols,5+rows), facecolor='white')
fig = plt.figure(figsize=(4+cols,1+rows), facecolor='white')
# viz_range = [-10,10]
# numticks = 200

# # samps = samps.data.numpy()
# #Plot samples
# for i in range(len(samps)):
#     ax.plot([samps[i],samps[i]], [0,.1], linewidth=2, label=r'$z_q$')







# Cost function
cur_row = 0
ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=2)
# ax.axis('off')
# ax.set_yticks([])
# ax.set_xticks([])

objective = lambda x: (x-.4)**2

x = np.linspace(-2,2, num=30)
y = objective(x)
ax.plot(x, y, linewidth=2, label=r'$(x-.4)^2$')

ax.legend(fontsize=9, loc=2)
cur_row+=1







print('REINFORCE')

# samp = dist.sample()
# logprob = dist.log_prob(samp)
# print (samp, torch.exp(logprob))

n_samps = 1000

bern_param_value = .7
bern_param = torch.tensor([bern_param_value], requires_grad=True)
# dist = Bernoulli(torch.Tensor([bern_param]))
dist = Bernoulli(bern_param)
losses = []
grads = []
for i in range(n_samps):
    samp = dist.sample()
    logprob = dist.log_prob(samp)
    logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
    loss = objective(samp)
    losses.append(loss.numpy()[0])
    grads.append((logprobgrad*loss).numpy()[0])
print (
    'Real Param:', bern_param.data.numpy()[0], 
    ' Avg loss:', np.mean(losses), 
    ' Grad Mean:', np.mean(grads),
    ' Grad Var:', np.var(grads),  )



fasfd







# plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=.5)
plt.tight_layout()
name_file = home+'/Documents/Grad_Estimators/plot.png'
plt.savefig(name_file)
print ('Saved fig', name_file)

print ('Done.')




















