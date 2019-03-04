

from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



import numpy as np

import torch

# from torch.distributions.bernoulli import Bernoulli
# from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
# from torch.distributions.normal import Normal
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical
# from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli
# from torch.distributions.categorical import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

def to_print2(x):
    return x.data.cpu().numpy()

seed=1
torch.manual_seed(seed)

rows = 1
cols = 1
fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)


col =0
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

n_cats = 2

# needsoftmax_mixtureweight = torch.tensor(np.ones(n_cats), requires_grad=True)
# needsoftmax_mixtureweight = torch.tensor([], requires_grad=True)
# weights = torch.softmax(needsoftmax_mixtureweight, dim=0).float()
theta = .99
weights =  torch.tensor([1-theta,theta], requires_grad=True).float()
cat = RelaxedOneHotCategorical(probs=weights, temperature=torch.tensor([1.]))

val = 1.
val2 = 0
val3 = 0
cmap='Blues'
alpha =1.
xlimits=[val3, val]
ylimits=[val2, val]
numticks = 51
x = np.linspace(*xlimits, num=numticks)
y = np.linspace(*ylimits, num=numticks)
X, Y = np.meshgrid(x, y)
aaa = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
aaa = torch.tensor(aaa).float()
logprob = cat.log_prob(aaa)

logprob = to_print2(logprob)
logprob = logprob.reshape(X.shape)
prob = np.exp(logprob)
cs = plt.contourf(X, Y, prob, cmap=cmap, alpha=alpha)

# ax.text(0,0,str(to_print2(weights)))

samps0 = []
samps1 = []
samps = []
for i in range (300):
    samp = cat.sample()
    samp =to_print2(samp)
    if samp[0]>samp[1]:
        samps0.append(samp)
    else:
        samps1.append(samp)
    samps.append(samp)
    # print(samp)
samps0 = np.array(samps0)
samps1 = np.array(samps1)
samps = np.array(samps)

print (len(samps0)/len(samps))
print (len(samps1)/len(samps))
# print(samps.shape)

# print (np.mean(samps.T[0]))
# print (np.mean(samps.T[1]))

ax.scatter(samps0.T[0], samps0.T[1], c='r', alpha=.1)
ax.scatter(samps1.T[0], samps1.T[1], c='blue', alpha=.1)

x=np.linspace(val2,val,20)
x1=np.linspace(val3,val,20)
ax.plot(x,x, c='orange')

ax.plot(x1,np.ones(len(x))*np.mean(samps.T[1]), c='b')
ax.plot(np.ones(len(x))*np.mean(samps.T[0]),x, c='r')

ax.set_ylabel('Class 1')
ax.set_xlabel('Class 0')

ax.set_aspect('equal')







# col =0
# row = 1
# ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

# cs = plt.contourf(X, Y, prob, cmap=cmap, alpha=alpha)
# ax.plot(x,x, c='orange')
# ax.plot(x1,np.ones(len(x))*np.mean(samps.T[1]), c='blue')
# ax.plot(np.ones(len(x))*np.mean(samps.T[0]),x, c='r')

# class1_mean = np.mean(samps.T[1])
# class0_mean = np.mean(samps.T[0])
# # samps4 = []
# # for i in range(len(samps)):
# #     if samps[i][0]>class0_mean and samps[i][1]>class1_mean:
# #         samps4.append(samps[i])
# # samps4 = np.array(samps4)

# # ax.scatter(samps4.T[0], samps4.T[1], alpha=.1, c='black')

# # print ('n class1', len(samps1))
# # print ('n class speical', len(samps4))
# # print ('frac', len(samps4)/len(samps1))

# ax.set_ylabel('Class 1')
# ax.set_xlabel('Class 0')
# ax.set_aspect('equal')


# #is symmetric?
# above = []
# above_class0 = []
# for i in range(len(samps)):
#     if samps[i][1] > class1_mean:
#         above.append(1)
#     else:
#         above.append(0)

#     if samps[i][0] > class0_mean:
#         above_class0.append(1)
#     else:
#         above_class0.append(0)

# print ('sym1:', np.mean(above))
# print ('sym0:', np.mean(above_class0))







save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'2dplot_thetaspace.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()

