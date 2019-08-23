

from os.path import expanduser
home = expanduser("~")


import numpy as np
import torch

import scipy.io

from scipy import stats


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os


import torch.distributions as d



def numpy(x):
    return x.data.cpu().numpy()



exp_dir = home+'/Documents/glow_clevr/dists/'

if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    print ('Made dir', exp_dir) 






# aa =m.sample()  

# print (m.sample())
# print (m.log_prob(aa))




n_samples = 500



cols = 2
rows = 2
fig = plt.figure(figsize=(7+cols,2+rows), facecolor='white', dpi=150)




x = np.linspace(-8,8,200)
x = torch.from_numpy(x).float()
# print (m.log_prob(x))
m = d.Cauchy(torch.tensor([0.0]), torch.tensor([1.]))
probs = torch.exp(m.log_prob(x))

m = d.Normal(torch.tensor([0.0]), torch.tensor([1.]))
probs2 = torch.exp(m.log_prob(x))

m = d.StudentT(torch.tensor([2.0]))
probs3 = torch.exp(m.log_prob(x))

ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

ax.plot(numpy(x), numpy(probs), label='Cauchy')
ax.plot(numpy(x), numpy(probs2), label='Normal')
ax.plot(numpy(x), numpy(probs3), label='StudentT')
ax.legend()








m = d.Cauchy(torch.tensor([0.0]), torch.tensor([1.]))
samps = m.sample([n_samples])

m = d.Normal(torch.tensor([0.0]), torch.tensor([1.]))
samps2 = m.sample([n_samples])

m = d.StudentT(torch.tensor([2.0]))
samps3 = m.sample([n_samples])

# samps = torch.cat([samps, samps2, samps3], 1)

# print (numpy(samps2.view(-1)).shape)
samps = numpy(samps.view(-1))

print ('cauchy', np.min(samps), np.max(samps))
samps = np.clip(samps, a_min=-8, a_max=8)
density = stats.kde.gaussian_kde(samps)


samps2 = numpy(samps2.view(-1))
density2 = stats.kde.gaussian_kde(samps2)
print ('normal', np.min(samps2), np.max(samps2))


samps3 = numpy(samps3.view(-1))
density3 = stats.kde.gaussian_kde(samps3)
print ('student', np.min(samps3), np.max(samps3))


ax = plt.subplot2grid((rows,cols), (1,0), frameon=False)

# ax.plot(numpy(x), numpy(probs), label='Cauchy')
# ax.plot(numpy(x), numpy(probs2), label='Normal')
# ax.plot(numpy(x), numpy(probs3), label='StudentT')
# ax.hist(numpy(samps), bins=200, label=['Cauchy','Normal','StudentT'])
ax.plot(numpy(x), density(x), label='Cauchy')
ax.fill_between(numpy(x), density(x), alpha=.2)
ax.plot(numpy(x), density2(x), label='Normal')
ax.fill_between(numpy(x), density2(x), alpha=.2)
ax.plot(numpy(x), density3(x), label='StudentT')
ax.fill_between(numpy(x), density3(x), alpha=.2)
# ax.set_xlim(left=-4, right=4)
ax.legend()














x = np.linspace(-1,1,200)
x = torch.from_numpy(x).float()
# print (m.log_prob(x))
m = d.Cauchy(torch.tensor([0.0]), torch.tensor([.01]))
probs = torch.exp(m.log_prob(x))

m = d.Normal(torch.tensor([0.0]), torch.tensor([.01]))
probs2 = torch.exp(m.log_prob(x))

ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)

ax.plot(numpy(x), numpy(probs), label='Cauchy')
ax.plot(numpy(x), numpy(probs2), label='Normal')
ax.legend()








m = d.Cauchy(torch.tensor([0.0]), torch.tensor([.01]))
samps = m.sample([n_samples])

m = d.Normal(torch.tensor([0.0]), torch.tensor([.01]))
samps2 = m.sample([n_samples])

ax = plt.subplot2grid((rows,cols), (1,1), frameon=False)


samps = torch.cat([samps, samps2], 1)
# print (samps.shape)
# fsadf

ax.hist(numpy(samps), 200, label=['Cauchy','Normal'])
# ax.hist(numpy(x), numpy(probs2), label='Normal')
ax.set_xlim(left=-1, right=1)
ax.legend()


# print ()



# plt.tight_layout()
plt_path = exp_dir + 'dists.png'
plt.savefig(plt_path)
print ('saved viz',plt_path)
plt.close(fig)






