

from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch



#gumbel on cats

print (np.log(.7))
print (np.log(.2))
print (np.log(.1))


def gumbel_logprob(x, theta):
    x = x - np.log(theta)
    # return np.exp(-(x+np.exp(-x)))
    return -(x+np.exp(-x))



rows = 3
cols = 1
fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

col =0
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

theta = .95
probs = []
xs = np.linspace(-10.,15., 100)
for x in xs:
    logprob = gumbel_logprob(x, theta)
    prob = np.exp(logprob)
    probs.append(prob)
ax.plot(xs,probs)
ax.plot([np.log(theta),np.log(theta)], [0,.4] )
ax.set_title(str(theta))

col =0
row = 1
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

theta = .03
probs = []
xs = np.linspace(-10.,15., 100)
for x in xs:
    logprob = gumbel_logprob(x, theta)
    prob = np.exp(logprob)
    probs.append(prob)
ax.plot(xs,probs)
ax.plot([np.log(theta),np.log(theta)], [0,.4] )
ax.set_title(str(theta))

col =0
row = 2
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

theta = .02
probs = []
xs = np.linspace(-10.,15., 100)
for x in xs:
    logprob = gumbel_logprob(x, theta)
    prob = np.exp(logprob)
    probs.append(prob)
ax.plot(xs,probs)
ax.plot([np.log(theta),np.log(theta)], [0,.4] )
ax.set_title(str(theta))

save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'gumbel_with_cats.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()
####
fsdafd










def to_print2(x):
    return x.data.cpu().numpy()

def H(soft):
    return torch.argmax(soft, dim=0)



def sample(probs):
    u = torch.rand(len(probs)) #*.5 + .5
    gumbels = -torch.log(-torch.log(u))
    # print (gumbels)
    logits = torch.log(probs)
    return gumbels + logits



# def sample(probs):
#   u = torch.rand(len(probs)) #*.5 + .5
#   gumbels = torch.exp(-torch.log(-torch.log(u)))
#   # print (gumbels)
#   logits = torch.log(probs)
#   return gumbels + logits



# def sample(probs):
#   u = torch.rand(len(probs)) #*.5 + .5
#   gumbels = torch.exp(-torch.log(-torch.log(u)))
#   # print (gumbels)
#   # logits = torch.log(probs)
#   return gumbels + probs



# def log_prob(values, probs):
#     logits = torch.log(probs)
#     dif = logits - values
#     return dif


# theta = .2
# weights =  torch.tensor([1-theta,theta], requires_grad=True).float()
weights =  torch.tensor([.7,.2,.1], requires_grad=True).float()

n_cats = 3
n= 10000
cs = np.zeros(n_cats)

for i in range(10000):
    samp = sample(weights)
    c = H(samp)
    # cs.append(to_print2(c))
    cs[to_print2(c)] +=1
# print (np.mean(cs))
print (cs/n)
























