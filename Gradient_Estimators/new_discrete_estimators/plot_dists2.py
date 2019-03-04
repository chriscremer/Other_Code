

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



####LOOKING AT GUMBEL
xs =[]
for i in range(9000):
    x= -np.log(-np.log(np.random.rand(1)))
    xs.append(x[0])
print (xs[:5])
print (np.mean(xs))
rows = 2
cols = 1
fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)
col =0
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.hist(xs,50, density=True)
ax.set_xlim(-10., 15.)

def gumbel_logprob(x):
    # return np.exp(-(x+np.exp(-x)))
    return -(x+np.exp(-x))

probs = []
xs = np.linspace(-10.,15., 100)
for x in xs:
    logprob = gumbel_logprob(x)
    prob = np.exp(logprob)
    probs.append(prob)

col =0
row = 1
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(xs,probs)

save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'gumbel.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()
fsdafd
####






seed=1
torch.manual_seed(seed)

rows = 2
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
cat = ExpRelaxedCategorical(probs=weights, temperature=torch.tensor([1.]))

val = .5
val2 = -2.5
val3 = -11
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







col =0
row = 1
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

cs = plt.contourf(X, Y, prob, cmap=cmap, alpha=alpha)
ax.plot(x,x, c='orange')
ax.plot(x1,np.ones(len(x))*np.mean(samps.T[1]), c='blue')
ax.plot(np.ones(len(x))*np.mean(samps.T[0]),x, c='r')

class1_mean = np.mean(samps.T[1])
class0_mean = np.mean(samps.T[0])
samps4 = []
for i in range(len(samps)):
    if samps[i][0]>class0_mean and samps[i][1]>class1_mean:
        samps4.append(samps[i])
samps4 = np.array(samps4)

ax.scatter(samps4.T[0], samps4.T[1], alpha=.1, c='black')

print ('n class1', len(samps1))
print ('n class speical', len(samps4))
print ('frac', len(samps4)/len(samps1))

ax.set_ylabel('Class 1')
ax.set_xlabel('Class 0')
ax.set_aspect('equal')


#is symmetric?
above = []
above_class0 = []
for i in range(len(samps)):
    if samps[i][1] > class1_mean:
        above.append(1)
    else:
        above.append(0)

    if samps[i][0] > class0_mean:
        above_class0.append(1)
    else:
        above_class0.append(0)

print ('sym1:', np.mean(above))
print ('sym0:', np.mean(above_class0))







save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'2dplot.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()


fsdafddfa

print (logprob.shape)

fdsfa




probs = []

samps = []
grads = []
# xs = np.linspace(.0001,.9999, 40)
xs = np.linspace(-10.,10., 40)
for x in xs:
    theta = torch.tensor([0.999], requires_grad=True)
    dist = LogitRelaxedBernoulli(temperature=torch.Tensor([1.]), probs=theta)
    # print (dist.sample())
    # print (dist.log_prob(dist.sample()))
    # # # fsa
    # prob = torch.exp(dist.log_prob(dist.sample()))
    # # print (prob)
    # # fsaf
    logprob = dist.log_prob(torch.tensor([x]))
    prob = torch.exp(logprob)

    samp = dist.rsample()
    logprob2 = dist.log_prob(samp.detach())
    grad_logprob = torch.autograd.grad(outputs=logprob2, inputs=(theta), retain_graph=False)[0]
    # print(grad_logprob)
    # fdsf

    probs.append(to_print2(prob)[0])
    grads.append(to_print2(grad_logprob)[0])
    samps.append(to_print2(samp)[0])
# fsdafas
# print (probs)





col =0
row = 1
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.scatter(samps, grads)
ax.set_xlim(-10., 10.)
# ax.set_title('With sigmoid')

save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'distplot.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()


fasdfa















# def sigmoid(x):
#     return 1. / (1. + np.exp(-x))

# def my_logprob_v4(y, theta):
#     return np.log((theta - theta**2)/ (-2*theta*y + theta + y)**2)

# # are these logistic or gumbel??
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
# z_samps = []
# sig_samps = []
# theta = .99
# temp = 5. #doesnt work for my sigmoid logprob since I didnt derive with temp, needs to be 1
# for i in range(400):
#     y = my_sample_nosigmoid(theta, temp=temp)
#     z_samps.append(y)
#     sig_samps.append(sigmoid(y))
#     samps.append(float(y>0))
# print (np.mean(samps))
# print (np.mean(z_samps))


dist = RelaxedBernoulli(temperature=torch.Tensor([1.]), probs=torch.tensor([.2]))

rows = 1
cols = 1
fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

col =0
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

probs = []
xs = np.linspace(.0001,.9999, 40)
for x in xs:
    # print (dist.sample())
    # print (dist.log_prob(dist.sample()))
    # # # fsa
    # prob = torch.exp(dist.log_prob(dist.sample()))
    # # print (prob)
    # # fsaf
    prob = torch.exp(dist.log_prob(torch.tensor([x])))
    probs.append(to_print2(prob)[0])
# fsdafas
# print (probs)
ax.plot(xs, probs)

# ax.set_title('With sigmoid')

save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'distplot.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()


fasdfa













# sum_ = 0
# for i in range(20):
#     sum_+= i+5
# print (sum_)
# fds 290


# print (m.log_prob(2.))
xs = np.linspace(.001,.999, 30)

# dist = RelaxedBernoulli(temperature=torch.Tensor([1.]), logits=torch.tensor([0.]))
dist = RelaxedBernoulli(temperature=torch.Tensor([.2]), probs=torch.tensor([0.3]))

ys = []
samps = []
for x in xs:
    # samp = dist.sample()
    # samps.append(samp.data.numpy()[0])
    # print (samp)
    # print (torch.exp(dist.log_prob(samp)))
    # print ()
    # print (m.log_prob(x))

    prob = torch.exp(dist.log_prob(torch.tensor([x]))).numpy()[0]
    # print (x)
    # print (prob)
    # component_i = ().numpy()[0]
    ys.append(prob)

samps = []
b_samps = []
for i in range(1000):
    samp = dist.sample()
    samps.append(samp.data.numpy()[0]) 
    b_samps.append(samp.data.numpy()[0] > .5)
ax.hist(samps, density=True, bins=30)

ax.plot(xs, ys)
ax.set_title('.2')
print (np.mean(samps))
print (np.mean(b_samps))
print ()





col =1
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

dist = RelaxedBernoulli(temperature=torch.Tensor([1.]), probs=torch.tensor([0.3]))

ys = []
samps = []
for x in xs:
    prob = torch.exp(dist.log_prob(torch.tensor([x]))).numpy()[0]
    ys.append(prob)

samps = []
b_samps = []
for i in range(1000):
    samp = dist.sample()
    samps.append(samp.data.numpy()[0]) 
    b_samps.append(samp.data.numpy()[0] > .5)
ax.hist(samps, density=True, bins=30)

ax.plot(xs, ys)
ax.set_title('1.')
print (np.mean(samps))
print (np.mean(b_samps))
print ()




col =2
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

dist = RelaxedBernoulli(temperature=torch.Tensor([5.]), probs=torch.tensor([0.3]))

ys = []
samps = []
for x in xs:
    prob = torch.exp(dist.log_prob(torch.tensor([x]))).numpy()[0]
    ys.append(prob)

samps = []
b_samps = []
for i in range(1000):
    samp = dist.sample()
    samps.append(samp.data.numpy()[0]) 
    b_samps.append(samp.data.numpy()[0] > .5)
ax.hist(samps, density=True, bins=30)

ax.plot(xs, ys)
ax.set_title('5.')
print (np.mean(samps))
print (np.mean(b_samps))







# # sum_ = np.zeros(len(xs))

# C = 20
# for c in range(C):


#     m = Normal(torch.tensor([c*10.]).float(), torch.tensor([5.0]).float())

    
#     # xs = torch.tensor(xs)
#     # print (m.log_prob(lin))

#     ys = []
#     for x in xs:
#         # print (m.log_prob(x))
#         component_i = (torch.exp(m.log_prob(x) )* ((c+5.) / 290.)).numpy()
#         ys.append(component_i)

#     ys = np.reshape(np.array(ys), [-1])

#     # print (sum_.shape)
#     # print (ys.shape)

#     sum_ += ys

#     # print (sum_)
#     # fsdfa


#     ax.plot(xs, ys, label='')


#     # ax.grid(True, alpha=.3)
#     # ax.set_title(r'$f=(b-.4)^2$', size=6, family='serif')
#     # ax.tick_params(labelsize=6)
#     # ax.set_ylabel(ylabel, size=6, family='serif')
#     # ax.set_xlabel(xlabel, size=6, family='serif')
#     # ax.legend(prop={'size':5}) #, loc=2)  #upper left


# ax.plot(xs, sum_, label='')




save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'relaxedbernoulli_plot.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()
















