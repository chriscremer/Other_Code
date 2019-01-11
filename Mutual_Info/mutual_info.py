
# this compares 2 ways of sampling conditinal gaussian samples



from os.path import expanduser
home = expanduser("~")

import sys
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath(home +'/anaconda3/envs/test_env/bin'))

# print (sys.path)

# import torch
# print (torch.__version__)

# fas



import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# print (matplotlib.__version__)


from scipy.stats import multivariate_normal

# import scipy
# print (scipy.__version__)

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# print (torch.__version__)


def get_MI(p, D):
    MI = (-D/2) * np.log(1-p**2)
    return MI

def get_p(MI, D):
    p = np.sqrt(-(np.exp((-2*MI)/D)-1))
    return p

# print(get_MI(p=.9, D=2))
# print(get_p(MI=10, D=20))



def samp_cond_gaus(n, p):



    x1 = np.random.normal(loc=0, scale=1, size=n)

    cov = [[1., p], [p, 1.]]
    cov_inv = np.linalg.inv(cov)

    # print(cov_inv)

    x2_mean = -x1*cov_inv[0,1]/cov_inv[0,0]

    x2 = np.random.normal(loc=0, scale=1, size=n)*np.sqrt(1./cov_inv[0,0]) + x2_mean

    # print()
    # print (x1)
    # print (x2_mean)

    # fsdsd

    # x2 = np.random.normal(loc=0, scale=1, size=D)
    # rv2 = multivariate_normal(mean, cov)
    # rv2.rvs()

    # fdsda

    return x1, x2




def do_plot():

    rows = 1
    cols = 2
    fig = plt.figure(figsize=(6+cols,4+rows), facecolor='white', dpi=150)   
    


    # x = np.linspace(0, 5, 10, endpoint=False)
    # y = multivariate_normal.pdf(x, mean=2.5, cov=0.5)
    # plt.plot(x, y)

    p = .9

    lim = 4
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y

    mean = [0., 0.]
    cov = [[1.0, p], [p, 1.0]]
    rv = multivariate_normal(mean, cov)

    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False, colspan=1, rowspan=1)
    

    samp = rv.rvs(size=1000)
    # print (samp.shape)
    plt.scatter(samp[:,0], samp[:,1], alpha=.3, s=5)
    plt.contour(x, y, rv.pdf(pos), cmap='Blues')
    ax.axis('equal')
    ax.grid(True, alpha=.3)
    # ax.set_ylim([-lim, lim])
    # ax.set_xlim([-lim, lim])
    ax.set_xlim(xmin=-lim, xmax=lim)
    ax.set_ylim(ymin=-lim, ymax=lim)
    ax.set_title('Sampling using scipy')


    x2,y2 = samp_cond_gaus(n=1000, p=.9)

    ax = plt.subplot2grid((rows,cols), (0,1), frameon=False, colspan=1, rowspan=1)
    
    plt.scatter(x2, y2, alpha=.3, s=5)
    plt.contour(x, y, rv.pdf(pos), cmap='Blues')
    ax.axis('equal')
    ax.grid(True, alpha=.3)
    # ax.set_ylim([-lim, lim])
    # ax.set_xlim(left=-lim, right=lim)
    ax.set_xlim(xmin=-lim, xmax=lim)
    ax.set_ylim(ymin=-lim, ymax=lim)
    ax.set_title('Sampling via conditional Gaussian')

    # plt.show()
    # fsadssa

    # plt_path = home+'/Downloads/plt.png'
    plt_path = home+'/Documents/Mutual_Info/plt.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()

    heyhey













class Net(nn.Module):
    def __init__(self, in_features):
        super(Net, self).__init__()

        L = 200
        n_layers = 3

        block = [  nn.Linear(in_features,L),
                        nn.LeakyReLU(),]        

        for i in range(n_layers):

            block += [  nn.Linear(L,L),
                    nn.LeakyReLU(),]

        block += [ nn.Linear(L,1)]

        self.sequential = nn.Sequential(*block)

    def forward(self, x):
        return self.sequential(x)


# print (np.eye(2).T)

# fsad

do_plot()


fsdfa




D = 4
B = 128 #64
p = .9 # .2

MI = get_MI(p=p, D=D)
print ('MI is:', MI)

mean = np.zeros(D) 
# cov = np.ones([D,D]) * np.eye(D)
cov = [[1., p], [p, 1.]]

samp_cond_gaus(mean, cov)

fasd


rv = multivariate_normal(mean, cov)












fasds


net = Net(D)
net.cuda()


lr = .0004
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=.0000001)





fsafs

step = 0
max_steps = 10000
while step < max_steps:

    samps = rv.rvs(size=B)
    samps = torch.from_numpy(samps).cuda().float()

    rand = torch.randperm(B)
    shuffled_y = samps[:,1][rand]
    shuffled_samps = torch.cat([samps[:,0].view(B,1), shuffled_y.view(B,1)],1)


    optimizer.zero_grad()

    output = net(samps)
    output_shuf = net(shuffled_samps)
    obj = torch.mean(output) - torch.log(torch.mean(torch.exp(output_shuf)))
    loss = -obj

    loss.backward()
    optimizer.step()

    if step % 50 ==0:
        print (step, obj.data.cpu().numpy(), MI)

    step +=1



print ('Done.')


















