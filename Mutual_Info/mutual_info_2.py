

#trains MINE, not biases-corrected, doesnt work for D=20, but is fine for D=1



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


# from scipy.stats import multivariate_normal

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
# fsd

def logmeanexp(input_):
    # [B,1] -> [1]
    max_ = torch.max(input_)
    out = torch.log(torch.mean(torch.exp(input_ - max_), 0)) + max_
    return out






def samp_cond_gaus(n, p):

    x1 = np.random.normal(loc=0, scale=1, size=n)

    cov = [[1., p], [p, 1.]]
    cov_inv = np.linalg.inv(cov)
    x2_mean = -x1*cov_inv[0,1]/cov_inv[0,0]

    x2 = np.random.normal(loc=0, scale=1, size=n)*np.sqrt(1./cov_inv[0,0]) + x2_mean
    return np.concatenate([np.expand_dims(x1,1),np.expand_dims(x2,1)], 1)


def samp_cond_gaus_2(n, p, cov_inv):

    x1 = np.random.normal(loc=0, scale=1, size=n)
    x2_mean = -x1*cov_inv[0,1]/cov_inv[0,0]
    x2 = np.random.normal(loc=0, scale=1, size=n)*np.sqrt(1./cov_inv[0,0]) + x2_mean
    return np.concatenate([np.expand_dims(x1,1),np.expand_dims(x2,1)], 1)




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






class ResBlock(nn.Module):
    def __init__(self, in_features):
        super(ResBlock, self).__init__()
     
        n_layers = 3
        L = in_features
        block = []
        for i in range(n_layers):

            block += [  nn.Linear(L,L),
                    nn.LeakyReLU(),]

        block += [ nn.Linear(L,L)]

        self.sequential = nn.Sequential(*block)

    def forward(self, x):
        return x + self.sequential(x)



class Net2(nn.Module):
    def __init__(self, in_features):
        super(Net2, self).__init__()

        L = 200
        n_layers = 3

        block = [  nn.Linear(in_features,L),
                        nn.LeakyReLU(),]        

        for i in range(n_layers):

            block += [ResBlock(L)]

        block += [ nn.Linear(L,1)]

        self.sequential = nn.Sequential(*block)

    def forward(self, x):
        return self.sequential(x)



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

# do_plot()


# fsdfa
np.random.seed(seed=0)
torch.cuda.manual_seed_all(seed=0)




D = 20
B = 64
p = .79 #.99 #.2 #.79 #.9 # .2

MI = get_MI(p=p, D=D)
print ('MI is:', MI)


cov = [[1., p], [p, 1.]]
cov_inv = np.linalg.inv(cov)

# mean = np.zeros(D) 
# cov = np.ones([D,D]) * np.eye(D)
# cov = [[1., p], [p, 1.]]

# samp_cond_gaus(mean, cov)

# fasd

# rv = multivariate_normal(mean, cov)

# fasds


net = Net2(D*2)
net.cuda()

lr = .000004
# optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=.0000000001)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.)


# ema = 0.
# alpha = .9

step = 0
max_steps = 100000
while step < max_steps:

    # samps = rv.rvs(size=B)
    batch = []
    for i in range(B):
        samp = samp_cond_gaus_2(n=D, p=p, cov_inv=cov_inv)
        batch.append(samp)
    batch = np.array(batch)
    batch = torch.from_numpy(batch).cuda().float() #[B,D,2]

    x1 = batch[:,:,0]
    x2 = batch[:,:,1]
    # print (x1)
    # print (x2)

    batch_v1 = torch.cat([x1,x2], 1)
    # print (batch_v1)
    # print (batch_v1.shape)

    # print (x2)
    # rand = torch.randperm(B)
    # print (rand)
    # print (x2[rand])
    rand = np.array(range(B))+1
    rand[-1] = 0


    # print (rand)
    # print (x2[rand])
    # fsaa




    shuffled_y = x2[rand]
    # print (shuffled_y)
    batch_shuf = torch.cat([x1,shuffled_y], 1)
    # print (batch_shuf)
    # print (batch_shuf.shape)
    # fdsa

    # print (batch_v1)
    # print(batch_shuf)
    # fsda

    
    # shuffled_samps = torch.cat([samps[:,0].view(B,1), shuffled_y.view(B,1)],1)

    # if step % 200 ==0 and step!=0:
    #     print (step, obj.data.cpu().numpy(), val1.data.cpu().numpy(), val2.data.cpu().numpy(), MI)

    optimizer.zero_grad()

    output = net(batch_v1)
    output_shuf = net(batch_shuf)
    # obj = torch.mean(output) - torch.log(torch.mean(torch.exp(output_shuf)))
    val1 = torch.mean(output)    
    # val2 = torch.log(torch.clamp(torch.mean(torch.exp(output_shuf)), min=1e-5))


    # MINE
    # val2 = logmeanexp(output_shuf)[0]
    # f-divergence / MINE-f
    val2 = torch.mean(torch.exp(output_shuf-1.))


    #EMA attempt 
    # cur = torch.exp(val2.detach())
    # ema = ema + alpha*(cur - ema)
    # obj = val1 - val2*(cur/ema)


    obj = val1 - val2
    
    loss = -obj

    loss.backward()
    optimizer.step()

    if step % 100 ==0:
        print (step, obj.data.cpu().numpy(), val1.data.cpu().numpy(), torch.mean(output_shuf).data.cpu().numpy(), val2.data.cpu().numpy(), MI)

    step +=1




print ('Done.')


















