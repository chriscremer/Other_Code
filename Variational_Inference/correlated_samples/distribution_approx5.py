



# L_M and L_arc

#used to chekc to see if the elbos are differnet given same parameters

#Result:
# L_M: about -.50
# L_AR: about -.52


import numpy as np


import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from os.path import expanduser
home = expanduser("~")


import matplotlib.pyplot as plt

import math


def plot_isocontours(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)
    bbb = func(Variable(aaa))

    bbb = bbb.data
    zs = torch.exp(bbb)
    Z = zs.view(X.shape)
    Z=Z.numpy()
    cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)
    # if legend:
    #     nm, lbl = cs.legend_elements()
    #     plt.legend(nm, lbl, fontsize=4, bbox_to_anchor=(0.7, 0.1)) 
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')



def lognormal4_2(x, mean, logvar):
    '''
    x: [B,X]
    mean,logvar: [X]
    output: [B]
    '''
    D = x.size()[1]
    term1 = D * torch.log(torch.FloatTensor([2.*math.pi])) #[1]
    return -.5 * (Variable(term1) + logvar.sum(0) + ((x - mean).pow(2)/torch.exp(logvar)).sum(1))


def logpz(z):
    # p(z): function that takes in [B,Z], Outputs: [B,1]

    cccc = lognormal4_2(z, Variable(torch.zeros(2)+2), Variable(torch.zeros(2)))

    aaa = torch.clamp(cccc, min=-30)
    # print aaa
    # bbb = torch.clamp(lognormal4_2(torch.Tensor(z), Variable(torch.zeros(2)), Variable(torch.zeros(2))), min=-30)
    bbb = torch.clamp(lognormal4_2(z, Variable(torch.zeros(2)), Variable(torch.zeros(2))), min=-30)

    return torch.log(.5*torch.exp(aaa) + .5*torch.exp(bbb))
    # return lognormal4_2(z, Variable(torch.zeros(2)+2), Variable(torch.zeros(2)))




# def plot_it(model):

#     plt.cla()

#     ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
#     plot_isocontours(ax, logpz, cmap='Blues')

#     func = lambda zs: lognormal4_2(zs, model.mean, model.logvar)
#     plot_isocontours(ax, func, cmap='Reds')

#     plt.draw()
#     plt.pause(1.0/30.0)






def plot_it2(e, model, elbo):

    plt.cla()

    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
    plot_isocontours(ax, logpz, cmap='Blues')

    func = lambda zs: lognormal4_2(zs, model.mean1, model.logvar1)
    plot_isocontours(ax, func, cmap='Reds')

    # func = lambda zs: lognormal4_2(zs, model.mean2, model.logvar2)
    # plot_isocontours(ax, func, cmap='Greens')

    func = lambda zs: lognormal4_2(zs, (model.mean1*model.linear_transform)+model.bias_transform, torch.log(torch.exp(model.logvar2) + model.linear_transform.pow(2)*torch.exp(model.logvar1)))
    plot_isocontours(ax, func, cmap='Greens')   

    ax.annotate('iter:'+str(e), xytext=(.1, 1.), xy=(0, 1), textcoords='axes fraction')
    ax.annotate('elbo'+str(elbo), xytext=(.1, .95), xy=(0, 1), textcoords='axes fraction')



    # plt.draw()
    # plt.pause(1.0/30.0)

    plt.savefig(home+'/Documents/tmp/'+str(e)+'thing.png')
    print 'Saved fig'


def train(model, 
            path_to_load_variables='', path_to_save_variables='', 
            epochs=10, batch_size=20, display_epoch=2, k=1):
    

    if path_to_load_variables != '':
        model.load_state_dict(torch.load(path_to_load_variables))
        print 'loaded variables ' + path_to_load_variables

    optimizer = optim.Adam(model.params, lr=.05)

    for epoch in range(1, epochs + 1):

        optimizer.zero_grad()

        elbo, logpz, logqz = model.forward(k=k)
        loss = -(elbo)

        loss.backward()
        # optimizer.step()

        if epoch%display_epoch==0:
            test_score =test(model,k=k,batch_size=20000)
            print 'Train Epoch: {}/{}'.format(epoch, epochs), \
                'Loss:{:.4f}'.format(loss.data[0]), \
                'logpz:{:.4f}'.format(logpz.data[0]), \
                'logqz:{:.4f}'.format(logqz.data[0]), \
                'test', test_score

            plot_it2(epoch, model, test_score)

            # plt.savefig(home+'/Documents/tmp/thing'+str(epoch)+'.png')
            # print 'Saved fig'

            print model.linear_transform.data, model.bias_transform.data, model.logvar2.data
            print model.mean1, model.logvar1


    if path_to_save_variables != '':
        torch.save(model.state_dict(), path_to_save_variables)
        print 'Saved variables to ' + path_to_save_variables




def test(model, path_to_load_variables='', batch_size=300, display_epoch=4, k=10):
    

    if path_to_load_variables != '':
        model.load_state_dict(torch.load(path_to_load_variables))
        print 'loaded variables ' + path_to_load_variables

    elbos = []
    data_index= 0
    for i in range(batch_size):

        # batch = data_x[data_index:data_index+batch_size]
        # data_index += batch_size

        elbo, logpz, logqz = model(k=k)
        elbos.append(elbo.data[0])

        # if i%display_epoch==0:
        #     print i,len(data_x)/ batch_size, elbo.data[0]

    return np.mean(elbos)






# class IWG(nn.Module):
#     #Importance Weighted Gaussian
#     def __init__(self, dim, logpz):
#         super(IWG, self).__init__()

#         torch.manual_seed(1000)

#         self.z_size = dim

#         self.mean = Variable(torch.zeros(self.z_size), requires_grad=True)
#         self.logvar = Variable(torch.randn(self.z_size)-3., requires_grad=True)
#         self.params = [self.mean, self.logvar]
#         self.logpz = logpz


#     def sample(self, mu, logvar, k):
#         eps = Variable(torch.FloatTensor(k, self.z_size).normal_()) #[P,Z]
#         z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,Z]
#         logqz = lognormal4_2(z, mu.detach(), logvar.detach())
#         return z, logqz


#     def forward(self, k=1):
        
#         z, logqz = self.sample(self.mean, self.logvar, k=k) #[P,B,Z], 
#         logpz = self.logpz(z) 

#         elbo = logpz - logqz  #[P,B]

#         if k>1:
#             max_ = torch.max(elbo, 0)[0] #[B]
#             elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

#         elbo = torch.mean(elbo) #[1]

#         logpz = torch.mean(logpz)
#         logqz = torch.mean(logqz)

#         return elbo, logpz, logqz



variables = {}
variables['mean1'] = [ 0.5752, 0.5130]
variables['logvar1'] = [  0.4593, 0.4308]
variables['linear_transform'] = [-0.0112, 0.1881]
variables['bias_transform'] = [0.8124, 0.8293]
variables['logvar2'] = [0.4942, 0.4098]




class MCGS(nn.Module):
    #Multiple Covaried Gaussian Samples 
    def __init__(self, dim, logpz):
        super(MCGS, self).__init__()

        torch.manual_seed(1000)

        self.z_size = dim

        # self.mean1 = Variable(torch.zeros(self.z_size), requires_grad=True)
        # self.logvar1 = Variable(torch.randn(self.z_size)-3., requires_grad=True)

        # self.linear_transform = Variable(torch.zeros(self.z_size), requires_grad=True)
        # self.bias_transform = Variable(torch.zeros(self.z_size), requires_grad=True)
        # self.logvar2 = Variable(torch.randn(self.z_size)-3., requires_grad=True)

        self.mean1 = Variable(torch.Tensor(variables['mean1']), requires_grad=True)
        self.logvar1 = Variable(torch.Tensor(variables['logvar1']), requires_grad=True)

        self.linear_transform = Variable(torch.Tensor(variables['linear_transform']), requires_grad=True)
        self.bias_transform = Variable(torch.Tensor(variables['bias_transform']), requires_grad=True)
        self.logvar2 = Variable(torch.Tensor(variables['logvar2']), requires_grad=True)


        self.params = [self.mean1, self.logvar1, self.linear_transform, self.bias_transform, self.logvar2]


        self.logpz = logpz


    def sample(self, mu, logvar):
        eps = Variable(torch.FloatTensor(1, self.z_size).normal_()) #[1,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mu  #[1,Z]
        logqz = lognormal4_2(z, mu.detach(), logvar.detach())
        return z, logqz


    def forward(self, k=1):
        
        z, logqz = self.sample(self.mean1, self.logvar1) #[1,Z]
        # z = torch.squeeze(z) #[Z]
        z2, logqz2 = self.sample((z*self.linear_transform)+self.bias_transform, self.logvar2) #[1,Z]

        #comment this out to get l_arc
        #We want z2 under marginal q2
        # logqz2 = lognormal4_2(z2, (self.mean1*self.linear_transform)+self.bias_transform, torch.log(torch.exp(self.logvar2) + self.linear_transform.pow(2)*torch.exp(self.logvar1)))



        # z = torch.unsqueeze(z,0)  #this gave a weird bug where the variacne was less for some reason on q2
        logpz = self.logpz(z) 
        logpz2 = self.logpz(z2) 

        logpz = logpz + logpz2
        logqz = logqz + logqz2

        elbo = logpz - logqz  #[P,B]

        # if k>1:
        #     max_ = torch.max(elbo, 0)[0] #[B]
        #     elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

        elbo = torch.mean(elbo) #[1]

        logpz = torch.mean(logpz)
        logqz = torch.mean(logqz)

        return elbo, logpz, logqz






# class LF(nn.Module):
#     #LF
#     def __init__(self, dim, logpz):
#         super(LF, self).__init__()

#         torch.manual_seed(1000)

#         self.z_size = dim

#         self.mean1 = Variable(torch.zeros(self.z_size), requires_grad=True)
#         self.logvar1 = Variable(torch.randn(self.z_size)-3., requires_grad=True)

#         # self.linear_transform = Variable(torch.zeros(self.z_size), requires_grad=True)
#         # self.bias_transform = Variable(torch.zeros(self.z_size), requires_grad=True)
#         self.mean2 = Variable(torch.zeros(self.z_size), requires_grad=True)
#         self.logvar2 = Variable(torch.randn(self.z_size)-3., requires_grad=True)
#         # self.logvar2 = self.logvar1



#         # self.params = [self.mean1, self.logvar1, self.linear_transform, self.bias_transform, self.logvar2]
#         self.params = [self.mean1, self.logvar1, self.mean2, self.logvar2]#, self.logvar2]


#         self.logpz = logpz


#     def sample(self, mu, logvar):
#         eps = Variable(torch.FloatTensor(1, self.z_size).normal_()) #[1,Z]
#         z = eps.mul(torch.exp(.5*logvar)) + mu  #[1,Z]
#         logqz = lognormal4_2(z, mu.detach(), logvar.detach())
#         return z, logqz


#     def forward(self, k=1):
        
#         z, logqz = self.sample(self.mean1, self.logvar1) #[1,Z]
#         # z = torch.squeeze(z) #[Z]
#         z2, logqz2 = self.sample(self.mean2, self.logvar2) #[1,Z]

#         #We want z2 under marginal q2
#         # z2 = torch.unsqueeze(z2,0)
#         # logqz2 = lognormal4_2(z2, (self.mean1*self.linear_transform)+self.bias_transform, self.logvar2 + self.linear_transform.pow(2)*torch.exp(self.logvar1))
#         # logqz2 = lognormal4_2(z2, (self.mean1*self.linear_transform)+self.bias_transform, torch.log(torch.exp(self.logvar2) + self.linear_transform.pow(2)*torch.exp(self.logvar1)))



#         # z = torch.unsqueeze(z,0)
#         logpz = self.logpz(z) 
#         logpz2 = self.logpz(z2) 

#         logpz = logpz + logpz2
#         logqz = logqz + logqz2

#         elbo = logpz - logqz  #[P,B]

#         # if k>1:
#         #     max_ = torch.max(elbo, 0)[0] #[B]
#         #     elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

#         elbo = torch.mean(elbo) #[1]

#         logpz = torch.mean(logpz)
#         logqz = torch.mean(logqz)

#         return elbo, logpz, logqz

















rows = 1
cols = 1
fig = plt.figure(figsize=(4+cols,4+rows), facecolor='white')
plt.ion()
plt.show(block=False)



# model = IWG(dim=2, logpz=logpz)
model = MCGS(dim=2, logpz=logpz)
# model = LF(dim=2, logpz=logpz)



path_to_load_variables=''
# path_to_load_variables=home+'/Documents/tmp/pytorch_first.pt'
# path_to_save_variables=home+'/Documents/tmp/pytorch_first.pt'
path_to_save_variables=''

train(model=model, 
            path_to_load_variables=path_to_load_variables, 
            path_to_save_variables=path_to_save_variables, 
            epochs=1000, batch_size=4, display_epoch=40, k=10)






























































