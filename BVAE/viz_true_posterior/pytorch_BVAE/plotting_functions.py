

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from utils import lognormal4 
from utils import log_bernoulli


def plot_isocontours(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1.):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)
    # print '111'
    # print aaa
    bbb = func(aaa)
    # bbb = Variable(bbb)
    # if bbb.data:
    #     bbb = bbb.data
    # bbb = bbb.data.numpy()
    zs = torch.exp(bbb)

    # Z = zs.reshape(X.shape)
    Z = zs.view(X.shape)
    Z=Z.numpy()
    plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')


def plot_isocontoursNoExp(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)
    # print '111'
    # print aaa
    bbb = func(aaa)
    # bbb = Variable(bbb)
    # if bbb.data:
    #     bbb = bbb.data
    # bbb = bbb.data.numpy()
    # zs = torch.exp(bbb)
    zs = bbb


    # Z = zs.reshape(X.shape)
    Z = zs.view(X.shape)
    Z=Z.numpy()
    cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)

    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4) 
        # plt.legend(nm, fontsize=4) 



    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')



def plot_isocontours2(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)
    # print '111'
    # print aaa
    bbb = func(aaa)
    # bbb = Variable(bbb)
    # if bbb.data:
    #     bbb = bbb.data
    # print bbb
    zs = bbb.data.numpy()
    # zs = np.exp(bbb)

    Z = zs.reshape(X.shape)
    # Z = zs.view(X.shape)
    # Z=Z.numpy()
    cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)

    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4) 


    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')


def plot_isocontours2_exp(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)
    # print '111'
    # print aaa
    bbb = func(aaa)
    # bbb = Variable(bbb)
    # if bbb.data:
    #     bbb = bbb.data
    # print bbb
    zs = bbb.data.numpy()
    zs = np.exp(zs/784)

    Z = zs.reshape(X.shape)
    # Z = zs.view(X.shape)
    # Z=Z.numpy()
    cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)

    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4) 


    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')










def plot_isocontours_expected(ax, model, data, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1.):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)

    n_samps = 20000
    if len(data) < n_samps:
        n_samps = len(data)


    for samp_i in range(n_samps):
        if samp_i % 1000 == 0:
            print samp_i
        mean, logvar = model.encode(Variable(torch.unsqueeze(data[samp_i],0)))
        func = lambda zs: lognormal4(torch.Tensor(zs), torch.squeeze(mean.data), torch.squeeze(logvar.data))
        bbb = func(aaa)
        zs = torch.exp(bbb)
        if samp_i ==0:
            sum_of_all = zs
        else:
            sum_of_all = sum_of_all + zs

    avg_of_all = sum_of_all / n_samps

    Z = avg_of_all.view(X.shape)
    Z=Z.numpy()
    plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')











def plot_isocontours_expected_W(ax, model, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1.):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)

    n_samps = 20000
    if len(data) < n_samps:
        n_samps = len(data)


    for samp_i in range(n_samps):
        if samp_i % 1000 == 0:
            print samp_i
        mean, logvar = model.encode(Variable(torch.unsqueeze(data[samp_i],0)))
        func = lambda zs: lognormal4(torch.Tensor(zs), torch.squeeze(mean.data), torch.squeeze(logvar.data))
        bbb = func(aaa)
        zs = torch.exp(bbb)
        if samp_i ==0:
            sum_of_all = zs
        else:
            sum_of_all = sum_of_all + zs

    avg_of_all = sum_of_all / n_samps

    Z = avg_of_all.view(X.shape)
    Z=Z.numpy()
    plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')






def plot_isocontours_expected_W(ax, model, samp, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)


    for i in range(50):
        if i % 10 ==0: print i

        Ws, logpW, logqW = model.sample_W()  #_ , [1], [1]   
        func = lambda zs: log_bernoulli(model.decode(Ws, Variable(torch.unsqueeze(zs,1))), Variable(torch.unsqueeze(samp,0)))+ Variable(torch.unsqueeze(lognormal4(torch.Tensor(zs), torch.zeros(2), torch.zeros(2)), 1))


        bbb = func(aaa)
        zs = bbb.data.numpy()
        zs = np.exp(zs/784)

        if i ==0:
            sum_of_all = zs
        else:
            sum_of_all = sum_of_all + zs


    avg_of_all = sum_of_all / 50

    Z = avg_of_all.reshape(X.shape)
    # Z = zs.view(X.shape)
    # Z=Z.numpy()
    cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)

    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4) 


    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')

















