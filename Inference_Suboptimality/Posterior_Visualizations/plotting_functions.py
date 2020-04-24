

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from utils import lognormal4 
from utils import log_bernoulli

import scipy.stats as st

def plot_isocontours(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
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
    zs = torch.exp(bbb)

    # Z = zs.reshape(X.shape)
    Z = zs.view(X.shape)
    Z=Z.numpy()


    zs_sum = np.sum(Z)
    Z = Z / zs_sum



    # print 'prior sum:', np.sum(Z)

    cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)


    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4, bbox_to_anchor=(0.7, 0.1)) 

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




def plot_isocontours2_exp_norm(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
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


    # zs = bbb.data.numpy()

    zs = bbb.data.cpu().numpy()

    # zs = np.exp(zs/784)




    max_ = np.max(zs)
    zs_sum = np.log(np.sum(np.exp(zs-max_))) + max_

    zs = zs - zs_sum
    zs = np.exp(zs)

    # print np.max(zs)
    # fadsf


    Z = zs.reshape(X.shape)
    # Z = zs.view(X.shape)
    # Z=Z.numpy()
    # cs = plt.contourf(X, Y, Z, cmap=cmap, alpha=alpha)
    cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)


    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4) 


    # ax.patch.set_edgecolor('black')

    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')









def plot_isocontours_expected(ax, model, data, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)

    n_samps = 10
    if len(data) < n_samps:
        n_samps = len(data)


    for samp_i in range(n_samps):
        # if samp_i % 1000 == 0:
        #     print samp_i
        mean, logvar = model.encode(Variable(torch.unsqueeze(data[samp_i],0)))
        func = lambda zs: lognormal4(torch.Tensor(zs), torch.squeeze(mean.data), torch.squeeze(logvar.data))
        # print aaa.size()
        bbb = func(aaa)

        # print 'sum:1', torch.sum(bbb)
        ddd = torch.exp(bbb)

        # print 'sum:', torch.sum(ddd)
        # print ddd.size()



        # fdsa

        if samp_i ==0:
            sum_of_all = ddd
        else:
            sum_of_all = sum_of_all + ddd

    avg_of_all = sum_of_all / n_samps

    Z = avg_of_all.view(X.shape)
    Z=Z.numpy()

    # print 'sum:', np.sum(Z)

    cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)


    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4) 


    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')


    return Z









def plot_isocontours_expected_norm(ax, model, data, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False, n_samps=10, cs_to_use=None):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)

    # n_samps = 10
    if len(data) < n_samps:
        n_samps = len(data)


    for samp_i in range(n_samps):
        # if samp_i % 1000 == 0:
        #     print samp_i
        mean, logvar = model.encode(Variable(torch.unsqueeze(data[samp_i],0)))
        func = lambda zs: lognormal4(torch.Tensor(zs), torch.squeeze(mean.data), torch.squeeze(logvar.data))
        # print aaa.size()
        bbb = func(aaa)

        # print 'sum:1', torch.sum(bbb)
        # ddd = torch.exp(bbb)

        # print 'sum:', torch.sum(ddd)
        # print ddd.size()




        # bbb = func(aaa)
        zs = bbb.numpy()
        max_ = np.max(zs)
        zs_sum = np.log(np.sum(np.exp(zs-max_))) + max_
        zs = zs - zs_sum
        ddd = np.exp(zs)




        if samp_i ==0:
            sum_of_all = ddd
        else:
            sum_of_all = sum_of_all + ddd

    avg_of_all = sum_of_all / n_samps

    Z = avg_of_all.reshape(X.shape)
    # Z = avg_of_all.view(X.shape)
    # Z=Z.numpy()

    # print 'sum:', np.sum(Z)

    if cs_to_use != None:
        cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha, levels=cs_to_use.levels)
    else:
        cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)

    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4) 


    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')


    return Z, cs




def plot_isocontours_expected_norm_ind(ax, model, data, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False, n_samps=10, cs_to_use=None):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)

    # n_samps = 10
    if len(data) < n_samps:
        n_samps = len(data)


    for samp_i in range(n_samps):
        # if samp_i % 1000 == 0:
        #     print samp_i
        mean, logvar = model.encode(Variable(torch.unsqueeze(data[samp_i],0)))
        func = lambda zs: lognormal4(torch.Tensor(zs), torch.squeeze(mean.data), torch.squeeze(logvar.data))
        # print aaa.size()
        bbb = func(aaa)

        zs = bbb.numpy()
        max_ = np.max(zs)
        zs_sum = np.log(np.sum(np.exp(zs-max_))) + max_
        zs = zs - zs_sum
        ddd = np.exp(zs)
        Z = ddd
        Z = Z.reshape(X.shape)
        if cs_to_use != None:
            cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha, levels=cs_to_use.levels)
        else:
            cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)


    #     if samp_i ==0:
    #         sum_of_all = ddd
    #     else:
    #         sum_of_all = sum_of_all + ddd

    # avg_of_all = sum_of_all / n_samps

    # Z = avg_of_all.reshape(X.shape)
    # print 'sum:', np.sum(Z)


    # if legend:
    #     nm, lbl = cs.legend_elements()
    #     plt.legend(nm, lbl, fontsize=4) 


    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')


    return Z, cs




# def plot_isocontours_expected_W(ax, model, xlimits=[-6, 6], ylimits=[-6, 6],
#                      numticks=101, cmap=None, alpha=1.):
#     x = np.linspace(*xlimits, num=numticks)
#     y = np.linspace(*ylimits, num=numticks)
#     X, Y = np.meshgrid(x, y)
#     # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
#     aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)

#     n_samps = 20000
#     if len(data) < n_samps:
#         n_samps = len(data)


#     for samp_i in range(n_samps):
#         if samp_i % 1000 == 0:
#             print samp_i
#         mean, logvar = model.encode(Variable(torch.unsqueeze(data[samp_i],0)))
#         func = lambda zs: lognormal4(torch.Tensor(zs), torch.squeeze(mean.data), torch.squeeze(logvar.data))
#         bbb = func(aaa)
#         zs = torch.exp(bbb)
#         if samp_i ==0:
#             sum_of_all = zs
#         else:
#             sum_of_all = sum_of_all + zs

#     avg_of_all = sum_of_all / n_samps

#     Z = avg_of_all.view(X.shape)
#     Z=Z.numpy()
#     plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)
#     ax.set_yticks([])
#     ax.set_xticks([])
#     plt.gca().set_aspect('equal', adjustable='box')






def plot_isocontours_expected_W(ax, model, samp, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)

    n_Ws = 10

    for i in range(n_Ws):
        # if i % 10 ==0: print i

        Ws, logpW, logqW = model.sample_W()  #_ , [1], [1]   
        func = lambda zs: log_bernoulli(model.decode(Ws, Variable(torch.unsqueeze(zs,1))), Variable(torch.unsqueeze(samp,0)))+ Variable(torch.unsqueeze(lognormal4(torch.Tensor(zs), torch.zeros(2), torch.zeros(2)), 1))


        bbb = func(aaa)
        zs = bbb.data.numpy()
        # zs = np.exp(zs/784)

        # print zs.shape
        max_ = np.max(zs)
        # print max_

        zs_sum = np.log(np.sum(np.exp(zs-max_))) + max_

        zs = zs - zs_sum
        zs = np.exp(zs)

        if i ==0:
            sum_of_all = zs
        else:
            sum_of_all = sum_of_all + zs



    avg_of_all = sum_of_all / n_Ws

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




















def plot_isocontours_expected_true_posterior(ax, model, data, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False, n_samps=10, cs_to_use=None):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)


    # n_samps = n_samps
    if len(data) < n_samps:
        n_samps = len(data)


    for samp_i in range(n_samps):
        # if samp_i % 100 == 0:
        #     print samp_i

        samp = data[samp_i]

        n_Ws = 1
        for i in range(n_Ws):
            # if i % 10 ==0: print i
            # print i
            Ws, logpW, logqW = model.sample_W()  #_ , [1], [1]   
            func = lambda zs: log_bernoulli(model.decode(Ws, Variable(torch.unsqueeze(zs,1))), Variable(torch.unsqueeze(samp,0)))+ Variable(torch.unsqueeze(lognormal4(torch.Tensor(zs), torch.zeros(2), torch.zeros(2)), 1))
            bbb = func(aaa)
            zs = bbb.data.numpy()
            max_ = np.max(zs)
            zs_sum = np.log(np.sum(np.exp(zs-max_))) + max_
            zs = zs - zs_sum
            zs = np.exp(zs)

            if i ==0:
                sum_of_all_i = zs
            else:
                sum_of_all_i = sum_of_all_i + zs



        if samp_i ==0:
            sum_of_all = sum_of_all_i
        else:
            sum_of_all = sum_of_all + sum_of_all_i




    avg_of_all = sum_of_all / n_samps


    # print 'sum:', np.sum(avg_of_all)


    Z = avg_of_all.reshape(X.shape)

    if cs_to_use != None:
        cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha, levels=cs_to_use.levels)
    else:
        cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)


    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4) 

    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')

    return Z







def plot_isocontours_expected_true_posterior_ind(ax, model, data, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False, n_samps=10, cs_to_use=None):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)


    # n_samps = n_samps
    if len(data) < n_samps:
        n_samps = len(data)


    for samp_i in range(n_samps):
        # if samp_i % 100 == 0:
        #     print samp_i

        samp = data[samp_i]

        n_Ws = 1
        for i in range(n_Ws):
            # if i % 10 ==0: print i
            # print i
            Ws, logpW, logqW = model.sample_W()  #_ , [1], [1]   
            func = lambda zs: log_bernoulli(model.decode(Ws, Variable(torch.unsqueeze(zs,1))), Variable(torch.unsqueeze(samp,0)))+ Variable(torch.unsqueeze(lognormal4(torch.Tensor(zs), torch.zeros(2), torch.zeros(2)), 1))
            bbb = func(aaa)
            zs = bbb.data.numpy()
            max_ = np.max(zs)
            zs_sum = np.log(np.sum(np.exp(zs-max_))) + max_
            zs = zs - zs_sum
            zs = np.exp(zs)



            Z = zs.reshape(X.shape)

            if cs_to_use != None:
                cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha, levels=cs_to_use.levels)
            else:
                cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)


            # if i ==0:
            #     sum_of_all_i = zs
            # else:
            #     sum_of_all_i = sum_of_all_i + zs



    #     if samp_i ==0:
    #         sum_of_all = sum_of_all_i
    #     else:
    #         sum_of_all = sum_of_all + sum_of_all_i




    # avg_of_all = sum_of_all / n_samps


    # print 'sum:', np.sum(avg_of_all)


    # if legend:
    #     nm, lbl = cs.legend_elements()
    #     plt.legend(nm, lbl, fontsize=4) 

    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')

    return Z









def plot_isocontours_of_matrix(ax, matrix, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False, cs_to_use=None):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)


    # cs = plt.contour(X, Y, matrix, cmap=cmap, alpha=alpha)
    # print 'dif sum:', np.sum(matrix)


    if cs_to_use != None:
        cs = plt.contour(X, Y, matrix, cmap=cmap, alpha=alpha, levels=cs_to_use.levels)
    else:
        cs = plt.contour(X, Y, matrix, cmap=cmap, alpha=alpha)

    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4) 

    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')





def plot_means(ax, model, data, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False, n_samps=10, cs_to_use=None):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)

    if len(data) < n_samps:
        n_samps = len(data)

    means = []
    for samp_i in range(n_samps):
        # if samp_i % 1000 == 0:
        #     print samp_i
        mean, logvar = model.encode(Variable(torch.unsqueeze(data[samp_i],0)))
        # print mean.data[0][0]
        means.append(np.array([mean.data[0][0],mean.data[0][1]]))
        # print mean
        # print mean[0][0].data[0]
    means=np.array(means)
    # print means.T[0]
    # plt.scatter(means.T[0],means.T[1], marker='x', s=3, alpha=alpha)
    plt.scatter(means.T[0],means.T[1], s=.1, alpha=alpha)


    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')









def plot_scatter(ax, samps, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False, n_samps=10, cs_to_use=None):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)

    # if len(data) < n_samps:
    #     n_samps = len(data)

    # means = []
    # for samp_i in range(n_samps):
    #     # if samp_i % 1000 == 0:
    #     #     print samp_i
    #     mean, logvar = model.encode(Variable(torch.unsqueeze(data[samp_i],0)))
    #     # print mean.data[0][0]
    #     means.append(np.array([mean.data[0][0],mean.data[0][1]]))
    #     # print mean
    #     # print mean[0][0].data[0]
    # means=np.array(means)
    # # print means.T[0]
    # plt.scatter(means.T[0],means.T[1], marker='x', s=3, alpha=alpha)
    # print (samps.T[0])
    # fsa
    plt.scatter(samps.T[0],samps.T[1], s=.01, alpha=alpha)


    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    plt.gca().set_aspect('equal', adjustable='box')














# def plot_isocontours_true_posterior_try2(ax, model, data, xlimits=[-6, 6], ylimits=[-6, 6],
#                      numticks=101, cmap=None, alpha=1., legend=False, n_samps=10, cs_to_use=None):
#     x = np.linspace(*xlimits, num=numticks)
#     y = np.linspace(*ylimits, num=numticks)
#     X, Y = np.meshgrid(x, y)
#     # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
#     aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)


#     # n_samps = n_samps
#     if len(data) < n_samps:
#         n_samps = len(data)


#     for samp_i in range(n_samps):
#         if samp_i % 100 == 0:
#             print samp_i

#         samp = data[samp_i]

#         n_Ws = 1
#         for i in range(n_Ws):
#             # if i % 10 ==0: print i
#             # print i
#             Ws, logpW, logqW = model.sample_W()  #_ , [1], [1]   
#             func = lambda zs: log_bernoulli(model.decode(Ws, Variable(torch.unsqueeze(zs,1))), Variable(torch.unsqueeze(samp,0)))+ Variable(torch.unsqueeze(lognormal4(torch.Tensor(zs), torch.zeros(2), torch.zeros(2)), 1))
#             bbb = func(aaa)
#             zs = bbb.data.numpy()
#             max_ = np.max(zs)
#             zs_sum = np.log(np.sum(np.exp(zs-max_))) + max_
#             zs = zs - zs_sum
#             zs = np.exp(zs)



#             Z = zs.reshape(X.shape)

#             if cs_to_use != None:
#                 cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha, levels=cs_to_use.levels)
#             else:
#                 cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)


#             # if i ==0:
#             #     sum_of_all_i = zs
#             # else:
#             #     sum_of_all_i = sum_of_all_i + zs



#     #     if samp_i ==0:
#     #         sum_of_all = sum_of_all_i
#     #     else:
#     #         sum_of_all = sum_of_all + sum_of_all_i




#     # avg_of_all = sum_of_all / n_samps


#     # print 'sum:', np.sum(avg_of_all)


#     # if legend:
#     #     nm, lbl = cs.legend_elements()
#     #     plt.legend(nm, lbl, fontsize=4) 

#     ax.set_yticks([])
#     ax.set_xticks([])
#     plt.gca().set_aspect('equal', adjustable='box')

#     return Z















def plot_isocontours2_exp_norm_logspace(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)
    # print '111'
    # print aaa

    aaa = Variable(aaa)
    bbb = func(aaa)
    # bbb = Variable(bbb)
    # if bbb.data:
    #     bbb = bbb.data
    # print bbb
    zs = bbb.data.numpy()
    # zs = np.exp(zs/784)




    max_ = np.max(zs)
    zs_sum = np.log(np.sum(np.exp(zs-max_))) + max_

    zs = zs - zs_sum
    # zs = np.exp(zs)

    # print np.max(zs)
    # fadsf


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




def plot_kde(ax, samps, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False):


    

    samps = np.array(samps)
    x_ = samps[:, 0]
    y_ = samps[:, 1]

    values = np.vstack([x_, y_])
    kernel = st.gaussian_kde(values)


    # Peform the kernel density estimate
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)

    positions = np.vstack([X.ravel(), Y.ravel()])

    f = np.reshape(kernel(positions).T, X.shape)

    # cfset = ax.contourf(X, Y, f, cmap=cmap)
    cfset = ax.contour(X, Y, f, cmap=cmap, linewidths=1.)


    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')












