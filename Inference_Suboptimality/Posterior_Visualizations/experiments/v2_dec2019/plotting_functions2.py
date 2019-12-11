import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from utils import lognormal4 
from utils import log_bernoulli

import scipy.stats as st






def plot_isocontours_new(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False, levels=[]):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor)


    dz = (xlimits[1] - xlimits[0]) / numticks

    logpxz = func(aaa)
    logpxz = logpxz.data.cpu().numpy()


    max_ = np.max(logpxz)
    logpx = np.log(dz**2 * np.sum(np.exp(logpxz-max_))) + max_

    # print (zs_sum)
    # px = zs_sum 

    log_posterior = logpxz - logpx # zs_sum
    posterior = np.exp(log_posterior)

    # Integal
    print (np.sum(posterior) * dz**2)
    print (np.max(posterior))
    print ()

    # print np.max(zs)
    # fadsf

    # zs = np.exp(zs/784)


    Z = posterior.reshape(X.shape)
    # Z = zs.view(X.shape)
    # Z=Z.numpy()
    # cs = plt.contourf(X, Y, Z, cmap=cmap, alpha=alpha)

    if len(levels) > 0:
        cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha, levels=levels)
    else:
        cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)


    # levels = []
    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4) 

        # print (nm)
        # print (lbl)
        # print (cs.levels)
    levels = cs.levels




    # ax.patch.set_edgecolor('black')

    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')

    return levels










