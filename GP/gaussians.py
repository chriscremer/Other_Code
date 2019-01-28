

import numpy as np 
import matplotlib.pyplot as plt
import math

from scipy.stats import multivariate_normal as mvn



def log_normal_pdf(x, mean=0., cov=0.):
    #x: [N,2]
    D=2
    mean = np.array([0.,0.])
    cov = np.array([[1.,cov],[cov,1.]])

    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)

    term1 = D * np.log(2.*math.pi)
    term2 = cov_det

    logprobs = []
    for i in range(len(x)):

        x_i = x[i]

        x_i = np.reshape(x_i, [2,1])
        mean = np.reshape(mean, [2,1])
        cov = np.reshape(cov, [2,2])
        term3 = np.dot(np.dot((x_i - mean).T, cov_inv), (x_i - mean))

        logprobs.append(-.5 * (term1 + term2 + term3))

    return logprobs



def plot_isocontours(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z, cmap=cmap)
    # plt.contourf(X, Y, Z, cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')






fig = plt.figure(figsize=(12,5), facecolor='white')
# ax1 = plt.subplot2grid((3, 3), (0, 0))#, colspan=3)

mean = np.array([0.,0.])

covs = np.array(range(0,5)) /5.
covs2 = np.array(range(1,5))[::-1] /-5.
covs = np.concatenate([covs2, covs])

# covs = [.8]

for i in range(len(covs)):

    ax1 = plt.subplot2grid((1,9), (0,i))#, colspan=3)

    print (covs[i])
    
    cov = np.array([[1.,covs[i]],[covs[i],1.]])
    rv = mvn(mean, cov)


    # func = lambda x: np.exp(log_normal_pdf(x,cov=covs[i]))

    # plot_isocontours(ax1, func, xlimits=[-6, 6], ylimits=[-6, 6], numticks=101, cmap=None)
    # plot_isocontours(ax1, rv.pdf, numticks=101, cmap='Blues')
    plot_isocontours(ax1, rv.pdf, numticks=101, cmap=None)

    ax1.set_title(str(covs[i]))




plt.show()




















