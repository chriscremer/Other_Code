

# Implements black-box variational inference, where the variational
# distribution use normalizing flows.

from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd.scipy.misc import logsumexp

from autograd import grad
from autograd.optimizers import adam



def diag_gaussian_log_density(x, mu, log_std):

    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def sample_diag_gaussian(mean, log_std, num_samples, rs):

    return rs.randn(num_samples, D) * np.exp(log_std) + mean


def build_nf_bbsvi(logprob, num_samples, k=10, rs=npr.RandomState(0)):

    def init_params(D, rs=npr.RandomState(0), **kwargs):

        init_mean    = -1 * np.ones(D) + rs.randn(D) * .1
        init_log_std = -5 * np.ones(D) + rs.randn(D) * .1
        # gauss_params = np.concatenate([init_mean, init_log_std])

        u = rs.randn(D)
        w = rs.randn(D)
        b = rs.randn(1)
    
        return [init_mean, init_log_std, u, w, b]


    def elbo(params, t):

        mean = params[0]
        log_std = params[1]
        u = params[2]
        w = params[3]
        b = params[4]

        samples = sample_diag_gaussian(mean, log_std, num_samples, rs)

        logp_z = logprob(samples)
        logq_z = diag_gaussian_log_density(samples, mean, log_std)

        # elbo =  - log_gauss_density(z_0) + np.log(abs(1+np.dot(u, (1.-np.tanh(np.dot(w,z_0)+b)**2)*w)))
        elbo = logp_z - logq_z
        return np.mean(elbo) #over samples

    def variational_log_density(params, samples):
        mean = params[0]
        log_std = params[1]

        return diag_gaussian_log_density(samples, mean, log_std)

    def sample_variational_density(params):
        mean = params[0]
        log_std = params[1]

        return sample_diag_gaussian(mean, log_std, num_samples, rs)   


    return init_params, elbo, variational_log_density, sample_variational_density


if __name__ == '__main__':

    # Specify an inference problem by its unnormalized log-density.
    D = 2
    def log_density(x):
        x_, y_ = x[:, 0], x[:, 1]
        sigma_density = norm.logpdf(y_, 0, 1.35)
        mu_density = norm.logpdf(x_, -0.5, np.exp(y_))
        sigma_density2 = norm.logpdf(y_, 0.1, 1.35)
        mu_density2 = norm.logpdf(x_, 0.5, np.exp(y_))
        return np.logaddexp(sigma_density + mu_density, sigma_density2 + mu_density2)


    init_var_params, elbo, variational_log_density, variational_sampler = \
        build_nf_bbsvi(log_density, num_samples=40, k=10)

    def objective(params, t):
        return -elbo(params, t)

    # Set up plotting code
    def plot_isocontours(ax, func, xlimits=[-6, 6], ylimits=[-6, 4],
                         numticks=101, cmap=None):
        x = np.linspace(*xlimits, num=numticks)
        y = np.linspace(*ylimits, num=numticks)
        X, Y = np.meshgrid(x, y)
        zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
        Z = zs.reshape(X.shape)
        plt.contour(X, Y, Z, cmap=cmap)
        ax.set_yticks([])
        ax.set_xticks([])

    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    num_plotting_samples = 51

    def callback(params, t, g):

        # log_weights = params[:10] - logsumexp(params[:10])
        print("Iteration {} lower bound {}".format(t, -objective(params, t)))
        # print (np.exp(log_weights))

        plt.cla()
        target_distribution = lambda x: np.exp(log_density(x))
        var_distribution    = lambda x: np.exp(variational_log_density(params, x))
        plot_isocontours(ax, target_distribution)
        plot_isocontours(ax, var_distribution, cmap=plt.cm.bone)
        ax.set_autoscale_on(False)


        # rs = npr.RandomState(0)
        # samples = variational_sampler(params, num_plotting_samples, rs)
        # plt.plot(samples[:, 0], samples[:, 1], 'x')

        plt.draw()
        plt.pause(1.0/30.0)

    print("Optimizing variational parameters...")
    variational_params = adam(grad(objective), init_var_params(D), step_size=0.1,
                              num_iters=2000, callback=callback)







