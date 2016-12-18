



# Switch from planar to radial flows


# Implements black-box variational inference, where the variational
# distribution uses normalizing flows.

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

        # u = rs.randn(D,1)
        # w = rs.randn(D,1)
        # b = rs.randn(1)
        norm_flow_params = [[rs.randn(D,1), rs.randn(1), rs.randn(1)] for x in range(k)]

        # norm_flow_params = np.array(norm_flow_params)
        # print (norm_flow_params.shape)
        # fadfsa
    
        return [init_mean, init_log_std, norm_flow_params]


    def normalizing_flows(z_0, norm_flow_params):
        '''
        z_0: [n_samples, D]
        u: [D,1]
        w: [D,1]
        b: [1]
        '''

        current_z = z_0
        all_zs = []
        all_zs.append(z_0)
        for params_k in norm_flow_params:

            z_0_mean = params_k[0]
            a = np.abs(params_k[1])
            b = params_k[2]

            # m_x = -1. + np.log(1.+np.exp(np.dot(w.T,u)))
            # u_k = u + (m_x - np.dot(w.T,u)) *  (w/np.linalg.norm(w))

            # print (a.shape)

            # print (current_z.shape)
            z_0_mean = np.reshape(z_0_mean, [len(current_z[0])])

            h = 1./(a + np.abs(current_z - z_0_mean))
            term1 = b * h * (current_z - z_0_mean)
            current_z = current_z + term1

            all_zs.append(current_z)

        return current_z, all_zs

    def variational_log_density(params, samples):
        '''
        samples: [n_samples, D]
        u: [D,1]
        w: [D,1]
        b: [1]
        Returns: [num_samples]
        '''
        n_samples = len(samples)
        d = len(samples[0])

        mean = params[0]
        log_std = params[1]
        norm_flow_params = params[2]

        # print (samples.shape)

        # samples = sample_diag_gaussian(mean, log_std, num_samples, rs)
        z_k, all_zs = normalizing_flows(samples, norm_flow_params)

        logp_zk = logprob(z_k)
        logp_zk = np.reshape(logp_zk, [n_samples, 1])

        logq_z0 = diag_gaussian_log_density(samples, mean, log_std)
        logq_z0 = np.reshape(logq_z0, [n_samples, 1])

        sum_nf = np.zeros([n_samples,1])
        for params_k in range(len(norm_flow_params)):
            z_0_mean = norm_flow_params[params_k][0]
            a = np.abs(norm_flow_params[params_k][1])
            b = norm_flow_params[params_k][2]

            # m_x = -1. + np.log(1.+np.exp(np.dot(w.T,u)))
            # u_k = u + (m_x - np.dot(w.T,u)) *  (w/np.linalg.norm(w))

            # [n_samples, D]
            # phi = np.dot((1.-np.tanh(np.dot(all_zs[params_k],w)+b)**2), w.T)
            # [n_samples, 1]
            current_z = all_zs[params_k]

            z_0_mean = np.reshape(z_0_mean, [len(current_z[0])])

            h = 1./(a + np.abs(current_z - z_0_mean))
            h_prime = -1*(a+np.abs(current_z-z_0_mean))**2 * (np.abs(current_z)/current_z)

            term1 = (1+b*h)**(d-1)
            term2 = 1+ b * h + b * h_prime * np.abs(current_z-z_0_mean)
            term3 = term1 * term2

            sum_nf = np.log(np.abs(term3))
            sum_nf += sum_nf

        # return logq_z0 - sum_nf
        print (logq_z0.shape)
        log_qz = np.reshape(logq_z0 - sum_nf, [n_samples])
        return log_qz

    def sample_variational_density(params):
        mean = params[0]
        log_std = params[1]
        norm_flow_params = params[2]

        samples = sample_diag_gaussian(mean, log_std, num_samples, rs) 

        z_k, all_zs = normalizing_flows(samples, norm_flow_params)

        return z_k


    def elbo(params, t):
        '''
        samples: [n_samples, D]
        u: [D,1]
        w: [D,1]
        b: [1]
        '''

        # beta = t/1000 + .001

        # if beta > .99:
        #     beta = 1.

        beta = 1.

        mean = params[0]
        log_std = params[1]
        norm_flow_params = params[2]

        samples = sample_diag_gaussian(mean, log_std, num_samples, rs)
        z_k, all_zs = normalizing_flows(samples, norm_flow_params)

        logp_zk = logprob(z_k)
        logp_zk = np.reshape(logp_zk, [num_samples, 1])

        logq_zk = variational_log_density(params, samples)
        logq_zk = np.reshape(logq_zk, [num_samples, 1])

        elbo = (beta*logp_zk) - logq_zk
  
        return np.mean(elbo) #over samples



    return init_params, elbo, variational_log_density, sample_variational_density


if __name__ == '__main__':

    # Specify an inference problem by its unnormalized log-density.
    D = 2
    k = 5
    num_samples = 100


    def log_density(x):
        '''
        x: [n_samples, D]
        return: [n_samples]
        '''
        x_, y_ = x[:, 0], x[:, 1]
        sigma_density = norm.logpdf(y_, 0, 1.35)
        mu_density = norm.logpdf(x_, -1.5, np.exp(y_))
        sigma_density2 = norm.logpdf(y_, 0.1, 1.35)
        mu_density2 = norm.logpdf(x_, 1.5, np.exp(y_))
        return np.logaddexp(sigma_density + mu_density, sigma_density2 + mu_density2)


    init_var_params, elbo, variational_log_density, variational_sampler = \
        build_nf_bbsvi(log_density, num_samples=num_samples, k=k)

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

        # mean = params[0]
        # log_std = params[1]
        # norm_flow_params = params[2]
        # print (len(params[2][0][0]))
        # print ('u', params[2][0])
        print ('u0', params[2][0][0])
        print ('u1', params[2][1][0])
        print ('w0', params[2][0][1])
        print ('w1', params[2][1][1])
        print ('b0', params[2][0][2])
        print ('b1', params[2][1][2])
        # print ('b', params[2][2])



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







