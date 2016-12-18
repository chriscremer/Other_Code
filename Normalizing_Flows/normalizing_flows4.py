


# Since its not working, going to try linear transformations


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

def sample_diag_gaussian(mean, log_std, n_samples, rs):

    return npr.randn(n_samples, D) * np.exp(log_std) + mean


def build_nf_bbsvi(logprob, n_samples, k=10, rs=npr.RandomState(0)):

    def init_params(D, rs=npr.RandomState(0), **kwargs):

        init_mean    = -1 * np.ones(D) + rs.randn(D) * .1
        init_log_std = -5 * np.ones(D) + rs.randn(D) * .1
        # gauss_params = np.concatenate([init_mean, init_log_std])

        # u = rs.randn(D,1)
        # w = rs.randn(D,1)
        # b = rs.randn(1)
        norm_flow_params = [[rs.randn(D,1), rs.randn(D,1), rs.randn(1)] for x in range(k)]

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


        current_z = z_0 * np.array([2,2])# + 10

        # for params_k in norm_flow_params:

        #     u = params_k[0]
        #     w = params_k[1]
        #     b = params_k[2]

        #     # m_x = -1. + np.log(1.+np.exp(np.dot(w.T,u)))
        #     # u_k = u + (m_x - np.dot(w.T,u)) *  (w/np.linalg.norm(w))
        #     u_k = u

        #     # [D,1]
        #     # term1 = np.tanh(np.dot(current_z,w)+b)
        #     term1 = np.dot(current_z,w)+b

        #     # [n_samples, D]
        #     term1 = np.dot(term1,u_k.T)
        #     # [n_samples, D]
        #     current_z = current_z + term1
        #     all_zs.append(current_z)

        return current_z, all_zs

    def variational_log_density(params, datapoints):
        '''
        samples: [n_samples, D]
        u: [D,1]
        w: [D,1]
        b: [1]
        Returns: [num_samples]
        '''
        n_samples = len(datapoints)

        mean = params[0]
        log_std = params[1]
        norm_flow_params = params[2]



        # print (samples.shape)

        # z_k, all_zs = normalizing_flows(datapoints, norm_flow_params)

        # logp_zk = logprob(z_k)
        # logp_zk = np.reshape(logp_zk, [n_samples, 1])

        logq_z0 = diag_gaussian_log_density(datapoints, mean, log_std)

        # print (logq_z0.shape)
        logq_z0 = np.reshape(logq_z0, [n_samples, 1])

        log_qk = logq_z0 + np.log(.5)

        # sum_nf = np.zeros([n_samples,1])
        # for params_k in range(len(norm_flow_params)):
        #     u = norm_flow_params[params_k][0]
        #     w = norm_flow_params[params_k][1]
        #     b = norm_flow_params[params_k][2]

        #     # m_x = -1. + np.log(1.+np.exp(np.dot(w.T,u)))
        #     # u_k = u + (m_x - np.dot(w.T,u)) *  (w/np.linalg.norm(w))
        #     u_k = u

        #     # [n_samples, D]
        #     # phi = np.dot((1.-np.tanh(np.dot(all_zs[params_k],w)+b)**2), w.T)
        #     phi = np.reshape(w, [2])
        #     ones = np.ones([n_samples,2])
        #     phi = ones*phi


        #     # [n_samples, 1]
        #     sum_nf = np.log(np.abs(1+np.dot(phi, u_k)))
        #     sum_nf += sum_nf

        # return logq_z0 - sum_nf
        # log_qz = np.reshape(logq_z0 - sum_nf, [n_samples])

        log_qz = np.reshape(log_qk, [n_samples])

        return log_qz

    def sample_variational_density(params):
        mean = params[0]
        log_std = params[1]
        norm_flow_params = params[2]

        samples = sample_diag_gaussian(mean, log_std, n_samples, rs) 

        z_k, all_zs = normalizing_flows(samples, norm_flow_params)

        return z_k


    def elbo(params, t):
        '''
        samples: [n_samples, D]
        u: [D,1]
        w: [D,1]
        b: [1]
        '''

        beta = t/100 + .001

        if beta > .99:
            beta = 1.

        beta = 1

        mean = params[0]
        log_std = params[1]
        norm_flow_params = params[2]

        samples = sample_diag_gaussian(mean, log_std, n_samples, rs)
        z_k, all_zs = normalizing_flows(samples, norm_flow_params)

        logp_zk = logprob(z_k)
        logp_zk = np.reshape(logp_zk, [n_samples, 1])

        logq_zk = variational_log_density(params, samples)
        logq_zk = np.reshape(logq_zk, [n_samples, 1])

        elbo = (beta*logp_zk) - logq_zk 
  
        return np.mean(elbo) #over samples



    return init_params, elbo, variational_log_density, sample_variational_density


if __name__ == '__main__':

    # Specify an inference problem by its unnormalized log-density.
    D = 2
    k = 2
    n_samples = 20


    def log_density(x):
        '''
        x: [n_samples, D]
        return: [n_samples]
        '''
        x_, y_ = x[:, 0], x[:, 1]
        sigma_density = norm.logpdf(y_, 0, 1.35)
        mu_density = norm.logpdf(x_, -2.2, np.exp(y_))
        sigma_density2 = norm.logpdf(y_, 0.1, 1.35)
        mu_density2 = norm.logpdf(x_, 2.2, np.exp(y_))
        return np.logaddexp(sigma_density + mu_density, sigma_density2 + mu_density2)

        # print ((.5*((np.linalg.norm(x, axis=1)-2)/.4)**2 - np.log(np.exp(-.5*((x_-2))/.6**2) + np.exp(-.5*((y_-2))/.6**2))).shape)
        # fafds

        # return .00001*((np.linalg.norm(x, axis=1)-2)/.4)**2 - np.log(np.exp(-.5*((x_))/.6**2) + np.exp(-.5*((y_))/.6**2))
        # return .1*((np.linalg.norm(x, axis=1)-2)/.4)**2

        # return np.logaddexp(norm.logpdf(x_, 0,1)+norm.logpdf(y_, 0,1), norm.logpdf(x_, 0,1)+norm.logpdf(y_, 0,1))




    init_var_params, elbo, variational_log_density, variational_sampler = \
        build_nf_bbsvi(log_density, n_samples=n_samples, k=k)

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

        mean = params[0]
        log_std = params[1]

        print ('mean', mean)
        print ('std', np.exp(log_std))

        # print ('u0', params[2][0][0])
        # print ('u1', params[2][1][0])
        # print ('w0', params[2][0][1])
        # print ('w1', params[2][1][1])
        # print ('b0', params[2][0][2])
        # print ('b1', params[2][1][2])



        # x_inverse = 

        plt.cla()
        target_distribution = lambda x: np.exp(log_density(x))
        var_distribution    = lambda x: np.exp(variational_log_density(params, x))
        plot_isocontours(ax, target_distribution)
        plot_isocontours(ax, var_distribution, cmap=plt.cm.bone)
        ax.set_autoscale_on(False)


        #PLot the z0 density
        var_distribution0 = lambda x: np.exp(diag_gaussian_log_density(x, mean, log_std))
        plot_isocontours(ax, var_distribution0)

        for transform in params[2]:

            xlimits=[-6, 6]
            w = transform[1]
            b = transform[2]
            x = np.linspace(*xlimits, num=101)
            plt.plot(x, (-w[0]*x - b)/w[1], '-')
            
            u = transform[0]
            plt.plot(x, (-u[0]*x)/u[1], '-')

        #PLot variational samples
        samples = variational_sampler(params)
        plt.plot(samples[:, 0], samples[:, 1], 'x')

        # #Plot q0 variational samples
        # rs = npr.RandomState(0)
        # samples = sample_diag_gaussian(mean, log_std, n_samples, rs) 
        # plt.plot(samples[:, 0], samples[:, 1], 'x')


        plt.draw()
        plt.pause(1.0/30.0)

    print("Optimizing variational parameters...")
    variational_params = adam(grad(objective), init_var_params(D), step_size=0.1,
                              num_iters=2000, callback=callback)







