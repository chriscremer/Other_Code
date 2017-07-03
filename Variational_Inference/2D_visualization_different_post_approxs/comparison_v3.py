





import numpy as np
import tensorflow as tf

import random
import math
from scipy.stats import norm
import scipy.stats as st

from os.path import expanduser
home = expanduser("~")

import matplotlib.pyplot as plt

import time





def plot_isocontours(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z, cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')



def plot_iw_isocontours(ax, p, q, sampler, k=3, xlimits=[-6, 6], ylimits=[-6, 6], numticks=101, 
                        n_sample_batches=3, cmap='Blues'):

    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T

    X_vec = np.reshape(X, [-1,1])
    Y_vec = np.reshape(Y, [-1,1])
    zipped = np.concatenate([X_vec, Y_vec], axis=1)

    log_p = np.log(p(zipped))
    log_q = np.log(q(zipped))
    log_w = log_p - log_q
    exp_log_p = np.exp(log_p)
    exp_log_w = np.exp(log_w)

    grid_values = np.zeros([numticks*numticks])

    # start_time = time.time()
    for b in range(n_sample_batches):
        # if b % 10 == 0:
        #     # elapsed_time = time.time() - start_time
        #     # start_time = time.time()
        #     # print elapsed_time
        #     print b

        if k >1:
            # samples = sampler(tf.convert_to_tensor(k-1, dtype='float32'))
            samples = sampler(k-1)
            log_p_ = np.log(p(samples))
            log_q_ = np.log(q(samples))
            log_w_ = log_p_ - log_q_
            sum_p_q = np.sum(np.exp(log_w_))
        else:
            sum_p_q = 0

        denominator = (exp_log_w + sum_p_q) / k
        grid_values += exp_log_p / denominator

    grid_values = grid_values / n_sample_batches
    grid_values = np.reshape(grid_values, [numticks, numticks])
    plt.contourf(X, Y, grid_values, cmap=cmap) #, vmin=10**(-10))
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')



def plot_hi_isocontours(ax, p, q, hamiltonian_sampling, k=3, xlimits=[-6, 6], ylimits=[-6, 6], numticks=101, 
                        n_sample_batches=3, cmap='Blues'):

    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T

    X_vec = np.reshape(X, [-1,1])
    Y_vec = np.reshape(Y, [-1,1])
    zipped = np.concatenate([X_vec, Y_vec], axis=1)

    log_p = np.log(p(zipped))
    log_q = np.log(q(zipped))
    log_w = log_p - log_q
    exp_log_p = np.exp(log_p)
    exp_log_w = np.exp(log_w)

    grid_values = np.zeros([numticks*numticks])

    
    for b in range(n_sample_batches):
        
        if k >1:

            start_time = time.time()
            z0, zT, v0, vT, log_q_z0, log_q_zT, log_q_v0, log_q_vT, n_steps, friction  = hamiltonian_sampling(k-1)
            print 'hamil took', time.time() - start_time

            log_p_ = np.log(p(zT))
            log_q_ = np.log(q(z0))

            log_w_ = log_p_ - log_q_ + log_q_vT - log_q_v0 + (n_steps*np.log(friction))

            sum_p_q = np.sum(np.exp(log_w_))
        else:
            sum_p_q = 0

        denominator = (exp_log_w + sum_p_q) / k
        grid_values += exp_log_p / denominator


    # #KDE
    # if b == 0:
    #     samps = zT
    # else:
    #     samps = np.concatenate([samps, zT], axis=0)
    # x_ = samps[:, 0]
    # y_ = samps[:, 1]
    # x = np.linspace(*xlimits, num=numticks)
    # y = np.linspace(*ylimits, num=numticks)
    # X, Y = np.meshgrid(x, y)
    # X = X.T
    # Y = Y.T
    # positions = np.vstack([X.ravel(), Y.ravel()])
    # values = np.vstack([x_, y_])
    # kernel = st.gaussian_kde(values)
    # f = np.reshape(kernel(positions).T, X.shape)
    # plt.contourf(X, Y, f, cmap=cmap) #, vmin=10**(-10))


    grid_values = grid_values / n_sample_batches
    grid_values = np.reshape(grid_values, [numticks, numticks])
    plt.contourf(X, Y, grid_values, cmap=cmap) #, vmin=10**(-10))
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')


def plot_scatter_hamil_points(z0, zT):

    plt.scatter(z0.T[0], z0.T[1], color='blue')
    plt.scatter(zT.T[0], zT.T[1], color='red')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.gca().set_aspect('equal', adjustable='box')



def log_normal(x, mean, log_var):
    '''
    x is [P, D]
    mean is [D]
    log_var is [D]
    '''
    term1 = 2 * tf.log(2*math.pi)
    term2 = tf.reduce_sum(log_var) #sum over dimensions, [1]
    term3 = tf.square(x - mean) / tf.exp(log_var)
    term3 = tf.reduce_sum(term3, 1) #sum over dimensions, [P]
    all_ = term1 + term2 + term3
    log_normal = -.5 * all_  
    return log_normal



def log_posteriors(index_, x):
    '''
    index_: which posterior you want
    x: [P,D]
    '''

    if index_ == 0:
        prob = (tf.exp(log_normal(x, [1,3], tf.ones(2)/100))
                    + tf.exp(log_normal(x, [3,0], tf.ones(2)/100))
                    + tf.exp(log_normal(x, [1,1], tf.ones(2)/100))
                    + tf.exp(log_normal(x, [-3,-1], tf.ones(2)/100))
                    + tf.exp(log_normal(x, [-1,-3], tf.ones(2)/100))
                    + tf.exp(log_normal(x, [2,2], tf.ones(2)))
                    + tf.exp(log_normal(x, [-2,-2], tf.ones(2)))
                    + tf.exp(log_normal(x, [0,0], tf.ones(2))))

    if index_ == 1:
        prob = (tf.exp(log_normal(x, [0.,4.], [2., -3.]))
                    + tf.exp(log_normal(x, [0.,-4.], [2., -3.])))

    if index_ == 2:
        prob = (tf.exp(log_normal(x, [-2.,2], [-2., 1.5]))
                    + tf.exp(log_normal(x, [2.,-1], [1.5, -2.])))
    if index_ == 3:
        prob = (tf.exp(log_normal(x, [-4.,4], [0., 0.])))

    prob = tf.maximum(prob, tf.exp(-40.))
    return tf.log(prob)

def proposal(x):

    return log_normal(x, [0.,0.], [1., 1.])


def sample_proposal(n_samples):

    n_samples = tf.convert_to_tensor(n_samples, dtype='int32')

    eps = tf.random_normal((n_samples, 2), 0, 1, dtype=tf.float32) #[P,D]
    sample_q = tf.add([0.,0.], tf.multiply(tf.sqrt(tf.exp([1.,1.])), eps)) #[P,D]

    return sample_q





def leapfrog_step(z, v, logprob, step_size):

    start_time = time.time()
    grad = tf.gradients(-logprob(z), [z])[0]
    print 'grad took', time.time() - start_time

    v = v - ((.5*step_size) * grad)

    z = z + (step_size * v)

    grad = tf.gradients(-logprob(z), [z])[0]
    v = v - ((.5*step_size) * grad)

    

    return z, v


def hamiltonian_sampling(n_samples, target_dist, proposal_dist, proposal_sampler):

    n_steps = 10
    step_size = .1
    friction = tf.convert_to_tensor(.95, dtype='float32') 
    
    v_0 = tf.random_normal((n_samples, 2), 0, .5, dtype=tf.float32) #[N,D]
    log_q_v0 = log_normal(v_0, [0.,0.], tf.sqrt(tf.exp([.5,.5])))

    z_0 = proposal_sampler(n_samples) #[N,D]
    log_q_z0 = proposal_dist(z_0) #[N]

    z = z_0
    v = v_0

    for i in range(n_steps):

        z, v = leapfrog_step(z, v, target_dist, step_size)

        v = v * friction

    log_q_vT = log_normal(v, [0.,0.], tf.sqrt(tf.exp([.5,.5])))
    log_q_zT = proposal_dist(z) #[N]

    return z_0, z, v_0, v, log_q_z0, log_q_zT, log_q_v0, log_q_vT, tf.convert_to_tensor(n_steps, dtype='float32'), friction






if __name__ == '__main__':


    sess = tf.Session()


    fig = plt.figure(figsize=(8,8), facecolor='white')
    


    #Grid 1,1
    ax = fig.add_subplot(441, frameon=False)
    target_distribution = lambda x: np.exp(sess.run(log_posteriors(0, tf.convert_to_tensor(x, dtype='float32'))))
    plot_isocontours(ax, target_distribution, cmap='Blues')
    plt.gca().set_aspect('equal', adjustable='box')
    ax.annotate('Posterior', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')
    
    #Grid 2,1
    ax = fig.add_subplot(445, frameon=False)
    target_distribution = lambda x: np.exp(sess.run(log_posteriors(1, tf.convert_to_tensor(x, dtype='float32'))))
    plot_isocontours(ax, target_distribution, cmap='Blues')

    #Grid 3,1
    ax = fig.add_subplot(449, frameon=False)
    target_distribution = lambda x: np.exp(sess.run(log_posteriors(2, tf.convert_to_tensor(x, dtype='float32'))))
    plot_isocontours(ax, target_distribution, cmap='Blues')

    #Grid 4,1
    ax = fig.add_subplot(4, 4, 13, frameon=False)
    target_distribution = lambda x: np.exp(sess.run(log_posteriors(3, tf.convert_to_tensor(x, dtype='float32'))))
    plot_isocontours(ax, target_distribution, cmap='Blues')




    #Grid 1,2
    ax = fig.add_subplot(4, 4, 2, frameon=False)
    target_distribution = lambda x: np.exp(sess.run(proposal(tf.convert_to_tensor(x, dtype='float32'))))
    plot_isocontours(ax, target_distribution, cmap='Reds')
    ax.annotate('Proposal', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

    #Grid 2,2
    ax = fig.add_subplot(4, 4, 6, frameon=False)
    target_distribution = lambda x: np.exp(sess.run(proposal(tf.convert_to_tensor(x, dtype='float32'))))
    plot_isocontours(ax, target_distribution, cmap='Reds')

    #Grid 3,2
    ax = fig.add_subplot(4, 4, 10, frameon=False)
    target_distribution = lambda x: np.exp(sess.run(proposal(tf.convert_to_tensor(x, dtype='float32'))))
    plot_isocontours(ax, target_distribution, cmap='Reds')

    #Grid 4,2
    ax = fig.add_subplot(4, 4, 14, frameon=False)
    target_distribution = lambda x: np.exp(sess.run(proposal(tf.convert_to_tensor(x, dtype='float32'))))
    plot_isocontours(ax, target_distribution, cmap='Reds')



    #Grid 1,3
    n_batches = 5
    k = 5
    ax = fig.add_subplot(4, 4, 3, frameon=False)
    target_distribution = lambda x: np.exp(sess.run(log_posteriors(0, tf.convert_to_tensor(x, dtype='float32'))))
    proposal_distribution = lambda x: np.exp(sess.run(proposal(tf.convert_to_tensor(x, dtype='float32'))))
    proposal_sampler = lambda x: sess.run(sample_proposal(tf.convert_to_tensor(x, dtype='int32')))
    plot_iw_isocontours(ax, target_distribution, proposal_distribution, proposal_sampler, k=k, n_sample_batches=n_batches, cmap='Reds')
    ax.annotate('Importance', xytext=(.2, 1.1), xy=(0, 1), textcoords='axes fraction')

    #Grid 2,3
    ax = fig.add_subplot(4, 4, 7, frameon=False)
    target_distribution = lambda x: np.exp(sess.run(log_posteriors(1, tf.convert_to_tensor(x, dtype='float32'))))
    plot_iw_isocontours(ax, target_distribution, proposal_distribution, proposal_sampler, k=k, n_sample_batches=n_batches, cmap='Reds')

    #Grid 3,3
    ax = fig.add_subplot(4, 4, 11, frameon=False)
    target_distribution = lambda x: np.exp(sess.run(log_posteriors(2, tf.convert_to_tensor(x, dtype='float32'))))
    plot_iw_isocontours(ax, target_distribution, proposal_distribution, proposal_sampler, k=k, n_sample_batches=n_batches, cmap='Reds')

    #Grid 4,3
    ax = fig.add_subplot(4, 4, 15, frameon=False)
    target_distribution = lambda x: np.exp(sess.run(log_posteriors(3, tf.convert_to_tensor(x, dtype='float32'))))
    plot_iw_isocontours(ax, target_distribution, proposal_distribution, proposal_sampler, k=k, n_sample_batches=n_batches, cmap='Reds')



    #Grid 4,1
    k = 5
    print 'g41'
    ax = fig.add_subplot(4, 4, 4, frameon=False)

    start_time = time.time()

    # tf.get_default_graph().finalize()

    target_distribution = lambda x: np.exp(sess.run(log_posteriors(0, tf.convert_to_tensor(x, dtype='float32'))))
    proposal_distribution = lambda x: np.exp(sess.run(proposal(tf.convert_to_tensor(x, dtype='float32'))))

    hamil_func = lambda x: sess.run(hamiltonian_sampling(n_samples=tf.convert_to_tensor(x, dtype='int32'), 
                                    target_dist=lambda y: log_posteriors(0,y), 
                                    proposal_dist=proposal, 
                                    proposal_sampler=sample_proposal))

    plot_hi_isocontours(ax, target_distribution, proposal_distribution, hamil_func, k=k, n_sample_batches=n_batches, cmap='Reds')
    print 'g41 first part time', time.time() - start_time

    z0, zT, v0, vT, log_q_z0, log_q_zT, log_q_v0, log_q_vT, n_steps, friction = sess.run(hamiltonian_sampling(n_samples=tf.convert_to_tensor(5, dtype='int32'), 
                                    target_dist=lambda y: log_posteriors(0,y), 
                                    proposal_dist=proposal, 
                                    proposal_sampler=sample_proposal))
    plot_scatter_hamil_points(z0, zT)

    ax.annotate('Hamiltonian', xytext=(.2, 1.1), xy=(0, 1), textcoords='axes fraction')

    print 'g41 time', time.time() - start_time


    #Grid 4,2
    print 'g42'
    ax = fig.add_subplot(4, 4, 8, frameon=False)
    start_time = time.time()
    target_distribution = lambda x: np.exp(sess.run(log_posteriors(1, tf.convert_to_tensor(x, dtype='float32'))))

    hamil_func = lambda x: sess.run(hamiltonian_sampling(n_samples=tf.convert_to_tensor(x, dtype='int32'), 
                                    target_dist=lambda y: log_posteriors(1,y), 
                                    proposal_dist=proposal, 
                                    proposal_sampler=sample_proposal))
    plot_hi_isocontours(ax, target_distribution, proposal_distribution, hamil_func, k=k, n_sample_batches=n_batches, cmap='Reds')
    print 'g42 first part time', time.time() - start_time

    z0, zT, v0, vT, log_q_z0, log_q_zT, log_q_v0, log_q_vT, n_steps, friction = sess.run(hamiltonian_sampling(n_samples=tf.convert_to_tensor(5, dtype='int32'), 
                                    target_dist=lambda y: log_posteriors(1,y), 
                                    proposal_dist=proposal, 
                                    proposal_sampler=sample_proposal))
    plot_scatter_hamil_points(z0, zT)
    print 'g42 time', time.time() - start_time

    #Grid 4,3
    print 'g43'
    ax = fig.add_subplot(4, 4, 12, frameon=False)
    start_time = time.time()
    target_distribution = lambda x: np.exp(sess.run(log_posteriors(2, tf.convert_to_tensor(x, dtype='float32'))))

    hamil_func = lambda x: sess.run(hamiltonian_sampling(n_samples=tf.convert_to_tensor(x, dtype='int32'), 
                                    target_dist=lambda y: log_posteriors(2,y), 
                                    proposal_dist=proposal, 
                                    proposal_sampler=sample_proposal))
    plot_hi_isocontours(ax, target_distribution, proposal_distribution, hamil_func, k=k, n_sample_batches=n_batches, cmap='Reds')
    print 'g43 first part time', time.time() - start_time

    z0, zT, v0, vT, log_q_z0, log_q_zT, log_q_v0, log_q_vT, n_steps, friction = sess.run(hamiltonian_sampling(n_samples=tf.convert_to_tensor(5, dtype='int32'), 
                                    target_dist=lambda y: log_posteriors(2,y), 
                                    proposal_dist=proposal, 
                                    proposal_sampler=sample_proposal))
    plot_scatter_hamil_points(z0, zT)
    print 'g43 time', time.time() - start_time

    #Grid 4,4
    print 'g44'
    ax = fig.add_subplot(4, 4, 16, frameon=False)
    start_time = time.time()
    target_distribution = lambda x: np.exp(sess.run(log_posteriors(3, tf.convert_to_tensor(x, dtype='float32'))))

    hamil_func = lambda x: sess.run(hamiltonian_sampling(n_samples=tf.convert_to_tensor(x, dtype='int32'), 
                                    target_dist=lambda y: log_posteriors(3,y), 
                                    proposal_dist=proposal, 
                                    proposal_sampler=sample_proposal))
    plot_hi_isocontours(ax, target_distribution, proposal_distribution, hamil_func, k=k, n_sample_batches=n_batches, cmap='Reds')
    print 'g44 first part time', time.time() - start_time

    z0, zT, v0, vT, log_q_z0, log_q_zT, log_q_v0, log_q_vT, n_steps, friction = sess.run(hamiltonian_sampling(n_samples=tf.convert_to_tensor(5, dtype='int32'), 
                                    target_dist=lambda y: log_posteriors(3,y), 
                                    proposal_dist=proposal, 
                                    proposal_sampler=sample_proposal))
    print z0, zT
    plot_scatter_hamil_points(z0, zT)
    print 'g44 time', time.time() - start_time




    plt.show()
    DSFSD
























