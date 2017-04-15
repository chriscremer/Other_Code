


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

    start_time = time.time()
    for b in range(n_sample_batches):
        if b % 10 == 0:
            # elapsed_time = time.time() - start_time
            # start_time = time.time()
            # print elapsed_time
            print b

        if k >1:
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
        return tf.log(  
                    tf.exp(log_normal(x, [1,3], tf.ones(2)/100))
                    + tf.exp(log_normal(x, [3,0], tf.ones(2)/100))
                    + tf.exp(log_normal(x, [1,1], tf.ones(2)/100))
                    + tf.exp(log_normal(x, [-3,-1], tf.ones(2)/100))
                    + tf.exp(log_normal(x, [-1,-3], tf.ones(2)/100))
                    + tf.exp(log_normal(x, [2,2], tf.ones(2)))
                    + tf.exp(log_normal(x, [-2,-2], tf.ones(2)))
                    + tf.exp(log_normal(x, [0,0], tf.ones(2)))
                    )
    if index_ == 1:
        return tf.log(  
                    tf.exp(log_normal(x, [0.,4.], [2., -3.]))
                    + tf.exp(log_normal(x, [0.,-4.], [2., -3.]))
                    )
    if index_ == 2:
        return tf.log(  
                    tf.exp(log_normal(x, [-2.,2], [-2., 1.5]))
                    + tf.exp(log_normal(x, [2.,-1], [1.5, -2.]))
                    )
    if index_ == 3:
        return tf.log(  
                    tf.exp(log_normal(x, [-4.,4], [0., 0.]))
                    )


def proposal(x):

    return log_normal(x, [0.,0.], [1., 1.])


def sample_proposal(n_samples):
        
    eps = tf.random_normal((n_samples, 2), 0, 1, dtype=tf.float32) #[P,D]
    sample_q = tf.add([0.,0.], tf.multiply(tf.sqrt(tf.exp([1.,1.])), eps)) #[P,D]

    return sample_q





def leapfrog_step(z, v, logprob, step_size):

    grad = tf.gradients(-logprob(z), [z])[0]
    v = v - ((.5*step_size) * grad)

    z = z + (step_size * v)

    grad = tf.gradients(-logprob(z), [z])[0]
    v = v - ((.5*step_size) * grad)

    return z, v


def hamiltonian_sampling(n_samples, target_dist, proposal_dist, proposal_sampler):

    n_steps = 10
    step_size = .1
    friction = .95
        
    v_0 = tf.random_normal((n_samples, 2), 0, .5, dtype=tf.float32) #[N,D]
    log_q_v0 = log_normal(v_0, [0.,0.], tf.sqrt(tf.exp([.5,.5])))

    z_0 = proposal_sampler(n_samples) #[N,D]
    log_q_z0 = proposal_dist(z_0) #[N]

    z = z_0
    v = v_0

    for i in range(n_steps):

        z, v = leapfrog_step(z, v, target_dist, step_size)

    log_q_vT = log_normal(v, [0.,0.], tf.sqrt(tf.exp([.5,.5])))
    log_q_zT = proposal_dist(z) #[N]


    # w = log_q_zT + log_q_zT - log_q_v0 - log_q_z0


    return z_0, z





if __name__ == '__main__':


    sess = tf.Session()


    fig = plt.figure(figsize=(8,8), facecolor='white')
    n_batches = 100


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



    # #Grid 1,3
    # k = 10
    # ax = fig.add_subplot(4, 4, 3, frameon=False)
    # target_distribution = lambda x: np.exp(sess.run(log_posteriors(0, tf.convert_to_tensor(x, dtype='float32'))))
    # proposal_distribution = lambda x: np.exp(sess.run(proposal(tf.convert_to_tensor(x, dtype='float32'))))
    # proposal_sampler = lambda x: sess.run(sample_proposal(tf.convert_to_tensor(x, dtype='int32')))
    # plot_iw_isocontours(ax, target_distribution, proposal_distribution, proposal_sampler, k=k, cmap='Reds')
    # ax.annotate('Importance', xytext=(.2, 1.1), xy=(0, 1), textcoords='axes fraction')

    # #Grid 2,3
    # ax = fig.add_subplot(4, 4, 7, frameon=False)
    # target_distribution = lambda x: np.exp(sess.run(log_posteriors(1, tf.convert_to_tensor(x, dtype='float32'))))
    # plot_iw_isocontours(ax, target_distribution, proposal_distribution, proposal_sampler, k=k, cmap='Reds')

    # #Grid 3,3
    # ax = fig.add_subplot(4, 4, 11, frameon=False)
    # target_distribution = lambda x: np.exp(sess.run(log_posteriors(2, tf.convert_to_tensor(x, dtype='float32'))))
    # plot_iw_isocontours(ax, target_distribution, proposal_distribution, proposal_sampler, k=k, cmap='Reds')

    # #Grid 4,3
    # ax = fig.add_subplot(4, 4, 15, frameon=False)
    # target_distribution = lambda x: np.exp(sess.run(log_posteriors(3, tf.convert_to_tensor(x, dtype='float32'))))
    # plot_iw_isocontours(ax, target_distribution, proposal_distribution, proposal_sampler, k=k, cmap='Reds')




    #Grid 4,1
    ax = fig.add_subplot(4, 4, 4, frameon=False)
    z0, zT = sess.run(hamiltonian_sampling(n_samples=tf.convert_to_tensor(5, dtype='int32'), 
                                    target_dist=lambda x: log_posteriors(0,x), 
                                    proposal_dist=proposal, 
                                    proposal_sampler=sample_proposal))
    plt.scatter(z0.T[0], z0.T[1], color='blue')
    plt.scatter(zT.T[0], zT.T[1], color='red')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.annotate('Hamiltonian', xytext=(.2, 1.1), xy=(0, 1), textcoords='axes fraction')


    #Grid 4,2
    ax = fig.add_subplot(4, 4, 8, frameon=False)
    z0, zT = sess.run(hamiltonian_sampling(n_samples=tf.convert_to_tensor(5, dtype='int32'), 
                                    target_dist=lambda x: log_posteriors(1,x), 
                                    proposal_dist=proposal, 
                                    proposal_sampler=sample_proposal))
    plt.scatter(z0.T[0], z0.T[1], color='blue')
    plt.scatter(zT.T[0], zT.T[1], color='red')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.gca().set_aspect('equal', adjustable='box')


    #Grid 4,3
    ax = fig.add_subplot(4, 4, 12, frameon=False)
    z0, zT = sess.run(hamiltonian_sampling(n_samples=tf.convert_to_tensor(5, dtype='int32'), 
                                    target_dist=lambda x: log_posteriors(2,x), 
                                    proposal_dist=proposal, 
                                    proposal_sampler=sample_proposal))
    plt.scatter(z0.T[0], z0.T[1], color='blue')
    plt.scatter(zT.T[0], zT.T[1], color='red')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.gca().set_aspect('equal', adjustable='box')


    #Grid 4,4
    ax = fig.add_subplot(4, 4, 16, frameon=False)
    z0, zT = sess.run(hamiltonian_sampling(n_samples=tf.convert_to_tensor(5, dtype='int32'), 
                                    target_dist=lambda x: log_posteriors(3,x), 
                                    proposal_dist=proposal, 
                                    proposal_sampler=sample_proposal))
    plt.scatter(z0.T[0], z0.T[1], color='blue')
    plt.scatter(zT.T[0], zT.T[1], color='red')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.gca().set_aspect('equal', adjustable='box')





    plt.show()
    DSFSD
























