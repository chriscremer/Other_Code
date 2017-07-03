


# compare qT approx to KDE


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


random_seed = 1


def plot_isocontours(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1.):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')


def plot_iw_isocontours(ax, p, q, sampler, k=3, xlimits=[-6, 6], 
                        ylimits=[-6, 6], numticks=101, 
                        n_sample_batches=3, cmap='Blues'):

    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T

    X_vec = np.reshape(X, [-1,1])
    Y_vec = np.reshape(Y, [-1,1])
    zipped = np.concatenate([X_vec, Y_vec], axis=1)

    # log_p = np.log(p(zipped))
    # log_q = np.log(q(zipped))
    log_p = p(zipped)
    log_q = q(zipped)
    log_w = log_p - log_q
    exp_log_p = np.exp(log_p)
    exp_log_w = np.exp(log_w)

    grid_values = np.zeros([numticks*numticks])

    for b in range(n_sample_batches):
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
    # print np.sum(grid_values) / len(grid_values)
    # print np.max(grid_values)
    grid_values = np.reshape(grid_values, [numticks, numticks])
    plt.contourf(X, Y, grid_values, cmap=cmap) #, vmin=10**(-10))
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')





def plot_hamil_kde_isocontours(ax, p, q, hamil_func, k=3, xlimits=[-6, 6], 
                        ylimits=[-6, 6], numticks=101, 
                        n_sample_batches=3, cmap='Blues'):

    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T

    X_vec = np.reshape(X, [-1,1])
    Y_vec = np.reshape(Y, [-1,1])
    zipped = np.concatenate([X_vec, Y_vec], axis=1)

    # log_p = np.log(p(zipped))
    # log_q = np.log(q(zipped))
    log_p = p(zipped)
    log_q = q(zipped)
    log_w = log_p - log_q
    exp_log_p = np.exp(log_p)
    exp_log_w = np.exp(log_w)

    grid_values = np.zeros([numticks*numticks])

    
    for b in range(n_sample_batches):



        if k >1:
            zT, log_w_ = hamil_func(k-1)

            #KDE
            if b == 0:
                samps = zT
            else:
                samps = np.concatenate([samps, zT], axis=0)

            # #HI
            # sum_p_q = np.sum(np.exp(log_w_))

        else:
            sum_p_q = 0

        # # #HI
        # denominator = (exp_log_w + sum_p_q) / k
        # grid_values += exp_log_p / denominator


    #KDE
    x_ = samps[:, 0]
    y_ = samps[:, 1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x_, y_])
    kernel = st.gaussian_kde(values)
    grid_values = np.reshape(kernel(positions).T, X.shape)

    # #HI
    # grid_values = np.reshape(grid_values, [numticks, numticks])
    # grid_values = grid_values / n_sample_batches


    plt.contourf(X, Y, grid_values, cmap=cmap) #, vmin=10**(-10))


    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')

    # print np.sum(grid_values) / len(grid_values)
    # print np.max(grid_values)





def plot_qT_approx_isocontours(ax, q0_z, q0_v, qT_v, sample_qT_v, reverse_hamil_func, k=3, xlimits=[-6, 6], 
                        ylimits=[-6, 6], numticks=101, 
                        n_sample_batches=3, cmap='Blues'):

    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T

    X_vec = np.reshape(X, [-1,1])
    Y_vec = np.reshape(Y, [-1,1])
    zipped = np.concatenate([X_vec, Y_vec], axis=1)

    grid_values = np.zeros([numticks*numticks])
    for k_i in range(k):

        # sample a velocites/momentums, and get their prob
        vT = sample_qT_v(1) #[1,D]
        log_qT_vT = qT_v(vT) #[1,1]
        N_ones = np.ones([numticks*numticks,1])
        vT_repeated = np.dot(N_ones, vT) #[N,D]

        # transform zT and vT back to z0,v0
        z0, v0 = reverse_hamil_func(numticks*numticks, zipped, vT_repeated) #[N,2]

        # get prob of z0,v0 under proposal
        # log_q0_z0 = np.log(np.maximum(q0_z(z0), np.exp(-40.))) #[N,1]
        log_q0_z0 = q0_z(z0) #[N,1]


        log_q0_v0 = q0_v(v0) #[N,1]


        # print log_q0_z0
        # print log_q0_v0
        # print log_qT_vT
        # prob of zT is average of k samples, of q(v0)q(z0)/p(vT)
        log_qT_zT = log_q0_z0 + log_q0_v0 - log_qT_vT #broadcasted I think
        # print log_qT_zT


        qT_zT = np.exp(log_qT_zT) #[N,1]
        # print qT_zT

        grid_values += qT_zT



    grid_values = np.reshape(grid_values, [numticks, numticks])
    grid_values = grid_values / k



    plt.contourf(X, Y, grid_values, cmap=cmap) #, vmin=10**(-10))

    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')







def plot_scatter_hamil_points(z0, zT):

    plt.scatter(z0.T[0], z0.T[1], color='yellow', marker='x')
    plt.scatter(zT.T[0], zT.T[1], color='green', marker='x')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.gca().set_aspect('equal', adjustable='box')


def plot_scatter_hamil_points3(z0, zT, reverse_z0):

    plt.scatter(z0.T[0], z0.T[1], color='yellow', marker='x')
    plt.scatter(zT.T[0], zT.T[1], color='green', marker='x')
    plt.scatter(reverse_z0.T[0], reverse_z0.T[1], color='orange', marker='x')
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



def log_posterior_0(x):
    '''
    x: [P,D]
    '''
    prob = (tf.exp(log_normal(x, [1,3], tf.ones(2)/100))
                + tf.exp(log_normal(x, [3,0], tf.ones(2)/100))
                + tf.exp(log_normal(x, [1,1], tf.ones(2)/100))
                + tf.exp(log_normal(x, [-3,-1], tf.ones(2)/100))
                + tf.exp(log_normal(x, [-1,-3], tf.ones(2)/100))
                + tf.exp(log_normal(x, [2,2], tf.ones(2)))
                + tf.exp(log_normal(x, [-2,-2], tf.ones(2)))
                + tf.exp(log_normal(x, [0,0], tf.ones(2))))
    prob = tf.maximum(prob, tf.exp(-40.))
    return tf.log(prob)

def log_posterior_1(x):
    '''
    x: [P,D]
    '''
    prob = (tf.exp(log_normal(x, [0.,4.], [2., -1.]))
                + tf.exp(log_normal(x, [0.,-4.], [2., -1.])))
    prob = tf.maximum(prob, tf.exp(-40.))
    return tf.log(prob)

def log_posterior_2(x):
    '''
    x: [P,D]
    '''
    prob = (tf.exp(log_normal(x, [-2.,2], [-2., 1.5]))
                    + tf.exp(log_normal(x, [2.,-1], [1.5, -2.])))
    prob = tf.maximum(prob, tf.exp(-40.))
    return tf.log(prob)

def log_posterior_3(x):
    '''
    x: [P,D]
    '''
    prob = (tf.exp(log_normal(x, [-4.,4], [0., 0.])))
    prob = tf.maximum(prob, tf.exp(-40.))
    return tf.log(prob)


def log_proposal(x):
    return log_normal(x, [0.,0.], [1., 1.])


def sample_proposal(n_samples):

    eps = tf.random_normal((n_samples, 2), 0, 1, dtype=tf.float32, seed=random_seed) #[P,D]
    sample_q = tf.add([0.,0.], tf.multiply(tf.sqrt(tf.exp([1.,1.])), eps)) #[P,D]

    return sample_q



def leapfrog_step2(prev_zv, current_step):
    '''
    prev_zv: [N,D+D]
    current_step: [1], ignore it
    '''

    z = tf.slice(prev_zv, [0,0], [n_samples_for_LF, 2]) #[N,2]
    v = tf.slice(prev_zv, [0,2], [n_samples_for_LF, 2]) #[N,2]

    grad = tf.gradients(-log_posterior_for_LF(z), [z])[0]
    v = v - ((.5*step_size_for_LF) * grad)

    z = z + (step_size_for_LF * v)

    grad = tf.gradients(-log_posterior_for_LF(z), [z])[0]
    v = v - ((.5*step_size_for_LF) * grad)

    v = v * friction_for_LF

    return tf.reshape(tf.concat([z, v], axis=1), [n_samples_for_LF, 4])



def reverse_leapfrog_step(zv, current_step):
    '''
    zv: [N,D+D]
    current_step: [1], ignore it
    '''

    zT = tf.slice(zv, [0,0], [n_samples_for_LF, 2]) #[N,2]
    vT = tf.slice(zv, [0,2], [n_samples_for_LF, 2]) #[N,2]

    vT = vT / friction_for_LF

    grad = tf.gradients(-log_posterior_for_LF(zT), [zT])[0]
    v_5 = vT + (.5*step_size_for_LF*grad)

    z0 = zT - (step_size_for_LF*v_5)

    grad = tf.gradients(-log_posterior_for_LF(z0), [z0])[0]
    v0= v_5 + (.5*step_size_for_LF * grad)

    return tf.reshape(tf.concat([z0, v0], axis=1), [n_samples_for_LF, 4])




def hamiltonian_sampling(n_samples, log_posterior, n_steps, friction):
    '''
    n_samples: [1]
    log_posterior: function that takes a 2d point
    '''

    global log_posterior_for_LF
    log_posterior_for_LF = log_posterior
    global n_samples_for_LF
    n_samples_for_LF = n_samples
    global friction_for_LF
    friction_for_LF = friction
    global step_size_for_LF
    step_size_for_LF = .1

    # n_steps = n_steps
    
    v_0 = tf.random_normal((n_samples, 2), 0, .5, dtype=tf.float32, seed=random_seed) #[N,D]
    log_q_v0 = log_normal(v_0, [0.,0.], tf.log(tf.square([.5,.5])))

    z_0 = sample_proposal(n_samples) #[N,D]
    log_q_z0 = log_proposal(z_0) #[N]

    scan_over_this = tf.ones([n_steps])  #*tf.cast(n_samples, 'float32') #[T]
    initializer = tf.concat([z_0, v_0], axis=1) # [N,D+D]

    zs_and_vs = tf.scan(leapfrog_step2, scan_over_this, initializer=initializer) # [T,N,2*D]

    #get last z and v
    zT = tf.slice(zs_and_vs, [n_steps-1,0,0], [1, n_samples, 2]) #[1,N,2]
    vT = tf.slice(zs_and_vs, [n_steps-1,0,2], [1, n_samples, 2]) #[1,N,2]
    zT = tf.reshape(zT, [n_samples, 2]) #[N,2]
    vT = tf.reshape(vT, [n_samples, 2]) #[N,2]

    # log_q_vT = log_normal(vT, [0.,0.], tf.log(tf.square([.5,.5])))
    log_q_zT = log_proposal(zT) #[N]

    logp_z = log_posterior(zT)

    log_w = logp_z + log_q_zT + (n_steps*tf.log(friction_for_LF)) - log_q_z0 - log_q_v0

    return zT, log_w, z_0, vT


def reverse_hamiltonian_sampling(n_samples, log_posterior, z_init, v_init, n_steps, friction):
    '''
    n_samples: [1]
    log_posterior: function that takes a 2d point
    '''

    global log_posterior_for_LF
    log_posterior_for_LF = log_posterior
    global n_samples_for_LF
    n_samples_for_LF = n_samples
    global friction_for_LF
    friction_for_LF = friction
    global step_size_for_LF
    step_size_for_LF = .1

    # n_steps = n_steps
    
    zT = z_init
    vT = v_init    
    # v_0 = tf.random_normal((n_samples, 2), 0, .5, dtype=tf.float32, seed=random_seed) #[N,D]
    # log_q_v0 = log_normal(v_0, [0.,0.], tf.sqrt(tf.exp([.5,.5])))

    # z_0 = sample_proposal(n_samples) #[N,D]
    # log_q_z0 = log_proposal(z_0) #[N]

    scan_over_this = tf.ones([n_steps])  #*tf.cast(n_samples, 'float32') #[T]
    initializer = tf.concat([zT, vT], axis=1) # [N,D+D]

    zs_and_vs = tf.scan(reverse_leapfrog_step, scan_over_this, initializer=initializer) # [T,N,2*D]

    #get last z and v
    z0 = tf.slice(zs_and_vs, [n_steps-1,0,0], [1, n_samples, 2]) #[1,N,2]
    v0 = tf.slice(zs_and_vs, [n_steps-1,0,2], [1, n_samples, 2]) #[1,N,2]
    z0 = tf.reshape(z0, [n_samples, 2]) #[N,2]
    v0 = tf.reshape(v0, [n_samples, 2]) #[N,2]

    # log_q_vT = log_normal(vT, [0.,0.], tf.sqrt(tf.exp([.5,.5])))
    # log_q_zT = log_proposal(zT) #[N]

    # logp_z = log_posterior(zT)

    # log_w = logp_z + log_q_zT + (n_steps*tf.log(friction_for_LF)) - log_q_z0 - log_q_v0

    return z0, v0











if __name__ == '__main__':


    for n_steps in range(21,61,5):
        
        print 'Initializing graph'
        print n_steps
        # n_steps = 6
        # friction = .9
        friction = 1.


        tf.reset_default_graph()

        

        #Placeholders - Inputs
        z = tf.placeholder(tf.float32, [None, 2])
        v = tf.placeholder(tf.float32, [None, 2])
        n_samples = tf.placeholder(tf.int32, [])


        #Functions
        logp0 = log_posterior_0(z)
        logp1 = log_posterior_1(z)
        logp2 = log_posterior_2(z)
        logp3 = log_posterior_3(z)

        q_samples = sample_proposal(n_samples)
        logq = log_proposal(z)

        zT_0, log_w0, z0_0, vT_0 = hamiltonian_sampling(n_samples, log_posterior_0, n_steps, friction)
        zT_1, log_w1, z0_1, vT_1 = hamiltonian_sampling(n_samples, log_posterior_1, n_steps, friction)
        zT_2, log_w2, z0_2, vT_2 = hamiltonian_sampling(n_samples, log_posterior_2, n_steps, friction)
        zT_3, log_w3, z0_3, vT_3 = hamiltonian_sampling(n_samples, log_posterior_3, n_steps, friction)


        r_z0_0, r_v0_0 = reverse_hamiltonian_sampling(n_samples, log_posterior_0, z, v, n_steps, friction)
        r_z0_1, r_v0_1 = reverse_hamiltonian_sampling(n_samples, log_posterior_1, z, v, n_steps, friction)
        r_z0_2, r_v0_2 = reverse_hamiltonian_sampling(n_samples, log_posterior_2, z, v, n_steps, friction)
        r_z0_3, r_v0_3 = reverse_hamiltonian_sampling(n_samples, log_posterior_3, z, v, n_steps, friction)


        # sample_v0 = tf.random_normal((n_samples, 2), 0, .5, dtype=tf.float32, seed=random_seed) #[N,D]
        log_q_v0 = log_normal(v, [0.,0.], tf.log(tf.square([.5,.5]))) #[N]

        # sample_vT = tf.random_normal((n_samples, 2), 0, .5*(.9**n_steps), dtype=tf.float32, seed=random_seed) #[N,D]
        # log_q_vT = log_normal(v, [0.,0.], tf.log(tf.square([.5*(.9**n_steps),.5*(.9**n_steps)]))) #[N]

        sample_vT = tf.random_normal((n_samples, 2), 0, .5, dtype=tf.float32, seed=random_seed) #[N,D]
        log_q_vT = log_normal(v, [0.,0.], tf.log(tf.square([.5,.5]))) #[N]

        

        #to make sure im not adding nodes to the graph
        tf.get_default_graph().finalize()

        print 'Plotting'


        fig = plt.figure(figsize=(8,8), facecolor='white')

        # sess = tf.Session()
        # sess.close()

        
        with tf.Session() as sess:

            # #Plot target distributions        
            # target_0 = lambda x: np.exp(sess.run(logp0, feed_dict={z: x}))
            # target_1 = lambda x: np.exp(sess.run(logp1, feed_dict={z: x}))
            # target_2 = lambda x: np.exp(sess.run(logp2, feed_dict={z: x}))
            # target_3 = lambda x: np.exp(sess.run(logp3, feed_dict={z: x}))

            #Plot target distributions        
            target_0 = lambda x: sess.run(logp0, feed_dict={z: x})
            target_1 = lambda x: sess.run(logp1, feed_dict={z: x})
            target_2 = lambda x: sess.run(logp2, feed_dict={z: x})
            target_3 = lambda x: sess.run(logp3, feed_dict={z: x})

            #Grid 1,1
            ax = fig.add_subplot(441, frameon=False)
            plot_isocontours(ax, target_0, cmap='Blues')
            ax.annotate('Posterior', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')
            #Grid 2,1
            ax = fig.add_subplot(445, frameon=False)
            plot_isocontours(ax, target_1, cmap='Blues')
            #Grid 3,1
            ax = fig.add_subplot(449, frameon=False)
            plot_isocontours(ax, target_2, cmap='Blues')
            #Grid 4,1
            ax = fig.add_subplot(4, 4, 13, frameon=False)
            plot_isocontours(ax, target_3, cmap='Blues')


            #Plot proposal distributions
            proposal_dist = lambda x: sess.run(logq, feed_dict={z: x})

            #Grid 1,2
            ax = fig.add_subplot(4, 4, 2, frameon=False)
            ax.annotate('Proposal', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')
            plot_isocontours(ax, proposal_dist, cmap='Reds')
            #Grid 2,2
            ax = fig.add_subplot(4, 4, 6, frameon=False)
            plot_isocontours(ax, proposal_dist, cmap='Reds')
            #Grid 3,2
            ax = fig.add_subplot(4, 4, 10, frameon=False)
            plot_isocontours(ax, proposal_dist, cmap='Reds')
            #Grid 4,2
            ax = fig.add_subplot(4, 4, 14, frameon=False)
            plot_isocontours(ax, proposal_dist, cmap='Reds')



        # with tf.Session() as sess:

            # qIW distributions
            n_batches = 5
            k = 1000
            alpha=.2
            # proposal_sampler = lambda x: sess.run(q_samples, feed_dict={n_samples: x})

            #Grid 1,3
            ax = fig.add_subplot(4, 4, 3, frameon=False)
            ax.annotate('qT KDE', xytext=(.2, 1.1), xy=(0, 1), textcoords='axes fraction')

            hamil_func = lambda x: sess.run([zT_0, log_w0], feed_dict={n_samples: x})
            plot_hamil_kde_isocontours(ax, target_0, proposal_dist, hamil_func, k=k, 
                                n_sample_batches=n_batches, cmap='Reds')
            plot_isocontours(ax, target_0, cmap='Blues', alpha=alpha)


            # plot_iw_isocontours(ax, target_0, proposal_dist, proposal_sampler, k=k, 
            #                     n_sample_batches=n_batches, cmap='Reds')
            # plot_isocontours(ax, target_0, cmap='Blues', alpha=alpha)
            


            # #Grid 2,3
            # ax = fig.add_subplot(4, 4, 7, frameon=False)
            # plot_iw_isocontours(ax, target_1, proposal_dist, proposal_sampler, k=k, 
            #                     n_sample_batches=n_batches, cmap='Reds')
            # plot_isocontours(ax, target_1, cmap='Blues', alpha=alpha)

            ax = fig.add_subplot(4, 4, 7, frameon=False)
            hamil_func = lambda x: sess.run([zT_1, log_w1], feed_dict={n_samples: x})
            plot_hamil_kde_isocontours(ax, target_1, proposal_dist, hamil_func, k=k, 
                                n_sample_batches=n_batches, cmap='Reds')
            plot_isocontours(ax, target_1, cmap='Blues', alpha=alpha)

            # #Grid 3,3
            # ax = fig.add_subplot(4, 4, 11, frameon=False)
            # plot_iw_isocontours(ax, target_2, proposal_dist, proposal_sampler, k=k, 
            #                     n_sample_batches=n_batches, cmap='Reds')
            # plot_isocontours(ax, target_2, cmap='Blues', alpha=alpha)

            ax = fig.add_subplot(4, 4, 11, frameon=False)
            hamil_func = lambda x: sess.run([zT_2, log_w2], feed_dict={n_samples: x})
            plot_hamil_kde_isocontours(ax, target_2, proposal_dist, hamil_func, k=k, 
                                n_sample_batches=n_batches, cmap='Reds')
            plot_isocontours(ax, target_2, cmap='Blues', alpha=alpha)


            # #Grid 4,3
            # ax = fig.add_subplot(4, 4, 15, frameon=False)
            # plot_iw_isocontours(ax, target_3, proposal_dist, proposal_sampler, k=k, 
            #                     n_sample_batches=n_batches, cmap='Reds')
            # plot_isocontours(ax, target_3, cmap='Blues', alpha=alpha)

            ax = fig.add_subplot(4, 4, 15, frameon=False)
            hamil_func = lambda x: sess.run([zT_3, log_w3], feed_dict={n_samples: x})
            plot_hamil_kde_isocontours(ax, target_3, proposal_dist, hamil_func, k=k, 
                                n_sample_batches=n_batches, cmap='Reds')
            plot_isocontours(ax, target_3, cmap='Blues', alpha=alpha)



        
        # with tf.Session() as sess:
            # Plot Hamiltonian distributions

            #skip it for now.


            # plt.show()
            save_to = home + '/Downloads/plots_'+str(n_steps)+'_steps_v5.png'
            plt.savefig(save_to)
            print 'saved to ' + save_to

            continue


            k = 1000

            # sess2 = tf.Session()
            sample_vT_func = lambda x: sess.run(sample_vT, feed_dict={n_samples: x})
            log_q_vT_func = lambda x: sess.run(log_q_vT, feed_dict={v: x})
            log_q_v0_func = lambda x: sess.run(log_q_v0, feed_dict={v: x})

            

            print 'qT approx0'
            #Grid 4,1

            ax = fig.add_subplot(4, 4, 4, frameon=False)
            reverse_hamil_func = lambda x1,x2,x3: sess.run([r_z0_0, r_v0_0], feed_dict={n_samples: x1, z: x2, v: x3})
            plot_qT_approx_isocontours(ax, proposal_dist, log_q_v0_func, log_q_vT_func, sample_vT_func, reverse_hamil_func, k=k, cmap='Reds')
            plot_isocontours(ax, target_0, cmap='Blues', alpha=alpha)
            ax.annotate('qT Reverse', xytext=(.2, 1.1), xy=(0, 1), textcoords='axes fraction')

            # zT, z0, r_z0 = sess.run([zT_0, z0_0, r_z0_0], feed_dict={n_samples: 5})
            # plot_scatter_hamil_points(z0, zT)
            # plot_scatter_hamil_points3(z0, zT, r_z0)


            print 'qT approx1'
            # #Grid 4,2

            ax = fig.add_subplot(4, 4, 8, frameon=False)
            reverse_hamil_func = lambda x1,x2,x3: sess.run([r_z0_1, r_v0_1], feed_dict={n_samples: x1, z: x2, v: x3})
            plot_qT_approx_isocontours(ax, proposal_dist, log_q_v0_func, log_q_vT_func, sample_vT_func, reverse_hamil_func, k=k, cmap='Reds')
            plot_isocontours(ax, target_1, cmap='Blues', alpha=alpha)

            # zT, z0, r_z0 = sess.run([zT_1, z0_1, r_z0_1], feed_dict={n_samples: 5})
            # # plot_scatter_hamil_points(z0, zT)
            # plot_scatter_hamil_points3(z0, zT, r_z0)

            print 'qT approx2'
            # #Grid 4,3
            # ax = fig.add_subplot(4, 4, 12, frameon=False)
            # hamil_func = lambda x: sess.run([zT_2, log_w2], feed_dict={n_samples: x})
            # plot_hamil_kde_isocontours(ax, target_2, proposal_dist, hamil_func, k=k, 
            #                     n_sample_batches=n_batches, cmap='Reds')
            # plot_isocontours(ax, target_2, cmap='Blues', alpha=alpha)
            ax = fig.add_subplot(4, 4, 12, frameon=False)
            reverse_hamil_func = lambda x1,x2,x3: sess.run([r_z0_2, r_v0_2], feed_dict={n_samples: x1, z: x2, v: x3})
            plot_qT_approx_isocontours(ax, proposal_dist, log_q_v0_func, log_q_vT_func, sample_vT_func, reverse_hamil_func, k=k, cmap='Reds')
            plot_isocontours(ax, target_2, cmap='Blues', alpha=alpha)

            # zT, z0, r_z0 = sess.run([zT_2, z0_2, r_z0_2], feed_dict={n_samples: 5})
            # # plot_scatter_hamil_points(z0, zT)
            # plot_scatter_hamil_points3(z0, zT, r_z0)



            print 'qT approx3'
            # #Grid 4,4
            # ax = fig.add_subplot(4, 4, 16, frameon=False)
            # hamil_func = lambda x: sess.run([zT_3, log_w3], feed_dict={n_samples: x})
            # plot_hamil_kde_isocontours(ax, target_3, proposal_dist, hamil_func, k=k, 
            #                     n_sample_batches=n_batches, cmap='Reds')
            # plot_isocontours(ax, target_3, cmap='Blues', alpha=alpha)

            ax = fig.add_subplot(4, 4, 16, frameon=False)
            reverse_hamil_func = lambda x1,x2,x3: sess.run([r_z0_3, r_v0_3], feed_dict={n_samples: x1, z: x2, v: x3})
            plot_qT_approx_isocontours(ax, proposal_dist, log_q_v0_func, log_q_vT_func, sample_vT_func, reverse_hamil_func, k=k, cmap='Reds')
            plot_isocontours(ax, target_3, cmap='Blues', alpha=alpha)

            # zT, z0, r_z0 = sess.run([zT_3, z0_3, r_z0_3], feed_dict={n_samples: 5})
            # # plot_scatter_hamil_points(z0, zT)
            # plot_scatter_hamil_points3(z0, zT, r_z0)





        # plt.show()
        save_to = home + '/Downloads/plots_'+str(n_steps)+'_steps_v5.png'
        plt.savefig(save_to)
        print 'saved to ' + save_to




    print 'Done.'
























