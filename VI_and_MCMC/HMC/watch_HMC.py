




#Hamiltonian Monte Carlo

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import scipy.stats as st


from HMC import HMC


def log_normal(x, mean, log_var, x_dim):
    '''
    x is [P, D]
    mean is [D]
    log_var is [D]
    x_dim is [1]
    '''

    log_var = tf.cast(log_var, tf.float32)
    mean = tf.cast(mean, tf.float32)
    term1 = tf.cast(x_dim * tf.log(2*math.pi), tf.float32) #[1]
    term2 = tf.reduce_sum(log_var) #sum over D, [1]
    term3 = tf.square(x - mean) / tf.exp(log_var)
    term3 = tf.reduce_sum(term3, 1) #sum over D, [P]
    all_ = term1 + term2 + term3 #broadcast to [P]
    log_normal_ = -.5 * all_  
    return log_normal_


def log_p_true(x, x_dim):

    return tf.log(  
                tf.exp(log_normal(x, [1,3], tf.ones(x_dim)/100, x_dim))
                + tf.exp(log_normal(x, [3,0], tf.ones(x_dim)/100, x_dim))
                + tf.exp(log_normal(x, [1,1], tf.ones(x_dim)/100, x_dim))
                + tf.exp(log_normal(x, [-3,-1], tf.ones(x_dim)/100, x_dim))
                + tf.exp(log_normal(x, [-1,-3], tf.ones(x_dim)/100, x_dim))
                + tf.exp(log_normal(x, [2,2], tf.ones(x_dim), x_dim))
                + tf.exp(log_normal(x, [-2,-2], tf.ones(x_dim), x_dim))
                + tf.exp(log_normal(x, [0,0], tf.ones(x_dim), x_dim))
                )


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



def plot_kde(ax, samples, xlimits=[-6, 6], ylimits=[-6, 6], numticks=101, cmap=None):

    x_ = samples[:, 0]
    y_ = samples[:, 1]

    xx, yy = np.mgrid[xlimits[0]:xlimits[1]:100j, ylimits[0]:ylimits[1]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x_, y_])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([])







if __name__ == '__main__':


    # sess = tf.Session() #For target distribution

    infmet = HMC(log_p_true) #Inference method


    fig = plt.figure(figsize=(12,5), facecolor='white')
    plt.ion()
    plt.show(block=False)


    ax = fig.add_subplot(141, frameon=False)
    target_distribution = lambda x: np.exp(infmet.sess.run(log_p_true(x, 2)))
    plot_isocontours(ax, target_distribution, cmap='Reds')
    plt.gca().set_aspect('equal', adjustable='box')
    

    # mean = [[0.,0.]]

    step_size = .1
    n_steps = 100
    pos = np.array([[0.,0.]])


    samps_kept = []

    for iters in range(100):


        positions_in_this_trajectory = []
        positions_in_this_trajectory.append(pos)

        # Sample momentum
        momentum = infmet.sess.run(infmet.sample_q)

        init_pos = pos
        init_momentum = momentum

        # Do leapfrogs to get new proposed state

        #Update velocity/momemtum a half step
        # pos = np.reshape(pos, [1,2])
        # aaa= (.5*step_size)
        # bbb = infmet.sess.run(infmet.log_p_grad, feed_dict={infmet.pos: pos})
        # grad = np.reshape(bbb[0], [2])
        # # print bbb[0].shape
        # thing = aaa * grad

        grad = -np.reshape(infmet.sess.run(infmet.log_p_grad, feed_dict={infmet.pos: pos})[0], [2])
        momentum = momentum - ((.5*step_size) * grad)

        # momentum = momentum - ((.5*step_size) * infmet.sess.run(infmet.log_p_grad, feed_dict={infmet.pos: pos}))
        
        #Update the position by a full step
        pos = pos + (step_size * momentum)

        for t in range(n_steps):

            #Update velocity/momemtum a full step
            grad = -np.reshape(infmet.sess.run(infmet.log_p_grad, feed_dict={infmet.pos: pos})[0], [2])
            momentum = momentum - (step_size * grad)

            #Update the position by a full step
            pos = pos + (step_size * momentum)

            # print pos
            positions_in_this_trajectory.append(pos)

            # momentum = .98*momentum

        #Update velocity/momemtum a half step
        grad = -np.reshape(infmet.sess.run(infmet.log_p_grad, feed_dict={infmet.pos: pos})[0], [2])
        momentum = momentum - ((.5*step_size) * grad)

        # Hamiltonian H = U + K
        H_0 = -infmet.sess.run(infmet.log_p, feed_dict={infmet.pos: init_pos}) + np.sum(init_momentum**2 / 2.)
        H_T = -infmet.sess.run(infmet.log_p, feed_dict={infmet.pos: pos}) + np.sum(momentum**2 / 2.)


        # print H_0, H_T
        # Accept/Reject
        # accept with prob min(1, exp(orig-current))
        p_accept = min(1., np.exp(H_0 - H_T))

        if p_accept < np.random.uniform():
            pos = init_pos
            #else its still pos

        samps_kept.append(pos)

        #     return x
        # else:
        #     return init_x



        # fasdf

 
        # mean = np.reshape(mean, [2])
        # samp = infmet.sess.run(infmet.sample_q, feed_dict={infmet.mean: mean})



        # proposal_distribution = lambda x: np.exp(infmet.sess.run(log_normal(x, mean, infmet.sess.run(infmet.log_var), 2)))


        ax = fig.add_subplot(142, frameon=False)
        plt.cla()

        plot_isocontours(ax, target_distribution, cmap='Reds')    
        # plot_isocontours(ax, proposal_distribution, cmap='Blues')
        plt.gca().set_aspect('equal', adjustable='box')
        ax.annotate('iter '+str(iters), xytext=(0, 1), xy=(0, 1), textcoords='axes fraction')
        
        # samp = np.reshape(samp, [2])
        # mean = np.reshape(mean, [2])
        # print samp, mean



        positions_in_this_trajectory = np.array(positions_in_this_trajectory)
        positions_in_this_trajectory = np.reshape(positions_in_this_trajectory, [n_steps+1, 2])

        # print positions_in_this_trajectory
        # fsdfa
        # print positions_in_this_trajectory.shape
        # for i in range(len(positions_in_this_trajectory-1)):

        init_pos = np.reshape(init_pos, [2])
        init_momentum = np.reshape(init_momentum, [2])

        plt.plot([init_pos[0], init_pos[0]+init_momentum[0]], [init_pos[1], init_pos[1]+init_momentum[1]])

        plt.plot(positions_in_this_trajectory.T[0], positions_in_this_trajectory.T[1])







        # samp = np.reshape(samp, [1, 2])
        # mean = np.reshape(mean, [1, 2])

        # p_x = infmet.sess.run(tf.exp(log_p_true(samp, 2)))
        # p_x_prev = infmet.sess.run(tf.exp(log_p_true(mean, 2)))

        # ratio = p_x / p_x_prev

        # if ratio >= 1.:
        #     #accept
        #     mean = samp
        # else:
        #     r = np.random.rand()
        #     #accept
        #     if ratio > r:
        #         mean = samp



        # mean = np.reshape(mean, [2])
        # samps_kept.append(mean)

        # print samps_kept_array.shape
        # fsdf

        #PLOT Scatter of samples
        samps_kept_array = np.array(samps_kept)
        samps_kept_array = np.reshape(samps_kept_array, [len(samps_kept_array), 2])

        ax = fig.add_subplot(143, frameon=False)
        plt.cla()
        plt.scatter(samps_kept_array.T[0], samps_kept_array.T[1])
        ax.set_yticks([])
        ax.set_xticks([])
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plot_isocontours(ax, target_distribution, cmap='Reds') 
        plt.gca().set_aspect('equal', adjustable='box')


        #PLOT KDE of samples
        if iters % 5 == 2:

            ax = fig.add_subplot(144, frameon=False)
            plt.cla()

            to_plot = samps_kept_array
            # print to_plot.shape
            plot_kde(ax, to_plot, cmap='Blues')

            plot_isocontours(ax, target_distribution, cmap='Reds') 
            plt.gca().set_aspect('equal', adjustable='box')




        plt.draw()
        plt.pause(1.0/100.0)





    ax.annotate('iter '+str(iters)+'done', xytext=(0, 1), xy=(0, 1), textcoords='axes fraction')
    plt.show(block=True)

    print 'Done'













