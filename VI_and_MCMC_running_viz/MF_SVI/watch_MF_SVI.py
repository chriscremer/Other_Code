
# import method, could be MCMC, VI, IW-VI, NF, HMC, MH
# 


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import scipy.stats as st


from MF_SVI import MF_SVI

# inference_method = 'MF_SVI'


def log_normal(x, mean, log_var, x_dim):
    '''
    x is [P, D]
    mean is [D]
    log_var is [D]
    x_dim is [1]
    '''

    log_var = tf.cast(log_var, tf.float32)
    mean = tf.cast(mean, tf.float32)
    # x_dim = tf.cast(mean, tf.float32)
    term1 = tf.cast(x_dim * tf.log(2*math.pi), tf.float32) #[1]
    term2 = tf.reduce_sum(log_var) #sum over D, [1]
    term3 = tf.square(x - mean) / tf.exp(log_var)
    term3 = tf.reduce_sum(term3, 1) #sum over D, [P]
    all_ = term1 + term2 + term3 #broadcast to [P]
    log_normal = -.5 * all_  
    return log_normal


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

    mf_svi = MF_SVI(log_p_true) #Inference method


    fig = plt.figure(figsize=(12,5), facecolor='white')
    plt.ion()
    plt.show(block=False)


    ax = fig.add_subplot(141, frameon=False)
    target_distribution = lambda x: np.exp(mf_svi.sess.run(log_p_true(x, 2)))
    plot_isocontours(ax, target_distribution, cmap='Reds')
    plt.gca().set_aspect('equal', adjustable='box')
    



    for iters in range(50):


        mf_svi.sess.run(mf_svi.optimizer)


        approx_distribution = lambda x: np.exp(mf_svi.sess.run(log_normal(x, mf_svi.mean, mf_svi.log_var, 2)))


        if iters % 10 == 0:

            samples = mf_svi.sess.run(mf_svi.sample_q)
            while len(samples)<100:
                batch = mf_svi.sess.run(mf_svi.sample_q)
                samples = np.concatenate([samples, batch], axis=0)

            ax = fig.add_subplot(144, frameon=False)
            plt.cla()

            plot_kde(ax, samples, cmap='Blues')
            plt.gca().set_aspect('equal', adjustable='box')



        ax = fig.add_subplot(143, frameon=False)
        plt.cla()

        plot_isocontours(ax, target_distribution, cmap='Reds')    
        plot_isocontours(ax, approx_distribution, cmap='Blues')
        plt.gca().set_aspect('equal', adjustable='box')
        ax.annotate('iter '+str(iters), xytext=(0, 1), xy=(0, 1), textcoords='axes fraction')
        


        plt.draw()
        plt.pause(1.0/100.0)





    ax.annotate('iter '+str(iters)+'done', xytext=(0, 1), xy=(0, 1), textcoords='axes fraction')
    plt.show(block=True)

    fsdafsd






    plt.cla()
    target_distribution = lambda x: np.exp(log_density(x, t))
    var_distribution    = lambda x: np.exp(variational_log_density(params, x))
    plot_isocontours(ax, target_distribution)
    plot_isocontours(ax, var_distribution, cmap=plt.cm.bone)
    ax.set_autoscale_on(False)


    rs = npr.RandomState(0)
    samples = variational_sampler(params, num_plotting_samples, rs)
    plt.plot(samples[:, 0], samples[:, 1], 'x')

    plt.draw()
    plt.pause(1.0/30.0)












