



import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import imageio
import time
import matplotlib.pyplot as plt
from scipy.stats import norm



def log_density(x):
    x_, y_ = x[:, 0], x[:, 1]
    return np.logaddexp(norm.logpdf(x_, 0,1)+norm.logpdf(y_, 0,1), norm.logpdf(x_, 3,1)+norm.logpdf(y_, 2,1))
    # return norm.logpdf(x_, 0,2)+norm.logpdf(y_, 0,1)

def log_variational(x, mean, logvar):
    x_, y_ = x[:, 0], x[:, 1]
    return norm.logpdf(x_, mean[0], np.sqrt(np.exp(logvar[0]))) + norm.logpdf(y_, mean[1], np.sqrt(np.exp(logvar[1])))




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


class Gaussian_SVI():

    def __init__(self, n_particles=1):

        self.n_particles = n_particles
        self.D = 2

        self.mean = tf.Variable([-2.,-2.])
        self.log_var = tf.Variable([1.,1.])

        #Sample
        self.eps = tf.random_normal((self.n_particles, self.D), 0, 1, dtype=tf.float32)
        self.sample_q = tf.add(self.mean, tf.mul(tf.sqrt(tf.exp(self.log_var)), self.eps))

        #Placeholders
        # self.x = tf.placeholder(tf.float32, [self.n_particles, self.D])
        self.log_px = tf.placeholder(tf.float32, [self.n_particles])
        self.log_qz = self.log_q_z(self.sample_q)
        # print self.log_qz #[P]

        #Objective
        self.elbo = tf.reduce_mean(self.log_px - self.log_qz) #[1]
        # Optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=.001, epsilon=1e-04).minimize(-self.elbo)

        self.grad = tf.gradients(self.elbo, self.mean)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def log_normal(self, x, mean, log_var):
        '''
        x is [P, D]
        mean is [D]
        log_var is [D]
        '''

        term1 = self.D * tf.log(2*math.pi)
        term2 = tf.reduce_sum(log_var) #sum over dimensions, now its a scalar [1]

        term3 = tf.square(x - mean) / tf.exp(log_var)
        term3 = tf.reduce_sum(term3, 1) #sum over dimensions so now its [particles]

        all_ = term1 + term2 + term3
        log_normal = -.5 * all_  

        return log_normal

    def log_q_z(self, x):

        log_q_z = self.log_normal(x, self.mean, self.log_var) 
        return log_q_z


    # def log_p_z(self, )



if __name__ == '__main__':

    n_particles = 50

    G_SVI = Gaussian_SVI(n_particles=n_particles)


    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)

    plt.ion()
    plt.show(block=False)

    print 'Performing SVI'
    for i in range(10000):

        #Sample q distribution
        x, eps = G_SVI.sess.run([G_SVI.sample_q, G_SVI.eps])
        # print x.shape [P,D]

        #Get log_p_x
        log_px = log_density(x)
        # print log_px.shape [P]

        #Optimize variational params 
        _ = G_SVI.sess.run(G_SVI.optimizer, feed_dict={G_SVI.eps: eps, G_SVI.log_px: log_px})

        if i%10==0:
            elbo, mean, logvar, grad = G_SVI.sess.run([G_SVI.elbo, G_SVI.mean, G_SVI.log_var, G_SVI.grad], feed_dict={G_SVI.eps: eps, G_SVI.log_px: log_px})
            print i, elbo, mean, logvar#, grad


            #Visualize while training
            plt.cla()

            target_distribution = lambda x: np.exp(log_density(x))
            plot_isocontours(ax, target_distribution, cmap='Blues')
            var_distribution    = lambda x: np.exp(log_variational(x, mean, logvar))
            plot_isocontours(ax, var_distribution, cmap='Reds')

            ax.set_autoscale_on(False)
            plt.draw()
            plt.pause(1.0/30.0)




    target_distribution = lambda x: np.exp(log_density(x))
    plot_isocontours(ax, target_distribution, cmap='Blues')

    var_distribution    = lambda x: np.exp(log_variational(x, mean, logvar))
    plot_isocontours(ax, var_distribution, cmap='Reds')

    plt.show()




















