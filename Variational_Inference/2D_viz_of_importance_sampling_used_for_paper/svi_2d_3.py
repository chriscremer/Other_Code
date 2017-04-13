



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
import scipy.stats as st



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


def iw_sampling(SVI):

    #Sample from the recognition model, k times, calculate their weights, sample one from a categorical dist
    samples, log_ws = SVI.sess.run([SVI.sample_q, SVI.log_w])
    # print 'sa'
    # print samples
    # print 'lw'
    # print log_ws

    max_ = np.max(log_ws)
    aaa = np.exp(log_ws - max_)
    normalized_ws = aaa / np.sum(aaa)

    if sum(normalized_ws) > 1.:
        normalized_ws = normalized_ws - .000001

    sampled = np.random.multinomial(1, normalized_ws)#, size=1)

    sampled = samples[np.argmax(sampled)]
    # print sampled
    # dfsaf

    # weighted_samples = []
    # for i in range(len(sampled)):
    #     for j in range(sampled[i]):
    #         weighted_samples.append(samples[i])
    return sampled






class Gaussian_SVI(object):

    def __init__(self, n_particles=1, mean=None, logvar=None):

        # tf.reset_default_graph()

        self.n_particles = n_particles
        self.D = 2

        if mean ==None:
            self.mean = tf.Variable([5.,-2.])
            self.log_var = tf.Variable([1.,1.])
        else:
            self.mean = tf.Variable(mean)
            self.log_var = tf.Variable(logvar)          

        #Sample
        self.eps = tf.random_normal((self.n_particles, self.D), 0, 1, dtype=tf.float32) #[P,D]
        self.sample_q = tf.add(self.mean, tf.mul(tf.sqrt(tf.exp(self.log_var)), self.eps)) #[P,D]

        #Calc log probs
        self.log_px = self.log_p_z(self.sample_q) #[P]
        self.log_qz = self.log_q_z(self.sample_q) #[P]
        self.log_w = self.log_px - self.log_qz #[P]

        #Redular Objective
        # self.elbo = tf.reduce_mean(self.log_w) #[1]

        # #IW Objective: logmeanexp
        max_ = tf.reduce_max(self.log_w)
        self.elbo = tf.log(tf.reduce_mean(tf.exp(self.log_w-max_))) +max_ #[1]

        # Optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=.01, epsilon=1e-04).minimize(-self.elbo)

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


    def log_p_z(self, x):

        # return self.log_normal(x, tf.zeros(self.D), tf.ones(self.D))
        return tf.log(  tf.exp(self.log_normal(x, tf.zeros(self.D)-3, tf.ones(self.D))) + tf.exp(self.log_normal(x, tf.zeros(self.D)+2, tf.ones(self.D))))



    def log_density(self, x):
        # x_, y_ = x[:, 0], x[:, 1]
        # return np.logaddexp(norm.logpdf(x_, 0,1)+norm.logpdf(y_, 0,1), norm.logpdf(x_, 3,1)+norm.logpdf(y_, 2,1))
        # return norm.logpdf(x_, 0,2)+norm.logpdf(y_, 0,1)

        # print x.shape
        # fdsaf

        ps = []
        for p in range(len(x)):

            # px = G_SVI.sess.run(G_SVI.log_px, feed_dict={G_SVI.sample_q: x[p]})
            hack = []
            hack.append(x[p])
            while len(hack) < self.n_particles:
                hack.append(np.zeros(self.D))

            p_x = G_SVI.sess.run(G_SVI.log_px, feed_dict={G_SVI.sample_q: hack})
            px = p_x[0]

            ps.append(px)

        return ps

        




if __name__ == '__main__':

    n_particles = 10
    n_particles_viz = 100

    #for training
    G_SVI = Gaussian_SVI(n_particles=n_particles)


    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)

    plt.ion()
    plt.show(block=False)

    print 'Performing SVI'
    for i in range(5000):

        #Sample q distribution
        # x, eps = G_SVI.sess.run([G_SVI.sample_q, G_SVI.eps]) #[P,D]

        #Get log_p_x
        # log_px = G_SVI.sess.run(G_SVI.log_px, feed_dict={G_SVI.sample_q: x})  #[P]

        #Optimize variational params 
        # _ = G_SVI.sess.run(G_SVI.optimizer, feed_dict={G_SVI.eps: eps, G_SVI.log_px: log_px})
        _ = G_SVI.sess.run(G_SVI.optimizer)


        if i%500==0:
            # elbo, mean, logvar = G_SVI.sess.run([G_SVI.elbo, G_SVI.mean, G_SVI.log_var], feed_dict={G_SVI.eps: eps, G_SVI.log_px: log_px})
            # print i, elbo, mean, logvar#, grad

            elbo, mean, logvar = G_SVI.sess.run([G_SVI.elbo, G_SVI.mean, G_SVI.log_var])
            print i, elbo, mean, logvar#, grad

            #Visualize while training
            plt.cla()

            mean_, log_var_ = G_SVI.sess.run([G_SVI.mean, G_SVI.log_var])
            G_SVI_for_viz = Gaussian_SVI(n_particles=n_particles_viz, mean=mean_, logvar=log_var_)
            

            # print x.shape
            w_samples = []
            for j in range(1000):
                w_samples.append(iw_sampling(G_SVI_for_viz))
            w_samples = np.array(w_samples)
            # print np.array(w_samples).shape

            x_ = w_samples[:, 0]
            y_ = w_samples[:, 1]

            # Peform the kernel density estimate
            xmin, xmax = -6, 6
            ymin, ymax = -6, 6

            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([x_, y_])
            kernel = st.gaussian_kde(values)
            f = np.reshape(kernel(positions).T, xx.shape)

            cfset = ax.contourf(xx, yy, f, cmap='Blues')
            plt.show()


            target_distribution = lambda x: np.exp(G_SVI.log_density(x))
            plot_isocontours(ax, target_distribution, cmap='Greens')
            var_distribution    = lambda x: np.exp(log_variational(x, mean, logvar))
            plot_isocontours(ax, var_distribution, cmap='Reds')

            ax.set_autoscale_on(False)
            plt.draw()
            plt.pause(1.0/30.0)




    target_distribution = lambda x: np.exp(G_SVI.log_density(x))
    plot_isocontours(ax, target_distribution, cmap='Blues')
    var_distribution    = lambda x: np.exp(log_variational(x, mean, logvar))
    plot_isocontours(ax, var_distribution, cmap='Reds')
    plt.show()




















