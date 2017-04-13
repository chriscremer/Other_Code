

# Mean-Field Stochastic Variational Inference


import numpy as np
import tensorflow as tf
import math


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






class MF_SVI(object):

    def __init__(self, log_p_true):

        # tf.reset_default_graph()
        # self.log_p_true_ = self.log_p_true
        self.n_particles = 5
        self.D = 2

        #Variational Approximation
        self.mean = tf.Variable([0.,0.])
        self.log_var = tf.Variable([1.,1.])

        #Sample
        self.eps = tf.random_normal((self.n_particles, self.D), 0, 1, dtype=tf.float32) #[P,D]
        self.sample_q = tf.add(self.mean, tf.multiply(tf.sqrt(tf.exp(self.log_var)), self.eps)) #[P,D]

        #ELBO Objective
        self.log_p = log_p_true(self.sample_q, self.D) #[P]
        self.log_q = log_normal(self.sample_q, self.mean, self.log_var, self.D)  #[P]
        self.log_w = self.log_p - self.log_q #[P]

        #Redular Objective
        self.elbo = tf.reduce_mean(self.log_w) #[1]

        # Optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=.1, epsilon=1e-04).minimize(-self.elbo)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # print self.sess.run(tf.gradients(self.elbo, [self.mean]))










