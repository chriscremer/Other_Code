
#Bayesian Neural Network

import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle




class BNN(object):

    def __init__(self, network_architecture):
        
        tf.reset_default_graph()

        #Model hyperparameters
        self.act_func = tf.nn.softplus #tf.tanh
        self.learning_rate = .0001
        self.batch_size = 20
        self.n_particles = 3
        self.rs = 0
        # self.n_input = network_architecture["n_input"]


        #Model
        self.model(network_architecture)


        
        # # #Variables
        # # self.network_weights, self.n_decoder_weights = self._initialize_weights(network_architecture)

        # #Sample weights
        # self.sampled_theta, log_p_theta_, log_q_theta_ = self.sample_weights(self.network_weights['decoder_weights'])

        # #Encoder - Recognition model - q(z|x): recog_mean,z_log_std_sq=[batch_size, n_z]
        # self.recog_means, self.recog_log_vars = self._recognition_network(self.x, self.network_weights['encoder_weights']) #, self.network_weights['encoder_biases'])
        
        # #Sample z
        # eps = tf.random_normal((self.n_particles, self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        # self.z = tf.add(self.recog_means, tf.multiply(tf.sqrt(tf.exp(self.recog_log_vars)), eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]
        
        # #Decoder - Generative model - p(x|z)
        # self.x_reconstr_mean_no_sigmoid = self._generator_network(self.z, self.sampled_theta)#, self.network_weights['decoder_biases']) #no sigmoid

        # #Objective
        # self.elbo = self.elbo(self.x, self.x_reconstr_mean_no_sigmoid, self.z, self.recog_means, self.recog_log_vars, log_p_theta_, log_q_theta_)

        # # Use ADAM optimizer
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(-self.elbo)

        # #For evaluation
        # self.log_w = self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid) + self._log_p_z(self.z) - self._log_q_z_given_x(self.z, self.recog_means, self.recog_log_vars)
        # self.x_reconstr_mean = tf.nn.sigmoid(self.x_reconstr_mean_no_sigmoid)



        #to make sure im not adding nodes to the graph
        tf.get_default_graph().finalize()


    def log_normal(self, position, mean, log_var):
        '''
        Log of normal distribution
        position is [P, B, D]
        mean is [B, D]
        log_var is [B, D]
        output is [P, B]
        '''

        n_D = tf.shape(mean)[1]
        term1 = n_D * tf.log(2*math.pi)
        term2 = tf.reduce_sum(log_var, reduction_indices=1) #sum over D,[B]
        dif_cov = tf.square(position - mean) / tf.exp(log_var)
        term3 = tf.reduce_sum(dif_cov, 2) #sum over D, [P, B]
        all_ = term1 + term2 + term3
        log_normal_ = -.5 * all_

        return log_normal_



    def model(self, net):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


        #Placeholders - Inputs [B,X]
        self.x = tf.placeholder(tf.float32, [None, net[0]])

        cur_val = self.x

        log_p_W_sum = 0
        log_q_W_sum = 0

        for layer_i in range(len(net)-1):

            input_size_i = net[layer_i]+1 #plus 1 for bias
            output_size_i = net[layer_i+1]

            #Define variables
            W_means = tf.Variable(xavier_init(input_size_i, output_size_i))
            W_logvars = tf.Variable(xavier_init(input_size_i, output_size_i))

            #Sample weights [B,P,I,O]
            eps = tf.random_normal((self.n_particles, input_size_i, output_size_i), 0, 1, seed=self.rs)
            W = tf.add(W_means, tf.multiply(tf.sqrt(tf.exp(W_logvars)), eps))

            # #Compute probs of samples 
            # log_p_W_sum += tf.reduce_sum(self.log_normal(W, tf.zeros([self.batch_size,self.n_particles,]), tf.ones()))
            # log_q_W_sum += tf.reduce_sum(self.log_normal(W, tf.reshape(W_means, [self.batch_size,], W_logvars)))

        #     #Concat 1 to input for biases
        #     cur_val = tf.concat([cur_val,tf.ones([self.n_particles*self.batch_size,1])], axis=1)

        #     #Forward Propagate [B,P,O]
        #     cur_val = self.transfer_fct(tf.matmul(z, W))

        # return cur_val, log_p_W_sum, log_q_W_sum




if __name__ == '__main__':

    net = [784,200,10] 

    model = BNN(net)
    
    print 'Done.'












