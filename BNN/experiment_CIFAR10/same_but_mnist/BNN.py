





#Bayesian Neural Network

import numpy as np
import numpy.random as npr
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle

from utils import log_normal3


class BNN(object):

    def __init__(self, network_architecture, act_func):
        

        #Model hyperparameters
        self.act_func = act_func
        # self.learning_rate = .0001
        self.rs = 0
        self.input_size = network_architecture[0]
        self.output_size = network_architecture[-1]
        self.net = network_architecture
        # self.batch_size = batch_size
        # self.n_particles = n_particles
        # self.batch_fraction_of_dataset = batch_frac
        # print 'bnn'
        self.W_means, self.W_logvars = self.init_weights()


    def init_weights(self):

        # def xavier_init(fan_in, fan_out, constant=1): 
        #     """ Xavier initialization of network weights"""
        #     # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
        #     low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
        #     high = constant*np.sqrt(6.0/(fan_in + fan_out))
        #     return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        W_means = []
        W_logvars = []

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Define variables [IS,OS]
            # W_means.append(tf.Variable(xavier_init(input_size_i, output_size_i)))
            W_means.append(tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1)))

            # W_logvars.append(tf.Variable(xavier_init(input_size_i, output_size_i) - 10.))
            W_logvars.append(tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1))-5.)

        return W_means, W_logvars



    def sample_weights(self, scale_log_probs=False):

        Ws = []

        log_p_W_sum = 0
        log_q_W_sum = 0


        W_dim_count = 0.


        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Get vars [I,O]
            W_means = self.W_means[layer_i]
            W_logvars = self.W_logvars[layer_i]

            #Sample weights [IS,OS]*[IS,OS]=[IS,OS]
            eps = tf.random_normal((input_size_i, output_size_i), 0, 1, seed=self.rs)
            W = tf.add(W_means, tf.multiply(tf.sqrt(tf.exp(W_logvars)), eps))

            # W = W_means

            #Compute probs of samples  [1]
            flat_w = tf.reshape(W,[input_size_i*output_size_i]) #[IS*OS]
            flat_W_means = tf.reshape(W_means, [input_size_i*output_size_i]) #[IS*OS]
            flat_W_logvars = tf.reshape(W_logvars, [input_size_i*output_size_i]) #[IS*OS]
            log_p_W_sum += log_normal3(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))
            # log_p_W_sum += log_normal3(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])*100.))

            log_q_W_sum += log_normal3(flat_w, flat_W_means, flat_W_logvars)

            # print W

            W_dim_count += tf.cast(tf.shape(flat_w)[0], tf.float32)

            Ws.append(W)

        # afsasd

        # return Ws, log_p_W_sum, log_q_W_sum

        if scale_log_probs:
            return Ws, (log_p_W_sum)/(W_dim_count), (log_q_W_sum)/(W_dim_count)
        else:
            return Ws, log_p_W_sum, log_q_W_sum




 
    def sample_weight_means(self):

        Ws = []
        log_p_W_sum = 0

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Get vars [I,O]
            W_means = self.W_means[layer_i]

            flat_w = tf.reshape(W_means,[input_size_i*output_size_i]) #[IS*OS]

            log_p_W_sum += log_normal3(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))


            Ws.append(W_means)

        return Ws, log_p_W_sum






    def feedforward(self, x, W_list):
        '''
        x: [B,X]
        y_hat: [B,O]
        '''

        B = tf.shape(x)[0]

        net = self.net

        #[B,X]
        cur_val = x

        for layer_i in range(len(net)-1):

            #[X,X']
            W = W_list[layer_i]

            #Concat 1 to input for biases  [B,P,X]->[B,P,X+1]
            cur_val = tf.concat([cur_val,tf.ones([B, 1])], axis=1)


            if self.act_func[layer_i] != None:
                cur_val = self.act_func[layer_i](tf.matmul(cur_val, W))
            else:
                cur_val = tf.matmul(cur_val, W)



        return cur_val










