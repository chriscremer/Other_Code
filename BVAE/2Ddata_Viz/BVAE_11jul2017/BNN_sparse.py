





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
        self.learning_rate = .0001
        self.rs = 0
        self.input_size = network_architecture[0]
        self.output_size = network_architecture[-1]
        self.net = network_architecture
        # self.batch_size = batch_size
        # self.n_particles = n_particles
        # self.batch_fraction_of_dataset = batch_frac
        # print 'bnn'
        self.W_means, self.W_logvars, self.s_means, self.s_logvars = self.init_weights()


    def init_weights(self):

        # def xavier_init(fan_in, fan_out, constant=1): 
        #     """ Xavier initialization of network weights"""
        #     # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
        #     low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
        #     high = constant*np.sqrt(6.0/(fan_in + fan_out))
        #     return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        W_means = []
        W_logvars = []

        s_means = []
        s_logvars = []

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Define variables [IS,OS]
            # W_means.append(tf.Variable(xavier_init(input_size_i, output_size_i)))
            W_means.append(tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1)))
            W_logvars.append(tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1))-5.)


            s_means.append(tf.Variable(tf.random_normal([input_size_i], stddev=0.1)))
            s_logvars.append(tf.Variable(tf.random_normal([input_size_i], stddev=0.1))-5.)

        return W_means, W_logvars, s_means, s_logvars



    def sample_weights(self, scale_log_probs):

        Ws = []

        log_p_W_sum = 0.
        log_q_W_sum = 0.



        log_p_s_sum = 0.
        log_q_s_sum = 0.

        W_dim_count = 0.
        s_dim_count = 0.


        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1



            #Get vars [I]
            s_means = self.s_means[layer_i]
            s_logvars = self.s_logvars[layer_i]

            #Sample scales [I]*[I]=[I]
            eps = tf.random_normal([input_size_i], 0, 1, seed=self.rs)
            s = tf.add(s_means, tf.multiply(tf.sqrt(tf.exp(s_logvars)), eps))

            #Compute probs of s samples  [1]
            log_p_s_sum += tf.reduce_sum(tf.abs(s))
            log_q_s_sum += log_normal3(s, s_means, s_logvars)





            #Get vars [I,O]
            W_means = self.W_means[layer_i]
            W_logvars = self.W_logvars[layer_i]

            s = tf.reshape(s, [-1,1])

            W_means_s = W_means*s #[I,O]
            W_vars_s = tf.exp(W_logvars)*tf.square(s) #[I,O]


            #Sample weights [IS,OS]*[IS,OS]=[IS,OS]
            eps = tf.random_normal((input_size_i, output_size_i), 0, 1, seed=self.rs)
            W = tf.add(W_means_s, tf.multiply(tf.sqrt(W_vars_s), eps))



            #Compute probs of samples  [1]
            flat_w = tf.reshape(W,[input_size_i*output_size_i]) #[IS*OS]
            flat_W_means = tf.reshape(W_means_s, [input_size_i*output_size_i]) #[IS*OS]
            flat_W_logvars = tf.log(tf.reshape(W_vars_s, [input_size_i*output_size_i])) #[IS*OS]

            log_squared_s = tf.reshape(tf.log(tf.square(s)), [-1,1])
            log_squared_s = tf.tile(log_squared_s, [1,output_size_i])
            flat_log_squared_s = tf.reshape(log_squared_s, [input_size_i*output_size_i])

            log_q_W_sum += log_normal3(flat_w, flat_W_means, flat_W_logvars)
            log_p_W_sum += log_normal3(flat_w, tf.zeros([input_size_i*output_size_i]), flat_log_squared_s)

            W_dim_count += tf.cast(tf.shape(flat_w)[0], tf.float32)
            s_dim_count += tf.cast(tf.shape(s)[0], tf.float32)

            Ws.append(W)

        # afsasd

        
        if scale_log_probs:
            return Ws, (log_p_W_sum+log_p_s_sum)/(W_dim_count+s_dim_count), (log_q_W_sum+log_q_s_sum)/(W_dim_count+s_dim_count)
        # return Ws, (log_p_W_sum+log_p_s_sum)-tf.log(W_dim_count+s_dim_count), (log_q_W_sum+log_q_s_sum)-tf.log(W_dim_count+s_dim_count)
        else:
            return Ws, (log_p_W_sum+log_p_s_sum), (log_q_W_sum+log_q_s_sum)




    def feedforward(self, W_list, x):
        '''
        W: list of layers weights
        x: [B,X]
        y: [B,Y]
        '''


        #[B,X]
        cur_val = x
        # #[B,X]->[B,1,X]
        # cur_val = tf.reshape(cur_val, [self.batch_size, 1, self.input_size])

        for layer_i in range(len(self.net)-1):

            #[X,X']
            W = W_list[layer_i]
            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Concat 1 to input for biases  [B,X]->[B,X+1]
            cur_val = tf.concat([cur_val,tf.ones([tf.shape(cur_val)[0], 1])], axis=1)
            # #[X,X']->[B,X,X']
            # W = tf.reshape(W, [1, input_size_i, output_size_i])
            # W = tf.tile(W, [self.batch_size, 1,1])

            #Forward Propagate  [B,X]*[X,X']->[B,X']
            if layer_i != len(self.net)-2:
                cur_val = self.act_func(tf.matmul(cur_val, W))
            else:
                cur_val = tf.matmul(cur_val, W)

        # #[B,P,1,X']->[B,P,X']
        # cur_val = tf.reshape(cur_val, [self.batch_size,P,output_size_i])
        #[B,Y]
        y = cur_val

        return y











