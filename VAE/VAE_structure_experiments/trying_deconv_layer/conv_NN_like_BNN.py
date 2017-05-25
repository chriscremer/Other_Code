




#This is a NN but can sample weights (which is just the mean)


# using the conv2d transpose
# https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose



# Neural Network

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


class conv_NN(object):

    def __init__(self, x_size, z_size, decoder_deconvnet):
        

        #Model hyperparameters
        # self.act_func = act_func
        self.learning_rate = .0001
        self.rs = 0
        # self.input_size = network_architecture[0]
        # self.output_size = network_architecture[-1]
        # self.net = network_architecture
        # self.batch_size = batch_size
        # self.n_particles = n_particles
        # self.batch_fraction_of_dataset = batch_frac

        #Model hyperparameters
        self.act_func = tf.tanh
        # self.input_size = network_architecture[0]
        # self.output_size = network_architecture[-1]
        # self.net = [784,72,10,10*2] 
        self.net = [z_size, decoder_deconvnet[6], decoder_deconvnet[5], x_size]  

        # self.batch_size = batch_size

        # self.filter_height = 5
        # self.filter_width = 5
        # self.n_in_channels = 5
        # self.n_out_channels = 1
        # self.strides = 4

        self.filter_height = decoder_deconvnet[0]
        self.filter_width = decoder_deconvnet[1]
        self.n_in_channels = decoder_deconvnet[3]
        self.n_out_channels = decoder_deconvnet[2]
        self.strides = decoder_deconvnet[4]

        self.W_means = self.init_weights()


        # Check model size
        total_size = 0
        # for v in tf.trainable_variables():
        for v in self.W_means:
                total_size += np.prod([int(s) for s in v.get_shape()])
                print(v.get_shape())
        print("Total number of trainable variables in decoder: {}".format(total_size))


    def init_weights(self):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        W_means = []
        # W_logvars = []

        # minus 2 becasue 1 for last layer and for conv
        for layer_i in range(len(self.net)-2):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Define variables [IS,OS]
            W_means.append(tf.Variable(xavier_init(input_size_i, output_size_i)))
            # W_logvars.append(tf.Variable(xavier_init(input_size_i, output_size_i) - 10.))

        #filter
        conv_weights = tf.Variable(tf.truncated_normal([self.filter_height, 
                                                        self.filter_width,
                                                        self.n_out_channels, 
                                                        self.n_in_channels], stddev=0.1))

        W_means.append(conv_weights)


        return W_means#, W_logvars



    def sample_weights(self):

        Ws = []

        log_p_W_sum = 0.
        log_q_W_sum = 0.

        for layer_i in range(len(self.net)-2):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Get vars [I,O]
            W = self.W_means[layer_i]
            # W_logvars = self.W_logvars[layer_i]

            #Sample weights [IS,OS]*[IS,OS]=[IS,OS]
            # eps = tf.random_normal((input_size_i, output_size_i), 0, 1, seed=self.rs)
            # W = tf.add(W_means, tf.multiply(tf.sqrt(tf.exp(W_logvars)), eps))
            # W = W_means

            # #Compute probs of samples  [1]
            flat_w = tf.reshape(W,[input_size_i*output_size_i]) #[IS*OS]
            # flat_W_means = tf.reshape(W_means, [input_size_i*output_size_i]) #[IS*OS]
            # flat_W_logvars = tf.reshape(W_logvars, [input_size_i*output_size_i]) #[IS*OS]
            log_p_W_sum += log_normal3(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))
            # log_q_W_sum += log_normal3(flat_w, flat_W_means, flat_W_logvars)

            # Ws.append(W)

        return self.W_means, log_p_W_sum, log_q_W_sum



    def feedforward(self, W_list, x):
        '''
        W: list of layers weights
        x: [B,X]
        y: [B,Y]
        '''

        batch_size = tf.shape(x)[0]


        #[B,X]
        cur_val = x
        # #[B,X]->[B,1,X]
        # cur_val = tf.reshape(cur_val, [self.batch_size, 1, self.input_size])

        for layer_i in range(len(self.net)-2):

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


        # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose

        # print cur_val
        # fasdf
        cur_val = tf.reshape(cur_val, [batch_size, 6 , 6 , self.n_in_channels])
        # print cur_val
        cur_val = tf.nn.conv2d_transpose(cur_val,  self.W_means[-1], [batch_size, 28,28,1] ,strides=[1, self.strides, self.strides, 1],padding='VALID')
        # print cur_val
        cur_val = tf.reshape(cur_val, [batch_size, -1])

        #[B,Y]
        y = cur_val

        return y











