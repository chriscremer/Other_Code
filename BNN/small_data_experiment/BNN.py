


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

    def __init__(self, network_architecture, act_func, batch_size):
        

        #Model hyperparameters
        self.act_func = act_func
        self.learning_rate = .0001
        self.rs = 0
        self.input_size = network_architecture[0]
        self.output_size = network_architecture[-1]
        self.net = network_architecture
        self.batch_size = batch_size
        # self.n_particles = n_particles
        # self.batch_fraction_of_dataset = batch_frac

        self.W_means, self.W_logvars = self.init_weights()


    def init_weights(self):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        W_means = []
        W_logvars = []

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Define variables [IS,OS]
            W_means.append(tf.Variable(xavier_init(input_size_i, output_size_i)))
            W_logvars.append(tf.Variable(xavier_init(input_size_i, output_size_i) - 10.))

        return W_means, W_logvars



    def sample_weights(self):

        Ws = []

        log_p_W_sum = 0
        log_q_W_sum = 0

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Get vars [I,O]
            W_means = self.W_means[layer_i]
            W_logvars = self.W_logvars[layer_i]

            #Sample weights [IS,OS]*[IS,OS]=[IS,OS]
            eps = tf.random_normal((input_size_i, output_size_i), 0, 1, seed=self.rs)
            W = tf.add(W_means, tf.multiply(tf.sqrt(tf.exp(W_logvars)), eps))

            #Compute probs of samples  [1]
            flat_w = tf.reshape(W,[input_size_i*output_size_i]) #[IS*OS]
            flat_W_means = tf.reshape(W_means, [input_size_i*output_size_i]) #[IS*OS]
            flat_W_logvars = tf.reshape(W_logvars, [input_size_i*output_size_i]) #[IS*OS]
            log_p_W_sum += log_normal3(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))
            log_q_W_sum += log_normal3(flat_w, flat_W_means, flat_W_logvars)

            Ws.append(W)

        return Ws, log_p_W_sum, log_q_W_sum



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






    def train(self, train_x, train_y, valid_x=[], valid_y=[], display_step=5, path_to_load_variables='', path_to_save_variables='', epochs=10, batch_size=20, n_particles=3):
        '''
        Train.
        '''
        random_seed=1
        rs=npr.RandomState(random_seed)
        n_datapoints = len(train_y)
        arr = np.arange(n_datapoints)

        if path_to_load_variables == '':
            self.sess.run(self.init_vars)

        else:
            #Load variables
            self.saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables

        #start = time.time()
        for epoch in range(epochs):

            #shuffle the data
            rs.shuffle(arr)
            train_x = train_x[arr]
            train_y = train_y[arr]

            data_index = 0
            for step in range(n_datapoints/batch_size):

                #Make batch
                batch = []
                batch_y = []
                while len(batch) != batch_size:
                    batch.append(train_x[data_index]) 
                    one_hot=np.zeros(10)
                    one_hot[train_y[data_index]]=1.
                    batch_y.append(one_hot)
                    data_index +=1

                # Fit training using batch data
                _ = self.sess.run((self.optimizer), feed_dict={self.x: batch, self.y: batch_y, 
                                                        self.batch_size: batch_size,
                                                        self.n_particles: n_particles, 
                                                        self.batch_fraction_of_dataset: 1./float(n_datapoints)})

                # Display logs per epoch step
                if step % display_step == 0:

                    cost,logpy,logpW,logqW,pred = self.sess.run((self.elbo,self.logpy, self.logpW, self.logqW,self.prediction), feed_dict={self.x: batch, self.y: batch_y, 
                                                        self.batch_size: batch_size, 
                                                        self.n_particles:n_particles, 
                                                        self.batch_fraction_of_dataset: 1./float(n_datapoints)})


                    # cost = -cost #because I want to see the NLL
                    print "Epoch", str(epoch+1)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "elbo=", "{:.6f}".format(float(cost)),logpy,logpW,logqW #, 'time', time.time() - start
                    # print 'target'
                    print ["{:.2f}".format(float(x)) for x in batch_y[0]] 
                    # print 'prediciton'
                    # print pred.shape
                    print ["{:.2f}".format(float(x)) for x in pred[0][0]] 
                    print


        if path_to_save_variables != '':
            self.saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables










