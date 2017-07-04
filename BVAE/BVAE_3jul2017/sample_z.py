





import numpy as np
import tensorflow as tf
import pickle
from os.path import expanduser
home = expanduser("~")

from utils import log_normal as log_norm
from utils import log_bernoulli as log_bern
from utils import split_mean_logvar  

from NN2 import NN
from BNN import BNN

slim=tf.contrib.slim




class Sample_z(object):


    def __init__(self, batch_size, z_size, n_z_particles, n_transformations=3):

        self.batch_size = batch_size
        self.z_size = z_size
        self.n_z_particles = n_z_particles

        self.rs = 0



        

        self.n_transitions = n_transformations

        self.variables = []

        for t in range(self.n_transitions):


            net1 = NN([z_size, 100], [tf.tanh], batch_size)
            net2 = NN([100, z_size], [None], batch_size)
            net3 = NN([100, z_size], [tf.nn.sigmoid], batch_size)

            self.variables.append([net1, net2, net3])









    def random_bernoulli(self, shape, p=0.5):
        if isinstance(shape, (list, tuple)):
            shape = tf.stack(shape)
        return tf.where(tf.random_uniform(shape) < p, tf.ones(shape), tf.zeros(shape))



    def transform_sample(self, z):
        '''
        z: [P,B,Z]
        '''

        P = tf.shape(z)[0]
        B = tf.shape(z)[1]
        # Z = tf.shape(z)[2]


        z = tf.reshape(z, [P*B,self.z_size])

        

        logdet_sum = tf.zeros([P*B])
        #Flows z0 -> zT
        for t in range(self.n_transitions):

            # print mask*z

            
            mask = self.random_bernoulli(tf.shape(z), p=0.5)

            # h = slim.stack(mask*z,slim.fully_connected,[100])
            # mew_ = slim.fully_connected(h,self.z_size,activation_fn=None) 
            # sig_ = slim.fully_connected(h,self.z_size,activation_fn=tf.nn.sigmoid) 

            h = self.variables[t][0].feedforward(mask*z)
            mew_ = self.variables[t][1].feedforward(h)
            sig_ = self.variables[t][2].feedforward(h)


            z = (mask * z) + (1-mask)*(z*sig_ + (1-sig_)*mew_)

            logdet = tf.reduce_sum((1-mask)*tf.log(sig_), axis=1) #[PB]

            logdet_sum += logdet

        z = tf.reshape(z, [P,B,self.z_size])
        logdet_sum = tf.reshape(logdet_sum, [P,B])

        # print 'made'

        return z, logdet_sum



    def sample_z(self, x, encoder, decoder, W):
        '''
        z: [P,B,Z]
        log_pz: [P,B]
        log_qz: [P,B]
        '''

        for i in range(len(W)):

            if i ==0:
                flatten_W = tf.reshape(W[i], [-1])
                # print flatten_W
            else:
                flattt = tf.reshape(W[i], [-1])
                # print flattt
                flatten_W = tf.concat([flatten_W, flattt], axis=0)

        flatten_W = tf.reshape(flatten_W, [1,-1])
        tiled = tf.tile(flatten_W, [self.batch_size, 1])
        intput_ = tf.concat([x,tiled], axis=1)

        #Encode
        z_mean_logvar = encoder.feedforward(intput_) #[B,Z*2]
        z_mean = tf.slice(z_mean_logvar, [0,0], [self.batch_size, self.z_size]) #[B,Z] 
        z_logvar = tf.slice(z_mean_logvar, [0,self.z_size], [self.batch_size, self.z_size]) #[B,Z]

        #Sample z  [P,B,Z]
        eps = tf.random_normal((self.n_z_particles, self.batch_size, self.z_size), 0, 1, seed=self.rs) 
        z0 = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps)) #broadcast, [P,B,Z]
        log_qz0 = log_norm(z0, z_mean, z_logvar)


        #[P,B,Z], [P,B]
        z,logdet = self.transform_sample(z0)

        # Calc log probs [P,B]
        log_pzT = log_norm(z, tf.zeros([self.batch_size, self.z_size]), 
                                tf.log(tf.ones([self.batch_size, self.z_size])))
        
        log_pz = log_pzT  + logdet

        log_qz =  log_qz0 


        # # Calc log probs [P,B]
        # log_pz = log_norm(z, tf.zeros([self.batch_size, self.z_size]), 
        #                         tf.log(tf.ones([self.batch_size, self.z_size])))
        # log_qz = log_norm(z, z_mean, z_logvar)

        return z, log_pz, log_qz


































