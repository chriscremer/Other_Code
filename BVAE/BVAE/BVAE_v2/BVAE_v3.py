



#Bayesian VAE


# - sample weights first, so latent samples can depend on weights. 
	# need to add method to BNN where I sample the weights but dont pass input. 
	# so then latent can be sampled based on gradient of the weights.
	# see vae plus code, since I already did this before.

# - multiple latent samples


# not completed

import numpy as np
import numpy.random as npr
import tensorflow as tf
import random
from os.path import expanduser
home = expanduser("~")
import time
import pickle

from utils import log_normal, log_bernoulli

from BNN import BNN
from NN import NN



class BVAE(object):

    def __init__(self, network_architecture):
        
        tf.reset_default_graph()

        encoder_net = network_architecture[0]
        decoder_net = network_architecture[1]

        #Model hyperparameters
        self.act_func = tf.nn.elu #tf.nn.softplus #tf.tanh
        self.learning_rate = .0001
        self.rs = 0
        self.input_size = encoder_net[0]
        self.z_size = decoder_net[0]

        #Placeholders - Inputs/Targets [B,X]
        self.batch_size = tf.placeholder(tf.int32, None)
        self.n_z_particles = tf.placeholder(tf.int32, None)
        self.n_W_particles = tf.placeholder(tf.int32, None)
        self.batch_frac = tf.placeholder(tf.float32, None)
        self.x = tf.placeholder(tf.float32, [None, self.input_size])

        #Encoder + Decoder
        self.NN_encoder = NN(encoder_net, self.batch_size)
        self.BNN_decoder = BNN(decoder_net, self.batch_size, self.n_W_particles, self.batch_frac)
        
        #Objective
        self.elbo = self.objective(self.x, self.NN_encoder, self.BNN_decoder)

        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(-self.elbo)


        # To init variables
        self.init_vars = tf.global_variables_initializer()
        # For loadind/saving variables
        self.saver = tf.train.Saver()
        # #For debugging 
        # self.vars = tf.trainable_variables()
        # self.grads = tf.gradients(self.elbo, tf.trainable_variables()
        #to make sure im not adding nodes to the graph
        tf.get_default_graph().finalize()
        #Start session
        self.sess = tf.Session()





    def objective(self, x, encoder, decoder):
        '''
        elbo: [1]
        '''

        #Encode
        z_mean_logvar = encoder.model(x) #[B,Z*2]
        z_mean = tf.slice(z_mean_logvar, [0,0], [self.batch_size, self.z_size]) #[B,Z] 
        z_logvar = tf.slice(z_mean_logvar, [0,self.z_size], [self.batch_size, self.z_size]) #[B,Z]

        # #Sample z
        # eps = tf.random_normal((self.batch_size, self.n_z_particles, self.z_size), 0, 1, dtype=tf.float32) #[B,P,Z]
        # z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps)) #uses broadcasting,[B,P,Z]

        # Sample z  [B,Z]
        eps = tf.random_normal((self.batch_size, self.z_size), 0, 1, dtype=tf.float32, seed=self.rs) #[B,Z]
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps)) #[B,Z]

        # [B]
        log_pz = log_normal(z, tf.zeros([self.batch_size, self.z_size]), tf.log(tf.ones([self.batch_size, self.z_size])))
        log_qz = log_normal(z, z_mean, z_logvar)

        # Decode [B,P,X], [P], [P]
        x_mean, log_pW, log_qW = decoder.model(z)
        
        # Likelihood [B,P]
        log_px = log_bernoulli(x, x_mean)

        # Objective
        self.log_px = tf.reduce_mean(log_px) #over batch + W_particles
        self.log_pz = tf.reduce_mean(log_pz) #over batch
        self.log_qz = tf.reduce_mean(log_qz) #over batch 
        self.log_pW = tf.reduce_mean(log_pW) #W_particles
        self.log_qW = tf.reduce_mean(log_qW) #W_particles

        elbo = self.log_px + self.log_pz - self.log_qz + self.batch_frac*(self.log_pW - self.log_qW)

        self.z_elbo = self.log_px + self.log_pz - self.log_qz 

        return elbo












    def train(self, train_x, valid_x=[], display_step=5, path_to_load_variables='', path_to_save_variables='', epochs=10, batch_size=20, n_particles=3):
        '''
        Train.
        '''
        random_seed=1
        rs=npr.RandomState(random_seed)
        n_datapoints = len(train_x)
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

            data_index = 0
            for step in range(n_datapoints/batch_size):

                #Make batch
                batch = []
                while len(batch) != batch_size:
                    batch.append(train_x[data_index]) 
                    data_index +=1

                # Fit training using batch data
                _ = self.sess.run((self.optimizer), feed_dict={self.x: batch, 
                                                        self.batch_size: batch_size,
                                                        self.n_z_particles: n_particles, 
                                                        self.n_W_particles: n_particles, 
                                                        self.batch_frac: 1./float(n_datapoints)})

                # Display logs per epoch step
                if step % display_step == 0:

                    cost,z_elbo,log_px,log_pz,log_qz,log_pW,log_qW = self.sess.run((self.elbo, self.z_elbo, self.log_px, self.log_pz, self.log_qz, self.log_pW, self.log_qW), 
                                                    feed_dict={self.x: batch, 
                                                        self.batch_size: batch_size, 
                                                        self.n_z_particles: n_particles, 
                                                        self.n_W_particles: n_particles, 
                                                        self.batch_frac: 1./float(n_datapoints)})

                    print "Epoch", str(epoch+1)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "elbo=", "{:.6f}".format(float(cost)),z_elbo,log_px,log_pz,log_qz,log_pW,log_qW#,logpy,logpW,logqW #, 'time', time.time() - start


        if path_to_save_variables != '':
            self.saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables








if __name__ == '__main__':

    x_size = 784
    z_size = 10

    net = [[x_size,20,z_size*2],
            [z_size,20,x_size]]


    model = BVAE(net)

    print 'Loading data'
    with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
        mnist_data = pickle.load(f)

    train_x = mnist_data[0][0]
    train_y = mnist_data[0][1]
    valid_x = mnist_data[1][0]
    valid_y = mnist_data[1][1]
    test_x = mnist_data[2][0]
    test_y = mnist_data[2][1]

    # path_to_load_variables=home+'/Documents/tmp/vars.ckpt' 
    path_to_load_variables=''
    path_to_save_variables=home+'/Documents/tmp/vars2.ckpt'

    print 'Training'
    model.train(train_x=train_x,
                epochs=50, batch_size=20, n_particles=2, display_step=1000,
                path_to_load_variables=path_to_load_variables,
                path_to_save_variables=path_to_save_variables)


    print 'Done.'












