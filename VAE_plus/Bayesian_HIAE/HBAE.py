
# Hamiltonian Bayesian Autoencoder

import numpy as np
import tensorflow as tf
import pickle
from os.path import expanduser
home = expanduser("~")

from utils import log_normal4 as log_norm
from utils import log_bernoulli as log_bern

from NN import NN
from BNN import BNN


class HBAE(object):


    def __init__(self, hyperparams):

        #Model hyperparameters
        self.learning_rate = hyperparams['learning_rate']
        self.encoder_act_func = tf.nn.elu #tf.nn.softplus #tf.tanh
        self.decoder_act_func = tf.tanh
        self.encoder_net = hyperparams['encoder_net']
        self.decoder_net = hyperparams['decoder_net']
        self.z_size = hyperparams['z_size']
        self.x_size = hyperparams['x_size']
        self.rs = 0
        self.n_W_particles = hyperparams['n_W_particles']

        #Placeholders - Inputs/Targets
        self.x = tf.placeholder(tf.float32, [None, self.x_size])
        self.batch_size = tf.placeholder(tf.int32, None)
        self.n_z_particles = tf.placeholder(tf.int32, None)
        self.batch_frac = tf.placeholder(tf.float32, None)
        
        #Objective
        self.elbo = self.objective(self.x)

        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                epsilon=1e-02).minimize(-self.elbo)

        #Finalize Initilization
        self.init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()
        self.sess = tf.Session()



    def objective(self, x):
        '''
        Returns scalar to maximize
        '''

        encoder = NN(self.encoder_net, self.encoder_act_func, self.batch_size)
        decoder = BNN(self.decoder_net, self.decoder_act_func, self.batch_size)

        log_pW_list = []
        log_qW_list = []
        log_pz_list = []
        log_qz_list = []
        log_px_list = []

        for W_i in range(self.n_W_particles):

            # Sample decoder weights  __, [1], [1]
            W, log_pW, log_qW = decoder.sample_weights()

            # Sample z   [B,P,Z], [B,P], [B,P]
            z, log_pz, log_qz = self.sample_z(x, encoder, decoder, W)

            # Decode [B,P,X]
            y = decoder.feedforward(W, z)

            # Likelihood p(x|z)  [B,P]
            log_px = log_bern(x,y)

            #Store for later
            log_pW_list.append(tf.reduce_mean(log_pz))
            log_qW_list.append(tf.reduce_mean(log_pz))
            log_pz_list.append(tf.reduce_mean(log_pz))
            log_qz_list.append(tf.reduce_mean(log_pz))
            log_px_list.append(tf.reduce_mean(log_px))

        # Calculte log probs
        self.log_px = tf.reduce_mean(tf.stack(log_px)) #over batch + W_particles + z_particles
        self.log_pz = tf.reduce_mean(tf.stack(log_pz)) #over batch + z_particles
        self.log_qz = tf.reduce_mean(tf.stack(log_qz)) #over batch + z_particles
        self.log_pW = tf.reduce_mean(tf.stack(log_pW)) #W_particles
        self.log_qW = tf.reduce_mean(tf.stack(log_qW)) #W_particles

        self.z_elbo = self.log_px + self.log_pz - self.log_qz 


        #Calc elbo
        elbo = self.log_px + self.log_pz - self.log_qz + self.batch_frac*(self.log_pW - self.log_qW)

        return elbo



    def sample_z(self, x, encoder, decoder, W):
        '''
        z: [B,P,Z]
        log_pz: [B,P]
        log_qz: [B,P]
        '''

        #Encode
        z_mean_logvar = encoder.feedforward(x) #[B,Z*2]
        z_mean = tf.slice(z_mean_logvar, [0,0], [self.batch_size, self.z_size]) #[B,Z] 
        z_logvar = tf.slice(z_mean_logvar, [0,self.z_size], [self.batch_size, self.z_size]) #[B,Z]

        #Sample z  [B,P,Z]
        z_mean_1 = tf.reshape(z_mean, [self.batch_size, 1, self.z_size])
        z_logvar_1 = tf.reshape(z_mean, [self.batch_size, 1, self.z_size])

        eps = tf.random_normal((self.batch_size, self.n_z_particles, self.z_size), 0, 1, seed=self.rs) 
        z = tf.add(z_mean_1, tf.multiply(tf.sqrt(tf.exp(z_logvar_1)), eps)) #uses broadcasting,[B,P,Z]

        # Calc log probs [B,P]
        log_pz = log_norm(z, tf.zeros([self.batch_size, self.z_size]), 
                                tf.log(tf.ones([self.batch_size, self.z_size])))
        log_qz = log_norm(z, z_mean, z_logvar)

        return z, log_pz, log_qz


    def train(self, train_x, valid_x=[], display_step=5, 
                path_to_load_variables='', path_to_save_variables='', 
                epochs=10, batch_size=20, n_W_particles=2, n_z_particles=3):
        '''
        Train.
        '''
        random_seed=1
        rs=np.random.RandomState(random_seed)
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
                                                        self.n_z_particles: n_z_particles, 
                                                        self.batch_frac: 1./float(n_datapoints)})
                # Display logs per epoch step
                if step % display_step == 0:
                    cost,z_elbo,log_px,log_pz,log_qz,log_pW,log_qW = self.sess.run((self.elbo, self.z_elbo, 
                                                                                self.log_px, self.log_pz, 
                                                                                self.log_qz, self.log_pW, 
                                                                                self.log_qW), 
                                                    feed_dict={self.x: batch, 
                                                        self.batch_size: batch_size, 
                                                        self.n_z_particles: n_z_particles, 
                                                        self.batch_frac: 1./float(n_datapoints)})
                    print ("Epoch", str(epoch+1)+'/'+str(epochs), 
                            'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), 
                            "elbo=", "{:.6f}".format(float(cost)),
                            z_elbo,log_px,log_pz,log_qz,log_pW,log_qW)

        if path_to_save_variables != '':
            self.saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables







if __name__ == '__main__':

    x_size = 784
    z_size = 10

    hyperparams = {
        'learning_rate': .0001,
        'x_size': x_size,
        'z_size': z_size,
        'encoder_net': [x_size, 20, z_size*2],
        'decoder_net': [z_size, 20, x_size],
        'n_W_particles': 2}

    model = HBAE(hyperparams)

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
                epochs=50, batch_size=20, n_W_particles=1, n_z_particles=1, display_step=1,
                path_to_load_variables=path_to_load_variables,
                path_to_save_variables=path_to_save_variables)


    print 'Done.'



















