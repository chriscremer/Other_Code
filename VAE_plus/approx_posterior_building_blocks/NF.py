



# NF



import numpy as np
import tensorflow as tf
import pickle
from os.path import expanduser
home = expanduser("~")

from utils import log_normal as log_norm
from utils import log_bernoulli as log_bern
from utils import split_mean_logvar  
from utils import sample_Gaussian
from utils import log_normal2 as log_norm2

slim=tf.contrib.slim

from NN import NN

from VAE import VAE


class NF(VAE):



    def random_bernoulli(self, shape, p=0.5):
        if isinstance(shape, (list, tuple)):
            shape = tf.stack(shape)
        return tf.where(tf.random_uniform(shape) < p, tf.ones(shape), tf.zeros(shape))




    def sample_z(self, x, encoder, decoder, k):
        '''
        x: [B,X]
        z: [P,B,Z]
        log_pz: [P,B]
        log_qz: [P,B]
        '''

        #Encode
        z_mean, z_logvar = split_mean_logvar(encoder.feedforward(x)) #[B,Z]

        #Sample z  [P,B,Z]
        eps = tf.random_normal((k, self.batch_size, self.z_size), 0, 1, seed=self.rs) 
        z0 = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps)) #broadcast, [P,B,Z]
        log_qz0 = log_norm(z0, z_mean, z_logvar)

        #[P,B,Z], [P,B]
        z,logdet = self.transform_sample(z0)

        # Calc log probs [P,B]
        log_pzT = log_norm(z, tf.zeros([self.batch_size, self.z_size]), 
                                tf.log(tf.ones([self.batch_size, self.z_size])))
        
        log_pz = log_pzT  + logdet

        log_qz =  log_qz0 

        return z, log_pz, log_qz



    def transform_sample(self, z):
        '''
        z: [P,B,Z]
        '''

        P = tf.shape(z)[0]
        B = tf.shape(z)[1]
        # Z = tf.shape(z)[2]


        z = tf.reshape(z, [P*B,self.z_size])

        self.n_transitions = 3

        logdet_sum = tf.zeros([P*B])
        #Flows z0 -> zT
        for t in range(self.n_transitions):

            # print mask*z

            
            mask = self.random_bernoulli(tf.shape(z), p=0.5)
            h = slim.stack(mask*z,slim.fully_connected,[100])
            mew_ = slim.fully_connected(h,self.z_size,activation_fn=None) 
            sig_ = slim.fully_connected(h,self.z_size,activation_fn=tf.nn.sigmoid) 

            z = (mask * z) + (1-mask)*(z*sig_ + (1-sig_)*mew_)

            logdet = tf.reduce_sum((1-mask)*tf.log(sig_), axis=1) #[PB]

            logdet_sum += logdet

        z = tf.reshape(z, [P,B,self.z_size])
        logdet_sum = tf.reshape(logdet_sum, [P,B])

        return z, logdet_sum











if __name__ == '__main__':

    x_size = 784
    z_size = 10

    hyperparams = {
        'learning_rate': .0001,
        'x_size': x_size,
        'z_size': z_size,
        'encoder_net': [x_size, 20, z_size*2],
        'decoder_net': [z_size, 20, x_size],
        # 'n_W_particles': 1,
        'n_z_particles': 2,
        'n_z_particles_test': 10,
        'lmba': .0000001}

    model = NF(hyperparams)

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
                epochs=2, batch_size=20, display_step=1,
                path_to_load_variables=path_to_load_variables,
                path_to_save_variables=path_to_save_variables)



    print 'Eval'
    iwae_elbo = model.eval(data=test_x, batch_size=20, display_step=10,
                path_to_load_variables=path_to_save_variables, data2=train_x)

    print iwae_elbo

    print 'Done.'























