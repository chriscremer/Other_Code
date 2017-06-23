

# HVI


#FOUDN BUG, im only taking gradient of p(x|z), shoild be p(x,z)
# fixed


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



from NN import NN

from VAE import VAE


class HVI(VAE):




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

        z, log_qv0, log_rvT = self.transform_sample(z0, decoder, x, k)

        # Calc log probs [P,B]
        log_pzT = log_norm(z, tf.zeros([self.batch_size, self.z_size]), 
                                tf.log(tf.ones([self.batch_size, self.z_size])))
        
        log_pz = log_pzT + log_rvT 

        log_qz =  log_qz0 + log_qv0

        return z, log_pz, log_qz



    def transform_sample(self, z0, decoder, x, k):
    	'''
    	z0: P,B,Z
    	'''

        # q(v0|x,z0) 
        net2 = NN([self.x_size+self.z_size, 100, self.z_size*2], tf.nn.elu)
        # r(v|x,z)
        net3 = NN([self.x_size+self.z_size, 100, self.z_size*2], tf.nn.elu)


        # Sample v0 and calc log_qv0
        z0_reshaped = tf.reshape(z0, [k*self.batch_size, self.z_size]) #[PB,Z]
        x_tiled = tf.tile(x, [k, 1]) #[PB,X]
        xz = tf.concat([x_tiled,z0_reshaped], axis=1) #[PB,X+Z]
        v0_mean, v0_logvar = split_mean_logvar(net2.feedforward(xz)) #[PB,Z]
        v0 = sample_Gaussian(v0_mean, v0_logvar, 1) #[1,PB,Z]
        v0 = tf.reshape(v0, [k*self.batch_size, self.z_size]) #[PB,Z]
        log_qv0 = log_norm2(v0, v0_mean, v0_logvar) #[PB]
        log_qv0 = tf.reshape(log_qv0, [k, self.batch_size]) #[P,B]
        v0 = tf.reshape(v0, [k, self.batch_size, self.z_size]) #[P,B,Z]


        self.n_transitions = 3


        # Transform [P,B,Z]
        zT, vT = self.leapfrogs(z0,v0,decoder,x,k)


        # Reverse model
        z_reshaped = tf.reshape(zT, [k*self.batch_size, self.z_size]) #[PB,Z]
        xz = tf.concat([x_tiled,z_reshaped], axis=1) #[PB,X+Z]
        vt_mean, vt_logvar = split_mean_logvar(net3.feedforward(xz)) #[PB,Z]
        vT = tf.reshape(vT, [k*self.batch_size, self.z_size]) #[PB,Z]
        log_rv = log_norm2(vT, vt_mean, vt_logvar) #[PB]
        log_rv = tf.reshape(log_rv, [k, self.batch_size]) #[P,B]

        return zT, log_qv0, log_rv



    def leapfrogs(self, z, v, p_xlz, x, k):
        '''
        z,v: [P,B,Z]
        x: [B,X] 
        '''


        self.step_size = tf.Variable([.1])

        for t in range(self.n_transitions):

            z_intermediate = z + ((.5*self.step_size) * v) 

            log_p = self._log_px(p_xlz, x, z_intermediate, k)
            grad = -tf.gradients(log_p, [z_intermediate])[0]
            v = v + (self.step_size * grad)

            z = z_intermediate + ((.5*self.step_size) * v)

        return z, v



    def _log_px(self, p_xlz, x, z, k):
        '''
        x: [B,X]
        z: [P,B,Z]
        output: [P,B]
        '''

        z_reshaped = tf.reshape(z, [k*self.batch_size, self.z_size]) #[PB,Z]
        x_mean = p_xlz.feedforward(z_reshaped) #[PB,X]
        x_mean = tf.reshape(x_mean, [k, self.batch_size, self.x_size]) #[P,B,Z]

        log_px = log_bern(x,x_mean) #[P,B]

        log_pz = log_norm(z, tf.zeros([self.batch_size, self.z_size]), 
                                tf.log(tf.ones([self.batch_size, self.z_size])))

        return log_px + log_pz









if __name__ == '__main__':

    x_size = 784
    z_size = 10

    # hyperparams = {
    #     'learning_rate': .0001,
    #     'x_size': x_size,
    #     'z_size': z_size,
    #     'encoder_net': [x_size, 20, z_size*2],
    #     'decoder_net': [z_size, 20, x_size],
    #     # 'n_W_particles': 1,
    #     'n_z_particles': 1,
    #     'n_z_particles_test': 10,
    #     'lmba': .0000001}

    hyperparams = {
        'learning_rate': lr,
        'x_size': x_size,
        'z_size': z_size,
        'encoder_net': [x_size, h1_size, z_size*2],
        'decoder_net': [z_size, h1_size, x_size],
        # 'n_z_particles': k_training,
        # 'n_z_particles_test': k_evaluation
        }

    model = HVI(hyperparams)

    print 'Loading data'
    with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
        mnist_data = pickle.load(f)

    train_x = mnist_data[0][0]
    train_y = mnist_data[0][1]
    valid_x = mnist_data[1][0]
    valid_y = mnist_data[1][1]
    test_x = mnist_data[2][0]
    test_y = mnist_data[2][1]

    path_to_load_variables=home+'/Documents/tmp/HVI_epochs9999999_2.ckpt' 
    # path_to_load_variables=''
    # path_to_save_variables=home+'/Documents/tmp/vars2.ckpt'
    path_to_save_variables=home+''




    with tf.Session() as model.sess:
        if path_to_load_variables == '':
            model.sess.run(model.init_vars)
        else:
            #Load variables
            model.saver.restore(model.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables

    print model.sess.run(model.step_size)

    fsada


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























