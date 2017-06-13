



# IWAE



import numpy as np
import tensorflow as tf
import pickle
from os.path import expanduser
home = expanduser("~")

from utils import log_normal as log_norm
from utils import log_bernoulli as log_bern

from NN import NN

from VAE import VAE


class IWAE(VAE):


    def __init__(self, hyperparams):

        tf.reset_default_graph()

        #Model hyperparameters
        self.learning_rate = hyperparams['learning_rate']
        self.encoder_act_func = tf.nn.elu #tf.nn.softplus #tf.tanh
        self.decoder_act_func = tf.tanh
        self.encoder_net = hyperparams['encoder_net']
        self.decoder_net = hyperparams['decoder_net']
        self.z_size = hyperparams['z_size']  #Z
        self.x_size = hyperparams['x_size']  #X
        self.rs = 0
        # self.n_W_particles = hyperparams['n_W_particles']  #S
        self.n_z_particles = hyperparams['n_z_particles']  #P

        # self.qW_weight = hyperparams['qW_weight']
        self.lmba = hyperparams['lmba']

        #Placeholders - Inputs/Targets
        self.x = tf.placeholder(tf.float32, [None, self.x_size])
        self.batch_size = tf.shape(self.x)[0]   #B
        self.batch_frac = tf.placeholder(tf.float32, None)

       
        self.encoder = NN(self.encoder_net, self.encoder_act_func, self.batch_size)
        self.decoder = NN(self.decoder_net, self.decoder_act_func, self.batch_size)

        self.l2_sum = self.encoder.weight_decay()




        #Objective
        log_px, log_pz, log_qz = self.log_probs(self.x, self.encoder, self.decoder)

        #ONLY CHANGE FROM VAE
        self.elbo = self.iwae_objective_test(log_px, log_pz, log_qz)
        # self.iwae_elbo = self.iwae_objective(*log_probs)

        self.iwae_elbo_test = self.iwae_objective_test(log_px, log_pz, log_qz)

        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                epsilon=1e-02).minimize(-self.elbo)



        #Finalize Initilization
        self.init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()
        self.sess = tf.Session()









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
        'n_z_particles': 1,
        'lmba': .0000001}

    model = IWAE(hyperparams)

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
                epochs=1, batch_size=20, display_step=[1,1000],
                path_to_load_variables=path_to_load_variables,
                path_to_save_variables=path_to_save_variables)



    print 'Eval'
    iwae_elbo = model.eval(data=test_x, batch_size=20, display_step=10,
                path_to_load_variables=path_to_save_variables, data2=train_x)

    print iwae_elbo

    print 'Done.'





















