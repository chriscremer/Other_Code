

# Markov Chain Variational Inference


#not completed. still need to finish the init and the main.

# I confused the MCMC steps and the Hamiltonian steps here.
# when T=1, there shoudlnt be any list of q and r..
# ya use T=1. so no loop
# also the reverse model needs z, right now its only given x.

import numpy as np
import tensorflow as tf
import pickle
from os.path import expanduser
home = expanduser("~")

from utils import log_normal_noSamps as log_norm
from utils import log_bernoulli_noSamps as log_bern

from NN import NN
from BNN import BNN




class MCVI(object):


    def __init__(self, hyperparams):

        tf.reset_default_graph()

        #Model hyperparameters
        self.learning_rate = hyperparams['learning_rate']

        self.z_size = hyperparams['z_size']  #Z
        self.x_size = hyperparams['x_size']  #X
        self.rs = 0
        self.n_W_particles = hyperparams['n_W_particles']  #S
        # self.n_W_particles = tf.placeholder(tf.int32, None)  #S
        self.n_z_particles = hyperparams['n_z_particles']  #P
        # self.n_z_particles = tf.placeholder(tf.int32, None)  #P


        self.encoder_act_func = tf.nn.elu #tf.nn.softplus #tf.tanh
        self.decoder_act_func = tf.tanh
        self.encoder_net = hyperparams['encoder_net']
        self.decoder_net = hyperparams['decoder_net']

        #Placeholders - Inputs/Targets
        self.x = tf.placeholder(tf.float32, [None, self.x_size])

        # self.batch_frac = tf.placeholder(tf.float32, None)
        self.batch_size = tf.shape(self.x)[0]   #B

        #Define networks

        # q(z|x)
        net1 = NN(self.encoder_net, self.encoder_act_func, self.batch_size)
        # q(z|x,z)
        net2 = NN(self.decoder_net, self.decoder_act_func, self.batch_size)
        # r(z|x,z)
        net2 = NN(self.decoder_net, self.decoder_act_func, self.batch_size)
        # p(x|z)
        net2 = NN(self.decoder_net, self.decoder_act_func, self.batch_size)
        

        #Objective
        self.elbo = self.log_probs(self.x, encoder, net1, net2, net3, net4)

        # self.elbo = self.objective(*log_probs)
        # self.iwae_elbo = self.iwae_objective(*log_probs)


        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                epsilon=1e-02).minimize(-self.elbo)


        # for var in tf.global_variables():
        #     print var


        #Finalize Initilization
        self.init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()
        # self.sess = tf.Session()



    def sample_z(self, x, encoder, decoder, W):
        '''
        z: [P,B,Z]
        log_pz: [P,B]
        log_qz: [P,B]
        '''

        #Encode
        z_mean_logvar = encoder.feedforward(x) #[B,Z*2]
        z_mean = tf.slice(z_mean_logvar, [0,0], [self.batch_size, self.z_size]) #[B,Z] 
        z_logvar = tf.slice(z_mean_logvar, [0,self.z_size], [self.batch_size, self.z_size]) #[B,Z]

        #Sample z  [P,B,Z]
        eps = tf.random_normal((self.n_z_particles, self.batch_size, self.z_size), 0, 1, seed=self.rs) 
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps)) #broadcast, [P,B,Z]

        # Calc log probs [P,B]
        log_pz = log_norm(z, tf.zeros([self.batch_size, self.z_size]), 
                                tf.log(tf.ones([self.batch_size, self.z_size])))
        log_qz = log_norm(z, z_mean, z_logvar)

        return z, log_pz, log_qz




    def split_mean_logvar(self, mean_logvar):
        '''
        mean_logvar: [B,Z*2]
        output: [B,Z]
        '''

        mean = tf.slice(mean_logvar, [0,0], [self.batch_size, self.z_size]) #[B,Z] 
        logvar = tf.slice(mean_logvar, [0,self.z_size], [self.batch_size, self.z_size]) #[B,Z]

        return mean, logvar


    def sample_Gaussian(self, mean, logvar):

        eps = tf.random_normal((self.batch_size, self.z_size), 0, 1, seed=self.rs) 
        z = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(logvar)), eps)) # [B,Z]

        return z



    def log_probs(self, x, q_zlx, q_zlxz, r_zlxz, p_xlz):
        '''
        x: [B,X]
        '''

        # log_px_list = []
        # log_pz_list = []
        log_qz_list = []
        log_rz_list = []


        # Sample z0 and calc log_qz0
        z0_mean, z0_logvar = self.split_mean_logvar(q_zlx.feedforward(x)) #[B,Z]
        z0 = self.sample_Gaussian(z0_mean, z0_logvar) #[B,Z]
        log_qz0 = log_norm(z0, z0_mean, z0_logvar) #[B]

        for t in range(self.n_transitions):

            #reverse model
            zt_mean, zt_logvar = self.split_mean_logvar(r_zlxz.feedforward(x)) #[B,Z]
            log_rzt = log_norm(z_minus1, zt_mean, zt_logvar) #[B]
            log_rz_list.append(log_rzt)

            #new sample
            xz = tf.concat([x,z], axis=1) #[B,X+Z]
            z_mean, z_logvar = self.split_mean_logvar(q_zlxz.feedforward(xz)) #[B,Z]
            z = self.sample_Gaussian(z_mean, z_logvar) #[B,Z]
            log_qz = log_norm(z, z_mean, z_logvar) #[B]
            log_qz_list.append(log_qz)


        log_rzs = tf.stack(log_rz_list) #[T,B]
        log_qzs = tf.stack(log_qz_list) #[T,B]

        log_pz = log_norm(z, tf.zeros([self.batch_size, self.z_size]), 
                                tf.log(tf.ones([self.batch_size, self.z_size]))) #[B]

        # Sample z   [P,B,Z], [P,B], [P,B]
        # z, log_pz, log_qz = self.sample_z(x, encoder, decoder, W)
        # z: [PB,Z]
        # z = tf.reshape(z, [self.n_z_particles*self.batch_size, self.z_size])

        # Decode [B,X]
        x_mean = p_xlz.feedforward(z)
        log_px = log_bern(x,x_mean)

        # y: [P,B,X]
        # y = tf.reshape(y, [self.n_z_particles, self.batch_size, self.x_size])

        # Likelihood p(x|z)  [B]
        

        #Store for later
        # log_px_list.append(log_px)
        # log_pz_list.append(log_pz)
        # log_qz_list.append(log_qz)

        # log_px = tf.stack(log_px_list) #[P,B]
        # log_pz = tf.stack(log_pz_list) #[P,B]
        # log_qz = tf.stack(log_qz_list) #[P,B]

        #[B]
        elbo = log_px + log_pz + tf.reduce_sum(log_rzs,axis=0) - log_qz0 - tf.reduce_sum(log_qzs, axis=0)

        return tf.reduce_mean(elbo) #over batch 



    # def objective(self, log_px, log_pz, log_qz, log_pW, log_qW):
    #     '''
    #     Returns scalar to maximize
    #     '''

    #     # Calculte log probs for printing
    #     self.log_px = tf.reduce_mean(log_px)
    #     self.log_pz = tf.reduce_mean(log_pz)
    #     self.log_qz = tf.reduce_mean(log_qz)
    #     self.log_pW = tf.reduce_mean(log_pW)
    #     self.log_qW = tf.reduce_mean(log_qW)
    #     self.z_elbo = self.log_px + self.log_pz - self.log_qz 

    #     #Calc elbo
    #     elbo = self.log_px + self.log_pz - self.log_qz + self.batch_frac*(self.log_pW - self.log_qW)

    #     return elbo



    # def iwae_objective(self, log_px, log_pz, log_qz, log_pW, log_qW):
    #     '''
    #     Returns scalar to maximize
    #     x: [B,X]
    #     '''

    #     # Log mean exp over S and P, mean over B
    #     temp_elbo = tf.reduce_mean(log_px + log_pz - log_qz, axis=2)   #[S,P]
    #     log_pW = tf.reshape(log_pW, [self.n_W_particles, 1]) #[S,1]
    #     log_qW = tf.reshape(log_qW, [self.n_W_particles, 1]) #[S,1]
    #     temp_elbo = temp_elbo + (self.batch_frac*(log_pW - log_qW)) #broadcast, [S,P]
    #     temp_elbo = tf.reshape(temp_elbo, [self.n_W_particles*self.n_z_particles]) #[SP]
    #     max_ = tf.reduce_max(temp_elbo, axis=0) #[1]
    #     iwae_elbo = tf.log(tf.reduce_mean(tf.exp(temp_elbo-max_))) + max_  #[1]

    #     return iwae_elbo







    def train(self, train_x, valid_x=[], display_step=5, 
                path_to_load_variables='', path_to_save_variables='', 
                epochs=10, batch_size=20, n_W_particles=2, n_z_particles=3):
        '''
        Train.
        '''
        with tf.Session() as self.sess:
            # self.sess = tf.Session()
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
                                                            self.batch_frac: 1./float(n_datapoints)})
                    # Display logs per epoch step
                    if step % display_step == 0:
                        elbo,log_px,log_pz,log_qz,log_pW,log_qW, i_elbo = self.sess.run((self.elbo, 
                                                                                    self.log_px, self.log_pz, 
                                                                                    self.log_qz, self.log_pW, 
                                                                                    self.log_qW, self.iwae_elbo), 
                                                        feed_dict={self.x: batch, 
                                                            self.batch_frac: 1./float(n_datapoints)})
                        print ("Epoch", str(epoch+1)+'/'+str(epochs), 
                                'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), 
                                "elbo={:.4f}".format(float(elbo)),
                                log_px,log_pz,log_qz,log_pW,log_qW, i_elbo)

            if path_to_save_variables != '':
                self.saver.save(self.sess, path_to_save_variables)
                print 'Saved variables to ' + path_to_save_variables




    # def eval(self, data, display_step=5, path_to_load_variables='',
    #          batch_size=20, n_W_particles=2, n_z_particles=3, data2=[]):
    #     '''
    #     Evaluate.
    #     '''
    #     with tf.Session() as self.sess:

    #         n_datapoints = len(data)
    #         n_datapoints_for_frac = len(data2)

    #         if path_to_load_variables == '':
    #             self.sess.run(self.init_vars)
    #         else:
    #             #Load variables
    #             self.saver.restore(self.sess, path_to_load_variables)
    #             print 'loaded variables ' + path_to_load_variables

    #         iwae_elbos = []
    #         elbos=[]
    #         logpxs=[]
    #         logpzs=[]
    #         logqzs=[]
    #         logpWs=[]
    #         logqWs=[]

    #         data_index = 0
    #         for step in range(n_datapoints/batch_size):

    #             #Make batch
    #             batch = []
    #             while len(batch) != batch_size:
    #                 batch.append(data[data_index]) 
    #                 data_index +=1
    #             # Calc iwae elbo on test set
    #             iwae_elbo, elbo, log_px,log_pz,log_qz,log_pW,log_qW = self.sess.run((self.iwae_elbo, self.elbo,
    #                                                                         self.log_px, self.log_pz, 
    #                                                                         self.log_qz, self.log_pW, 
    #                                                                         self.log_qW), 
    #                                                     feed_dict={self.x: batch, 
    #                                                     self.batch_frac: 1./float(n_datapoints_for_frac)})
    #             iwae_elbos.append(iwae_elbo)
    #             elbos.append(elbo)
    #             logpxs.append(log_px)
    #             logpzs.append(log_pz)
    #             logqzs.append(log_qz)
    #             logpWs.append(log_pW)
    #             logqWs.append(log_qW)

    #     return [np.mean(iwae_elbos), np.mean(elbos), np.mean(logpxs), np.mean(logpzs), np.mean(logqzs), np.mean(logpWs), np.mean(logqWs)]






# class BIWAE(BVAE):


#     def __init__(self, hyperparams):


#         tf.reset_default_graph()

#         #Model hyperparameters
#         self.learning_rate = hyperparams['learning_rate']
#         self.encoder_act_func = tf.nn.elu #tf.nn.softplus #tf.tanh
#         self.decoder_act_func = tf.tanh
#         self.encoder_net = hyperparams['encoder_net']
#         self.decoder_net = hyperparams['decoder_net']
#         self.z_size = hyperparams['z_size']  #Z
#         self.x_size = hyperparams['x_size']  #X
#         self.rs = 0
#         self.n_W_particles = hyperparams['n_W_particles']  #S
#         # self.n_W_particles = tf.placeholder(tf.int32, None)  #S
#         self.n_z_particles = hyperparams['n_z_particles']  #P
#         # self.n_z_particles = tf.placeholder(tf.int32, None)  #P

#         #Placeholders - Inputs/Targets
#         self.x = tf.placeholder(tf.float32, [None, self.x_size])

#         self.batch_frac = tf.placeholder(tf.float32, None)
#         self.batch_size = tf.shape(self.x)[0]   #B

#         #Define endocer and decoder
#         with tf.variable_scope("encoder"):
#             encoder = NN(self.encoder_net, self.encoder_act_func, self.batch_size)

#         with tf.variable_scope("decoder"):
#             decoder = BNN(self.decoder_net, self.decoder_act_func, self.batch_size)
        
#         #Objective
#         log_probs = self.log_probs(self.x, encoder, decoder)

#         self.elbo = self.iwae_objective(*log_probs)

#         self.iwae_elbo = self.iwae_objective(*log_probs)

#         #for printing
#         stuff = self.objective(*log_probs)


#         # Minimize negative ELBO
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
#                                                 epsilon=1e-02).minimize(-self.elbo)

#         #Finalize Initilization
#         self.init_vars = tf.global_variables_initializer()
#         self.saver = tf.train.Saver()
#         tf.get_default_graph().finalize()
#         self.sess = tf.Session()

















if __name__ == '__main__':

    x_size = 784
    z_size = 10

    hyperparams = {
        'learning_rate': .0001,
        'x_size': x_size,
        'z_size': z_size,
        'encoder_net': [x_size, 20, z_size*2],
        'decoder_net': [z_size, 20, x_size],
        'n_W_particles': 1,
        'n_z_particles': 1}

    model = BVAE(hyperparams)

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
                epochs=2, batch_size=20, n_W_particles=1, n_z_particles=1, display_step=1000,
                path_to_load_variables=path_to_load_variables,
                path_to_save_variables=path_to_save_variables)


    # print 'Eval'
    # iwae_elbo = model.eval(data=test_x, batch_size=20, n_W_particles=2, n_z_particles=3, display_step=10,
    #             path_to_load_variables=path_to_load_variables, data2=train_x)

    # print iwae_elbo

    print 'Done.'





















