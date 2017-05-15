

# Hamiltonian Variational Inference


#not completed. still need to finish the init and the main.
# allow for multiple samples

# I confused the MCMC steps and the Hamiltonian steps here.
# when T=1, there shoudlnt be any list of q and r..
# ya use T=1. so no loop
# also the reverse model needs z, right now its only given x.

import numpy as np
import tensorflow as tf
import pickle
from os.path import expanduser
home = expanduser("~")

from utils import log_normal as log_norm
from utils import log_normal2 as log_norm2
from utils import log_bernoulli as log_bern
from utils import split_mean_logvar
from utils import sample_Gaussian

from NN import NN
from BNN import BNN




class HVI(object):


    def __init__(self, hyperparams):

        tf.reset_default_graph()

        #Model hyperparameters
        self.learning_rate = hyperparams['learning_rate']

        self.z_size = hyperparams['z_size']  #Z
        self.x_size = hyperparams['x_size']  #X
        self.rs = 0
        # self.n_W_particles = hyperparams['n_W_particles']  #S
        # self.n_W_particles = tf.placeholder(tf.int32, None)  #S
        self.n_z_particles = hyperparams['n_z_particles']  #P
        # self.n_z_particles = tf.placeholder(tf.int32, None)  #P
        self.n_transitions = hyperparams['leapfrog_steps'] #this is leapfrog steps, whereas T=1 like the paper


        # self.encoder_act_func = tf.nn.elu #tf.nn.softplus #tf.tanh
        # self.decoder_act_func = tf.tanh
        # self.encoder_net = hyperparams['encoder_net']
        # self.decoder_net = hyperparams['decoder_net']

        #Placeholders - Inputs/Targets
        self.x = tf.placeholder(tf.float32, [None, self.x_size])

        # self.batch_frac = tf.placeholder(tf.float32, None)
        self.batch_size = tf.shape(self.x)[0]   #B

        #Define networks q_zlx, q_vlxz, r_vlxz, p_xlz

        # q(z|x)
        net1 = NN([self.x_size, 300, 300, self.z_size*2], tf.nn.elu)
        # q(v|x,z) 
        net2 = NN([self.x_size+self.z_size, 300, 300, self.z_size*2], tf.nn.elu)
        # r(v|x,z)
        net3 = NN([self.x_size+self.z_size, 300, 300, self.z_size*2], tf.nn.elu)
        # p(x|z)
        net4 = NN([self.z_size, 300, 300, self.x_size], tf.tanh)
        

        #Objective
        log_probs_list = self.log_probs(self.x, net1, net2, net3, net4)

        self.elbo = self.objective(*log_probs_list)
        self.iwae_elbo = self.iwae_objective(*log_probs_list)


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



    # def sample_z(self, x, encoder, decoder, W):
    #     '''
    #     z: [P,B,Z]
    #     log_pz: [P,B]
    #     log_qz: [P,B]
    #     '''

    #     #Encode
    #     z_mean_logvar = encoder.feedforward(x) #[B,Z*2]
    #     z_mean = tf.slice(z_mean_logvar, [0,0], [self.batch_size, self.z_size]) #[B,Z] 
    #     z_logvar = tf.slice(z_mean_logvar, [0,self.z_size], [self.batch_size, self.z_size]) #[B,Z]

    #     #Sample z  [P,B,Z]
    #     eps = tf.random_normal((self.n_z_particles, self.batch_size, self.z_size), 0, 1, seed=self.rs) 
    #     z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps)) #broadcast, [P,B,Z]

    #     # Calc log probs [P,B]
    #     log_pz = log_norm(z, tf.zeros([self.batch_size, self.z_size]), 
    #                             tf.log(tf.ones([self.batch_size, self.z_size])))
    #     log_qz = log_norm(z, z_mean, z_logvar)

    #     return z, log_pz, log_qz


    def _log_px(self, p_xlz, x, z):
        '''
        x: [B,X]
        z: [P,B,Z]
        output: [P,B]
        '''

        z_reshaped = tf.reshape(z, [self.n_z_particles*self.batch_size, self.z_size]) #[PB,Z]
        x_mean = p_xlz.feedforward(z_reshaped) #[PB,X]
        x_mean = tf.reshape(x_mean, [self.n_z_particles, self.batch_size, self.x_size]) #[P,B,Z]

        log_px = log_bern(x,x_mean) #[P,B]

        return log_px




    def leapfrogs(self, z, v, p_xlz, x):
        '''
        z,v: [P,B,Z]
        x: [B,X] 
        '''


        self.step_size = tf.Variable([.1])

        for t in range(self.n_transitions):

            z_intermediate = z + ((.5*self.step_size) * v) 

            log_p = self._log_px(p_xlz, x, z_intermediate)
            grad = -tf.gradients(log_p, [z_intermediate])[0]
            v = v + (self.step_size * grad)

            z = z_intermediate + ((.5*self.step_size) * v)

        return z, v





    def log_probs(self, x, q_zlx, q_vlxz, r_vlxz, p_xlz):
        '''
        x: [B,X]
        '''

        # Sample z0 and calc log_qz0
        z0_mean, z0_logvar = split_mean_logvar(q_zlx.feedforward(x)) #[B,Z]
        z0 = sample_Gaussian(z0_mean, z0_logvar, self.n_z_particles) #[P,B,Z]
        log_qz0 = log_norm(z0, z0_mean, z0_logvar) #[P,B]

        # Sample v0 and calc log_qv0
        z0_reshaped = tf.reshape(z0, [self.n_z_particles*self.batch_size, self.z_size]) #[PB,Z]
        x_tiled = tf.tile(x, [self.n_z_particles, 1]) #[PB,X]
        xz = tf.concat([x_tiled,z0_reshaped], axis=1) #[PB,X+Z]
        v0_mean, v0_logvar = split_mean_logvar(q_vlxz.feedforward(xz)) #[PB,Z]
        v0 = sample_Gaussian(v0_mean, v0_logvar, 1) #[1,PB,Z]
        v0 = tf.reshape(v0, [self.n_z_particles*self.batch_size, self.z_size]) #[PB,Z]
        # v0 = tf.reshape(v0, [self.n_z_particles, self.batch_size, self.z_size]) #[P,B,Z]
        # v0_mean = tf.reshape(v0_mean, [self.n_z_particles, self.batch_size, self.z_size])  #[P,B,Z]
        # v0_logvar = tf.reshape(v0_logvar, [self.n_z_particles, self.batch_size, self.z_size]) #[P,B,Z]
        log_qv0 = log_norm2(v0, v0_mean, v0_logvar) #[PB]
        log_qv0 = tf.reshape(log_qv0, [self.n_z_particles, self.batch_size]) #[P,B]
        v0 = tf.reshape(v0, [self.n_z_particles, self.batch_size, self.z_size]) #[P,B,Z]

        # Transform [P,B,Z]
        zT, vT = self.leapfrogs(z0,v0,p_xlz,x)

        # Reverse model
        z_reshaped = tf.reshape(zT, [self.n_z_particles*self.batch_size, self.z_size]) #[PB,Z]
        xz = tf.concat([x_tiled,z_reshaped], axis=1) #[PB,X+Z]
        vt_mean, vt_logvar = split_mean_logvar(r_vlxz.feedforward(xz)) #[PB,Z]
        vT = tf.reshape(vT, [self.n_z_particles*self.batch_size, self.z_size]) #[PB,Z]
        log_rv = log_norm2(vT, vt_mean, vt_logvar) #[PB]
        log_rv = tf.reshape(log_rv, [self.n_z_particles, self.batch_size]) #[PB]


        log_pz = log_norm(zT, tf.zeros([self.batch_size, self.z_size]), 
                                tf.log(tf.ones([self.batch_size, self.z_size]))) #[P,B]

        # # Decode [P,B,X]
        # zT_reshaped = tf.reshape(zT, [self.n_z_particles*self.batch_size, self.z_size]) #[PB,Z]
        # x_mean = p_xlz.feedforward(zT_reshaped) #[PB,X]
        # x_mean = tf.reshape(x_mean, [self.n_z_particles, self.batch_size, self.z_size]) #[P,B,Z]
        # log_px = log_bern(x,x_mean) #[P,B]

        # [P,B]
        log_px = self._log_px(p_xlz, x, zT)

        # #[B]
        # elbo = log_px + log_pz + tf.reduce_sum(log_rzs,axis=0) - log_qz0 - tf.reduce_sum(log_qzs, axis=0)

        # return tf.reduce_mean(elbo) #over batch 
        return [log_px, log_pz, log_qz0, log_qv0, log_rv]





    def objective(self, log_px, log_pz, log_qz, log_qv, log_rv):
        '''
        Input: [P,B]
        Output: [1]
        '''

        # Calculte log probs for printing
        self.log_px = tf.reduce_mean(log_px)
        self.log_pz = tf.reduce_mean(log_pz)
        self.log_qz = tf.reduce_mean(log_qz)
        self.log_qv = tf.reduce_mean(log_qv)
        self.log_rv = tf.reduce_mean(log_rv)
        # self.z_elbo = self.log_px + self.log_pz - self.log_qz 

        #Calc elbo
        elbo = self.log_px + self.log_pz + self.log_rv - self.log_qz - self.log_qv

        return elbo



    def iwae_objective(self, log_px, log_pz, log_qz, log_qv, log_rv):
        '''
        Input: [P,B]
        Output: [1]
        '''

        # print log_px, log_pz, log_qz, log_qv, log_rv

        # mean over B, Log mean exp over P, 
        temp_elbo = tf.reduce_mean(log_px + log_pz + log_rv - log_qz - log_qv, axis=1)  #[P]
        max_ = tf.reduce_max(temp_elbo, axis=0) #[1]
        iwae_elbo = tf.log(tf.reduce_mean(tf.exp(temp_elbo-max_))) + max_  #[1]

        return iwae_elbo







    def train(self, train_x, valid_x=[], display_step=5, 
                path_to_load_variables='', path_to_save_variables='', 
                epochs=10, batch_size=20):
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
                    _ = self.sess.run((self.optimizer), feed_dict={self.x: batch})
                    # Display logs per epoch step
                    if step % display_step == 0:
                        elbo,log_px,log_pz,log_qz, i_elbo = self.sess.run((self.elbo, 
                                                                                    self.log_px, self.log_pz, 
                                                                                    self.log_qz, self.iwae_elbo), 
                                                        feed_dict={self.x: batch})
                        print ("Epoch", str(epoch+1)+'/'+str(epochs), 
                                'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), 
                                "elbo={:.4f}".format(float(elbo)),
                                log_px,log_pz,log_qz, i_elbo)

            if path_to_save_variables != '':
                self.saver.save(self.sess, path_to_save_variables)
                print 'Saved variables to ' + path_to_save_variables




    def eval(self, data, display_step=100, path_to_load_variables='',
             batch_size=20, data2=[]):
        '''
        Evaluate.
        '''
        with tf.Session() as self.sess:

            n_datapoints = len(data)
            # n_datapoints_for_frac = len(data2)

            if path_to_load_variables == '':
                self.sess.run(self.init_vars)
            else:
                #Load variables
                self.saver.restore(self.sess, path_to_load_variables)
                print 'loaded variables ' + path_to_load_variables

            iwae_elbos = []
            elbos=[]
            logpxs=[]
            logpzs=[]
            logqzs=[]
            # logpWs=[]
            # logqWs=[]

            data_index = 0
            for step in range(n_datapoints/batch_size):

                if step % display_step == 0:
                    print step

                #Make batch
                batch = []
                while len(batch) != batch_size:
                    batch.append(data[data_index]) 
                    data_index +=1
                # Calc iwae elbo on test set
                iwae_elbo, elbo, log_px,log_pz,log_qz = self.sess.run((self.iwae_elbo, self.elbo,
                                                                            self.log_px, self.log_pz, 
                                                                            self.log_qz), 
                                                        feed_dict={self.x: batch})
                iwae_elbos.append(iwae_elbo)
                elbos.append(elbo)
                logpxs.append(log_px)
                logpzs.append(log_pz)
                logqzs.append(log_qz)
                # logpWs.append(log_pW)
                # logqWs.append(log_qW)

        return [np.mean(iwae_elbos), np.mean(elbos), np.mean(logpxs), np.mean(logpzs), np.mean(logqzs)]#, np.mean(logpWs), np.mean(logqWs)]






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
    x_size = 784
    z_size = 30
    hyperparams = {
        'learning_rate': .0001,
        'x_size': x_size,
        'z_size': z_size,
        'n_z_particles': 2,
        'leapfrog_steps': 5}
    
    model = HVI(hyperparams)
    model.train(train_x=train_x,
                epochs=1, batch_size=20, display_step=1000,
                path_to_load_variables=path_to_load_variables,
                path_to_save_variables=path_to_save_variables)


    print 'Eval'
    hyperparams = {
        'learning_rate': .0001,
        'x_size': x_size,
        'z_size': z_size,
        'n_z_particles': 10,
        'leapfrog_steps': 5}
    model = HVI(hyperparams)
    iwae_elbo = model.eval(data=test_x, batch_size=2, display_step=100,
                path_to_load_variables=path_to_save_variables, data2=train_x)

    print iwae_elbo

    print 'Done.'





















