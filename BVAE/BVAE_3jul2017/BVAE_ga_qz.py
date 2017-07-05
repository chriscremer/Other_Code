


# Bayesian Variational Autoencoder



import numpy as np
import tensorflow as tf
import pickle
from os.path import expanduser
home = expanduser("~")

from utils import log_normal as log_norm
from utils import log_bernoulli as log_bern
from utils import split_mean_logvar  

from NN import NN
from BNN import BNN
from sample_z import Sample_z

slim=tf.contrib.slim


class BVAE(object):


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
        self.n_W_particles = hyperparams['n_W_particles']  #S
        self.n_z_particles = hyperparams['n_z_particles']  #P

        self.qW_weight = hyperparams['qW_weight']
        self.lmba = hyperparams['lmba']

        #Placeholders - Inputs/Targets
        self.x = tf.placeholder(tf.float32, [None, self.x_size])
        self.batch_size = tf.shape(self.x)[0]   #B
        self.batch_frac = tf.placeholder(tf.float32, None)

        n_transformations = 3

        #Define endocer and decoder
        with tf.variable_scope("encoder"):
            encoder = NN(self.encoder_net, self.encoder_act_func, self.batch_size)

        self.l2_sum = encoder.weight_decay()

        with tf.variable_scope("decoder"):
            decoder = BNN(self.decoder_net, self.decoder_act_func, self.batch_size)

        with tf.variable_scope("sample_z"):
            sample_z = Sample_z(self.batch_size, self.z_size, self.n_z_particles, n_transformations)

        with tf.variable_scope("log_probs"):
            log_probs = self.log_probs(self.x, encoder, decoder, sample_z)

        with tf.variable_scope("objectives"):
            self.elbo = self.objective(*log_probs)
            self.iwae_elbo_test = self.iwae_objective_test(*log_probs)


        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                epsilon=1e-02).minimize(-self.elbo)


        # for var in tf.global_variables():
        #     print var

        # fasdf

        # FOR INSPECING MODEL

        self.decoder_means = decoder.W_means
        self.decoder_logvars = decoder.W_logvars

        # print 'right here'

        # with tf.variable_scope("is_it_this"):
        #     self.recons, self.priors = self.get_x_samples(self.x, encoder, decoder, sample_z)


        # print 'prob here'


        #Finalize Initilization
        self.init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()
        # self.sess = tf.Session()

        # fadsfasf

    def log_probs(self, x, encoder, decoder, sample_z):

        # log_px_list = []
        # log_pz_list = []
        # log_qz_list = []
        # log_pW_list = []
        # log_qW_list = []

        def foo(i, log_pxi, log_pzi, log_qzi, log_pWi, log_qWi):

            # Sample decoder weights  __, [1], [1]
            W, log_pW, log_qW = decoder.sample_weights()
            log_pW = tf.reshape(log_pW, [1])
            log_qW = tf.reshape(log_qW, [1])

            # Sample z   [P,B,Z], [P,B], [P,B]
            z, log_pz, log_qz = sample_z.sample_z(x, encoder, decoder, W)
            log_pz = tf.reshape(log_pz, [1, self.n_z_particles, self.batch_size])
            log_qz = tf.reshape(log_qz, [1, self.n_z_particles, self.batch_size])
            # z: [PB,Z]
            z = tf.reshape(z, [self.n_z_particles*self.batch_size, self.z_size])

            # Decode [PB,X]
            y = decoder.feedforward(W, z)
            # y: [P,B,X]
            y = tf.reshape(y, [self.n_z_particles, self.batch_size, self.x_size])

            # Likelihood p(x|z)  [P,B]
            log_px = log_bern(x,y)
            log_px = tf.reshape(log_px, [1, self.n_z_particles, self.batch_size])

            # #Store for later
            # log_px_list.append(log_px)
            # log_pz_list.append(log_pz)
            # log_qz_list.append(log_qz)
            # log_pW_list.append(log_pW)
            # log_qW_list.append(log_qW)

            log_px = tf.concat([log_pxi, log_px], axis=0)
            log_pz = tf.concat([log_pzi, log_pz], axis=0)
            log_qz = tf.concat([log_qzi, log_qz], axis=0)
            log_pW = tf.concat([log_pWi, log_pW], axis=0)
            log_qW = tf.concat([log_qWi, log_qW], axis=0)

            return [i+1, log_px, log_pz, log_qz, log_pW, log_qW]


        i0 = tf.constant(0)
        log_px0 = tf.zeros([1,self.n_z_particles, self.batch_size]) 
        log_pz0 = tf.zeros([1,self.n_z_particles, self.batch_size]) 
        log_qz0 = tf.zeros([1,self.n_z_particles, self.batch_size]) 
        log_pW0 = tf.zeros([1]) 
        log_qW0 = tf.zeros([1]) 


        c = lambda i, log_px, log_pz, log_qz, log_pW, log_qW: i < self.n_W_particles
        it, log_px, log_pz, log_qz, log_pW, log_qW = tf.while_loop(c, foo, 
                            loop_vars=[i0, log_px0, log_pz0, log_qz0, log_pW0, log_qW0], 
                            shape_invariants=[i0.get_shape(), 
                                                tf.TensorShape([None, self.n_z_particles, None]), 
                                                tf.TensorShape([None, self.n_z_particles, None]),
                                                tf.TensorShape([None, self.n_z_particles, None]),
                                                tf.TensorShape([None]),
                                                tf.TensorShape([None])])


        #remove the inits
        log_px = tf.slice(log_px, [1, 0, 0], [self.n_W_particles, self.n_z_particles, self.batch_size])
        log_pz = tf.slice(log_pz, [1, 0, 0], [self.n_W_particles, self.n_z_particles, self.batch_size])
        log_qz = tf.slice(log_qz, [1, 0, 0], [self.n_W_particles, self.n_z_particles, self.batch_size])
        log_pW = tf.slice(log_pW, [1], [self.n_W_particles])
        log_qW = tf.slice(log_qW, [1], [self.n_W_particles])






        # fsadf



        # log_px_list = []
        # log_pz_list = []
        # log_qz_list = []
        # log_pW_list = []
        # log_qW_list = []

        # for W_i in range(self.n_W_particles):

        #     # Sample decoder weights  __, [1], [1]
        #     W, log_pW, log_qW = decoder.sample_weights()

        #     # Sample z   [P,B,Z], [P,B], [P,B]
        #     z, log_pz, log_qz = sample_z.sample_z(x, encoder, decoder, W)
        #     # z: [PB,Z]
        #     z = tf.reshape(z, [self.n_z_particles*self.batch_size, self.z_size])

        #     # Decode [PB,X]
        #     y = decoder.feedforward(W, z)
        #     # y: [P,B,X]
        #     y = tf.reshape(y, [self.n_z_particles, self.batch_size, self.x_size])

        #     # Likelihood p(x|z)  [P,B]
        #     log_px = log_bern(x,y)

        #     #Store for later
        #     log_px_list.append(log_px)
        #     log_pz_list.append(log_pz)
        #     log_qz_list.append(log_qz)
        #     log_pW_list.append(log_pW)
        #     log_qW_list.append(log_qW)


        # log_px = tf.stack(log_px_list) #[S,P,B]
        # log_pz = tf.stack(log_pz_list) #[S,P,B]
        # log_qz = tf.stack(log_qz_list) #[S,P,B]
        # log_pW = tf.stack(log_pW_list) #[S]
        # log_qW = tf.stack(log_qW_list) #[S]
        # print log_pW
        # log_pW = tf.reshape(log_pW, [])
        # log_qW = tf.reshape(log_qW, [])
        # print log_px, log_pz, log_qz, log_pW, log_qW
        # fafds

        return [log_px, log_pz, log_qz, log_pW, log_qW]  



    def objective(self, log_px, log_pz, log_qz, log_pW, log_qW):
        '''
        log_px, log_pz, log_qz: [S,P,B]
        log_pW, log_qW: [S]
        Output: [1]
        '''

        # Calculte log probs for printing
        self.log_px = tf.reduce_mean(log_px)
        self.log_pz = tf.reduce_mean(log_pz)
        self.log_qz = tf.reduce_mean(log_qz)
        self.log_pW = tf.reduce_mean(log_pW)
        self.log_qW = tf.reduce_mean(log_qW)
        # self.z_elbo = self.log_px + self.log_pz - self.log_qz 

        #Calc elbo
        elbo = self.log_px + self.log_pz - self.log_qz + self.batch_frac*(self.log_pW - (self.log_qW*self.qW_weight)) - (self.lmba*self.l2_sum)

        return elbo



    # def iwae_objective(self, log_px, log_pz, log_qz, log_pW, log_qW):
    #     '''
    #     log_px, log_pz, log_qz: [S,P,B]
    #     log_pW, log_qW: [S]
    #     Output: [1]
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



    # def objective_test(self, log_px, log_pz, log_qz, log_pW, log_qW):
    #     '''
    #     Uses approximate posterior
    #     log_px, log_pz, log_qz: [S,P,B]
    #     log_pW, log_qW: [S]
    #     Output: [1]
    #     '''

    #     # Calculte log probs for printing
    #     log_px = tf.reduce_mean(log_px)
    #     log_pz = tf.reduce_mean(log_pz)
    #     log_qz = tf.reduce_mean(log_qz)
    #     # self.log_pW = tf.reduce_mean(log_pW)
    #     # self.log_qW = tf.reduce_mean(log_qW)
    #     # self.z_elbo = self.log_px + self.log_pz - self.log_qz 

    #     #Calc elbo
    #     elbo = self.log_px + self.log_pz - self.log_qz# + self.batch_frac*(self.log_pW - self.log_qW)

    #     return elbo



    def iwae_objective_test(self, log_px, log_pz, log_qz, log_pW, log_qW):
        '''
        Uses approximate posterior
        log_px, log_pz, log_qz: [S,P,B]
        log_pW, log_qW: [S]
        Output: [1]
        '''

        # print log_px, log_pz, log_qz, log_pW, log_qW

        # log_px_eval = tf.reduce_mean(log_px)   #[1]
        # log_pz_eval = tf.reduce_mean(log_pz)   #[1]
        # log_qz_eval = tf.reduce_mean(log_qz)   #[1]


        # Log mean exp over S and P, mean over B
        temp_elbo = tf.reduce_mean(log_px + log_pz - log_qz, axis=2)   #[S,P]
        # log_pW = tf.reshape(log_pW, [self.n_W_particles, 1]) #[S,1]
        # log_qW = tf.reshape(log_qW, [self.n_W_particles, 1]) #[S,1]
        # temp_elbo = temp_elbo #+ (self.batch_frac*(log_pW - log_qW)) #broadcast, [S,P]
        temp_elbo = tf.reshape(temp_elbo, [self.n_W_particles*self.n_z_particles]) #[SP]
        max_ = tf.reduce_max(temp_elbo, axis=0) #[1]
        iwae_elbo = tf.log(tf.reduce_mean(tf.exp(temp_elbo-max_))) + max_  #[1]

        return iwae_elbo



    # def random_bernoulli(self, shape, p=0.5):
    #     if isinstance(shape, (list, tuple)):
    #         shape = tf.stack(shape)
    #     return tf.where(tf.random_uniform(shape) < p, tf.ones(shape), tf.zeros(shape))



    # def transform_sample(self, z):
    #     '''
    #     z: [P,B,Z]
    #     '''

    #     P = tf.shape(z)[0]
    #     B = tf.shape(z)[1]
    #     # Z = tf.shape(z)[2]


    #     z = tf.reshape(z, [P*B,self.z_size])

    #     self.n_transitions = 3

    #     logdet_sum = tf.zeros([P*B])
    #     #Flows z0 -> zT
    #     for t in range(self.n_transitions):

    #         # print mask*z

            
    #         mask = self.random_bernoulli(tf.shape(z), p=0.5)
    #         h = slim.stack(mask*z,slim.fully_connected,[100])
    #         mew_ = slim.fully_connected(h,self.z_size,activation_fn=None) 
    #         sig_ = slim.fully_connected(h,self.z_size,activation_fn=tf.nn.sigmoid) 

    #         z = (mask * z) + (1-mask)*(z*sig_ + (1-sig_)*mew_)

    #         logdet = tf.reduce_sum((1-mask)*tf.log(sig_), axis=1) #[PB]

    #         logdet_sum += logdet

    #     z = tf.reshape(z, [P,B,self.z_size])
    #     logdet_sum = tf.reshape(logdet_sum, [P,B])

    #     print 'made'

    #     return z, logdet_sum




    # # def transform_sample(self, z):
    # #     '''
    # #     z: [P,B,Z]
    # #     '''

    # #     P = tf.shape(z)[0]
    # #     B = tf.shape(z)[1]
    # #     # # Z = tf.shape(z)[2]


    # #     # z = tf.reshape(z, [P*B,self.z_size])

    # #     # self.n_transitions = 3

    # #     logdet_sum = tf.zeros([P*B])
    # #     # #Flows z0 -> zT
    # #     # for t in range(self.n_transitions):

    # #     #     # print mask*z

            
    # #     #     mask = self.random_bernoulli(tf.shape(z), p=0.5)
    # #     #     h = slim.stack(mask*z,slim.fully_connected,[100])
    # #     #     mew_ = slim.fully_connected(h,self.z_size,activation_fn=None) 
    # #     #     sig_ = slim.fully_connected(h,self.z_size,activation_fn=tf.nn.sigmoid) 

    # #     #     z = (mask * z) + (1-mask)*(z*sig_ + (1-sig_)*mew_)

    # #     #     logdet = tf.reduce_sum((1-mask)*tf.log(sig_), axis=1) #[PB]

    # #     #     logdet_sum += logdet

    # #     # z = tf.reshape(z, [P,B,self.z_size])
    # #     # logdet_sum = tf.reshape(logdet_sum, [P,B])

    # #     return z, logdet_sum




    # def sample_z(self, x, encoder, decoder, W):
    #     '''
    #     z: [P,B,Z]
    #     log_pz: [P,B]
    #     log_qz: [P,B]
    #     '''

    #     for i in range(len(W)):

    #         if i ==0:
    #             flatten_W = tf.reshape(W[i], [-1])
    #             # print flatten_W
    #         else:
    #             flattt = tf.reshape(W[i], [-1])
    #             # print flattt
    #             flatten_W = tf.concat([flatten_W, flattt], axis=0)

    #     flatten_W = tf.reshape(flatten_W, [1,-1])
    #     tiled = tf.tile(flatten_W, [self.batch_size, 1])
    #     intput_ = tf.concat([x,tiled], axis=1)

    #     #Encode
    #     z_mean_logvar = encoder.feedforward(intput_) #[B,Z*2]
    #     z_mean = tf.slice(z_mean_logvar, [0,0], [self.batch_size, self.z_size]) #[B,Z] 
    #     z_logvar = tf.slice(z_mean_logvar, [0,self.z_size], [self.batch_size, self.z_size]) #[B,Z]

    #     #Sample z  [P,B,Z]
    #     eps = tf.random_normal((self.n_z_particles, self.batch_size, self.z_size), 0, 1, seed=self.rs) 
    #     z0 = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps)) #broadcast, [P,B,Z]
    #     log_qz0 = log_norm(z0, z_mean, z_logvar)


    #     #[P,B,Z], [P,B]
    #     z,logdet = self.transform_sample(z0)

    #     # Calc log probs [P,B]
    #     log_pzT = log_norm(z, tf.zeros([self.batch_size, self.z_size]), 
    #                             tf.log(tf.ones([self.batch_size, self.z_size])))
        
    #     log_pz = log_pzT  + logdet

    #     log_qz =  log_qz0 


    #     # # Calc log probs [P,B]
    #     # log_pz = log_norm(z, tf.zeros([self.batch_size, self.z_size]), 
    #     #                         tf.log(tf.ones([self.batch_size, self.z_size])))
    #     # log_qz = log_norm(z, z_mean, z_logvar)

    #     return z, log_pz, log_qz





    def get_x_samples(self, x, encoder, decoder, sample_z):

        recons = []
        priors = []

        for W_i in range(self.n_W_particles):

            # Sample decoder weights  __, [1], [1]
            W, log_pW, log_qW = decoder.sample_weights()

            # Sample z   [P,B,Z], [P,B], [P,B]
            z, log_pz, log_qz = sample_z.sample_z(x, encoder, decoder, W)
            # z: [PB,Z]
            z = tf.reshape(z, [self.n_z_particles*self.batch_size, self.z_size])

            # Decode [PB,X]
            y = decoder.feedforward(W, z)
            # y: [P,B,X]
            y = tf.reshape(y, [self.n_z_particles, self.batch_size, self.x_size])
            y = tf.sigmoid(y)
            recons.append(y)

            # # Likelihood p(x|z)  [P,B]
            # log_px = log_bern(x,y)

            # #Store for later
            # log_px_list.append(log_px)

            #Sample prior
            z = tf.random_normal((self.n_z_particles, self.batch_size, self.z_size), 0, 1, seed=self.rs) 
            z = tf.reshape(z, [self.n_z_particles*self.batch_size, self.z_size])

            # Decode [PB,X]
            y = decoder.feedforward(W, z)
            # y: [P,B,X]
            y = tf.reshape(y, [self.n_z_particles, self.batch_size, self.x_size])
            y = tf.sigmoid(y)
            priors.append(y)


        recons = tf.stack(recons)
        priors = tf.stack(priors)

        return recons, priors



    def train(self, train_x, valid_x=[], display_step=[1,100], 
                path_to_load_variables='', path_to_save_variables='', 
                epochs=10, batch_size=20):
        '''
        Train.
        Display step: [0] every x epochs, [1] every x steps
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

            # elbos = []
            # logpxs=[]
            # logpzs=[]
            # logqzs=[]
            # logpWs=[]
            # logqWs=[]
            # l2_sums=[]
            values =[]

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
                    if step % display_step[1] == 0 and epoch % display_step[0] == 0:
                        elbo,log_px,log_pz,log_qz,log_pW,log_qW,l2_sum = self.sess.run((self.elbo, 
                                                                                    self.log_px, self.log_pz, 
                                                                                    self.log_qz, self.log_pW, 
                                                                                    self.log_qW, self.l2_sum), 
                                                        feed_dict={self.x: batch, 
                                                            self.batch_frac: 1./float(n_datapoints)})
                        print ("Epoch", str(epoch+1)+'/'+str(epochs), 
                                'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), 
                                "elbo={:.4f}".format(float(elbo)),
                                log_px,log_pz,log_qz,log_pW,log_qW)


                        values.append([epoch, elbo,log_px,log_pz,-log_qz,log_pW,-log_qW,-l2_sum])

            if path_to_save_variables != '':
                self.saver.save(self.sess, path_to_save_variables)
                print 'Saved variables to ' + path_to_save_variables

        labels = ['epoch', 'elbo','log_px','log_pz','-log_qz','log_pW','-log_qW','-l2_sum']

        return np.array(values), labels










    def train2(self, train_x, valid_x=[], display_step=[1,100], 
                path_to_load_variables='', path_to_save_variables='', 
                epochs=10, batch_size=20):
        '''
        Train. THis train also computes the test scores
        Display step: [0] every x epochs, [1] every x steps
        '''
        with tf.Session() as self.sess:
            # self.sess = tf.Session()
            random_seed=1
            rs=np.random.RandomState(random_seed)
            n_datapoints = len(train_x)
            n_datapoints_valid = len(valid_x)
            arr = np.arange(n_datapoints)

            if path_to_load_variables == '':
                self.sess.run(self.init_vars)
            else:
                #Load variables
                self.saver.restore(self.sess, path_to_load_variables)
                print 'loaded variables ' + path_to_load_variables

            values =[]
            valid_values =[]

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
                        if data_index == n_datapoints:
                            data_index = 0


                    # Fit training using batch data
                    _ = self.sess.run((self.optimizer), feed_dict={self.x: batch, 
                                                            self.batch_frac: 1./float(n_datapoints)})




                    # Display logs per epoch step
                    # if step % display_step[1] == 0 and epoch % display_step[0] == 0:
                    if epoch % display_step == 0 and step ==0:


                        #Get scores on training set
                        elbo,log_px,log_pz,log_qz,log_pW,log_qW,l2_sum = self.sess.run((self.elbo, 
                                                                                    self.log_px, self.log_pz, 
                                                                                    self.log_qz, self.log_pW, 
                                                                                    self.log_qW, self.l2_sum), 
                                                        feed_dict={self.x: batch, 
                                                            self.batch_frac: 1./float(n_datapoints)})
                        print ("Epoch", str(epoch+1)+'/'+str(epochs), 
                                'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), 
                                "elbo={:.4f}".format(float(elbo)),
                                log_px,log_pz,log_qz,log_pW,log_qW)

                        if epoch != 0:
                            values.append([epoch, elbo,log_px,log_pz,-log_qz,log_pW,-log_qW,-l2_sum])


                            #Run over test set, just a subset, all would take too long I think, maybe not becasue using less particles..
                            iwae_elbos = []
                            logpxs=[]
                            logpzs=[]
                            logqzs=[]
                            logpWs=[]
                            logqWs=[]
                            l2_sums=[]

                            # data_index2 = 0
                            # for step2 in range(n_datapoints_valid/batch_size):
                            for step2 in range(1000/batch_size):


                                #Make batch
                                batch = []
                                while len(batch) != batch_size:
                                    # batch.append(valid_x[data_index2]) 
                                    batch.append(valid_x[np.random.randint(0,len(valid_x))]) 
                                    # data_index2 +=1
                                # Calc iwae elbo on test set
                                iwae_elbo, log_px,log_pz,log_qz,log_pW,log_qW,l2_sum = self.sess.run((self.iwae_elbo_test,
                                                                                            self.log_px, self.log_pz, 
                                                                                            self.log_qz, self.log_pW, 
                                                                                            self.log_qW, self.l2_sum), 
                                                                        feed_dict={self.x: batch})
                                iwae_elbos.append(iwae_elbo)
                                logpxs.append(log_px)
                                logpzs.append(log_pz)
                                logqzs.append(log_qz)
                                logpWs.append(log_pW)
                                logqWs.append(log_qW)
                                l2_sums.append(l2_sum)

                            test_results = [epoch, np.mean(iwae_elbos), np.mean(logpxs), np.mean(logpzs), -np.mean(logqzs), np.mean(logpWs), -np.mean(logqWs), -np.mean(l2_sums)]

                            valid_values.append(test_results)

                        

            if path_to_save_variables != '':
                self.saver.save(self.sess, path_to_save_variables)
                print 'Saved variables to ' + path_to_save_variables

        labels = ['epoch', 'elbo','log_px','log_pz','-log_qz','log_pW','-log_qW','-l2_sum']
        test_labels = ['epoch', 'test_iwae_elbo','test_log_px','test_log_pz','-test_log_qz','test_log_pW','-test_log_qW','-test_l2_sum']

        return np.array(values), labels, np.array(valid_values), test_labels










    def eval(self, data, display_step=5, path_to_load_variables='',
             batch_size=20, data2=[]):
        '''
        Evaluate.
        '''
        with tf.Session() as self.sess:

            n_datapoints = len(data)
            n_datapoints_for_frac = len(data2)

            if path_to_load_variables == '':
                self.sess.run(self.init_vars)
            else:
                #Load variables
                self.saver.restore(self.sess, path_to_load_variables)
                print 'loaded variables ' + path_to_load_variables

            iwae_elbos = []
            logpxs=[]
            logpzs=[]
            logqzs=[]
            logpWs=[]
            logqWs=[]
            l2_sums=[]

            data_index = 0
            for step in range(n_datapoints/batch_size):

                #Make batch
                batch = []
                while len(batch) != batch_size:
                    batch.append(data[data_index]) 
                    data_index +=1
                # Calc iwae elbo on test set
                iwae_elbo, log_px,log_pz,log_qz,log_pW,log_qW,l2_sum = self.sess.run((self.iwae_elbo_test,
                                                                            self.log_px, self.log_pz, 
                                                                            self.log_qz, self.log_pW, 
                                                                            self.log_qW, self.l2_sum), 
                                                        feed_dict={self.x: batch})
                iwae_elbos.append(iwae_elbo)
                logpxs.append(log_px)
                logpzs.append(log_pz)
                logqzs.append(log_qz)
                logpWs.append(log_pW)
                logqWs.append(log_qW)
                l2_sums.append(l2_sum)

            test_results = [np.mean(iwae_elbos), np.mean(logpxs), np.mean(logpzs), np.mean(logqzs), np.mean(logpWs), np.mean(logqWs), np.mean(l2_sums)]
            test_labels = ['iwae_elbo','log_px','log_pz','log_qz','log_pW','log_qW','l2_sum']


            #get training info too
            batch = []
            rs=np.random.RandomState(0)
            while len(batch) != batch_size:
                batch.append(data2[rs.randint(0,len(data2))]) 
 
            elbo,log_px,log_pz,log_qz,log_pW,log_qW, l2_sum, batch_frac = self.sess.run((self.iwae_elbo_test, 
                                                                        self.log_px, self.log_pz, 
                                                                        self.log_qz, self.log_pW, 
                                                                        self.log_qW, self.l2_sum,
                                                                        self.batch_frac), 
                                                feed_dict={self.x: batch, 
                                                    self.batch_frac: 1./n_datapoints_for_frac})

            train_results = [elbo,log_px,log_pz,log_qz,log_pW,log_qW,l2_sum,batch_frac]

            train_labels = ['elbo','log_px','log_pz','log_qz','log_pW','log_qW','l2_sum','batch_frac']


            return test_results, train_results, test_labels, train_labels




    def get_means_logvars(self, path_to_load_variables=''):

        with tf.Session() as self.sess:

            if path_to_load_variables == '':
                self.sess.run(self.init_vars)
            else:
                #Load variables
                self.saver.restore(self.sess, path_to_load_variables)
                print 'loaded variables ' + path_to_load_variables

            means, logvars = self.sess.run((self.decoder_means, self.decoder_logvars))

            return means, logvars


    def get_recons_and_priors(self, data, batch_size, path_to_load_variables=''):


        with tf.Session() as self.sess:

            if path_to_load_variables == '':
                self.sess.run(self.init_vars)
            else:
                #Load variables
                self.saver.restore(self.sess, path_to_load_variables)
                print 'loaded variables ' + path_to_load_variables

            data_index = 0
            batch = []
            rs=np.random.RandomState(0)
            while len(batch) != batch_size:
                
                batch.append(data[rs.randint(0,len(data))]) 
                # batch.append(data[data_index])     
                # data_index +=1
                # if data_index >= len(data):
                #     data_index = 0
            batch = np.array(batch)

            recons, priors = self.sess.run((self.recons, self.priors), feed_dict={self.x: batch})

            return batch, recons, priors




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

#         #Placeholders - Inputs/Targets
#         self.x = tf.placeholder(tf.float32, [None, self.x_size])
#         self.n_z_particles = tf.placeholder(tf.int32, None)  #P
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
                epochs=1, batch_size=20, display_step=[1,1000],
                path_to_load_variables=path_to_load_variables,
                path_to_save_variables=path_to_save_variables)



    print 'Eval'
    iwae_elbo = model.eval(data=test_x, batch_size=20, display_step=10,
                path_to_load_variables=path_to_save_variables, data2=train_x)

    print iwae_elbo

    print 'Done.'























