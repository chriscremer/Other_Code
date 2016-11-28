
#Generative Autoencoder classes
import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle




class VAE():

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=5, n_particles=3):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_particles = n_particles
        self.n_z = network_architecture["n_z"]
        self.n_input = network_architecture["n_input"]

        #Placeholders - Inputs
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        #Variables
        self.network_weights = self._initialize_weights(**self.network_architecture)
        #Encoder - Recognition model - p(z|x): recog_mean,z_log_std_sq=[batch_size, n_z]
        self.recog_mean, self.recog_log_std_sq = self._recognition_network(self.network_weights["weights_recog"], self.network_weights["biases_recog"])
        #Sample
        eps = tf.random_normal((self.n_particles, self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.recog_mean, tf.mul(tf.sqrt(tf.exp(self.recog_log_std_sq)), eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]
        #Decoder - Generative model - p(x|z)
        self.x_reconstr_mean_no_sigmoid = self._generator_network(self.network_weights["weights_gener"], self.network_weights["biases_gener"]) #no sigmoid
        # self.x_reconstr_mean = tf.nn.sigmoid(self.x_reconstr_mean_no_sigmoid) #shape=[n_particles, n_batch, n_input]

        #Objective
        self.elbo = self.log_likelihood() + self._log_p_z() - self._log_p_z_given_x()

        self.iwae_elbo = self.iwae_elbo_calc()

        # self.w = tf.exp(self.i_log_likelihood() + self.i_log_p_z() - self.i_log_p_z_given_x())
        self.w = self.i_log_likelihood() + self.i_log_p_z() - self.i_log_p_z_given_x()

        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-04).minimize(-self.elbo)


    def _recognition_network(self, weights, biases):

        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])) 

        z_mean_t = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        z_log_sigma_sq_t = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])

        return (z_mean_t, z_log_sigma_sq_t)


    def _generator_network(self, weights, biases):

        z = tf.reshape(self.z, [self.n_particles*self.batch_size, self.n_z])

        layer_1 = self.transfer_fct(tf.add(tf.matmul(z, weights['h1']), 
                                           biases['b1'])) #shape is now [p*b,l1]

        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 

        x_reconstr_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                     biases['out_mean'])

        x_reconstr_mean = tf.reshape(x_reconstr_mean, [self.n_particles, self.batch_size, self.n_input])

        return x_reconstr_mean


    def _log_p_z_(self, z):
    	'''
    	Log of normal distribution with zero mean and one var

    	z is [n_particles, batch_size, n_z]
    	output is [n_particles, batch_size]
    	'''

        # term1 = 0
        term2 = self.n_z * tf.log(2*math.pi)
        term3 = tf.reduce_sum(tf.square(z), 2) #sum over dimensions n_z so now its [particles, batch]

        all_ = term2 + term3
        log_p_z = -.5 * all_

        return log_p_z


    def _log_p_z_given_x_(self, z, mean, log_var):
    	'''
    	Log of normal distribution

    	z is [n_particles, batch_size, n_z]
    	mean is [batch_size, n_z]
    	log_var is [batch_size, n_z]
    	output is [n_particles, batch_size]
    	'''

        # term1 = tf.log(tf.reduce_prod(tf.exp(log_var_sq), reduction_indices=1))
        term1 = tf.reduce_sum(log_var, reduction_indices=1) #sum over dimensions n_z so now its [batch]

        term2 = self.n_z * tf.log(2*math.pi)
        dif = tf.square(z - mean)
        dif_cov = dif / tf.exp(log_var)
        # term3 = tf.reduce_sum(dif_cov * dif, 1) 
        term3 = tf.reduce_sum(dif_cov, 2) #sum over dimensions n_z so now its [particles, batch]

        all_ = term1 + term2 + term3
        log_p_z_given_x = -.5 * all_

        return log_p_z_given_x


    def _log_likelihood(self, t, pred_no_sig):
		'''
		Log of bernoulli distribution

		t is [batch_size, n_input]
		pred_no_sig is [n_particles, batch_size, n_input] 
		output is [n_particles, batch_size]
		'''

        reconstr_loss = \
                tf.reduce_sum(tf.maximum(pred_no_sig, 0) 
                            - pred_no_sig * t
                            + tf.log(1 + tf.exp(-tf.abs(pred_no_sig))),
                             2) #sum over dimensions

        #negative because the above calculated the NLL, so this is returning the LL
        return -reconstr_loss



    def train2(self, train_x, valid_x=[], display_step=5, path_to_load_variables='', path_to_save_variables='', starting_stage=0):
        '''
        This training method is the IWAE one where they do many passes over the data with decreasing LR
        One difference is that I look at the validation NLL after each stage and save the variables
        '''

        data_to_save = []

        n_datapoints = len(train_x)
        
        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            self.sess.run(tf.initialize_all_variables())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables

        total_stages= 6
        #start = time.time()
        for stage in range(starting_stage,total_stages+1):

            self.learning_rate = .001 * 10.**(-stage/float(total_stages))
            print 'learning rate', self.learning_rate

            passes_over_data = 3**stage

            for pass_ in range(passes_over_data):

                #shuffle the data
                arr = np.arange(len(train_x))
                np.random.shuffle(arr)

                data_index = 0
                for step in range(n_datapoints/self.batch_size):

                    #Make batch
                    batch = []
                    while len(batch) != self.batch_size:
                        datapoint = train_x[data_index]
                        batch.append(datapoint)
                        data_index +=1

                    # Fit training using batch data
                    # nothing = self.partial_fit(batch)
                    _ = self.sess.run((self.optimizer), feed_dict={self.x: batch})
                    
                    # Display logs per epoch step
                    if step % display_step == 0:

                        cost = self.sess.run((self.elbo), feed_dict={self.x: batch})
                        cost = -cost #because I want to see the NLL

                        print "Stage:" + str(stage)+'/7', "Pass", str(pass_)+'/'+str(passes_over_data-1), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/self.batch_size), "cost=", "{:.6f}".format(cost)#, 'time', time.time() - start


                print 'calculating elbos'
                #Select 1000 training and validation datapoints
                arr = np.arange(1000)
                np.random.shuffle(arr)

                train_x_check = train_x[arr]
                valid_x_check = valid_x[arr]

                #Calculate the elbos for all the data and average them
                data_index_11 = 0
                elbos = []
                elbos_iwae = []
                data_index11 = 0
                for step in range(1000/self.batch_size):

                    #Make batch
                    batch = []
                    while len(batch) != self.batch_size:
                        datapoint = train_x_check[data_index11]
                        batch.append(datapoint)
                        data_index11 +=1
                    train_elbo, train_iwae_elbo = self.sess.run((self.elbo, self.iwae_elbo), feed_dict={self.x: batch})
                    elbos.append(train_elbo)
                    elbos_iwae.append(train_iwae_elbo)

                train_elbo_ = np.mean(elbos)
                train_elbo_iwae_ = np.mean(elbos_iwae)

                data_index11 = 0
                for step in range(1000/self.batch_size):

                    #Make batch
                    batch = []
                    while len(batch) != self.batch_size:
                        datapoint = valid_x_check[data_index11]
                        batch.append(datapoint)
                        data_index11 +=1
                    valid_elbo, valid_iwae_elbo = self.sess.run((self.elbo, self.iwae_elbo), feed_dict={self.x: batch})
                    elbos.append(valid_elbo)
                    elbos_iwae.append(valid_iwae_elbo)
                    
                valid_elbo_ = np.mean(elbos)
                valid_elbo_iwae_ = np.mean(elbos_iwae)

                data_to_save.append([train_elbo_, train_elbo_iwae_, valid_elbo_, valid_elbo_iwae_])

            #Save data
            with open(home+ '/data/training_data/elbos_stage'+ str(stage)+'.pkl', 'wb') as f:
                pickle.dump(data_to_save, f)

            #TODO: save what stage the variables are
            path_to_save_variables = path_to_save_variables.split('.')[0]
            path_to_save_variables = path_to_save_variables + '_stage' + str(stage) + '.ckpt'

            if path_to_save_variables != '':
                print 'saving variables to ' + path_to_save_variables
                saver.save(self.sess, path_to_save_variables)
                print 'Saved variables to ' + path_to_save_variables

























class IWAE(VAE):

	def __init__():








