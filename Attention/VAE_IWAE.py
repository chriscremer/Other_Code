
#Generative Autoencoder classes
import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle




class VAE(object):

    def __init__(self, batch_size):
        
        # tf.reset_default_graph()

        self.network_architecture = dict(n_input=784, # 784 image
                                         encoder_net=[200,200], 
                                         n_z=20,  # dimensionality of latent space
                                         decoder_net=[200,200]) 

        self.transfer_fct = tf.nn.softplus #tf.tanh
        self.learning_rate = 0.0001
        self.batch_size = batch_size
        self.n_particles = 1
        self.n_z = self.network_architecture["n_z"]
        self.z_size = self.n_z
        self.n_input = self.network_architecture["n_input"]
        self.input_size = self.n_input
        self.reg_param = .000001

        with tf.name_scope('model_input'):
            #Placeholders - Inputs
            self.x = tf.placeholder(tf.float32, [None, self.n_input])
        
        #Variables
        self.params_dict, self.params_list = self._initialize_weights(self.network_architecture)
        
        #Encoder - Recognition model - q(z|x): recog_mean,z_log_std_sq=[batch_size, n_z]
        self.recog_means, self.recog_log_vars = self._recognition_network(self.x)
        
        #Sample
        eps = tf.random_normal((self.n_particles, self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.recog_means, tf.mul(tf.sqrt(tf.exp(self.recog_log_vars)), eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]
        
        #Decoder - Generative model - p(x|z)
        self.x_reconstr_mean_no_sigmoid = self._generator_network(self.z) #no sigmoid

        #Objective
        self.elbo = self.elbo(self.x, self.x_reconstr_mean_no_sigmoid, self.z, self.recog_means, self.recog_log_vars)
        self.cost = -self.elbo + (self.reg_param * self.l2_regularization())

        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost)

        #For evaluation
        self.log_w = self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid) + self._log_p_z(self.z) - self._log_q_z_given_x(self.z, self.recog_means, self.recog_log_vars)
        self.x_reconstr_mean = tf.nn.sigmoid(self.x_reconstr_mean_no_sigmoid)
        self.generate_samples = tf.nn.sigmoid(self._generator_network(eps))


    def _initialize_weights(self, network_architecture):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


        params_dict = dict()

        #Recognition/Inference net q(z|z-1,u,x)
        # all_weights['encoder_weights'] = {}
        # all_weights['encoder_biases'] = {}


        with tf.name_scope('encoder_vars'):

            # params_dict['conv_weights'] = tf.Variable(tf.truncated_normal([self.filter_height, self.filter_width, self.n_channels, self.filter_out_channels1], stddev=0.1))
            # params_dict['conv_biases'] = tf.Variable(tf.truncated_normal([self.filter_out_channels1], stddev=0.1))

            for layer_i in range(len(network_architecture['encoder_net'])):
                if layer_i == 0:
                    params_dict['encoder_weights_l'+str(layer_i)] = tf.Variable(xavier_init(self.input_size, network_architecture['encoder_net'][layer_i]))
                    params_dict['encoder_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))
                else:
                    params_dict['encoder_weights_l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['encoder_net'][layer_i-1], network_architecture['encoder_net'][layer_i]))
                    params_dict['encoder_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

            params_dict['encoder_weights_out_mean'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.z_size))
            params_dict['encoder_weights_out_log_var'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.z_size))
            params_dict['encoder_biases_out_mean'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))
            params_dict['encoder_biases_out_log_var'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))


        #Generator net p(x|z)
        # all_weights['decoder_weights'] = {}
        # all_weights['decoder_biases'] = {}

        with tf.name_scope('decoder_vars'):

            for layer_i in range(len(network_architecture['decoder_net'])):
                if layer_i == 0:
                    params_dict['decoder_weights_l'+str(layer_i)] = tf.Variable(xavier_init(self.z_size, network_architecture['decoder_net'][layer_i]))
                    params_dict['decoder_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['decoder_net'][layer_i]], dtype=tf.float32))
                else:
                    params_dict['decoder_weights_l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['decoder_net'][layer_i-1], network_architecture['decoder_net'][layer_i]))
                    params_dict['decoder_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['decoder_net'][layer_i]], dtype=tf.float32))

            params_dict['decoder_weights_out_mean'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.input_size))
            params_dict['decoder_biases_out_mean'] = tf.Variable(tf.zeros([self.input_size], dtype=tf.float32))

            # params_dict['decoder_weights_reward_mean'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.reward_size))
            # params_dict['decoder_biases_reward_mean'] = tf.Variable(tf.zeros([self.reward_size], dtype=tf.float32))
            # params_dict['decoder_weights_reward_log_var'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.reward_size))
            # params_dict['decoder_biases_reward_log_var'] = tf.Variable(tf.zeros([self.reward_size], dtype=tf.float32))
         
            # # #dont need this if output is bernoulli mean
            # params_dict['decoder_weights_out_log_var'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.input_size))
            # params_dict['decoder_biases_out_log_var'] = tf.Variable(tf.zeros([self.input_size], dtype=tf.float32))
            





        # all_weights = dict()

        # #Recognition net
        # all_weights['encoder_weights'] = {}
        # all_weights['encoder_biases'] = {}

        # for layer_i in range(len(network_architecture['encoder_net'])):
        #     if layer_i == 0:
        #         all_weights['encoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.n_input, network_architecture['encoder_net'][layer_i]))
        #         all_weights['encoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))
        #     else:
        #         all_weights['encoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['encoder_net'][layer_i-1], network_architecture['encoder_net'][layer_i]))
        #         all_weights['encoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

        # all_weights['encoder_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.n_z))
        # all_weights['encoder_weights']['out_log_var'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.n_z))
        # all_weights['encoder_biases']['out_mean'] = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32))
        # all_weights['encoder_biases']['out_log_var'] = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32))


        # #Generator net
        # all_weights['decoder_weights'] = {}
        # all_weights['decoder_biases'] = {}

        # for layer_i in range(len(network_architecture['decoder_net'])):
        #     if layer_i == 0:
        #         all_weights['decoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.n_z, network_architecture['decoder_net'][layer_i]))
        #         all_weights['decoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['decoder_net'][layer_i]], dtype=tf.float32))
        #     else:
        #         all_weights['decoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['decoder_net'][layer_i-1], network_architecture['decoder_net'][layer_i]))
        #         all_weights['decoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

        # all_weights['decoder_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.n_input))
        # # all_weights['decoder_weights']['out_log_var'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.n_input))
        # all_weights['decoder_biases']['out_mean'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        # # all_weights['decoder_biases']['out_log_var'] = tf.Variable(tf.zeros([sefl.n_input], dtype=tf.float32))

        # return all_weights

        params_list = []
        for layer in params_dict:

            params_list.append(params_dict[layer])

        return params_dict, params_list



    def l2_regularization(self):


        with tf.name_scope('L2_reg'):

            sum_ = 0
            for layer in self.params_dict:

                sum_ += tf.reduce_sum(tf.square(self.params_dict[layer]))

        return sum_



    def _recognition_network(self, x):


        # n_layers = len(weights) - 2 #minus 2 for the mean and var outputs
        # for layer_i in range(n_layers):

        #     x = self.transfer_fct(tf.add(tf.matmul(x, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 

        # z_mean = tf.add(tf.matmul(x, weights['out_mean']), biases['out_mean'])
        # z_log_var = tf.add(tf.matmul(x, weights['out_log_var']), biases['out_log_var'])

        # return z_mean, z_log_var


        with tf.name_scope('recognition_net'):

            input_ = x
            n_layers = len(self.network_architecture['encoder_net'])

            for layer_i in range(n_layers):

                input_ = self.transfer_fct(tf.contrib.layers.layer_norm(tf.add(tf.matmul(input_, self.params_dict['encoder_weights_l'+str(layer_i)]), self.params_dict['encoder_biases_l'+str(layer_i)])))
                
            z_mean = tf.add(tf.matmul(input_, self.params_dict['encoder_weights_out_mean']), self.params_dict['encoder_biases_out_mean'])
            z_log_var = tf.add(tf.matmul(input_, self.params_dict['encoder_weights_out_log_var']), self.params_dict['encoder_biases_out_log_var'])

        return z_mean, z_log_var




    def _generator_network(self, z):

        z = tf.reshape(z, [self.n_particles*self.batch_size, self.n_z])

        # n_layers = len(weights) - 1 #minus 1 for the mean output
        # for layer_i in range(n_layers):

        #     # print z
        #     # print weights['l'+str(layer_i)]

        #     z = self.transfer_fct(tf.add(tf.matmul(z, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 

        # #notive no sigmoid
        # x_mean = tf.add(tf.matmul(z, weights['out_mean']), biases['out_mean'])

        # x_reconstr_mean = tf.reshape(x_mean, [self.n_particles, self.batch_size, self.n_input])


        # return x_reconstr_mean

        with tf.name_scope('observation_net'):

            input_ = z

            n_layers = len(self.network_architecture['decoder_net'])

            for layer_i in range(n_layers):

                input_ = self.transfer_fct(tf.contrib.layers.layer_norm(tf.add(tf.matmul(input_, self.params_dict['decoder_weights_l'+str(layer_i)]), self.params_dict['decoder_biases_l'+str(layer_i)])))

            x_mean = tf.add(tf.matmul(input_, self.params_dict['decoder_weights_out_mean']), self.params_dict['decoder_biases_out_mean'])
            # x_log_var = tf.add(tf.matmul(input_, self.params_dict['decoder_weights_out_log_var']), self.params_dict['decoder_biases_out_log_var'])

            # reward_mean = tf.add(tf.matmul(input_, self.params_dict['decoder_weights_reward_mean']), self.params_dict['decoder_biases_reward_mean'])
            # reward_log_var = tf.add(tf.matmul(input_, self.params_dict['decoder_weights_reward_log_var']), self.params_dict['decoder_biases_reward_log_var'])


        return x_mean#, x_log_var





    def _log_p_z(self, z):
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


    def _log_q_z_given_x(self, z, mean, log_var):
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


        # reconstr_loss = \
        #         tf.reduce_sum(tf.maximum(pred_no_sig, 0) 
        #                     - pred_no_sig * t
        #                     + tf.log(1 + tf.exp(-tf.abs(pred_no_sig))),
        #                      2) #sum over dimensions


        # [P,B,X]
        reconstr_loss = tf.maximum(pred_no_sig, 0) - pred_no_sig * t + tf.log(1 + tf.exp(-tf.abs(pred_no_sig)))

        attention = np.concatenate( (np.ones([392]), np.zeros([392])), axis=0)
        attention = np.reshape(attention, [1,1,784])

        reconstr_loss = reconstr_loss * attention
        reconstr_loss = tf.reduce_sum(reconstr_loss, 2) #sum over dimensions

        #negative because the above calculated the NLL, so this is returning the LL
        return -reconstr_loss



    def elbo(self, x, x_recon, z, mean, log_var):

        elbo = self._log_likelihood(x, x_recon) + self._log_p_z(z) - self._log_q_z_given_x(z, mean, log_var)

        elbo = tf.reduce_mean(elbo, 1) #average over batch
        elbo = tf.reduce_mean(elbo) #average over particles

        return elbo



    def train(self, train_x, valid_x=[], display_step=20, path_to_load_variables='', path_to_save_variables='', epochs=10):
        '''
        Train.
        Use early stopping, actually no, because I want it to be equal for each model. Time? Epochs? 
        I'll do stages for now.
        '''

        n_datapoints = len(train_x)
        
        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            # self.sess.run(tf.initialize_all_variables())
            self.sess.run(tf.global_variables_initializer())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables

        #start = time.time()
        # for stage in range(starting_stage,ending_stage+1):

            # self.learning_rate = .001 * 10.**(-stage/float(ending_stage))
            # print 'learning rate', self.learning_rate
            # print 'stage', stage

            # passes_over_data = 3**stage

        for epoch in range(epochs):

            #shuffle the data
            arr = np.arange(len(train_x))
            np.random.shuffle(arr)
            train_x = train_x[arr]

            data_index = 0
            for step in range(n_datapoints/self.batch_size):

                #Make batch
                batch = []
                while len(batch) != self.batch_size:
                    datapoint = train_x[data_index]
                    batch.append(datapoint)
                    data_index +=1

                # Fit training using batch data
                _ = self.sess.run((self.optimizer), feed_dict={self.x: batch})

                # print self.sess.run((self.asdf), feed_dict={self.x: batch})
                # fasdfa
                
                # Display logs per epoch step
                if step % display_step == 0:

                    cost = self.sess.run((self.elbo), feed_dict={self.x: batch})
                    cost = -cost #because I want to see the NLL

                    print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/self.batch_size), "cost=", "{:.6f}".format(float(cost))#, 'time', time.time() - start

                    # lw= self.sess.run((self.log_weights), feed_dict={self.x: batch})
                    # print np.exp(lw)

        if path_to_save_variables != '':
            # print 'saving variables to ' + path_to_save_variables
            saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables








    def encode(self, data):

        return self.sess.run([self.recog_means, self.recog_log_vars], feed_dict={self.x:data})


    def decode(self, sample):

        return self.sess.run(self.x_reconstr_mean_no_sigmoid, feed_dict={self.z:sample})



    def load_parameters(self, path_to_load_variables):

        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            # self.sess.run(tf.initialize_all_variables())
            print 'No path tpo variables'
            error
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables



    def reconstruct(self, sampling, data):

        # if data==-1:
        #     #Ramdomly select a batch
        batch = []
        while len(batch) != self.batch_size:
            datapoint = data[np.random.randint(0,len(data))]
            batch.append(datapoint)
        # else:
        #     batch = data


        if sampling == 'vae':

            #Encode and get p and q
            log_ws, recons = self.sess.run((self.log_w, self.x_reconstr_mean), feed_dict={self.x: batch})

            # print log_ws.shape
            # print recons.shape

            return recons, np.array(batch)

        if sampling == 'iwae':

            recons_resampled = []
            for i in range(self.n_particles):

                #Encode and get p and q.. log_ws [K,B,1], reons [K,B,X]
                log_ws, recons = self.sess.run((self.log_w, self.x_reconstr_mean), feed_dict={self.x: batch})

                #log normalize
                max_ = np.max(log_ws, axis=0)
                lse = np.log(np.sum(np.exp(log_ws-max_), axis=0)) + max_
                log_norm_ws = log_ws - lse

                # ws = np.exp(log_ws)
                # sums = np.sum(ws, axis=0)
                # norm_ws = ws / sums


                # print log_ws
                # print
                # print lse
                # print
                # print log_norm_ws
                # print 
                # print np.exp(log_norm_ws)
                # fsdfa

                #sample one based on cat(w)

                samps = []
                for j in range(self.batch_size):

                    samp = np.argmax(np.random.multinomial(1, np.exp(log_norm_ws.T[j])-.000001))
                    samps.append(recons[samp][j])
                    # print samp

                # print samps
                # print samps.shape
                # fasdf
                recons_resampled.append(np.array(samps))

            recons_resampled = np.array(recons_resampled)
            # print recons_resampled.shape


            return recons_resampled, batch


    def generate(self):

        samps = self.sess.run(self.generate_samples)

        # print log_ws.shape
        # print recons.shape

        return samps






class IWAE(VAE):

    def elbo(self, x, x_recon, z, mean, log_var):

        # [P, B]
        temp_elbo = self._log_likelihood(x, x_recon) + self._log_p_z(z) - self._log_q_z_given_x(z, mean, log_var)

        max_ = tf.reduce_max(temp_elbo, reduction_indices=0) #over particles? so its [B]

        elbo = tf.log(tf.reduce_mean(tf.exp(temp_elbo-max_), 0)) + max_  #mean over particles so its [B]

        elbo = tf.reduce_mean(elbo) #over batch

        return elbo



























