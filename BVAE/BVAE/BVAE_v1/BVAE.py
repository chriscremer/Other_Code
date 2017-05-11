



#Generative Autoencoder classes
import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle




class BVAE(object):

    def __init__(self, network_architecture, learning_rate=0.0001, batch_size=5, n_particles=3):
        
        tf.reset_default_graph()

        self.network_architecture = network_architecture
        self.transfer_fct = tf.nn.softplus #tf.tanh
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_particles = n_particles
        self.n_z = network_architecture["n_z"]
        self.n_input = network_architecture["n_input"]

        #Placeholders - Inputs
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        
        #Variables
        self.network_weights, self.n_decoder_weights = self._initialize_weights(network_architecture)

        #Sample weights
        self.sampled_theta, log_p_theta_, log_q_theta_ = self.sample_weights(self.network_weights['decoder_weights'])

        #Encoder - Recognition model - q(z|x): recog_mean,z_log_std_sq=[batch_size, n_z]
        self.recog_means, self.recog_log_vars = self._recognition_network(self.x, self.network_weights['encoder_weights']) #, self.network_weights['encoder_biases'])
        
        #Sample z
        eps = tf.random_normal((self.n_particles, self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.recog_means, tf.multiply(tf.sqrt(tf.exp(self.recog_log_vars)), eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]
        
        #Decoder - Generative model - p(x|z)
        self.x_reconstr_mean_no_sigmoid = self._generator_network(self.z, self.sampled_theta)#, self.network_weights['decoder_biases']) #no sigmoid

        #Objective
        self.elbo = self.elbo(self.x, self.x_reconstr_mean_no_sigmoid, self.z, self.recog_means, self.recog_log_vars, log_p_theta_, log_q_theta_)

        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(-self.elbo)

        #For evaluation
        self.log_w = self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid) + self._log_p_z(self.z) - self._log_q_z_given_x(self.z, self.recog_means, self.recog_log_vars)
        self.x_reconstr_mean = tf.nn.sigmoid(self.x_reconstr_mean_no_sigmoid)


    def _initialize_weights(self, network_architecture):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


        all_weights = dict()

        #Recognition net
        all_weights['encoder_weights'] = {}
        # all_weights['encoder_biases'] = {}

        for layer_i in range(len(network_architecture['encoder_net'])):
            if layer_i == 0:
                all_weights['encoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.n_input+1, network_architecture['encoder_net'][layer_i]))
                # all_weights['encoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['encoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['encoder_net'][layer_i-1]+1, network_architecture['encoder_net'][layer_i]))
                # all_weights['encoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

        all_weights['encoder_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1]+1, self.n_z))
        all_weights['encoder_weights']['out_log_var'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1]+1, self.n_z))
        # all_weights['encoder_biases']['out_mean'] = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32))
        # all_weights['encoder_biases']['out_log_var'] = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32))


        #Generator net
        all_weights['decoder_weights'] = {}
        # all_weights['decoder_biases'] = {}

        n_decoder_weights = 0

        for layer_i in range(len(network_architecture['decoder_net'])):
            if layer_i == 0:
                all_weights['decoder_weights']['l'+str(layer_i)+'mean'] = tf.Variable(xavier_init(self.n_z+1, network_architecture['decoder_net'][layer_i]))
                all_weights['decoder_weights']['l'+str(layer_i)+'logvar'] = tf.Variable(xavier_init(self.n_z+1, network_architecture['decoder_net'][layer_i]))
                n_decoder_weights += (self.n_z+1) * network_architecture['decoder_net'][layer_i]
                # all_weights['decoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['decoder_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['decoder_weights']['l'+str(layer_i)+'mean'] = tf.Variable(xavier_init(network_architecture['decoder_net'][layer_i-1]+1, network_architecture['decoder_net'][layer_i]))
                all_weights['decoder_weights']['l'+str(layer_i)+'logvar'] = tf.Variable(xavier_init(network_architecture['decoder_net'][layer_i-1]+1, network_architecture['decoder_net'][layer_i]))
                n_decoder_weights += network_architecture['decoder_net'][layer_i-1]+1 * network_architecture['decoder_net'][layer_i]
                # all_weights['decoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

        all_weights['decoder_weights']['out_mean_mean'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1]+1, self.n_input))
        all_weights['decoder_weights']['out_mean_logvar'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1]+1, self.n_input))
        n_decoder_weights += network_architecture['decoder_net'][-1]+1 * self.n_input

        # all_weights['decoder_weights']['out_log_var'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.n_input))
        # all_weights['decoder_biases']['out_mean'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        # all_weights['decoder_biases']['out_log_var'] = tf.Variable(tf.zeros([sefl.n_input], dtype=tf.float32))

        return all_weights, n_decoder_weights


    def _recognition_network(self, x, weights):
        '''
        x: [B,D]
        '''

        n_layers = len(weights) - 2 #minus 2 for the mean and var outputs
        for layer_i in range(n_layers):

            #concat 1 to input for biases
            x = tf.concat([x,tf.ones([self.batch_size,1])], axis=1)

            # x = self.transfer_fct(tf.add(tf.matmul(x, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 
            x = self.transfer_fct(tf.matmul(x, weights['l'+str(layer_i)])) #, biases['l'+str(layer_i)])) 

        x = tf.concat([x,tf.ones([self.batch_size,1])], axis=1)

        z_mean = tf.matmul(x, weights['out_mean'])#, biases['out_mean'])
        z_log_var = tf.matmul(x, weights['out_log_var'])#, biases['out_log_var'])

        return z_mean, z_log_var

    def sample_weights(self, weights):

        log_p = 0
        log_q = 0

        sampled_weights = []
        for layer_i in range(len(self.network_architecture['decoder_net'])):

            if layer_i == 0:
                eps = tf.random_normal((self.n_z+1, self.network_architecture['decoder_net'][layer_i]), 0, 1, dtype=tf.float32)
                weights_ = tf.add(weights['l'+str(layer_i)+'mean'], tf.multiply(tf.sqrt(tf.exp(weights['l'+str(layer_i)+'logvar'])), eps))
                n_decoder_weights = (self.n_z+1) * self.network_architecture['decoder_net'][layer_i]
                log_p += self.log_p_theta(weights_, n_decoder_weights)
                log_q += self.log_q_theta(weights_, weights['l'+str(layer_i)+'mean'], weights['l'+str(layer_i)+'logvar'], n_decoder_weights)
            else:
                eps = tf.random_normal((self.network_architecture['decoder_net'][layer_i-1]+1, self.network_architecture['decoder_net'][layer_i]), 0, 1, dtype=tf.float32)
                weights_ = tf.add(weights['l'+str(layer_i)+'mean'], tf.multiply(tf.sqrt(tf.exp(weights['l'+str(layer_i)+'logvar'])), eps))
                n_decoder_weights = self.network_architecture['decoder_net'][layer_i-1]+1 * self.network_architecture['decoder_net'][layer_i]
                log_p += self.log_p_theta(weights_, n_decoder_weights)
                log_q += self.log_q_theta(weights_, weights['l'+str(layer_i)+'mean'], weights['l'+str(layer_i)+'logvar'], n_decoder_weights)

            sampled_weights.append(weights_)

        eps = tf.random_normal((self.network_architecture['decoder_net'][-1]+1, self.n_input), 0, 1, dtype=tf.float32)
        weights_ = tf.add(weights['out_mean_mean'], tf.multiply(tf.sqrt(tf.exp(weights['out_mean_logvar'])), eps))
        sampled_weights.append(weights_)
        n_decoder_weights = self.network_architecture['decoder_net'][-1]+1 * self.n_input
        log_p += self.log_p_theta(weights_, n_decoder_weights)
        log_q += self.log_q_theta(weights_, weights['out_mean_mean'], weights['out_mean_logvar'], n_decoder_weights)

        # print log_p
        # print log_q
        # fasdf

        return sampled_weights, log_p, log_q


    def _generator_network(self, z, weights):

        z = tf.reshape(z, [self.n_particles*self.batch_size, self.n_z])

        # n_layers = len(weights) - 1 #minus 1 for the mean output
        # for layer_i in range(n_layers):

        for layer_i in range(len(self.network_architecture['decoder_net'])):

            # print z
            # print weights['l'+str(layer_i)]

            #concat 1 to input for biases
            z = tf.concat([z,tf.ones([self.n_particles*self.batch_size,1])], axis=1)

            if layer_i == 0:

                #sample weights
                # eps = tf.random_normal((self.n_z+1, self.network_architecture['decoder_net'][layer_i]), 0, 1, dtype=tf.float32)
                # weights_ = tf.add(weights['l'+str(layer_i)+'mean'], tf.multiply(tf.sqrt(tf.exp(weights['l'+str(layer_i)+'logvar'])), eps))
                z = self.transfer_fct(tf.matmul(z, weights[layer_i]))#, biases['l'+str(layer_i)]))

            else:

                #sample weights
                # eps = tf.random_normal((self.network_architecture['decoder_net'][layer_i-1]+1, self.network_architecture['decoder_net'][layer_i]), 0, 1, dtype=tf.float32)
                # weights_ = tf.add(weights['l'+str(layer_i)+'mean'], tf.multiply(tf.sqrt(tf.exp(weights['l'+str(layer_i)+'logvar'])), eps))
                z = self.transfer_fct(tf.matmul(z, weights[layer_i]))#, biases['l'+str(layer_i)]))


        z = tf.concat([z,tf.ones([self.n_particles*self.batch_size,1])], axis=1)

        #notice no sigmoid
        # x_mean = tf.matmul(z, weights['out_mean'])#, biases['out_mean'])
        x_mean = tf.matmul(z, weights[-1])#, biases['out_mean'])


        x_reconstr_mean = tf.reshape(x_mean, [self.n_particles, self.batch_size, self.n_input])


        return x_reconstr_mean


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


        reconstr_loss = \
                tf.reduce_sum(tf.maximum(pred_no_sig, 0) 
                            - pred_no_sig * t
                            + tf.log(1 + tf.exp(-tf.abs(pred_no_sig))),
                             2) #sum over dimensions

        #negative because the above calculated the NLL, so this is returning the LL
        return -reconstr_loss


    def log_q_theta(self, theta, mean, log_var, size):
        '''
        theta: [input, ouput]
        '''

        term1 = tf.reduce_sum(log_var)#, reduction_indices=1) #sum over dimensions,[batch]

        term2 = size * tf.log(2*math.pi)
        dif = tf.square(theta - mean)
        dif_cov = dif / tf.exp(log_var)
        # term3 = tf.reduce_sum(dif_cov * dif, 1) 
        term3 = tf.reduce_sum(dif_cov)#, 2) #sum over dimensions n_z so now its [particles, batch]

        all_ = term1 + term2 + term3
        log_p_z_given_x = -.5 * all_

        return log_p_z_given_x




    def log_p_theta(self, theta, size):
        '''
        theta: [input, ouput]
        '''
        # term1 = 0
        term2 = size * tf.log(2*math.pi)
        term3 = tf.reduce_sum(tf.square(theta))#, 2) #sum over dimensions n_z so now its [particles, batch]
        all_ = term2 + term3
        log_p_z = -.5 * all_

        return log_p_z



    def elbo(self, x, x_recon, z, mean, log_var, log_p_theta, log_q_theta):

        elbo = self._log_likelihood(x, x_recon) + self._log_p_z(z) - self._log_q_z_given_x(z, mean, log_var)#+ self.log_p_theta(theta) - self.log_q_theta(theta, theta_mean, theta_logvar))

        elbo = tf.reduce_mean(elbo, 1) #average over batch
        elbo = tf.reduce_mean(elbo) #average over particles

        elbo = elbo + (log_p_theta -  log_q_theta)*(1./50000.)

        return elbo



    def train(self, train_x, valid_x=[], display_step=5, path_to_load_variables='', path_to_save_variables='', epochs=10):
        '''
        Train.
        Use early stopping, actually no, because I want it to be equal for each model. Time? Epochs? 
        I'll do stages for now.
        '''

        n_datapoints = len(train_x)
        
        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
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

        # #Ramdomly select a batch
        # batch = []
        # while len(batch) != self.batch_size:
        #     datapoint = data[np.random.randint(0,len(data))]
        #     batch.append(datapoint)
        batch = data


        if sampling == 'vae':

            #Encode and get p and q
            log_ws, recons = self.sess.run((self.log_w, self.x_reconstr_mean), feed_dict={self.x: batch})

            # print log_ws.shape
            # print recons.shape

            return recons, batch

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







# class IWAE(VAE):

#     def elbo(self, x, x_recon, z, mean, log_var):

#         # [P, B]
#         temp_elbo = self._log_likelihood(x, x_recon) + self._log_p_z(z) - self._log_q_z_given_x(z, mean, log_var)

#         max_ = tf.reduce_max(temp_elbo, reduction_indices=0) #over particles? so its [B]

#         elbo = tf.log(tf.reduce_mean(tf.exp(temp_elbo-max_), 0)) + max_  #mean over particles so its [B]

#         elbo = tf.reduce_mean(elbo) #over batch

#         return elbo





    # def train(self, train_x, valid_x=[], display_step=5, path_to_load_variables='', path_to_save_variables='', starting_stage=0, ending_stage=5, path_to_save_training_info=''):
    #     '''
    #     Train.
    #     Use early stopping, actually no, because I want it to be equal for each model. Time? Epochs? 
    #     I'll do stages for now.
    #     '''

    #     n_datapoints = len(train_x)
        
    #     saver = tf.train.Saver()
    #     self.sess = tf.Session()

    #     if path_to_load_variables == '':
    #         self.sess.run(tf.initialize_all_variables())
    #     else:
    #         #Load variables
    #         saver.restore(self.sess, path_to_load_variables)
    #         print 'loaded variables ' + path_to_load_variables

    #     #start = time.time()
    #     for stage in range(starting_stage,ending_stage+1):

    #         # self.learning_rate = .001 * 10.**(-stage/float(ending_stage))
    #         # print 'learning rate', self.learning_rate
    #         print 'stage', stage

    #         passes_over_data = 3**stage

    #         for pass_ in range(passes_over_data):

    #             #shuffle the data
    #             arr = np.arange(len(train_x))
    #             np.random.shuffle(arr)
    #             train_x = train_x[arr]

    #             data_index = 0
    #             for step in range(n_datapoints/self.batch_size):

    #                 #Make batch
    #                 batch = []
    #                 while len(batch) != self.batch_size:
    #                     datapoint = train_x[data_index]
    #                     batch.append(datapoint)
    #                     data_index +=1

    #                 # Fit training using batch data
    #                 _ = self.sess.run((self.optimizer), feed_dict={self.x: batch})

    #                 # print self.sess.run((self.asdf), feed_dict={self.x: batch})
    #                 # fasdfa
                    
    #                 # Display logs per epoch step
    #                 if step % display_step == 0:

    #                     cost = self.sess.run((self.elbo), feed_dict={self.x: batch})
    #                     cost = -cost #because I want to see the NLL

    #                     print "Stage:" + str(stage)+'/' + str(ending_stage), "Pass", str(pass_)+'/'+str(passes_over_data-1), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/self.batch_size), "cost=", "{:.6f}".format(float(cost))#, 'time', time.time() - start

    #                     # lw= self.sess.run((self.log_weights), feed_dict={self.x: batch})
    #                     # print np.exp(lw)

    #     if path_to_save_variables != '':
    #         # print 'saving variables to ' + path_to_save_variables
    #         saver.save(self.sess, path_to_save_variables)
    #         print 'Saved variables to ' + path_to_save_variables




if __name__ == '__main__':

    n_a0 = \
        dict(n_input=784, # 784 image
             encoder_net=[200,200], 
             n_z=5,  # dimensionality of latent space
             decoder_net=[200,200]) 

    model = BVAE(n_a0, batch_size=5, n_particles=3)
    fsaf




