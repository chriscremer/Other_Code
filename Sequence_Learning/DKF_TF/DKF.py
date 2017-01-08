

import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import imageio

from ball_sequence import make_ball_gif


class DKF():

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=5, n_time_steps=2):
        
        tf.reset_default_graph()

        self.transfer_fct = tf.tanh
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_time_steps = n_time_steps
        self.z_size = network_architecture["z_size"]
        self.input_size = network_architecture["input_size"]
        self.action_size = network_architecture["action_size"]


        # Graph Input
        #[batch, n_frames, vector_length]
        self.x = tf.placeholder(tf.float32, [None, None, self.input_size])
        self.actions = tf.placeholder(tf.float32, [None, None, self.action_size])
        
        #Variables
        self.network_weights = self._initialize_weights(self.network_architecture)

        #Encoder / Recognition model / q(z|z-1,x-1,u,x): z_mean,z_log_var: [B,T,Z]
        # Have to sample to get the distribution over the sequence
        # self.z_mean, self.z_log_var = self._recognition_network(self.x, self.actions, self.network_weights)
        #Sample: [B,T,P,Z]
        # eps = tf.random_normal((self.batch_size, self.n_time_steps, self.n_particles, self.n_z), 0, 1, dtype=tf.float32)
        # self.z = tf.add(self.recog_means, tf.mul(tf.sqrt(tf.exp(self.recog_log_vars)), eps))
        # [B,T,P,Z]

        # [B,T,P,Z], [B,T,P]
        particle_sequence, particle_sequence_probs = self.Inference_Model(self.x, self.actions, self.network_weights)
        
        #Decoder / Generative model / p(x|z): [B,T,P,X]
        # self.x_reconstr_mean_no_sigmoid = self._generator_network(self.z, self.network_weights)
        
        self.Generative_Model(self.x, self.actions, self.network_weights)

        #Objective
        self.elbo = self.elbo(self.x, self.x_reconstr_mean_no_sigmoid, self.z, self.recog_means, self.recog_log_vars)

        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(-self.elbo)



    def _initialize_weights(self, network_architecture):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


        all_weights = dict()

        #Recognition/Inference net q(z|z-1,x-1,u,x)
        all_weights['encoder_weights'] = {}
        all_weights['encoder_biases'] = {}

        for layer_i in range(len(network_architecture['encoder_net'])):
            if layer_i == 0:
                all_weights['encoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.input_size+self.input_size+self.action_size+self.z_size, network_architecture['encoder_net'][layer_i]))
                all_weights['encoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['encoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['encoder_net'][layer_i-1], network_architecture['encoder_net'][layer_i]))
                all_weights['encoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

        all_weights['encoder_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.z_size))
        all_weights['encoder_weights']['out_log_var'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.z_size))
        all_weights['encoder_biases']['out_mean'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))
        all_weights['encoder_biases']['out_log_var'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))


        #Generator net p(x|x-1,z)
        all_weights['decoder_weights'] = {}
        all_weights['decoder_biases'] = {}

        for layer_i in range(len(network_architecture['decoder_net'])):
            if layer_i == 0:
                all_weights['decoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.z_size+self.input_size, network_architecture['decoder_net'][layer_i]))
                all_weights['decoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['decoder_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['decoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['decoder_net'][layer_i-1], network_architecture['decoder_net'][layer_i]))
                all_weights['decoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

        all_weights['decoder_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.n_input))
        all_weights['decoder_biases']['out_mean'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))


        #Generator/Transition net q(z|z-1,x-1,u)
        all_weights['trans_weights'] = {}
        all_weights['trans_biases'] = {}

        for layer_i in range(len(network_architecture['trans_net'])):
            if layer_i == 0:
                all_weights['trans_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.input_size+self.action_size+self.z_size, network_architecture['trans_net'][layer_i]))
                all_weights['trans_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['trans_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['trans_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['trans_net'][layer_i-1], network_architecture['trans_net'][layer_i]))
                all_weights['trans_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['trans_net'][layer_i]], dtype=tf.float32))

        all_weights['trans_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['trans_net'][-1], self.z_size))
        all_weights['trans_weights']['out_log_var'] = tf.Variable(xavier_init(network_architecture['trans_net'][-1], self.z_size))
        all_weights['trans_biases']['out_mean'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))
        all_weights['trans_biases']['out_log_var'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))


        return all_weights

    def log_normal(self, z, mean, log_var):
        '''
        Log of normal distribution

        z is [B, Z]
        mean is [B, Z]
        log_var is [B, Z]
        output is [B]
        '''

        # term1 = tf.log(tf.reduce_prod(tf.exp(log_var_sq), reduction_indices=1))
        term1 = tf.reduce_sum(log_var, reduction_indices=1) #sum over dimensions n_z so now its [B]

        # [1]
        term2 = self.z_size * tf.log(2*math.pi)

        dif = tf.square(z - mean)
        dif_cov = dif / tf.exp(log_var)
        term3 = tf.reduce_sum(dif_cov, 1) #sum over dimensions so its [B]

        all_ = term1 + term2 + term3
        log_norm = -.5 * all_

        return log_norm


    def recognition_net(self, input_, network_weights):
        # input:[B,2X+A+Z]

        n_layers = len(network_weights['encoder_weights']) - 2 #minus 2 for the mean and var outputs
        weights = network_weights['encoder_weights']

        for layer_i in range(n_layers):

            input_ = self.transfer_fct(tf.add(tf.matmul(input_, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 

        z_mean = tf.add(tf.matmul(input_, weights['out_mean']), biases['out_mean'])
        z_log_var = tf.add(tf.matmul(input_, weights['out_log_var']), biases['out_log_var'])

        return z_mean, z_log_var


            
    def Inference_Model(self, x, actions, network_weights):
        '''
        x: [B,T,X]
        actions: [B,T,A]

        q(z|z-1,x-1,u,x) for each t

        output particles: [B,T,P,Z]
        and their probs: [B,T,P]
        '''

        sequence_particles = []
        sequence_particles_probs = []
        prev_x = tf.zeros([self.batch_size, self.input_size])
        z = tf.zeros([self.batch_size, self.n_particles, self.z_size])

        for t in range(self.n_time_steps):

            #slice current x, current action: [B,1,X] and [B,1,A]
            current_x = tf.slice(x, [0, t, 0], [self.batch_size, 1, self.input_size])
            current_a = tf.slice(actions, [0, t, 0], [self.batch_size, 1, self.action_size])
            #reshape [B,X] and [B,A]
            current_x = tf.reshape(current_x, [self.batch_size, self.input_size])
            current_a = tf.reshape(current_a, [self.batch_size, self.action_size])

            particles = []
            particle_probs = []
            for k in range(self.n_particles):

                #slice out one z [B,1,Z]
                z_k = tf.slice(z, [0, k, 0], [self.batch_size, 1, self.z_size])
                #reshape [B,Z]
                z_k = tf.reshape(z_k, [self.batch_size, self.z_size])

                #concatenate current x, current action, prev x, z_k: [B,2X+A+Z]
                concatenate_all = tf.concat(1, [tf.concat(1, [tf.concat(1, [current_x, prev_x]), z_k]), current_a])

                #predict z distribution: [B,Z]
                z_mean, z_log_var = self.recognition_net(concatenate_all, network_weights)

                #sample from it
                eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
                this_particle = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_var)), eps))
                particle_prob = self.log_normal(this_particle, z_mean, z_log_var)

                #store it
                particles.append(this_particle)
                particle_probs.append(particle_prob)

            # [B,P,Z]
            z = tf.pack(particles, axis=1)
            # [B,P]
            z_probs = tf.pack(particle_probs, axis=1)

            #append to lists
            sequence_particles.append(z)
            sequence_particles_probs.append(z_probs)

            #update prev x and z
            prev_x = current_x
            prev_z = z

        #pack [B,T,P,Z]
        sequence_particles = tf.pack(sequence_particles, axis=1)
        #pack [B,T,P]
        sequence_particles_probs = tf.pack(sequence_particles_probs, axis=1)

        return sequence_particles, sequence_particles_probs
















    def _generator_network(self, z, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # Used for reconstruction

        layer_1 = self.transfer_fct(tf.add(tf.matmul(z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 

        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                 biases['out_mean']))

        return x_reconstr_mean



    def _generator_network_no_sigmoid(self, z, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.

        layer_1 = self.transfer_fct(tf.add(tf.matmul(z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 

        x_reconstr_mean = \
            tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                 biases['out_mean'])

        return x_reconstr_mean



    def _generative_transition_prior_network(self, prev_z, action, weights, biases):

        concatenate_state_and_action = tf.concat(1, [prev_z, action])

        layer_1 = self.transfer_fct(tf.add(tf.matmul(concatenate_state_and_action, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])

        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
            
        return (z_mean, z_log_sigma_sq)
            


    def _create_loss_optimizer(self):

        #NEW APPROACH, ONE LOOP
        reconstr_loss_list = []
        latent_loss_list = []
        prev_z = tf.zeros([self.batch_size, self.network_architecture['n_z']])
        self.cost = 0
        reconstr_loss_sum = 0
        kl_sum = 0
        for time_step in range(self.n_time_steps):

            current_frame = tf.slice(self.x, [0,time_step,0], [self.batch_size, 1, self.network_architecture["n_input"]])
            current_frame = tf.reshape(current_frame, [self.batch_size,self.network_architecture["n_input"]])

            current_action = tf.slice(self.actions, [0,time_step,0], [self.batch_size, 1, self.network_architecture["n_actions"]])
            current_action = tf.reshape(current_action, [self.batch_size,self.network_architecture["n_actions"]])

            #Get prior on latent distribution, Given last latent sample from the approximation
            #If t=0, then its just a standard Normal
            if time_step == 0:
                prior_mean = tf.zeros([self.batch_size, self.network_architecture['n_z']])
                prior_log_sigma_sq = tf.ones([self.batch_size, self.network_architecture['n_z']])
                # prior_log_sigma_sq = tf.diag(tf.ones([self.network_architecture['n_z']]))
                # prior_log_sigma_sq = tf.expand_dims(prior_log_sigma_sq, 0)
                # prior_log_sigma_sq = tf.tile(prior_log_sigma_sq, [self.batch_size,1,1])
            else:
                prior_mean, prior_log_sigma_sq = self._generative_transition_prior_network(prev_z, current_action, self.network_weights["weights_trans"], self.network_weights["biases_trans"])


            #Get latent distribution approximation, Given previous sampled state, and current frame
            recog_mean, recog_log_sigma_sq = self._recognition_network(prev_z, current_frame, current_action, self.network_weights["weights_recog"], self.network_weights["biases_recog"])
            
            #Probably need to store the recog values, since Ill need them to reconstruct

            #Calc KL divergence of the prior and recognition, see https://arxiv.org/pdf/1609.09869.pdf
            #determinant of diagonal matrix is the prodcut of the diagonal
            # I guess Im reduing all of them to scalars, or keeping batches? ya keep batches


            a = tf.log(tf.reduce_prod(tf.exp(prior_log_sigma_sq), 1)) - tf.log(tf.reduce_prod(tf.exp(recog_log_sigma_sq), 1)) #shape=[batch, 1]
            b = tf.matrix_inverse(tf.matrix_diag(tf.exp(prior_log_sigma_sq)))
            b = tf.matrix_diag_part(tf.batch_matmul(b, tf.matrix_diag(tf.exp(recog_log_sigma_sq))))
            b = tf.reduce_sum(b, 1) #this is the trace, should have size [batch, 1] now
            dif = prior_mean - recog_mean 
            dif_transpose = tf.expand_dims(dif, 1)
            dif = tf.expand_dims(dif, 2)
            # print dif_transpose
            c = tf.matrix_inverse(tf.matrix_diag(tf.exp(prior_log_sigma_sq)))
            # print c
            c = tf.batch_matmul(dif_transpose, c)
            c = tf.batch_matmul(c, dif)
            # kl = .5*(a-self.network_architecture['n_z']+b+c)
            # kl = a+b+c

            #over the batches
            a = tf.reduce_mean(a)
            b = tf.reduce_mean(b)
            c = tf.reduce_mean(c)

            if time_step == 1:
                self.a = a
                self.b = b
                self.c = c
                self.prior_mean = prior_mean
                self.recog_mean = recog_mean
                self.prior_log_sigma_sq = prior_log_sigma_sq
                self.recog_log_sigma_sq = recog_log_sigma_sq

            #Sample state, Draw one sample z from Gaussian distribution, z = mu + sigma*epsilon
            eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), 0, 1, dtype=tf.float32)
            z = tf.add(recog_mean, tf.mul(tf.sqrt(tf.exp(recog_log_sigma_sq)), eps))

            #Generate frame x_t
            reconstructed_mean = self._generator_network_no_sigmoid(z, self.network_weights["weights_gener"], self.network_weights["biases_gener"])

            #Calc reconstruction error
            #this sum is over the dimensions 
            reconstr_loss = \
                    tf.reduce_sum(tf.maximum(reconstructed_mean, 0) 
                                - reconstructed_mean * current_frame
                                + tf.log(1 + tf.exp(-abs(reconstructed_mean))),
                                 1)

            #Store those values, reduce is over batches
            # self.cost += tf.reduce_mean(reconstr_loss + kl)

            #mean over batch
            reconstr_loss_sum += tf.reduce_mean(reconstr_loss)
            # kl_sum += tf.reduce_mean(kl)
            kl_sum += a + b + c
            # kl_sum += b+ c



            prev_z = z


        self.reconstr_loss = reconstr_loss_sum
        self.kl_loss = kl_sum
        # self.cost = reconstr_loss_sum + (.01*kl_sum)
        self.cost = reconstr_loss_sum + kl_sum

 
        # self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X, actions):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost, rec, kl, nw = self.sess.run((self.optimizer, self.cost, self.reconstr_loss, self.kl_loss, self.network_weights), 
                                  feed_dict={self.x: X, self.actions: actions})
        return cost, rec, kl, nw
    
    # def transform(self, X):
    #     """Transform data by mapping it into the latent space."""
    #     # Note: This maps to mean of distribution, we could alternatively
    #     # sample from Gaussian distribution
    #     return self.sess.run(self.z_mean, feed_dict={self.x: X})
    

    def call_generate(self):

        batch=[]
        while len(batch) != self.batch_size:
            sequence, actions =get_ball()
            # batch.append(frame)
            batch.append(actions)

        batch = np.array(batch)
        # batch_actions = np.array(batch_actions)

        return self.sess.run(self.generate_1_, feed_dict={self.actions:batch})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        # if z_mu is None:
        #     # z_mu = np.random.normal(size=self.network_architecture["n_z"])
        #     z_mu = np.random.normal(size=(self.batch_size, self.n_time_steps, self.network_architecture["n_z"]))


        zs=[]
        #z_0 from prior
        prior_mean = tf.zeros([self.batch_size, self.network_architecture['n_z']])
        prior_log_sigma_sq = tf.log(tf.ones([self.batch_size, self.network_architecture['n_z']]))
        eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), 0, 1, dtype=tf.float32)
        z = tf.add(prior_mean, tf.mul(tf.sqrt(tf.exp(prior_log_sigma_sq)), eps))
        # z = prior_mean
        zs.append(z)
        prev_z = z
        #get z_t for next time steps usign the generative transition network
        for t in range(self.n_time_steps-1):
            #sample action
            # batch_actions = []
            # while len(batch_actions) != self.batch_size:
            #     a = np.random.choice(4, 1, p=[.25, .25, .25, .25])[0]
            #     action = np.zeros([self.network_architecture["n_actions"]])
            #     action[a] = 1
            #     batch_actions.append(action)
            current_action = tf.slice(self.actions, [0,t,0], [self.batch_size, 1, self.network_architecture["n_actions"]])
            current_action = tf.reshape(current_action, [self.batch_size,self.network_architecture["n_actions"]])

            prior_mean, prior_log_sigma_sq = self._generative_transition_prior_network(prev_z, current_action, self.network_weights["weights_trans"], self.network_weights["biases_trans"])
            eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), 0, 1, dtype=tf.float32)
            z = tf.add(prior_mean, tf.mul(tf.sqrt(tf.exp(prior_log_sigma_sq)), eps))
            # z = prior_mean
            zs.append(z)
            prev_z = z

        #use generative network to reconstruct those z.
        reconstructed_frames = []
        for i in range(len(zs)):
            reconstructed_mean = self._generator_network(zs[i], self.network_weights["weights_gener"], self.network_weights["biases_gener"])
            reconstructed_frames.append(reconstructed_mean)

        return reconstructed_frames
        


    # def call_generate2(self):
    #     return self.sess.run(self.generate_2_)


    # def generate2(self, z_mu=None):
    #     """ Generate data by sampling from latent space.
        
    #     If z_mu is not None, data for this point in latent space is
    #     generated. Otherwise, z_mu is drawn from prior in latent 
    #     space.        
    #     """
    #     zs=[]
    #     frames = [] 
    #     #z_0 from prior
    #     prior_mean = tf.zeros([self.batch_size, self.network_architecture['n_z']])
    #     prior_log_sigma_sq = tf.log(tf.ones([self.batch_size, self.network_architecture['n_z']]))
    #     eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), 0, 1, dtype=tf.float32)
    #     z = tf.add(prior_mean, tf.mul(tf.sqrt(tf.exp(prior_log_sigma_sq)), eps))

    #     # print z

    #     recon_mean = self._generator_network(z, self.network_weights["weights_gener"], self.network_weights["biases_gener"])
    #     # frames.append(recon_mean)
    #     # print recon_mean

    #     prev_z = tf.zeros([self.batch_size, self.network_architecture['n_z']])
    #     recog_mean, recog_log_sigma_sq = self._recognition_network(prev_z, recon_mean, self.network_weights["weights_recog"], self.network_weights["biases_recog"])
    #     recon_mean2 = self._generator_network(recog_mean, self.network_weights["weights_gener"], self.network_weights["biases_gener"])

    #     recog_mean2, recog_log_sigma_sq2 = self._recognition_network(prev_z, recon_mean2, self.network_weights["weights_recog"], self.network_weights["biases_recog"])
    #     recon_mean3 = self._generator_network(recog_mean2, self.network_weights["weights_gener"], self.network_weights["biases_gener"])

    #     recog_mean3, recog_log_sigma_sq3 = self._recognition_network(prev_z, recon_mean3, self.network_weights["weights_recog"], self.network_weights["biases_recog"])
    #     recon_mean4 = self._generator_network(recog_mean3, self.network_weights["weights_gener"], self.network_weights["biases_gener"])

    #     recog_mean4, recog_log_sigma_sq4 = self._recognition_network(prev_z, recon_mean4, self.network_weights["weights_recog"], self.network_weights["biases_recog"])
    #     recon_mean5 = self._generator_network(recog_mean3, self.network_weights["weights_gener"], self.network_weights["biases_gener"])


    #     return recon_mean, recon_mean2, recon_mean3, recon_mean4, recon_mean5
        

    # def generate3(self, z_mu=None):
    #     """ Generate data by sampling from latent space.
        
    #     If z_mu is not None, data for this point in latent space is
    #     generated. Otherwise, z_mu is drawn from prior in latent 
    #     space.        
    #     """

    #     prior_mean = tf.zeros([self.batch_size, self.network_architecture['n_z']])
    #     prior_log_sigma_sq = tf.log(tf.ones([self.batch_size, self.network_architecture['n_z']]))
    #     eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), 0, 1, dtype=tf.float32)
    #     z = tf.add(prior_mean, tf.mul(tf.sqrt(tf.exp(prior_log_sigma_sq)), eps))

    #     recon_mean = self._generator_network(z, self.network_weights["weights_gener"], self.network_weights["biases_gener"])

    #     prev_z = tf.zeros([self.batch_size, self.network_architecture['n_z']])
    #     recog_mean, recog_log_sigma_sq = self._recognition_network(prev_z, recon_mean, self.network_weights["weights_recog"], self.network_weights["biases_recog"])
    #     prior_mean, prior_log_sigma_sq = self._generative_transition_prior_network(recog_mean, self.network_weights["weights_trans"], self.network_weights["biases_trans"])

    #     recon_mean2 = self._generator_network(prior_mean, self.network_weights["weights_gener"], self.network_weights["biases_gener"])

    #     recog_mean2, recog_log_sigma_sq2 = self._recognition_network(recog_mean, recon_mean2, self.network_weights["weights_recog"], self.network_weights["biases_recog"])
    #     prior_mean, prior_log_sigma_sq = self._generative_transition_prior_network(recog_mean2, self.network_weights["weights_trans"], self.network_weights["biases_trans"])

    #     recon_mean3 = self._generator_network(prior_mean, self.network_weights["weights_gener"], self.network_weights["biases_gener"])

    #     recog_mean3, recog_log_sigma_sq3 = self._recognition_network(recog_mean2, recon_mean3, self.network_weights["weights_recog"], self.network_weights["biases_recog"])
    #     prior_mean, prior_log_sigma_sq = self._generative_transition_prior_network(recog_mean3, self.network_weights["weights_trans"], self.network_weights["biases_trans"])

    #     recon_mean4 = self._generator_network(prior_mean, self.network_weights["weights_gener"], self.network_weights["biases_gener"])

    #     recog_mean4, recog_log_sigma_sq4 = self._recognition_network(recog_mean3, recon_mean4, self.network_weights["weights_recog"], self.network_weights["biases_recog"])
    #     prior_mean, prior_log_sigma_sq = self._generative_transition_prior_network(recog_mean4, self.network_weights["weights_trans"], self.network_weights["biases_trans"])

    #     recon_mean5 = self._generator_network(prior_mean, self.network_weights["weights_gener"], self.network_weights["biases_gener"])


    #     return recon_mean, recon_mean2, recon_mean3, recon_mean4, recon_mean5


    # def call_predict_next(self, X):
    #     return self.sess.run(self.predict_next_, feed_dict={self.x: X})

    # def predict_next(self, X):

    #     prev_z = tf.zeros([self.batch_size, self.network_architecture['n_z']])
    #     X = tf.reshape(X, [self.batch_size,self.network_architecture["n_input"]])
    #     recog_mean, recog_log_sigma_sq = self._recognition_network(prev_z, X, self.network_weights["weights_recog"], self.network_weights["biases_recog"])
    #     prior_mean, prior_log_sigma_sq = self._generative_transition_prior_network(recog_mean, self.network_weights["weights_trans"], self.network_weights["biases_trans"])
    #     recon_mean = self._generator_network(prior_mean, self.network_weights["weights_gener"], self.network_weights["biases_gener"])

    #     return recon_mean


    def call_predict_next2(self, X):
        return self.sess.run(self.predict_next_2, feed_dict={self.x: X})

    def predict_next2(self, X):

        prev_z = tf.zeros([self.batch_size, self.network_architecture['n_z']])
        X = tf.reshape(X, [self.batch_size,self.network_architecture["n_input"]])
        recog_mean, recog_log_sigma_sq = self._recognition_network(prev_z, X, self.network_weights["weights_recog"], self.network_weights["biases_recog"])
        prior_mean, prior_log_sigma_sq = self._generative_transition_prior_network(recog_mean, self.network_weights["weights_trans"], self.network_weights["biases_trans"])
        recon_mean1 = self._generator_network(prior_mean, self.network_weights["weights_gener"], self.network_weights["biases_gener"])
        prior_mean2, prior_log_sigma_sq2 = self._generative_transition_prior_network(prior_mean, self.network_weights["weights_trans"], self.network_weights["biases_trans"])
        recon_mean2 = self._generator_network(prior_mean2, self.network_weights["weights_gener"], self.network_weights["biases_gener"])

        return recon_mean1, recon_mean2



    def call_reconstruct(self, X):
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.x: X})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """

        prev_z = tf.zeros([self.batch_size, self.network_architecture['n_z']])
        reconstruction = []
        for time_step in range(self.n_time_steps):

            current_frame = tf.slice(X, [0,time_step,0], [self.batch_size, 1, self.network_architecture["n_input"]])
            current_frame = tf.reshape(current_frame, [self.batch_size,self.network_architecture["n_input"]])

            recog_mean, recog_log_sigma_sq = self._recognition_network(prev_z, current_frame, self.network_weights["weights_recog"], self.network_weights["biases_recog"])

            recon_mean = self._generator_network(recog_mean, self.network_weights["weights_gener"], self.network_weights["biases_gener"])
            reconstruction.append(recon_mean)

            prev_z = recog_mean

        return reconstruction


    # def call_reconstruct2(self, X):
    #     return self.sess.run(self.x_reconstr_mean2, feed_dict={self.x: X})

    # def reconstruct2(self, X):
    #     """ Use VAE to reconstruct given data. """

    #     prev_z = tf.zeros([self.batch_size, self.network_architecture['n_z']])
    #     # reconstruction = []/
    #     # for time_step in range(self.n_time_steps):

    #         # current_frame = tf.slice(X, [0,time_step,0], [self.batch_size, 1, self.network_architecture["n_input"]])
    #         # current_frame = tf.reshape(current_frame, [self.batch_size,self.network_architecture["n_input"]])
    #     X = tf.reshape(X, [self.batch_size,self.network_architecture["n_input"]])

    #     recog_mean, recog_log_sigma_sq = self._recognition_network(prev_z, X, self.network_weights["weights_recog"], self.network_weights["biases_recog"])

    #     recon_mean = self._generator_network(recog_mean, self.network_weights["weights_gener"], self.network_weights["biases_gener"])
    #     # reconstruction.append(recon_mean)

    #     # prev_z = recog_mean
    #     recog_mean2, recog_log_sigma_sq2 = self._recognition_network(prev_z, recon_mean, self.network_weights["weights_recog"], self.network_weights["biases_recog"])

    #     recon_mean2 = self._generator_network(recog_mean2, self.network_weights["weights_gener"], self.network_weights["biases_gener"])


    #     recog_mean3, recog_log_sigma_sq3 = self._recognition_network(prev_z, recon_mean2, self.network_weights["weights_recog"], self.network_weights["biases_recog"])

    #     recon_mean3 = self._generator_network(recog_mean3, self.network_weights["weights_gener"], self.network_weights["biases_gener"])

    #     return recon_mean, recon_mean2, recon_mean3


    def train(self, get_data, steps=1000, display_step=10, path_to_load_variables='', path_to_save_variables=''):


        # if self.path_to_load_variables == '':
        #     self.sess.run(init)
        # else:
        saver = tf.train.Saver()

        if path_to_load_variables != '':
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables


        # Training cycle
        for step in range(steps):
            # avg_cost = 0.
            # total_batch = int(n_samples / self.batch_size)
            # Loop over all batches
            # for i in range(total_batch):
            batch = []
            bactch_actions = []
            while len(batch) != self.batch_size:
                # frame=make_ball_gif(n_frames=1, f_height=14, f_width=14, ball_size=2, max_1=True, vector_output=True)
                sequence, actions=get_data()
                # batch_xs, _ = mnist.train.next_batch(batch_size)
                batch.append(sequence)
                bactch_actions.append(actions)

            # Fit training using batch data
            # cost1, rec1, kl1, nw1 = self.sess.run((self.cost, self.reconstr_loss, self.kl_loss, self.network_weights), 
            #                       feed_dict={self.x: batch})

            cost, rec, kl, nw = vae.partial_fit(batch, bactch_actions)

            # cost2, rec2, kl2, nw2 = self.sess.run((self.cost, self.reconstr_loss, self.kl_loss, self.network_weights), 
            #                       feed_dict={self.x: batch})
            # print vae.reconstr_loss.eval()
            # print vae.kl_loss.eval()


            

            # Compute average loss
            # avg_cost += cost / n_samples * self.batch_size
            if cost != cost:
                print 'here'
                print cost
                # rec_loss = self.sess.run(self.reconstr_loss_, feed_dict={self.x: batch})
                # print rec_loss
                # rec_loss = self.sess.run(self.another_test, feed_dict={self.x: batch})
                # print rec_loss
                # rec_loss = self.sess.run(self.x_reconstr_mean, feed_dict={self.x: batch})
                print rec_loss
                fasdfas
            # Display logs per epoch step
            if step % display_step == 0:
                print "Step:", '%04d' % (step+1), \
                      "cost=", "{:.5f}".format(cost),\
                      "recon=", "{:.5f}".format(rec),\
                      "kl=", "{:.5f}".format(kl)

                # print nw['weights_trans']['h1']

                # print "Step:", '%04d' % (step+1), \
                #       "cost1=", "{:.5f}".format(cost1),\
                #       "recon1=", "{:.5f}".format(rec1),\
                #       "kl1=", "{:.5f}".format(kl1)

                # print "Step:", '%04d' % (step+1), \
                #       "cost2=", "{:.5f}".format(cost2),\
                #       "recon2=", "{:.5f}".format(rec2),\
                #       "kl2=", "{:.5f}".format(kl2)

                # print 



                # #for seeing the means
                # cost2, rec2, kl2,a,b,c,pm,rm, pv, rv= self.sess.run((self.cost, self.reconstr_loss, self.kl_loss, self.a, self.b, self.c, self.prior_mean, self.recog_mean, self.prior_log_sigma_sq, self.recog_log_sigma_sq), 
                #       feed_dict={self.x: batch})
                # print "a=", "{:.5f}".format(a),\
                #       "b=", "{:.5f}".format(b),\
                #       "c=", "{:.5f}".format(c), '\n',\
                #       "pm=", str(pm[0]), '\n',\
                #       "rm=", str(rm[0]), '\n',\
                #       "pv=", str(pv[0]), '\n',\
                #       "rv=", str(rv[0])
                # print




        if path_to_save_variables != '':
            saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables

        return vae



if __name__ == "__main__":

    steps = 100
    f_height=30
    f_width=30
    ball_size=5
    n_time_steps = 10
    batch_size = 4
    # path_to_load_variables=home+'/Documents/tmp/vae2.ckpt'
    path_to_load_variables=''
    path_to_save_variables=home+'/Documents/tmp/vae_sequential_actions.ckpt'

    def get_ball():
        return make_ball_gif(n_frames=n_time_steps, f_height=f_height, f_width=f_width, ball_size=ball_size, max_1=True, vector_output=True)


    network_architecture = \
        dict(n_hidden_recog_1=100, # 1st layer encoder neurons
             n_hidden_recog_2=100, # 2nd layer encoder neurons
             n_hidden_gener_1=100, # 1st layer decoder neurons
             n_hidden_gener_2=100, # 2nd layer decoder neurons
             n_hidden_trans_1=100,
             n_hidden_trans_2=100,
             n_input=f_height*f_width, # image as vector
             n_z=20,  # dimensionality of latent space
             n_actions=4)

    vae = VAE(network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=batch_size, n_time_steps=n_time_steps)
    vae.train(get_data=get_ball, steps=steps, display_step=20, path_to_load_variables=path_to_load_variables, path_to_save_variables=path_to_save_variables)




    #GENERATE 
    # generated = []
    # # for i in range(5):
    # generated.append(vae.generate())
    generated = vae.call_generate()
    generated = np.array(generated)
    print 'generated', generated.shape #[timestep, batch, image_vector]
    # generated = np.reshape(generated, [batch_size*n_time_steps, f_height,f_width,1])
    one_sequence = []
    for t in range(len(generated)):
        one_sequence.append(generated[t][0])
    one_sequence = np.array(one_sequence)
    print one_sequence.shape
    one_sequence = np.reshape(one_sequence, [n_time_steps, f_height,f_width, 1])

    # print 'generated'
    # print one_sequence[0,:,0]

    for i in range(len(one_sequence)): 
        # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
        one_sequence[i] = one_sequence[i] * (255. / np.max(one_sequence[i]))
        one_sequence[i] = one_sequence[i].astype('uint8')

    # for i in range(len(generated)): 
    #     # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
    #     generated[i] = generated[i] * (255. / np.max(generated[i]))
    #     generated[i] = generated[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/gen_gif.gif", one_sequence, 'GIF', **kargs)



    one_sequence = []
    for t in range(len(generated)):
        one_sequence.append(generated[t][1])
    one_sequence = np.array(one_sequence)
    print one_sequence.shape
    one_sequence = np.reshape(one_sequence, [n_time_steps, f_height,f_width, 1])

    for i in range(len(one_sequence)): 
        # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
        one_sequence[i] = one_sequence[i] * (255. / np.max(one_sequence[i]))
        one_sequence[i] = one_sequence[i].astype('uint8')

    # for i in range(len(generated)): 
    #     # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
    #     generated[i] = generated[i] * (255. / np.max(generated[i]))
    #     generated[i] = generated[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/gen_gif1.gif", one_sequence, 'GIF', **kargs)



    one_sequence = []
    for t in range(len(generated)):
        one_sequence.append(generated[t][2])
    one_sequence = np.array(one_sequence)
    print one_sequence.shape
    one_sequence = np.reshape(one_sequence, [n_time_steps, f_height,f_width, 1])

    for i in range(len(one_sequence)): 
        # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
        one_sequence[i] = one_sequence[i] * (255. / np.max(one_sequence[i]))
        one_sequence[i] = one_sequence[i].astype('uint8')

    # for i in range(len(generated)): 
    #     # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
    #     generated[i] = generated[i] * (255. / np.max(generated[i]))
    #     generated[i] = generated[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/gen_gif2.gif", one_sequence, 'GIF', **kargs)

    print 'Generated 3 sequences'
    fasdfadf


    # #GENERATE 2

    # gen1, gen2, gen3, gen4, gen5 = vae.call_generate2()
    # # gen1 = gen1.eval()
    # # gen2 = gen2.eval()
    # # gen3 = gen3.eval()
    # # gen4 = gen4.eval()
    # # gen5 = gen5.eval()
    # print 'gen', gen1.shape, gen2.shape, gen3.shape, gen4.shape, gen5.shape

    # together = [gen1[0], gen2[0], gen3[0], gen4[0], gen5[0]]
    # together = np.array(together)
    # together = np.reshape(together, [5, f_height,f_width, 1])
    # for i in range(len(together)): 
    #     together[i] = together[i] * (255. / np.max(together[i]))
    #     together[i] = together[i].astype('uint8')

    # kargs = { 'duration': .6 }
    # imageio.mimsave(home+"/Downloads/generate2_gif.gif", together, 'GIF', **kargs)




    # #GENERATE 3

    # gen1, gen2, gen3, gen4, gen5 = vae.generate3()
    # gen1 = gen1.eval()
    # gen2 = gen2.eval()
    # gen3 = gen3.eval()
    # gen4 = gen4.eval()
    # gen5 = gen5.eval()
    # print 'gen', gen1.shape, gen2.shape, gen3.shape, gen4.shape, gen5.shape

    # together = [gen1[0], gen2[0], gen3[0], gen4[0], gen5[0]]
    # together = np.array(together)
    # together = np.reshape(together, [5, f_height,f_width, 1])
    # for i in range(len(together)): 
    #     together[i] = together[i] * (255. / np.max(together[i]))
    #     together[i] = together[i].astype('uint8')

    # kargs = { 'duration': .6 }
    # imageio.mimsave(home+"/Downloads/generate3_gif.gif", together, 'GIF', **kargs)




    #RECONSTRUCT 
    batch = []
    batch_actions = []
    while len(batch) != batch_size:
        sequence, actions =get_ball()
        batch.append(sequence)
        batch_actions.append(actions)

    batch = np.array(batch)
    batch_actions = np.array(batch_actions)
    print batch.shape
    #REMEMBER THE BATCH SHAPE AND RECONSTRUCTION ARENT THE SAME
    # batch is [bathc, timestep, frame]
    # recon is [timestep, batch, frame]

    #look at one batch sequence
    one_sequence = []
    for t in range(len(batch[0])):
        one_sequence.append(batch[0][t])
    one_sequence = np.array(one_sequence)
    print one_sequence.shape
    one_sequence = np.reshape(one_sequence, [n_time_steps, f_height,f_width, 1])

    for i in range(len(one_sequence)): 
        # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
        one_sequence[i] = one_sequence[i] * (255. / np.max(one_sequence[i]))
        one_sequence[i] = one_sequence[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/reconstr_input_gif.gif", one_sequence, 'GIF', **kargs)
    print 'saved', home+"/Downloads/reconstr_input_gif.gif"




    x_reconstruct = vae.call_reconstruct(batch)
    x_reconstruct = np.array(x_reconstruct)
    print x_reconstruct.shape

    one_sequence = []
    for t in range(len(x_reconstruct)):
        one_sequence.append(x_reconstruct[t][0])
    one_sequence = np.array(one_sequence)
    print one_sequence.shape
    one_sequence = np.reshape(one_sequence, [n_time_steps, f_height,f_width, 1])

    # print 'reconstructed'
    # print one_sequence[0,:,0]

    for i in range(len(one_sequence)): 
        # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
        one_sequence[i] = one_sequence[i] * (255. / np.max(one_sequence[i]))
        one_sequence[i] = one_sequence[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/reconstr_gif.gif", one_sequence, 'GIF', **kargs)
    print 'saved', home+"/Downloads/reconstr_gif.gif"



    # #RECONSRUCT 2
    # batch = []
    # while len(batch) != batch_size:
    #     frame=make_ball_gif(n_frames=1, f_height=f_height, f_width=f_width, ball_size=ball_size, max_1=True, vector_output=True)
    #     batch.append(frame)

    # batch = np.array(batch)
    # batch = np.reshape(batch, [batch_size, 1 , f_height*f_width])
    # print batch.shape

    # recon1, recon2, recon3 = vae.call_reconstruct2(batch)
    # # recon1 = recon1.eval()
    # # recon2 = recon2.eval()
    # # recon3 = recon3.eval()
    # print 'recon2', recon1.shape, recon2.shape, recon3.shape

    # # x_reconstruct = np.array(x_reconstruct)
    # # print x_reconstruct.shape

    # together = [batch[0][0], recon1[0], recon2[0], recon3[0]]
    # together = np.array(together)
    # together = np.reshape(together, [4, f_height,f_width, 1])
    # for i in range(len(together)): 
    #     together[i] = together[i] * (255. / np.max(together[i]))
    #     together[i] = together[i].astype('uint8')

    # kargs = { 'duration': .6 }
    # imageio.mimsave(home+"/Downloads/reconstruct2_gif.gif", together, 'GIF', **kargs)




    # batch = np.array(batch)
    # batch = np.reshape(batch, [batch_size*n_time_steps, f_height,f_width,1])

    # for i in range(len(batch)):
    #     # batch[i] = np.reshape(batch[i], [f_height,f_width,1])
    #     batch[i] = batch[i] * (255. / np.max(batch[i]))
    #     batch[i] = batch[i].astype('uint8')
    # # kargs = { 'duration': .8 }
    # # imageio.mimsave(home+"/Downloads/inputs_gif.gif", batch, 'GIF', **kargs)

    # x_reconstruct = np.reshape(x_reconstruct, [batch_size*n_time_steps, f_height,f_width,1])

    # for i in range(len(x_reconstruct)):
    #     # reconstruct_batch.append(np.reshape(x_reconstruct[i], [f_height,f_width,1]))
    #     x_reconstruct[i] = x_reconstruct[i] * (255. / np.max(x_reconstruct[i]))
    #     x_reconstruct[i] = x_reconstruct[i].astype('uint8')
    # # kargs = { 'duration': .5 }
    # # imageio.mimsave(home+"/Downloads/inputs_rec_gif.gif", reconstruct_batch, 'GIF', **kargs)

    # combined = []
    # for i in range(0,len(batch),3):
    #     combined.append(batch[i])
    #     combined.append(batch[i+1])
    #     combined.append(batch[i+2])

    #     combined.append(x_reconstruct[i])
    #     combined.append(x_reconstruct[i+1])
    #     combined.append(x_reconstruct[i+2])

    # kargs = { 'duration': .8 }
    # imageio.mimsave(home+"/Downloads/comb_gif.gif", combined, 'GIF', **kargs)





    #PREDICT

    batch = []
    while len(batch) != batch_size:
        frame=make_ball_gif(n_frames=1, f_height=f_height, f_width=f_width, ball_size=ball_size, max_1=True, vector_output=True)
        batch.append(frame)

    batch = np.array(batch)
    batch = np.reshape(batch, [batch_size, 1 , f_height*f_width])
    print batch.shape

    # predicted = vae.call_predict_next(batch)

    # print predicted.shape

    # together = [batch[0][0], predicted[0]]
    # together = np.array(together)
    # together = np.reshape(together, [2, f_height,f_width, 1])
    # for i in range(len(together)): 
    #     together[i] = together[i] * (255. / np.max(together[i]))
    #     together[i] = together[i].astype('uint8')

    # kargs = { 'duration': .6 }
    # imageio.mimsave(home+"/Downloads/pred_gif.gif", together, 'GIF', **kargs)



    predicted2, predicted3 = vae.call_predict_next2(batch)

    print predicted.shape

    together = [batch[0][0], predicted2[0], predicted3[0]]
    together = np.array(together)
    together = np.reshape(together, [3, f_height,f_width, 1])
    for i in range(len(together)): 
        together[i] = together[i] * (255. / np.max(together[i]))
        together[i] = together[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/pred2_gif.gif", together, 'GIF', **kargs)

    print 'DONE'









