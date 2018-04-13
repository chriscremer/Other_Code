

# adding multiple particles

# update May 2 2017: updated code for TF r1.0
    # tf.concat
    # tf.multiply
    # tf.pack to tf.stack



import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")



class DKF(object):

    def __init__(self, network_architecture, batch_size=5, n_particles=4):
        
        tf.reset_default_graph()

        self.batch_size = batch_size
        self.n_particles = n_particles

        self.transfer_fct=tf.nn.softplus #tf.tanh
        self.learning_rate = 0.001
        
        self.z_size = network_architecture["n_z"]
        self.input_size = network_architecture["n_input"]
        self.action_size = network_architecture["n_actions"]
        self.reg_param = .00001


        # Graph Input: [B,T,X], [B,T,A]
        self.x = tf.placeholder(tf.float32, [None, None, self.input_size])
        self.actions = tf.placeholder(tf.float32, [None, None, self.action_size])
        
        #Variables
        self.network_weights = self._initialize_weights(network_architecture)

        #Objective
        self.elbo = self.Model(self.x, self.actions)
        self.cost = -self.elbo + (self.reg_param * self.l2_regularization())

        #Optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost)

        #Evaluation
        self.prev_z_and_current_a_ = tf.placeholder(tf.float32, [self.batch_size, self.z_size+self.action_size])
        self.next_state = self.transition_net(self.prev_z_and_current_a_)

        self.current_z_ = tf.placeholder(tf.float32, [self.batch_size, self.z_size])
        self.current_emission = tf.sigmoid(self.observation_net(self.current_z_))

        self.prior_mean_ = tf.placeholder(tf.float32, [self.batch_size, self.z_size])
        self.prior_logvar_ = tf.placeholder(tf.float32, [self.batch_size, self.z_size])
        eps = tf.random_normal((self.batch_size, self.z_size), 0, 1, dtype=tf.float32)
        self.sample = tf.add(self.prior_mean_, tf.multiply(tf.sqrt(tf.exp(self.prior_logvar_)), eps))

        #to make sure im not adding nodes to the graph
        # tf.get_default_graph().finalize()


    def _initialize_weights(self, network_architecture):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


        all_weights = dict()

        #Recognition/Inference net q(z|z-1,u,x)
        all_weights['encoder_weights'] = {}
        all_weights['encoder_biases'] = {}

        for layer_i in range(len(network_architecture['encoder_net'])):
            if layer_i == 0:
                all_weights['encoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.input_size+self.action_size+self.z_size, network_architecture['encoder_net'][layer_i]))
                all_weights['encoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['encoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['encoder_net'][layer_i-1], network_architecture['encoder_net'][layer_i]))
                all_weights['encoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

        all_weights['encoder_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.z_size))
        all_weights['encoder_weights']['out_log_var'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.z_size))
        all_weights['encoder_biases']['out_mean'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))
        all_weights['encoder_biases']['out_log_var'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))


        #Generator net p(x|z)
        all_weights['decoder_weights'] = {}
        all_weights['decoder_biases'] = {}

        for layer_i in range(len(network_architecture['decoder_net'])):
            if layer_i == 0:
                all_weights['decoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.z_size, network_architecture['decoder_net'][layer_i]))
                all_weights['decoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['decoder_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['decoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['decoder_net'][layer_i-1], network_architecture['decoder_net'][layer_i]))
                all_weights['decoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

        all_weights['decoder_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.input_size))
        all_weights['decoder_biases']['out_mean'] = tf.Variable(tf.zeros([self.input_size], dtype=tf.float32))

        #Generator/Transition net q(z|z-1,u)
        all_weights['trans_weights'] = {}
        all_weights['trans_biases'] = {}

        for layer_i in range(len(network_architecture['trans_net'])):
            if layer_i == 0:
                all_weights['trans_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.action_size+self.z_size, network_architecture['trans_net'][layer_i]))
                all_weights['trans_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['trans_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['trans_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['trans_net'][layer_i-1], network_architecture['trans_net'][layer_i]))
                all_weights['trans_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['trans_net'][layer_i]], dtype=tf.float32))

        all_weights['trans_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['trans_net'][-1], self.z_size))
        all_weights['trans_weights']['out_log_var'] = tf.Variable(xavier_init(network_architecture['trans_net'][-1], self.z_size))
        all_weights['trans_biases']['out_mean'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))
        all_weights['trans_biases']['out_log_var'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))

        return all_weights


    def l2_regularization(self):

        sum_ = 0
        for net in self.network_weights:

            for weights_biases in self.network_weights[net]:

                sum_ += tf.reduce_sum(tf.square(self.network_weights[net][weights_biases]))

        return sum_


    def log_normal(self, z, mean, log_var):
        '''
        Log of normal distribution

        z is [B, Z]
        mean is [B, Z]
        log_var is [B, Z]
        output is [B]
        '''

        term1 = tf.reduce_sum(log_var, reduction_indices=1) #sum over dimensions, [B]

        # [1]
        term2 = self.z_size * tf.log(2*math.pi)

        dif = tf.square(z - mean)
        dif_cov = dif / tf.exp(log_var)
        term3 = tf.reduce_sum(dif_cov, 1) #sum over dimensions,[B]

        all_ = term1 + term2 + term3
        log_norm = -.5 * all_

        return log_norm



    def recognition_net(self, input_):
        # input:[B,X+A+Z]

        n_layers = len(self.network_weights['encoder_weights']) - 2 #minus 2 for the mean and var outputs
        weights = self.network_weights['encoder_weights']
        biases = self.network_weights['encoder_biases']

        for layer_i in range(n_layers):

            input_ = self.transfer_fct(tf.add(tf.matmul(input_, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 
            #add batch norm here

        z_mean = tf.add(tf.matmul(input_, weights['out_mean']), biases['out_mean'])
        z_log_var = tf.add(tf.matmul(input_, weights['out_log_var']), biases['out_log_var'])

        return z_mean, z_log_var


    def observation_net(self, input_):
        # input:[B,Z]

        n_layers = len(self.network_weights['decoder_weights']) - 1 #minus 2 for the mean and var outputs
        weights = self.network_weights['decoder_weights']
        biases = self.network_weights['decoder_biases']

        for layer_i in range(n_layers):

            input_ = self.transfer_fct(tf.add(tf.matmul(input_, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 
            #add batch norm here

        x_mean = tf.add(tf.matmul(input_, weights['out_mean']), biases['out_mean'])
        # x_log_var = tf.add(tf.matmul(input_, weights['out_log_var']), biases['out_log_var'])

        return x_mean


    def transition_net(self, input_):
        # input:[B,Z+A]

        n_layers = len(self.network_weights['trans_weights'])  - 2 #minus 2 for the mean and var outputs
        weights = self.network_weights['trans_weights']
        biases = self.network_weights['trans_biases']

        for layer_i in range(n_layers):

            input_ = self.transfer_fct(tf.add(tf.matmul(input_, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 
            #add batch norm here


        z_mean = tf.add(tf.matmul(input_, weights['out_mean']), biases['out_mean'])
        z_log_var = tf.add(tf.matmul(input_, weights['out_log_var']), biases['out_log_var'])

        return z_mean, z_log_var

            


    def Model(self, x, actions):
        '''
        x: [B,T,X]
        actions: [B,T,A]

        output: elbo scalar
        '''

        def fn_over_timesteps(particle_and_logprobs, xa_t):
            '''
            particle_and_logprobs: [B,PZ+3]
            xa_t: [B,X+A]

            Steps: encode, sample, decode, calc logprobs

            return [B,Z+3]
            '''

            #unpack previous particle_and_logprobs, prev_z:[B,PZ], ignore prev logprobs
            prev_particles = tf.slice(particle_and_logprobs, [0,0], [self.batch_size, self.n_particles*self.z_size])
            #unpack xa_t
            current_x = tf.slice(xa_t, [0,0], [self.batch_size, self.input_size]) #[B,X]
            current_a = tf.slice(xa_t, [0,self.input_size], [self.batch_size, self.action_size]) #[B,A]

            log_pzs = []
            log_qzs = []
            log_pxs = []
            new_particles = []

            for i in range(self.n_particles):

                #select particle
                prev_z = tf.slice(prev_particles, [0,i*self.z_size], [self.batch_size, self.z_size])

                #combine prez and current a (used for prior)
                prev_z_and_current_a = tf.concat([prev_z, current_a], axis=1) #[B,ZA]

                #ENCODE
                #Concatenate current x, current action, prev_z: [B,XA+Z]
                concatenate_all = tf.concat([xa_t, prev_z], axis=1)

                #Predict q(z|z-1,u,x): [B,Z] [B,Z]
                z_mean, z_log_var = self.recognition_net(concatenate_all)


                #SAMPLE from q(z|z-1,u,x)  [B,Z]
                eps = tf.random_normal((self.batch_size, self.z_size), 0, 1, dtype=tf.float32)
                this_particle = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_var)), eps))


                #DECODE  p(x|z): [B,X]
                x_mean = self.observation_net(this_particle)

                #CALC LOGPROBS

                #Prior p(z|z-1,u) [B,Z]
                prior_mean, prior_log_var = self.transition_net(prev_z_and_current_a) #[B,Z]
                log_p_z = self.log_normal(this_particle, prior_mean, prior_log_var) #[B]

                #Recognition q(z|z-1,x,u)
                log_q_z = self.log_normal(this_particle, z_mean, z_log_var)


                #Likelihood p(x|z)  Bernoulli
                reconstr_loss = \
                    tf.reduce_sum(tf.maximum(x_mean, 0) 
                                - x_mean * current_x
                                + tf.log(1 + tf.exp(-tf.abs(x_mean))),
                                 1) #sum over dimensions
                log_p_x = -reconstr_loss

                log_pzs.append(log_p_z)
                log_qzs.append(log_q_z)
                log_pxs.append(log_p_x)
                new_particles.append(this_particle)


            #Average the log probs
            log_p_z = tf.reduce_mean(tf.stack(log_pzs), axis=0) #over particles so its [B]

            log_q_z = tf.reduce_mean(tf.stack(log_qzs), axis=0) 
            log_p_x = tf.reduce_mean(tf.stack(log_pxs), axis=0) 
            #Organize output
            logprobs = tf.stack([log_p_x, log_p_z, log_q_z], axis=1) # [B,3]

            #Reshape particles
            new_particles = tf.stack(new_particles, axis=1) #[B,P,Z]
            new_particles = tf.reshape(new_particles, [self.batch_size, self.n_particles*self.z_size])

            # output = tf.concat(1, [new_particles, logprobs])# [B,Z+3]
            output = tf.concat([new_particles, logprobs], axis=1)# [B,Z+3]


            return output




        # Put obs and actions to together [B,T,X+A]
        x_and_a = tf.concat([x, actions], axis=2)

        # Transpose so that timesteps is first [T,B,X+A]
        x_and_a = tf.transpose(x_and_a, [1,0,2])

        #Make initializations for scan
        z_init = tf.zeros([self.batch_size, self.n_particles*self.z_size])
        logprobs_init = tf.zeros([self.batch_size, 3])
        # Put z and logprobs together [B,PZ+3]
        # initializer = tf.concat(1, [z_init, logprobs_init])
        initializer = tf.concat([z_init, logprobs_init], axis=1)


        # Scan over timesteps, returning particles and logprobs [T,B,Z+3]
        self.particles_and_logprobs = tf.scan(fn_over_timesteps, x_and_a, initializer=initializer)

        #unpack the logprobs  list([pz+3,T,B])
        particles_and_logprobs_unstacked = tf.unstack(self.particles_and_logprobs, axis=2)

        #[T,B]
        log_p_x_over_time = particles_and_logprobs_unstacked[self.z_size*self.n_particles]
        log_p_z_over_time = particles_and_logprobs_unstacked[self.z_size*self.n_particles+1]
        log_q_z_over_time = particles_and_logprobs_unstacked[self.z_size*self.n_particles+2]

        # sum over timesteps  [B]
        log_q_z_batch = tf.reduce_sum(log_q_z_over_time, axis=0)
        log_p_z_batch = tf.reduce_sum(log_p_z_over_time, axis=0)
        log_p_x_batch = tf.reduce_sum(log_p_x_over_time, axis=0)
        # average over batch  [1]
        self.log_q_z_final = tf.reduce_mean(log_q_z_batch)
        self.log_p_z_final = tf.reduce_mean(log_p_z_batch)
        self.log_p_x_final = tf.reduce_mean(log_p_x_batch)

        elbo = self.log_p_x_final + self.log_p_z_final - self.log_q_z_final

        return elbo



    def predict_future(self, prev_z, actions):
        '''
        prev_z [B,PZ]
        actions [B,timesteps_left, A]

        return [timesteps_left, P, X]
        '''

        #convert prev_z to a list of particle states
        particle_states = []
        for p in range(self.n_particles):
            particle_states.append(prev_z[0][p*self.z_size : p*self.z_size+self.z_size])


        #predict future 
        obs = []
        for t in range(len(actions[0])):

            this_timestep_obs = []
            for p in range(self.n_particles):

                #transition state
                prev_z_and_current_a = np.concatenate((np.reshape(particle_states[p], [1,self.z_size]), [actions[0][t]]), axis=1) #[B,ZA]

                # [B,Z]
                prior_mean, prior_log_var = self.sess.run(self.next_state, feed_dict={self.prev_z_and_current_a_: prev_z_and_current_a})

                #sample new state
                sample = self.sess.run(self.sample, feed_dict={self.prior_mean_: prior_mean, self.prior_logvar_: prior_log_var})

                #decode state
                x_mean = self.sess.run(self.current_emission, feed_dict={self.current_z_: sample})
                this_timestep_obs.append(x_mean)

                #set this sample as previous sample
                particle_states[p] = np.reshape(sample, [-1])

            obs.append(this_timestep_obs)

        return np.array(obs) #this will be [TL, P, X]



                

    def train(self, get_data, steps=1000, display_step=10, path_to_load_variables='', path_to_save_variables=''):


        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            self.sess.run(tf.global_variables_initializer())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print ('loaded variables ' + path_to_load_variables)


        # Training cycle
        for step in range(steps):

            batch = []
            batch_actions = []
            while len(batch) != self.batch_size:

                sequence, actions=get_data()
                batch.append(sequence)
                batch_actions.append(actions)

            _ = self.sess.run(self.optimizer, feed_dict={self.x: batch, self.actions: batch_actions})

            # Display
            if step % display_step == 0:

                cost = self.sess.run(self.elbo, feed_dict={self.x: batch, self.actions: batch_actions})
                cost = -cost #because I want to see the NLL

                p1,p2,p3 = self.sess.run([self.log_p_x_final, self.log_p_z_final, self.log_q_z_final ], feed_dict={self.x: batch, self.actions: batch_actions})


                print ("Step:", '%04d' % (step+1), "cost=", "{:.5f}".format(cost), 'logpx', p1, 'logpz', p2, 'logqz', p3)


        if path_to_save_variables != '':
            saver.save(self.sess, path_to_save_variables)
            print ('Saved variables to ' + path_to_save_variables)

        print ('Done training')






    def test(self, get_data, path_to_load_variables=''):

        #get actions and frames
        #give model all actions and only some frames
        # so its two steps, 
            # run normally while there are given frames, take last state
            # give last state and generate next states and obs
        #append gen to real 


        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            self.sess.run(tf.global_variables_initializer())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print ('loaded variables ' + path_to_load_variables)


        batch = []
        batch_actions = []
        while len(batch) != self.batch_size:

            sequence, actions=get_data()
            batch.append(sequence)
            batch_actions.append(actions)

        #chop up the sequence, only give it first 3 frames
        n_time_steps_given = 3
        given_sequence = []
        given_actions = []
        hidden_sequence = []
        hidden_actions = []
        for b in range(len(batch)):
            given_sequence.append(batch[b][:n_time_steps_given])
            given_actions.append(batch_actions[b][:n_time_steps_given])
            #and the ones that are not used
            hidden_sequence.append(batch[b][n_time_steps_given:])
            hidden_actions.append(batch_actions[b][n_time_steps_given:])

        #Get states [T,B,PZ+3]
        current_state = self.sess.run(self.particles_and_logprobs, feed_dict={self.x: given_sequence, self.actions: given_actions})
        # Unpack, get states
        current_state = current_state[n_time_steps_given-1][0][:self.z_size*self.n_particles]


        #Step 2: Predict future states and decode to frames

        # print np.array(hidden_actions).shape #[B,leftovertime, A]
        current_state = [current_state] #so it fits batch
        
        # [TL, P, B, X]
        obs = self.predict_future(current_state, hidden_actions) 

        # [P, T-TL, X]
        real_and_gen = []
        for p in range(self.n_particles):
            real_and_gen.append(list(given_sequence[0]))

        for obs_t in range(len(obs)):
            # print 'obs_t', obs_t
            for p in range(len(obs[obs_t])):
                # print 'p', p

                obs_t_p = np.reshape(obs[obs_t][p][0], [-1])

                real_and_gen[p].append(obs_t_p)

        #[T,P,X]
        real_and_gen = np.array(real_and_gen)
        # print real_and_gen.shape #[T,X]
        
        real_sequence = np.array(batch[0])
        actions = np.array(batch_actions[0])

        return real_sequence, actions, real_and_gen















