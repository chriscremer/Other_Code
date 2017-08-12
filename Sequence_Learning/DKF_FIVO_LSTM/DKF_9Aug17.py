


import numpy as np
import tensorflow as tf

from NN2 import NN

from utils import split_mean_logvar


class DKF_new(object):

    def __init__(self, network_architecture):


        tf.reset_default_graph()



        self.rs = 0
        self.act_fct=tf.nn.softplus #tf.tanh
        self.learning_rate = 0.001
        self.reg_param = .00001
        self.z_size = network_architecture["n_z"]
        self.input_size = network_architecture["n_input"]
        self.action_size = network_architecture["n_actions"]

        self.n_particles = network_architecture["n_particles"]
        self.n_timesteps = network_architecture["n_timesteps"]

        # Graph Input: [B,T,X], [B,T,A]
        self.x = tf.placeholder(tf.float32, [None, None, self.input_size])
        self.actions = tf.placeholder(tf.float32, [None, None, self.action_size])
        

        #Recognition/Inference net q(z|z-1,u,x)
        self.rec_net = NN([self.input_size+self.z_size+self.action_size, 100, 100, self.z_size*2], [tf.nn.softplus,tf.nn.softplus, None])
        #Emission net p(x|z)
        self.emiss_net = NN([self.z_size, 100, 100, self.input_size], [tf.nn.softplus,tf.nn.softplus, None])
        #Transition net p(z|z-1,u)
        self.trans_net = NN([self.z_size+self.action_size, 100, 100, self.z_size*2], [tf.nn.softplus,tf.nn.softplus, None])

        weight_decay = self.rec_net.weight_decay() + self.emiss_net.weight_decay() + self.trans_net.weight_decay()

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
        tf.get_default_graph().finalize()
        #Start session
        self.sess = tf.Session()







 


    def Model(self, x, actions):
        '''
        x: [B,T,X]
        actions: [B,T,A]

        output: elbo scalar
        '''


        #problem with loop is creates many insteaces of everythin in it..
        for t in range(self.n_timesteps):










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

















