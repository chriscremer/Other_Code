


import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
# import imageio

# from ball_sequence import make_ball_gif


class DKF():

    def __init__(self, network_architecture, batch_size=5):
        
        tf.reset_default_graph()

        self.batch_size = batch_size

        self.transfer_fct=tf.nn.softplus #tf.tanh
        self.learning_rate = 0.001

        # self.n_time_steps = n_time_steps #this shouldnt be used
        # self.n_particles = n_particles

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
        self.elbo = self.Model(self.x, self.actions, self.network_weights)
        self.cost = -self.elbo + (self.reg_param * self.l2_regularization())

        #Optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost)

        #Evaluation
        # self.generate = self.generate(self.x, self.actions)
        # self.current_z = tf.placeholder(tf.float32, [None, self.z_size])
        # self.prev_x = tf.placeholder(tf.float32, [None, self.input_size])
        # self.current_emission = self.observation_net(tf.concat(1, [self.current_z, self.prev_x]), self.network_weights)

        # self.prev_z = tf.placeholder(tf.float32, [None, self.z_size])
        # self.current_action = tf.placeholder(tf.float32, [None, self.action_size])
        # apz = tf.concat(1, [tf.concat(1, [self.current_action, self.prev_x]), self.prev_z])
        # self.next_state = self.transition_net(apz, self.network_weights)


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
        # all_weights['decoder_weights']['out_log_var'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.input_size))
        # all_weights['decoder_biases']['out_log_var'] = tf.Variable(tf.zeros([self.input_size], dtype=tf.float32))



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


    # def log_normal_x(self, x, mean, log_var):
    #     '''
    #     Log of normal distribution

    #     z is [B, Z]
    #     mean is [B, Z]
    #     log_var is [B, Z]
    #     output is [B]
    #     '''

    #     # term1 = tf.log(tf.reduce_prod(tf.exp(log_var_sq), reduction_indices=1))
    #     term1 = tf.reduce_sum(log_var, reduction_indices=1) #sum over dimensions n_z so now its [B]

    #     # [1]
    #     term2 = self.input_size * tf.log(2*math.pi)

    #     dif = tf.square(x - mean)
    #     dif_cov = dif / tf.exp(log_var)
    #     term3 = tf.reduce_sum(dif_cov, 1) #sum over dimensions so its [B]

    #     all_ = term1 + term2 + term3
    #     log_norm = -.5 * all_

    #     return log_norm


    def recognition_net(self, input_):
        # input:[B,2X+A+Z]

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
        # input:[B,X+Z]

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
        # input:[B,X+Z+A]

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

        #TODO allow for multiple particles

        def fn_over_timesteps(particle_and_logprobs, xa_t):
            '''
            particle_and_logprobs: [B,Z+3]
            xa_t: [B,X+A]

            Steps: encode, sample, decode, calc logprobs

            return [B,Z+3]
            '''

            #unpack previous particle_and_logprobs, prev_z:[B,Z], ignore prev logprobs
            prev_z = tf.slice(particle_and_logprobs, [0,0], [self.batch_size, self.z_size])
            #unpack xa_t
            current_x = tf.slice(xa_t, [0,0], [self.batch_size, self.input_size]) #[B,X]
            current_a = tf.slice(xa_t, [0,self.input_size], [self.batch_size, self.action_size]) #[B,A]
            #combine prez and current a (used for prior)
            prev_z_and_current_a = tf.concat(1, [prev_z, current_a]) #[B,ZA]
            


            #ENCODE
            #Concatenate current x, current action, prev_z: [B,XA+Z]
            concatenate_all = tf.concat(1, [xa_t, prev_z])
            #Predict q(z|z-1,u,x): [B,Z] [B,Z]
            z_mean, z_log_var = self.recognition_net(concatenate_all)


            #SAMPLE from q(z|z-1,u,x)  [B,Z]
            eps = tf.random_normal((self.batch_size, self.z_size), 0, 1, dtype=tf.float32)
            this_particle = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_var)), eps))


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
                             2) #sum over dimensions
			log_p_x = -reconstr_loss

			#Organize output
            logprobs = tf.pack([log_p_x, log_p_z, log_q_z], axis=1) # [B,3]
            output = tf.concat(1, [this_particle, logprobs])# [B,Z+3]

            return output







        # Put obs and actions to together [B,T,X+A]
        x_and_a = tf.concat(2, [x, actions])
        # Transpose so that timesteps is first [T,B,X+A]
        x_and_a = tf.transpose(x_and_a, [1,0,2])

        #Make initializations for scan
        z_init = tf.zeros([self.batch_size, self.z_size])
        log_probs_init = tf.zeros([self.batch_size, 3])
        # Put z and logprobs together [B,Z+3]
        initializer = tf.concat(1, [z_init, logprobs_init])

        # Scan over timesteps, returning particles and logprobs [T,B,Z+3]
        particles_and_logprobs = tf.scan(fn_over_timesteps, x_and_a, initializer=initializer)



        #unpack the logprobs  list([z+3,T,B])
        particles_and_logprobs_unstacked = tf.unstack(particles_and_logprobs, axis=2)

        #[T,B]
        log_p_x_over_time = particles_and_logprobs_unstacked[self.z_size]
        log_p_z_over_time = particles_and_logprobs_unstacked[self.z_size+1]
        log_q_z_over_time = particles_and_logprobs_unstacked[self.z_size+2]

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


                

    def train(self, get_data, steps=1000, display_step=10, path_to_load_variables='', path_to_save_variables=''):


        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            self.sess.run(tf.global_variables_initializer())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables


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


                print "Step:", '%04d' % (step+1), "cost=", "{:.5f}".format(cost), p1, p2, p3


        if path_to_save_variables != '':
            saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables

        print 'Done training'


    # def test(self, get_data, steps=1000, display_step=10, path_to_load_variables='', path_to_save_variables=''):



    #     # Training cycle
    #     for step in range(steps):

    #         batch = []
    #         batch_actions = []
    #         while len(batch) != self.batch_size:

    #             sequence, actions=get_data()
    #             batch.append(sequence)
    #             batch_actions.append(actions)

                
    #         # if len(np.array(batch_actions).shape) != 3:
    #         #     print batch_actions
    #         # _ = self.sess.run(self.optimizer, feed_dict={self.x: batch, self.actions: batch_actions})

    #         # Display
    #         if step % display_step == 0:

    #             cost = self.sess.run(self.elbo, feed_dict={self.x: batch, self.actions: batch_actions})
    #             cost = -cost #because I want to see the NLL

    #             p1,p2,p3 = self.sess.run([self.log_p_x_final, self.log_p_z_final, self.log_q_z_final ], feed_dict={self.x: batch, self.actions: batch_actions})


    #             print "Step:", '%04d' % (step+1), "cost=", "{:.5f}".format(cost), p1, p2, p3



    #     print 'Done test'



    # def get_current_emission(self, state, prev_emission):


    #     current_emission = self.sess.run(self.current_emission, feed_dict={self.current_z: state, self.prev_x: prev_emission})

    #     return current_emission


    # def get_next_state(self, prev_z, current_action, prev_x):


    #     next_state = self.sess.run(self.next_state, feed_dict={self.prev_z: prev_z, self.prev_x: prev_x, self.current_action: current_action})

    #     return next_state







# if __name__ == "__main__":

#     save_to = home + '/data/' #for boltz
#     # save_to = home + '/Documents/tmp/' # for mac

#     training_steps = 4000
#     f_height=30
#     f_width=30
#     ball_size=5
#     n_time_steps = 10
#     n_particles = 3
#     batch_size = 4
#     path_to_load_variables=save_to + 'dkf_ball_vars.ckpt'
#     # path_to_load_variables=''
#     path_to_save_variables=save_to + 'dkf_ball_vars2.ckpt'
#     # path_to_save_variables=''

#     train = 1
#     generate = 1


#     def get_ball():
#         return make_ball_gif(n_frames=n_time_steps, f_height=f_height, f_width=f_width, ball_size=ball_size, max_1=True, vector_output=True)


#     network_architecture = \
#         dict(   encoder_net=[100,100],
#                 decoder_net=[100,100],
#                 trans_net=[100,100],
#                 n_input=f_height*f_width, # image as vector
#                 n_z=20,  # dimensionality of latent space
#                 n_actions=4) #4 possible actions
    
#     print 'Initializing model..'
#     dkf = DKF(network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=batch_size, n_time_steps=n_time_steps, n_particles=n_particles)


#     if train:
#         print 'Training'
#         dkf.train(get_data=get_ball, steps=training_steps, display_step=20, path_to_load_variables=path_to_load_variables, path_to_save_variables=path_to_save_variables)


#     #GENERATE - Give it a sequence of actions and make it generate the frames
#     if generate:
#         print 'Generating'
        
#         # real: [B,T,X] gen:[T,K,B,X]
#         real_frames, gen_frames = dkf.run_generate(get_data=get_ball, path_to_load_variables=path_to_save_variables)


#         real_gif = []
#         gen_gif = []
#         for t in range(n_time_steps):

#             real_frame = real_frames[0][t]
#             real_frame = real_frame * (255. / np.max(real_frame))
#             real_frame = real_frame.astype('uint8')
#             real_gif.append(np.reshape(real_frame, [f_height,f_width, 1]))

#             gen_frame = gen_frames[t][0][0]
#             gen_frame = gen_frame * (255. / np.max(gen_frame))
#             gen_frame = gen_frame.astype('uint8')
#             gen_gif.append(np.reshape(gen_frame, [f_height,f_width, 1]))


#         real_gif = np.array(real_gif)
#         gen_gif = np.array(gen_gif)


#         kargs = { 'duration': .6 }
#         imageio.mimsave(save_to+'gen_gif.gif', gen_gif, 'GIF', **kargs)
#         print 'saved gif:' + save_to+'gen_gif.gif'

#         kargs = { 'duration': .6 }
#         imageio.mimsave(save_to+'real_gif.gif', real_gif, 'GIF', **kargs)
#         print 'saved gif:' +save_to+'real_gif.gif'

#     donedonedone












