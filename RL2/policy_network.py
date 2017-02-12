


import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")



class Policy():

    def __init__(self, network_architecture, model, batch_size, n_particles, n_timesteps):
        
        self.model = model

        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.transfer_fct=tf.nn.softplus #tf.tanh
        self.n_particles = n_particles
        self.learning_rate = 0.001
        self.reg_param = .00001

        self.network_architecture = network_architecture
        self.input_size = network_architecture["input_size"]
        self.z_size = network_architecture["z_size"]
        self.action_size = network_architecture["action_size"]

        # Graph Input: [B,T,Z]
        # self.input = tf.placeholder(tf.float32, [None, None, self.z_size])
        
        #Variables
        self.params_dict, self.params_list = self._initialize_weights(network_architecture)

        #Objective
        self.objective = self.J_equation()
        self.cost = -self.objective + (self.reg_param * self.l2_regularization())

        #Optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost, var_list=self.params_list)

        #Evaluate
        self.state_ = tf.placeholder(tf.float32, [self.batch_size, self.z_size])
        self.action_ = self.predict_action(self.state_)


    def _initialize_weights(self, network_architecture):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


        params_dict = dict()

        #Recognition/Inference net q(z|z-1,u,x)
        # params_dict['policy_weights'] = {}
        # params_dict['policy_biases'] = {}

        for layer_i in range(len(network_architecture['policy_net'])):
            if layer_i == 0:
                params_dict['policy_weights_l'+str(layer_i)] = tf.Variable(xavier_init(self.z_size, network_architecture['policy_net'][layer_i]))
                params_dict['policy_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['policy_net'][layer_i]], dtype=tf.float32))
            else:
                params_dict['policy_weights_l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['policy_net'][layer_i-1], network_architecture['policy_net'][layer_i]))
                params_dict['policy_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['policy_net'][layer_i]], dtype=tf.float32))

        params_dict['policy_weights_out_mean'] = tf.Variable(xavier_init(network_architecture['policy_net'][-1], self.action_size))
        params_dict['policy_biases_out_mean'] = tf.Variable(tf.zeros([self.action_size], dtype=tf.float32))


        params_list = []
        for layer in params_dict:

            params_list.append(params_dict[layer])

        return params_dict, params_list


    def l2_regularization(self):

        sum_ = 0
        for layer in self.params_dict:

            sum_ += tf.reduce_sum(tf.square(self.params_dict[layer]))

        return sum_




    def predict_action(self, prev_z):
        '''
        prev_z: [B, Z]
        return: [B, A]

        needs to be one hot because model has only seen one hot actions
        '''

        n_layers = len(self.network_architecture['policy_net'])
        # weights = self.params_dict['policy_weights']
        # biases = self.params_dict['policy_biases']

        input_ = prev_z

        for layer_i in range(n_layers):

            input_ = self.transfer_fct(tf.add(tf.matmul(input_, self.params_dict['policy_weights_l'+str(layer_i)]), self.params_dict['policy_biases_l'+str(layer_i)])) 

        action = tf.add(tf.matmul(input_, self.params_dict['policy_weights_out_mean']), self.params_dict['policy_biases_out_mean'])

        action = tf.argmax(action, axis=1) 

        action = tf.one_hot(indices=action, depth=self.action_size, axis=None)

        return action





    def reward_func(self, observation_t):
        '''
        observation_t: [B,X]
        action_tL [B,A], action isnt used for now, but it could be later
            - some actions could cost more than otehrs for example
        '''

        # forw = tf.range(tf.shape(observation_t)[1]/2, dtype='float32') 
        # #if this doesnt work, can make a tensor of zeros of length X, then call shape on it.

        # # back = tf.reverse_v2(tf.range(tf.shape(observation_t)[1]/2), axis=0)
        # back = tf.range(tf.shape(observation_t)[1]/2, dtype='float32') 

        # # [X]
        # reward_matrix = tf.concat(0, [forw, back])
        reward_matrix = tf.ones([self.input_size])


        range_f = np.array([float(x) for x in range(0,self.input_size/2)], dtype='float32')
        range_r = np.array([float(x) for x in range(0,self.input_size/2)], dtype='float32')[::-1]
        r_mat = np.concatenate([range_f, range_r], axis=0)
        # print r_mat
        reward_matrix = tf.pack(r_mat)

        reward_matrix = tf.reshape(reward_matrix, [1, self.input_size])

        rewards = tf.reduce_sum(tf.multiply(tf.sigmoid(observation_t),reward_matrix), axis=1)

        return rewards #of the batch? [B]



    # def policy_objective(self, generated_observations):
    #     '''
    #     generated_observations: [T,B,PX]
    #     '''

    #     def fn_over_timesteps(prev_reward, current_observation):
    #         '''
    #         prev_reward: [B] not used here
    #         current_observation: [B,X]

    #         return [B]
    #         '''

    #         particle_rewards = []

    #         for i in range(self.n_particles):

    #             #select particle [B,X]
    #             this_particle_obs = tf.slice(current_observation, [0,i*self.input_size], [self.batch_size, self.input_size])

    #             #calc reward [B]
    #             this_reward = self.reward_func(this_particle_obs)

    #             #save it 
    #             particle_rewards.append(this_reward)

    #         #Average the particles, goest from [P,B] to [B]
    #         avg_particle_reward = tf.reduce_mean(tf.pack(particle_rewards), axis=0) 

    #         return avg_particle_reward




    #     #Make initializations for scan
    #     reward_init = tf.zeros([self.batch_size])

    #     # Scan over timesteps, returns [T,B]
    #     rewards = tf.scan(fn_over_timesteps, generated_observations, initializer=reward_init) 

    #     #Average over batch [T]
    #     rewards = tf.reduce_mean(rewards, axis=1)

    #     #Sum over timesteps [1]
    #     reward = tf.reduce_sum(rewards, axis=0)

    #     #return scalar
    #     return reward



    def J_equation(self):
        #Steps:
            #Generate states using model and policy actions
            #Get emission of states
            #Get sum of rewards 

        #Generate states using model and policy actions
            # - could use scan on observations instead of looping over every timestep
            # this avoids making large graph
            # observations wouldnt be used
            # I should combine this with reward calc, to speed it up
            #
            # rewards = []
            # for t in timesteps
            #   a_t = policy(z_t-1)
            #   z_t = model(a_t, z_t-1)
            #   x_t = model(z_t)
            #   r_t = calc_reward(x_t)
            #   rewards.append(r_t)
            # sum_rewards

        def fn_over_timesteps(prev_reward_and_z, current_scan_over):
            '''
            prev_reward_and_z: [B, 1+Z]
            current_scan_over: [0]

            return [B, 1+Z]
            '''

            particles = []
            particle_rewards = []

            for i in range(self.n_particles):

                #select particle [B,Z]
                z_t_minus1 = tf.slice(prev_reward_and_z, [0,1+(i*self.z_size)], [self.batch_size, self.z_size])

                #use policy to get action [B,A]
                action = self.predict_action(z_t_minus1)

                #use model to get new state [B,Z]
                z_mean, z_log_var = self.model.transition_net(tf.concat(1,[z_t_minus1, action]))
                eps = tf.random_normal((self.batch_size, self.z_size), 0, 1, dtype=tf.float32)
                z_t = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_var)), eps))

                #use model to get emission of state [B,X]
                x_t = self.model.observation_net(z_t)

                #calc reward [B]
                r_t = self.reward_func(x_t)

                #save it 
                particles.append(z_t)
                particle_rewards.append(r_t)

            #Average the particles, goest from [P,B] to [B]
            avg_particle_reward = tf.reduce_mean(tf.pack(particle_rewards), axis=0) 
            avg_particle_reward = tf.reshape(avg_particle_reward, [self.batch_size, 1])

            new_particles = tf.pack(particles, axis=1) #[B,P,Z]
            new_particles = tf.reshape(new_particles, [self.batch_size, self.n_particles*self.z_size]) #[B,PZ]
            output = tf.concat(1, [avg_particle_reward, new_particles])

            return output


        #Make initializations for scan
        reward_init = tf.zeros([self.batch_size, 1])
        z_init = tf.zeros([self.batch_size, self.z_size*self.n_particles])
        initialization = tf.concat(1, [reward_init, z_init])

        scan_over = tf.zeros([self.n_timesteps])

        # Scan over timesteps, returns [T,B,1+PZ]
        rewards_and_particles = tf.scan(fn_over_timesteps, scan_over, initializer=initialization) 

        #Unpack rewards list([1+PZ,T,B])
        rewards = tf.unstack(rewards_and_particles, axis=2)

        #[T,B]
        rewards = rewards[0]

        #Average over batch [T]
        rewards = tf.reduce_mean(rewards, axis=1)

        #Sum over timesteps [1]
        reward = tf.reduce_sum(rewards, axis=0)

        #return scalar
        return reward























