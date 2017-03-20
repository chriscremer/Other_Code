



import numpy as np
import tensorflow as tf




class RNN():

    def __init__(self, network_architecture, batch_size):
        
        # tf.reset_default_graph()

        self.batch_size = batch_size
        self.transfer_fct=  tf.nn.softplus  #tf.nn.relu #tf.nn.softplus #tf.tanh
        self.learning_rate = 0.000001
        self.reg_param = .000001

        self.network_architecture = network_architecture
        self.z_size = network_architecture["n_z"]
        self.input_size = network_architecture["n_input"]
        self.action_size = network_architecture["n_actions"]
        self.reward_size = network_architecture["reward_size"]

        # Graph Input: [B,T,X], [B,T,A]
        # Graph Input: [T,B,X], [T,B,A]
        with tf.name_scope('model_input'):
            self.x = tf.placeholder(tf.float32, [None, self.batch_size, self.input_size], name='Observations_B_T_X')
            self.actions = tf.placeholder(tf.float32, [None, self.batch_size, self.action_size], name='Actions_B_T_A')
            self.rewards = tf.placeholder(tf.float32, [None, self.batch_size, self.reward_size], name='Rewards_B_T_R')
            self.sequence_lengths = tf.placeholder(tf.int32, [self.batch_size], name='Seq_length')

        
        #Variables
        self.params_dict, self.params_list = self._initialize_weights(network_architecture)

        #Objective
        with tf.name_scope('model_objective'):
            with tf.name_scope('model_elbo'):
                self.elbo = self.Model(self.x, self.actions, self.rewards)

            self.cost = -self.elbo + (self.reg_param * self.l2_regularization())

        #Optimize
        with tf.name_scope('model_optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost, var_list=self.params_list)




        #Evaluation
        with tf.name_scope('for_testing_model'):
            # self.prev_z_and_current_a_ = tf.placeholder(tf.float32, [self.batch_size, self.z_size+self.action_size])
            self.prev_z_ = tf.placeholder(tf.float32, [self.batch_size, self.z_size])
            self.current_a_ = tf.placeholder(tf.float32, [self.batch_size, self.action_size])

            prev_z_and_current_a_ = tf.concat(1, [self.prev_z_, self.current_a_])
            self.next_state_mean, self.next_state_logvar = self.transition_net(prev_z_and_current_a_)

            self.current_z_ = tf.placeholder(tf.float32, [self.batch_size, self.z_size])
            obs, obs_log_var, reward_mean, reward_log_var = self.observation_net(self.current_z_)
            # self.current_emission = tf.sigmoid(obs)
            self.current_emission = tf.sigmoid(obs), reward_mean


            self.prior_mean_ = tf.placeholder(tf.float32, [self.batch_size, self.z_size])
            self.prior_logvar_ = tf.placeholder(tf.float32, [self.batch_size, self.z_size])
            eps = tf.random_normal((self.batch_size, self.z_size), 0, 1, dtype=tf.float32)
            self.sample = tf.add(self.prior_mean_, tf.mul(tf.sqrt(tf.exp(self.prior_logvar_)), eps))

            #for testing policy
            #ENCODE
            self.current_observation = tf.placeholder(tf.float32, [self.batch_size, self.input_size])
            # self.prev_z_ = tf.placeholder(tf.float32, [self.batch_size, self.z_size])
            # self.current_a_ = tf.placeholder(tf.float32, [self.batch_size, self.action_size])
            #Concatenate current x, current action, prev_z: [B,XA+Z]
            # concatenate_all = tf.concat(1, [self.current_observation, self.current_a_])
            concatenate_all = tf.concat(1, [self.current_observation, self.prev_z_])
            #Predict q(z|z-1,u,x): [B,Z] [B,Z]
            self.z_mean_, z_log_var = self.recognition_net(concatenate_all)




    def _initialize_weights(self, network_architecture):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


        params_dict = dict()

        # ENCODER
        with tf.name_scope('encoder_vars'):

            # params_dict['conv_weights'] = tf.Variable(tf.truncated_normal([self.filter_height, self.filter_width, self.n_channels, self.filter_out_channels1], stddev=0.1))
            # params_dict['conv_biases'] = tf.Variable(tf.truncated_normal([self.filter_out_channels1], stddev=0.1))

            for layer_i in range(len(network_architecture['encoder_net'])):
                if layer_i == 0:
                    params_dict['encoder_weights_l'+str(layer_i)] = tf.Variable(xavier_init(self.input_size+self.z_size, network_architecture['encoder_net'][layer_i]))
                    params_dict['encoder_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))
                else:
                    params_dict['encoder_weights_l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['encoder_net'][layer_i-1], network_architecture['encoder_net'][layer_i]))
                    params_dict['encoder_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

            params_dict['encoder_weights_out_mean'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.z_size))
            params_dict['encoder_biases_out_mean'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))


        # DECODER
        with tf.name_scope('decoder_vars'):

            for layer_i in range(len(network_architecture['decoder_net'])):
                if layer_i == 0:
                    params_dict['decoder_weights_l'+str(layer_i)] = tf.Variable(xavier_init(self.z_size, network_architecture['decoder_net'][layer_i]))
                    params_dict['decoder_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['decoder_net'][layer_i]], dtype=tf.float32))
                else:
                    params_dict['decoder_weights_l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['decoder_net'][layer_i-1], network_architecture['decoder_net'][layer_i]))
                    params_dict['decoder_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

            params_dict['decoder_weights_out_mean'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.input_size))
            params_dict['decoder_biases_out_mean'] = tf.Variable(tf.zeros([self.input_size], dtype=tf.float32))

            params_dict['decoder_weights_reward_mean'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.reward_size))
            params_dict['decoder_biases_reward_mean'] = tf.Variable(tf.zeros([self.reward_size], dtype=tf.float32))


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



    def encoder(self, input_):
        # input:[B,X]
        # output: [B,E]  #E is the size of the encoded image, so its different than Z

        with tf.name_scope('encoder'):

            n_layers = len(self.network_architecture['encoder_net'])

            for layer_i in range(n_layers):
                input_ = self.transfer_fct(tf.contrib.layers.layer_norm(tf.add(tf.matmul(input_, self.params_dict['encoder_weights_l'+str(layer_i)]), self.params_dict['encoder_biases_l'+str(layer_i)])))
                
            z_mean = tf.add(tf.matmul(input_, self.params_dict['encoder_weights_out_mean']), self.params_dict['encoder_biases_out_mean'])

        return z_mean


    def decoder(self, input_):
        # input:[B,Z]
        # output: [B,X] and [B,R]

        with tf.name_scope('decoder'):

            n_layers = len(self.network_architecture['decoder_net'])

            for layer_i in range(n_layers):
                input_ = self.transfer_fct(tf.contrib.layers.layer_norm(tf.add(tf.matmul(input_, self.params_dict['decoder_weights_l'+str(layer_i)]), self.params_dict['decoder_biases_l'+str(layer_i)])))

            x_mean = tf.add(tf.matmul(input_, self.params_dict['decoder_weights_out_mean']), self.params_dict['decoder_biases_out_mean'])

            reward_mean = tf.add(tf.matmul(input_, self.params_dict['decoder_weights_reward_mean']), self.params_dict['decoder_biases_reward_mean'])

        return x_mean, reward_mean




    def transition(self, input_):
        # input:[B,Z+A+E]
        # output: [B,]

        with tf.name_scope('transition_net'):

            n_layers = len(self.network_architecture['trans_net'])
            # weights = self.network_weights['trans_weights']
            # biases = self.network_weights['trans_biases']

            for layer_i in range(n_layers):

                # input_ = tf.contrib.layers.layer_norm(input_)
                input_ = self.transfer_fct(tf.contrib.layers.layer_norm(tf.add(tf.matmul(input_, self.params_dict['trans_weights_l'+str(layer_i)]), self.params_dict['trans_biases_l'+str(layer_i)])))
                #add batch norm here


            z_mean = tf.add(tf.matmul(input_, self.params_dict['trans_weights_out_mean']), self.params_dict['trans_biases_out_mean'])
            z_log_var = tf.add(tf.matmul(input_, self.params_dict['trans_weights_out_log_var']), self.params_dict['trans_biases_out_log_var'])

        return z_mean, z_log_var

            



    #For now, I want to use their methods. So ill use a GRU on the frames and actions



    def objective(self, states):

    	#go over timesteps, usign scan, and calc dif between prediction and real frames
    	# also convert the state to predictions 






















