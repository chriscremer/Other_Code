
import numpy as np
import tensorflow as tf

class REINFORCE():


    def __init__(self, network_architecture):
        
        tf.reset_default_graph()

        self.sess = tf.Session()

        self.learning_rate = .0001
        self.discount_factor = .99
        self.reg_param = 0.001

        self.input_size = network_architecture["n_input"]
        self.action_size = network_architecture["n_actions"]
        self.all_rewards = [] #used to normalize the rewards
        self.max_reward_length = 100000

        #Inputs
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.taken_actions = tf.placeholder(tf.int32, (None))
        self.discounted_rewards = tf.placeholder(tf.float32, (None))   

        #Variables
        self.network_weights = self._initialize_weights(network_architecture)

        # print self.network_weights['policy_weights']['l0']

        #Actions
        self.logprobs = self.policy_net(self.x)

        self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logprobs, self.taken_actions)
        self.pg_loss            = tf.reduce_mean(self.cross_entropy_loss) #over action dimensions
        self.reg_loss           = tf.reduce_sum([tf.reduce_sum(tf.square(self.network_weights['policy_weights'][x])) for x in self.network_weights['policy_weights']])
        self.reg_loss2           = tf.reduce_sum([tf.reduce_sum(tf.square(self.network_weights['policy_biases'][x])) for x in self.network_weights['policy_biases']])

        self.loss               = self.pg_loss + self.reg_param * (self.reg_loss + self.reg_loss2)



        # Policy Gradients
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
        self.gradients = self.optimizer.compute_gradients(self.loss)

        for i, (grad, var) in enumerate(self.gradients):
            if grad is not None:
              self.gradients[i] = (grad * self.discounted_rewards, var)

        self.train_op = self.optimizer.apply_gradients(self.gradients)


    def _initialize_weights(self, network_architecture):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


        #Policy network
        all_weights = dict()
        all_weights['policy_weights'] = {}
        all_weights['policy_biases'] = {}

        for layer_i in range(len(network_architecture['policy_net'])):
            if layer_i == 0:
                all_weights['policy_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.input_size, network_architecture['policy_net'][layer_i]))
                all_weights['policy_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['policy_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['policy_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['policy_net'][layer_i-1], network_architecture['policy_net'][layer_i]))
                all_weights['policy_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['policy_net'][layer_i]], dtype=tf.float32))

        all_weights['policy_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['policy_net'][-1], self.action_size))
        all_weights['policy_biases']['out_mean'] = tf.Variable(tf.zeros([self.action_size], dtype=tf.float32))


        return all_weights

    def policy_net(self, input_):

        #outputs the logprobs

        n_layers = len(self.network_weights['policy_weights']) - 1 #minus 1 for the mean outputs
        weights = self.network_weights['policy_weights']
        biases = self.network_weights['policy_biases']

        for layer_i in range(n_layers):

            input_ = tf.tanh(tf.add(tf.matmul(input_, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 

        logprob = tf.add(tf.matmul(input_, weights['out_mean']), biases['out_mean'])

        return logprob



    def sampleAction(self, input_):

        def softmax(y):
            """ simple helper function here that takes unnormalized logprobs """
            maxy = np.amax(y)
            e = np.exp(y - maxy)
            return e / np.sum(e)

        logprobs = self.sess.run(self.logprobs, feed_dict={self.x: input_})

        # print logprobs.shape
        logprobs = np.reshape(logprobs, [self.action_size])
        logprobs = softmax(logprobs)
        # logprobs[0] += 1.- np.sum(logprobs)
        # logprobs /= np.sum(logprobs)

        action = np.random.choice(self.action_size, p=logprobs)

        return action




    def update_policy(self, states, actions, rewards):

        T = len(states)
        r = 0 # use discounted reward to approximate Q value

        #convert actions to one hot
        # actions_ = np.zeros((T, self.action_size))
        # actions_[np.arange(T), actions] = 1
        # actions = actions_

        # compute discounted future rewards
        discounted_rewards = np.zeros(T)
        for t in reversed(xrange(T-1)):
          # future discounted reward from now on
          r = rewards[t] + self.discount_factor * r
          discounted_rewards[t] = r

        # reduce gradient variance by normalization
        # self.all_rewards += discounted_rewards.tolist()
        self.all_rewards =  discounted_rewards.tolist() + self.all_rewards
        self.all_rewards = self.all_rewards[:self.max_reward_length]
        discounted_rewards -= np.mean(self.all_rewards)
        discounted_rewards /= np.std(self.all_rewards)

        # # update policy network with the rollout in batches
        # for t in xrange(N-1):

        #   # prepare inputs
        #   states  = self.state_buffer[t][np.newaxis, :]
        #   actions = np.array([self.action_buffer[t]])
        #   rewards = np.array([discounted_rewards[t]])

        #   # perform one update of training
        #   _ = self.session.run(self.train_op)



        # print discounted_rewards

        #Train, one at a time because need to multiply gradient individualy
        for t in range(T):
            _ = self.sess.run(self.train_op, feed_dict={self.x: [states[t]], self.taken_actions: [actions[t]], self.discounted_rewards: [discounted_rewards[t]]})




    def initialize_sess(self, path_to_load_variables=''):

        self.sess = tf.Session()

        if path_to_load_variables == '':
            self.sess.run(tf.global_variables_initializer())




















