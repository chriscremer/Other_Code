


#Hamiltonian Variational Inference

import numpy as np
import tensorflow as tf

from NN import NN


class HVI(object):

    def __init__(self, batch_size):
        

        #Model hyperparameters
        self.act_func = tf.nn.softplus



    def init_weights(self):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        W_means = []
        W_logvars = []

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Define variables [IS,OS]
            W_means.append(tf.Variable(xavier_init(input_size_i, output_size_i)))
            W_logvars.append(tf.Variable(xavier_init(input_size_i, output_size_i) - 10.))

        return W_means, W_logvars



    def sample_weights(self):

        Ws = []

        log_p_W_sum = 0
        log_q_W_sum = 0

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Get vars [I,O]
            W_means = self.W_means[layer_i]
            W_logvars = self.W_logvars[layer_i]

            #Sample weights [IS,OS]*[IS,OS]=[IS,OS]
            eps = tf.random_normal((input_size_i, output_size_i), 0, 1, seed=self.rs)
            W = tf.add(W_means, tf.multiply(tf.sqrt(tf.exp(W_logvars)), eps))

            #Compute probs of samples  [1]
            flat_w = tf.reshape(W,[input_size_i*output_size_i]) #[IS*OS]
            flat_W_means = tf.reshape(W_means, [input_size_i*output_size_i]) #[IS*OS]
            flat_W_logvars = tf.reshape(W_logvars, [input_size_i*output_size_i]) #[IS*OS]
            log_p_W_sum += log_normal3(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))
            log_q_W_sum += log_normal3(flat_w, flat_W_means, flat_W_logvars)

            Ws.append(W)

        return Ws, log_p_W_sum, log_q_W_sum



    def feedforward(self, W_list, x):
        '''
        W: list of layers weights
        x: [B,X]
        y: [B,Y]
        '''


        #[B,X]
        cur_val = x
        # #[B,X]->[B,1,X]
        # cur_val = tf.reshape(cur_val, [self.batch_size, 1, self.input_size])

        for layer_i in range(len(self.net)-1):

            #[X,X']
            W = W_list[layer_i]
            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Concat 1 to input for biases  [B,X]->[B,X+1]
            cur_val = tf.concat([cur_val,tf.ones([self.batch_size, 1])], axis=1)
            # #[X,X']->[B,X,X']
            # W = tf.reshape(W, [1, input_size_i, output_size_i])
            # W = tf.tile(W, [self.batch_size, 1,1])

            #Forward Propagate  [B,X]*[X,X']->[B,X']
            if layer_i != len(self.net)-2:
                cur_val = self.act_func(tf.matmul(cur_val, W))
            else:
                cur_val = tf.matmul(cur_val, W)

        # #[B,P,1,X']->[B,P,X']
        # cur_val = tf.reshape(cur_val, [self.batch_size,P,output_size_i])
        #[B,Y]
        y = cur_val

        return y














def leapfrogs(self, z, T, friction, step_size, x):


    v_0 = tf.random_normal((self.n_particles, self.batch_size, self.n_z), 0, 1, dtype=tf.float32)

    log_p = self._log_likelihood(x, self._generator_network(z, self.network_weights['decoder_weights'], self.network_weights['decoder_biases'])) + self._log_p_z(z)
    grad = -tf.gradients(log_p, [z])[0]

    v = v_0 - ((.5*step_size) * grad)
    z = z + (step_size * v)

    for t in range(T-1):

        log_p = self._log_likelihood(x, self._generator_network(z, self.network_weights['decoder_weights'], self.network_weights['decoder_biases'])) + self._log_p_z(z)
        grad = -tf.gradients(log_p, [z])[0]

        v = v - (step_size * grad)
        z = z + (step_size * v)

        v = friction * v

    log_p = self._log_likelihood(x, self._generator_network(z, self.network_weights['decoder_weights'], self.network_weights['decoder_biases'])) + self._log_p_z(z)
    grad = -tf.gradients(log_p, [z])[0]

    v = v - ((.5*step_size) * grad)

    return z, v_0, v


