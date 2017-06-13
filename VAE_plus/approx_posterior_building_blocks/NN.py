



#Neural Network

import tensorflow as tf
import numpy as np


class NN(object):

    def __init__(self, network_architecture, act_func, batch_size):
        
        #Model hyperparameters
        self.act_func = act_func
        self.input_size = network_architecture[0]
        self.output_size = network_architecture[-1]
        self.net = network_architecture
        # self.batch_size = batch_size

        self.Ws = self.init_weights()




    def init_weights(self):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        Ws = []

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Define variables [IS,OS]
            Ws.append(tf.Variable(xavier_init(input_size_i, output_size_i)))

        return Ws





    def weight_decay(self):

        l2 = 0
        for weight_layer in self.Ws:

            l2 += tf.reduce_sum(tf.square(weight_layer))

        return l2







    def feedforward(self, x):
        '''
        x: [B,X]
        y_hat: [B,O]
        '''

        B = tf.shape(x)[0]

        net = self.net

        # def xavier_init(fan_in, fan_out, constant=1): 
        #     """ Xavier initialization of network weights"""
        #     # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
        #     low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
        #     high = constant*np.sqrt(6.0/(fan_in + fan_out))
        #     return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        #[B,X]
        cur_val = x

        for layer_i in range(len(net)-1):

            W = self.Ws[layer_i]

            # input_size_i = net[layer_i]+1 #plus 1 for bias
            # output_size_i = net[layer_i+1] #plus 1 because we want layer i+1

            #Define variables [IS,OS]
            # W = tf.Variable(xavier_init(input_size_i, output_size_i))

            #Concat 1 to input for biases  [B,P,X]->[B,P,X+1]
            cur_val = tf.concat([cur_val,tf.ones([B, 1])], axis=1)

            if layer_i != len(net)-2:
                cur_val = self.act_func(tf.matmul(cur_val, W))
            else:
                cur_val = tf.matmul(cur_val, W)

        return cur_val















