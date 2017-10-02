

#Neural Network

import tensorflow as tf
import numpy as np


class NN(object):

    def __init__(self, network_architecture, act_func):
        
        #Model hyperparameters
        self.act_func = act_func
        self.input_size = network_architecture[0]
        self.output_size = network_architecture[-1]
        self.net = network_architecture

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






    def feedforward(self, x):
        '''
        x: [B,X]
        y_hat: [B,O]
        '''

        net = self.net

        batch_size = tf.shape(x)[0]   #B    

        #[B,X]
        cur_val = x

        for layer_i in range(len(net)-1):

            W = self.Ws[layer_i]

            #Concat 1 to input for biases  [B,P,X]->[B,P,X+1]
            cur_val = tf.concat([cur_val,tf.ones([batch_size, 1])], axis=1)

            cur_val = tf.matmul(cur_val, W)

            if layer_i != len(net)-2:
                cur_val = self.act_func(cur_val)
                

        return cur_val















