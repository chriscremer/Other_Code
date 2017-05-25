



#Neural Network

import tensorflow as tf
import numpy as np


class conv_NN(object):

    def __init__(self, x_size, z_size, encoder_convnet):
        
        #Model hyperparameters
        self.act_func = tf.tanh
        # self.input_size = network_architecture[0]
        # self.output_size = network_architecture[-1]
        self.net = [x_size, encoder_convnet[5], encoder_convnet[6], z_size*2] 
        # self.batch_size = batch_size

        # self.filter_height = 5
        # self.filter_width = 5
        # self.n_in_channels = 1
        # self.n_out_channels = 5
        # self.strides = 4

        self.filter_height = encoder_convnet[0]
        self.filter_width = encoder_convnet[1]
        self.n_in_channels = encoder_convnet[2]
        self.n_out_channels = encoder_convnet[3]
        self.strides = encoder_convnet[4]

        self.Ws = self.init_weights()

        # Check model size
        total_size = 0
        for v in tf.trainable_variables():
                total_size += np.prod([int(s) for s in v.get_shape()])
                print(v.get_shape())
        print("Total number of trainable variables in encoder: {}".format(total_size))


        # total_size = 0
        # for v in tf.trainable_variables():
        #         total_size += np.prod([int(s) for s in v.get_shape()])
        # print("Total number of trainable variables: {}".format(total_size))


    def init_weights(self):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        Ws = []

        conv_weights = tf.Variable(tf.truncated_normal([self.filter_height, 
                                                        self.filter_width, 
                                                        self.n_in_channels, 
                                                        self.n_out_channels], stddev=0.1))


        Ws.append(conv_weights)

        for layer_i in range(1,len(self.net)-1):

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

        batch_size = tf.shape(x)[0]

        net = self.net

        # def xavier_init(fan_in, fan_out, constant=1): 
        #     """ Xavier initialization of network weights"""
        #     # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
        #     low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
        #     high = constant*np.sqrt(6.0/(fan_in + fan_out))
        #     return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        #[B,X]
        cur_val = x

        cur_val = tf.reshape(cur_val, [batch_size, 28 ,28 , 1])
        # print cur_val
        cur_val = tf.nn.conv2d(cur_val,self.Ws[0],strides=[1, 4, 4, 1],padding='VALID')
        # print cur_val
        cur_val = tf.reshape(cur_val, [batch_size, -1])
        # print cur_val


        for layer_i in range(1, len(net)-1):

            W = self.Ws[layer_i]

            # input_size_i = net[layer_i]+1 #plus 1 for bias
            # output_size_i = net[layer_i+1] #plus 1 because we want layer i+1

            #Define variables [IS,OS]
            # W = tf.Variable(xavier_init(input_size_i, output_size_i))

            #Concat 1 to input for biases  [B,P,X]->[B,P,X+1]
            cur_val = tf.concat([cur_val,tf.ones([batch_size, 1])], axis=1)

            if layer_i != len(net)-2:
                cur_val = self.act_func(tf.matmul(cur_val, W))
            else:
                cur_val = tf.matmul(cur_val, W)

        return cur_val















