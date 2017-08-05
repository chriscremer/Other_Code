




#Bayesian Neural Network

import numpy as np
import numpy.random as npr
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle

from utils import log_normal3
from utils import log_normal2



class NVP(object):

    def __init__(self, network_architecture, act_func):
        

        #Model hyperparameters
        self.act_func = act_func
        # self.learning_rate = .0001
        self.rs = 0
        self.input_size = network_architecture[0]
        self.output_size = network_architecture[-1]
        self.net = network_architecture
        # self.batch_size = batch_size
        # self.n_particles = n_particles
        # self.batch_fraction_of_dataset = batch_frac
        # print 'bnn'
        # self.W_means, self.W_logvars = self.init_weights()

        self.W_means, self.W_logvars, self.z_means, self.z_logvars, self.fgk, self.cbb, self.fgk2 = self.init_weights()


    # def init_weights(self):

    #     # def xavier_init(fan_in, fan_out, constant=1): 
    #     #     """ Xavier initialization of network weights"""
    #     #     # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    #     #     low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    #     #     high = constant*np.sqrt(6.0/(fan_in + fan_out))
    #     #     return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    #     W_means = []
    #     W_logvars = []

    #     for layer_i in range(len(self.net)-1):

    #         input_size_i = self.net[layer_i]+1 #plus 1 for bias
    #         output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

    #         #Define variables [IS,OS]
    #         # W_means.append(tf.Variable(xavier_init(input_size_i, output_size_i)))
    #         W_means.append(tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1)))

    #         # W_logvars.append(tf.Variable(xavier_init(input_size_i, output_size_i) - 10.))
    #         W_logvars.append(tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1))-5.)

    #     return W_means, W_logvars



    def init_weights(self):

        W_means = []
        W_logvars = []

        z_means = []
        z_logvars = []

        fgk = []

        cbb = []

        fgk2 = []

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            W_means.append(tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1)))
            W_logvars.append(tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1))-5.)

            z_means.append(tf.Variable(tf.random_normal([input_size_i], stddev=0.1)))
            z_logvars.append(tf.Variable(tf.random_normal([input_size_i], stddev=0.1) - 5.))


            f = tf.Variable(tf.random_normal([input_size_i, 30], stddev=0.1))
            g = tf.Variable(tf.random_normal([30, input_size_i], stddev=0.1))
            k = tf.Variable(tf.random_normal([30, input_size_i], stddev=0.1))

            fgk.append([f,g,k])


            c = tf.Variable(tf.random_normal([1, input_size_i], stddev=0.1)) #[1,I]
            b1 = tf.Variable(tf.random_normal([input_size_i,1], stddev=0.1)) #[I,1]
            b2 = tf.Variable(tf.random_normal([input_size_i,1], stddev=0.1)) #[I,1]

            cbb.append([c,b1,b2])


            f2 = tf.Variable(tf.random_normal([input_size_i, 30], stddev=0.1))
            g2 = tf.Variable(tf.random_normal([30, input_size_i], stddev=0.1))
            k2 = tf.Variable(tf.random_normal([30, input_size_i], stddev=0.1))

            fgk2.append([f2,g2,k2])


        return W_means, W_logvars, z_means, z_logvars, fgk, cbb, fgk2



    def random_bernoulli(self, shape, p=0.5):
        if isinstance(shape, (list, tuple)):
            shape = tf.stack(shape)
        return tf.where(tf.random_uniform(shape) < p, tf.ones(shape), tf.zeros(shape))



    def sample_weights(self):

        Ws = []
        zs = []

        log_p_W_sum = 0
        log_q_W_sum = 0
        log_q_z_sum = 0
        log_r_z_sum = 0

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1


            #Sample z   [I]
            eps = tf.random_normal([input_size_i], 0, 1, seed=self.rs)
            z0 = tf.add(self.z_means[layer_i], tf.multiply(tf.sqrt(tf.exp(self.z_logvars[layer_i])), eps))
            z0 = tf.reshape(z0,[1,input_size_i]) #[1,I]
            log_q_z_sum += log_normal2(z0, self.z_means[layer_i], self.z_logvars[layer_i]) #[1]

            # Transform z0
            z = z0
            #Flows z0 -> zT  Right now its only a single flow, 
            #should allow it to be more. Should use similar code to original MNF
            mask = self.random_bernoulli(tf.shape(z), p=0.5)
            h = tf.matmul((mask * z), self.fgk[layer_i][0])  #[1,30]
            h = tf.tanh(h)
            mew_ = tf.matmul(h,self.fgk[layer_i][1])  #[1,I]
            sig_ = tf.nn.sigmoid(tf.matmul(h,self.fgk[layer_i][2]))  #[1,I]
            # zT
            zT = (mask * z) + (1-mask)*(z*sig_ + (1-sig_)*mew_)
            zT = tf.reshape(zT, [input_size_i,1])  #[I,1]

            logdet = tf.reduce_sum((1-mask)*tf.log(sig_), axis=1)
            log_q_z_sum -= logdet


            # Multiply mean by zT [I,O]
            W_means = self.W_means[layer_i] * zT #broadcast  [I,O]*[I,1] = [I,O]

            #Sample weights [IS,OS]*[IS,OS]=[IS,OS]
            eps = tf.random_normal((input_size_i, output_size_i), 0, 1, seed=self.rs)
            W = tf.add(W_means, tf.multiply(tf.sqrt(tf.exp(self.W_logvars[layer_i])), eps))


            # r(zT|W)
            cW = tf.tanh(tf.matmul(self.cbb[layer_i][0], W)) # [1,I]*[I,O] =[1,O]
            b1cW = tf.matmul(self.cbb[layer_i][1], cW) #[I,1]*[1,O]=[I,O]
            ones = tf.ones([output_size_i, 1]) / tf.to_float(output_size_i) #[O,1]
            b1cW = tf.matmul(b1cW, ones) #[I,O]*[O,1]=[I,1]
            b1cW = tf.reshape(b1cW, [input_size_i]) #[I]

            b2cW = tf.matmul(self.cbb[layer_i][2], cW) #[I,1]*[1,O]=[I,O] 
            b2cW = tf.matmul(b2cW, ones) #[I,O]*[O,1]=[I,1]
            b2cW = tf.reshape(b2cW, [input_size_i]) #[I]


            #Flows zT -> zB
            zT = tf.reshape(zT, [1,input_size_i])  #[1,I]
            mask = self.random_bernoulli(tf.shape(zT), p=0.5)
            h = tf.matmul((mask * zT), self.fgk2[layer_i][0])  #[1,30]
            h = tf.tanh(h)
            mew_ = tf.matmul(h,self.fgk2[layer_i][1])  #[1,I]
            sig_ = tf.nn.sigmoid(tf.matmul(h,self.fgk2[layer_i][2]))  #[1,I]

            zB = (mask * zT) + (1-mask)*(zT*sig_ + (1-sig_)*mew_)
            logdet = tf.reduce_sum((1-mask)*tf.log(sig_), axis=1)

            log_r_z_sum += log_normal2(zB, b1cW, b2cW) #[1]
            log_r_z_sum += logdet


            #Compute probs of samples  [1]
            flat_w = tf.reshape(W,[input_size_i*output_size_i]) #[IS*OS]
            flat_W_means = tf.reshape(W_means, [input_size_i*output_size_i]) #[IS*OS]
            flat_W_logvars = tf.reshape(self.W_logvars[layer_i], [input_size_i*output_size_i]) #[IS*OS]
            
            log_p_W_sum += log_normal3(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))
            log_q_W_sum += log_normal3(flat_w, flat_W_means, flat_W_logvars)

            Ws.append(W)
            zs.append(zT)


        log_p_W_sum = log_p_W_sum + log_r_z_sum
        log_q_W_sum = log_q_W_sum + log_q_z_sum


        return Ws, log_p_W_sum, log_q_W_sum








 
    def sample_weight_means(self):

        Ws = []
        zs = []

        log_p_W_sum = 0
        log_q_W_sum = 0
        log_q_z_sum = 0
        log_r_z_sum = 0

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1


            #Sample z   [I]
            eps = tf.random_normal([input_size_i], 0, 1, seed=self.rs)
            z0 = tf.add(self.z_means[layer_i], tf.multiply(tf.sqrt(tf.exp(self.z_logvars[layer_i])), eps))
            z0 = tf.reshape(z0,[1,input_size_i]) #[1,I]
            log_q_z_sum += log_normal2(z0, self.z_means[layer_i], self.z_logvars[layer_i]) #[1]

            # Transform z0
            z = z0
            #Flows z0 -> zT  Right now its only a single flow, 
            #should allow it to be more. Should use similar code to original MNF
            mask = self.random_bernoulli(tf.shape(z), p=0.5)
            h = tf.matmul((mask * z), self.fgk[layer_i][0])  #[1,30]
            h = tf.tanh(h)
            mew_ = tf.matmul(h,self.fgk[layer_i][1])  #[1,I]
            sig_ = tf.nn.sigmoid(tf.matmul(h,self.fgk[layer_i][2]))  #[1,I]
            # zT
            zT = (mask * z) + (1-mask)*(z*sig_ + (1-sig_)*mew_)
            zT = tf.reshape(zT, [input_size_i,1])  #[I,1]

            logdet = tf.reduce_sum((1-mask)*tf.log(sig_), axis=1)
            log_q_z_sum -= logdet


            # Multiply mean by zT [I,O]
            W_means = self.W_means[layer_i] * zT #broadcast  [I,O]*[I,1] = [I,O]

            #Sample weights [IS,OS]*[IS,OS]=[IS,OS]
            eps = tf.random_normal((input_size_i, output_size_i), 0, 1, seed=self.rs)
            W = tf.add(W_means, tf.multiply(tf.sqrt(tf.exp(self.W_logvars[layer_i])), eps))


            # r(zT|W)
            cW = tf.tanh(tf.matmul(self.cbb[layer_i][0], W)) # [1,I]*[I,O] =[1,O]
            b1cW = tf.matmul(self.cbb[layer_i][1], cW) #[I,1]*[1,O]=[I,O]
            ones = tf.ones([output_size_i, 1]) / tf.to_float(output_size_i) #[O,1]
            b1cW = tf.matmul(b1cW, ones) #[I,O]*[O,1]=[I,1]
            b1cW = tf.reshape(b1cW, [input_size_i]) #[I]

            b2cW = tf.matmul(self.cbb[layer_i][2], cW) #[I,1]*[1,O]=[I,O] 
            b2cW = tf.matmul(b2cW, ones) #[I,O]*[O,1]=[I,1]
            b2cW = tf.reshape(b2cW, [input_size_i]) #[I]


            #Flows zT -> zB
            zT = tf.reshape(zT, [1,input_size_i])  #[1,I]
            mask = self.random_bernoulli(tf.shape(zT), p=0.5)
            h = tf.matmul((mask * zT), self.fgk2[layer_i][0])  #[1,30]
            h = tf.tanh(h)
            mew_ = tf.matmul(h,self.fgk2[layer_i][1])  #[1,I]
            sig_ = tf.nn.sigmoid(tf.matmul(h,self.fgk2[layer_i][2]))  #[1,I]

            zB = (mask * zT) + (1-mask)*(zT*sig_ + (1-sig_)*mew_)
            logdet = tf.reduce_sum((1-mask)*tf.log(sig_), axis=1)

            log_r_z_sum += log_normal2(zB, b1cW, b2cW) #[1]
            log_r_z_sum += logdet


            #Compute probs of samples  [1]
            flat_w = tf.reshape(W,[input_size_i*output_size_i]) #[IS*OS]
            flat_W_means = tf.reshape(W_means, [input_size_i*output_size_i]) #[IS*OS]
            flat_W_logvars = tf.reshape(self.W_logvars[layer_i], [input_size_i*output_size_i]) #[IS*OS]
            
            log_p_W_sum += log_normal3(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))
            log_q_W_sum += log_normal3(flat_w, flat_W_means, flat_W_logvars)

            Ws.append(W)
            zs.append(zT)


        log_p_W_sum = log_p_W_sum + log_r_z_sum
        log_q_W_sum = log_q_W_sum + log_q_z_sum


        return Ws

        return Ws











    def feedforward(self, x, W_list):
        '''
        W: list of layers weights
        x: [B,X]
        y: [B,Y]
        '''



        B = tf.shape(x)[0]


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
            # cur_val = tf.concat([cur_val,tf.ones([tf.shape(cur_val)[0], 1])], axis=1)

            cur_val = tf.concat([cur_val,tf.ones([B, 1])], axis=1)


            # #[X,X']->[B,X,X']
            # W = tf.reshape(W, [1, input_size_i, output_size_i])
            # W = tf.tile(W, [self.batch_size, 1,1])


            # #Forward Propagate  [B,X]*[X,X']->[B,X']
            # if layer_i != len(self.net)-2:
            #     cur_val = self.act_func(tf.matmul(cur_val, W))
            # else:
            #     cur_val = tf.matmul(cur_val, W)

            cur_val = tf.matmul(cur_val, W)

            if self.act_func[layer_i] != None:
                cur_val = self.act_func[layer_i](cur_val)




        # #[B,P,1,X']->[B,P,X']
        # cur_val = tf.reshape(cur_val, [self.batch_size,P,output_size_i])
        #[B,Y]
        y = cur_val

        return y











