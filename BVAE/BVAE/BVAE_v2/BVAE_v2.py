

#Bayesian VAE

import numpy as np
import numpy.random as npr
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle

from BNN import BNN
from NN import NN



class BVAE(object):

    def __init__(self, network_architecture):
        
        tf.reset_default_graph()

        encoder_net = network_architecture[0]
        decoder_net = network_architecture[1]

        #Model hyperparameters
        self.act_func = tf.nn.softplus #tf.tanh #elu
        self.learning_rate = .0001
        self.rs = 0
        self.input_size = encoder_net[0]
        self.z_size = decoder_net[0]
        # self.net = network_architecture

        #Placeholders - Inputs/Targets [B,X]
        self.batch_size = tf.placeholder(tf.int32, None)
        self.n_z_particles = tf.placeholder(tf.int32, None)
        self.n_W_particles = tf.placeholder(tf.int32, None)
        self.batch_frac = tf.placeholder(tf.float32, None)
        self.x = tf.placeholder(tf.float32, [None, self.input_size])

        self.NN_encoder = NN(encoder_net, self.batch_size)
        self.BNN_decoder = BNN(decoder_net, self.batch_size, self.n_W_particles, self.batch_frac)
        
        #Objective
        self.elbo = self.objective(self.x, self.NN_encoder, self.BNN_decoder)

        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(-self.elbo)


        # To init variables
        self.init_vars = tf.global_variables_initializer()
        # For loadind/saving variables
        self.saver = tf.train.Saver()
        # #For debugging 
        # self.vars = tf.trainable_variables()
        # self.grads = tf.gradients(self.elbo, tf.trainable_variables()
        #to make sure im not adding nodes to the graph
        tf.get_default_graph().finalize()
        #Start session
        self.sess = tf.Session()





    def objective(self, x, encoder, decoder):
        '''
        elbo: [1]
        '''

        #Encode
        z_mean_logvar = encoder.model(x) #[B,Z*2]
        z_mean = tf.slice(z_mean_logvar, [0,0], [self.batch_size, self.z_size]) #[B,Z] 
        z_logvar = tf.slice(z_mean_logvar, [0,self.z_size], [self.batch_size, self.z_size]) #[B,Z]

        # #Sample z
        # eps = tf.random_normal((self.batch_size, self.n_z_particles, self.z_size), 0, 1, dtype=tf.float32) #[B,P,Z]
        # z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps)) #uses broadcasting,[B,P,Z]

        # Sample z  [B,Z]
        eps = tf.random_normal((self.batch_size, self.z_size), 0, 1, dtype=tf.float32) #[B,Z]
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps)) #[B,Z]

        # [B]
        log_pz = self.log_normal(z, tf.zeros([self.batch_size, self.z_size]), tf.log(tf.ones([self.batch_size, self.z_size])))
        log_qz = self.log_normal(z, z_mean, z_logvar)

        # Decode [B,P,X], [P], [P]
        x_mean, log_pW, log_qW = decoder.model(z)
        
        # Likelihood [B,P]
        log_px = self.log_bernoulli(x, x_mean)

        # Objective
        self.log_px = tf.reduce_mean(log_px) #over batch + W_particles
        self.log_pz = tf.reduce_mean(log_pz) #over batch
        self.log_qz = tf.reduce_mean(log_qz) #over batch 
        self.log_pW = tf.reduce_mean(log_pW) #W_particles
        self.log_qW = tf.reduce_mean(log_qW) #W_particles

        elbo = self.log_px + self.log_pz - self.log_qz + self.batch_frac*(self.log_pW - self.log_qW)

        self.z_elbo = self.log_px + self.log_pz - self.log_qz 

        return elbo




    # def log_normal(self, position, mean, log_var):
    #     '''
    #     Log of normal distribution
    #     position is [P, D]
    #     mean is [D]
    #     log_var is [D]
    #     output is [P]
    #     '''

    #     n_D = tf.to_float(tf.shape(mean)[0])
    #     term1 = n_D * tf.log(2*math.pi)
    #     term2 = tf.reduce_sum(log_var, 0) #sum over D [1]
    #     dif_cov = tf.square(position - mean) / tf.exp(log_var)
    #     term3 = tf.reduce_sum(dif_cov, 1) #sum over D [P]
    #     all_ = term1 + term2 + term3
    #     log_normal_ = -.5 * all_

    # #     return log_normal_



    # def log_normal(self, position, mean, log_var):
    #     '''
    #     Log of normal distribution
    #     position is [?, D]
    #     mean is [D]
    #     log_var is [D]
    #     output is [?]
    #     '''

    #     n_D = tf.to_float(tf.shape(mean)[0])
    #     term1 = n_D * tf.log(2*math.pi)
    #     term2 = tf.reduce_sum(log_var, 0) #sum over D [1]
    #     dif_cov = tf.square(position - mean) / tf.exp(log_var)
    #     term3 = tf.reduce_sum(dif_cov, 1) #sum over D [P]
    #     all_ = term1 + term2 + term3
    #     log_normal_ = -.5 * all_

    #     return log_normal_



    def log_normal(self, z, mean, log_var):
        '''
        Log of normal distribution

        z is [B, Z]
        mean is [B, Z]
        log_var is [B, Z]
        output is [B]
        '''

        n_D = tf.to_float(tf.shape(mean)[0])
        term1 = n_D * tf.log(2*math.pi) #[1]
        term2 = tf.reduce_sum(log_var, axis=1) #sum over Z, [B]
        dif_cov = tf.square(z - mean) / tf.exp(log_var)
        term3 = tf.reduce_sum(dif_cov, axis=1) #sum over Z, [B]
        all_ = term1 + term2 + term3
        log_N = -.5 * all_
        return log_N





    def log_bernoulli(self, t, pred_no_sig):
        '''
        Log of bernoulli distribution
        t is [B, X]
        pred_no_sig is [B, P, X] 
        output is [B, P]
        '''

        #[B,1,X]
        t = tf.reshape(t, [self.batch_size, 1, self.input_size])

        reconstr_loss = \
                tf.reduce_sum(tf.maximum(pred_no_sig, 0) 
                            - pred_no_sig * t
                            + tf.log(1 + tf.exp(-tf.abs(pred_no_sig))),
                             2) #sum over dimensions

        #negative because the above calculated the NLL, so this is returning the LL
        return -reconstr_loss



    # def log_likelihood(self, y, y_hat):
    #     '''
    #     Calculate p(y|x,theta)
    #     y: [B,Y]
    #     y_hat: [B,P,Y]
    #     '''
    #     self.y1 = y
    #     #tile y for each particle [B,P,Y]
    #     self.y2 = tf.tile(self.y1, [1,self.n_particles])
    #     self.y3 = tf.reshape(self.y2, [self.batch_size,self.n_particles,self.output_size])

    #     # [B,P]
    #     log_likelihood = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.y3)



    #     return -log_likelihood


    # def model(self, x):
    #     '''
    #     x: [B,X]
    #     y_hat: [B,P,O]
    #     log_p_W: [P]
    #     log_q_W: [P]
    #     '''

    #     net = self.net

    #     def xavier_init(fan_in, fan_out, constant=1): 
    #         """ Xavier initialization of network weights"""
    #         # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    #         low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    #         high = constant*np.sqrt(6.0/(fan_in + fan_out))
    #         return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    #     #[B,X]
    #     cur_val = x
    #     #tile x for each particle [B,P,X]
    #     cur_val = tf.tile(cur_val, [1,self.n_particles])
    #     cur_val = tf.reshape(cur_val, [self.batch_size,self.n_particles,self.input_size])


    #     log_p_W_sum = 0
    #     log_q_W_sum = 0

    #     for layer_i in range(len(net)-1):

    #         input_size_i = net[layer_i]+1 #plus 1 for bias
    #         output_size_i = net[layer_i+1] #plus 1 because we want layer i+1

    #         #Define variables [IS,OS]
    #         W_means = tf.Variable(xavier_init(input_size_i, output_size_i))
    #         W_logvars = tf.Variable(xavier_init(input_size_i, output_size_i) - 10.)

    #         #Sample weights [IS,OS]*[P,IS,OS]=[P,IS,OS]
    #         eps = tf.random_normal((self.n_particles, input_size_i, output_size_i), 0, 1, seed=self.rs)
    #         W = tf.add(W_means, tf.multiply(tf.sqrt(tf.exp(W_logvars)), eps))

    #         #Compute probs of samples  [B,P,1]
    #         flat_w = tf.reshape(W,[self.n_particles,input_size_i*output_size_i]) #[P,IS*OS]
    #         flat_W_means = tf.reshape(W_means, [input_size_i*output_size_i]) #[IS*OS]
    #         flat_W_logvars = tf.reshape(W_logvars, [input_size_i*output_size_i]) #[IS*OS]
    #         log_p_W_sum += self.log_normal(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))
    #         log_q_W_sum += self.log_normal(flat_w, flat_W_means, flat_W_logvars)

    #         # if layer_i==0:
    #         #     self.debug_pw = self.log_normal(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))

    #         #Concat 1 to input for biases  [B,P,X]->[B,P,X+1]
    #         cur_val = tf.concat([cur_val,tf.ones([self.batch_size,self.n_particles, 1])], axis=2)
    #         #[B,P,X]->[B,P,1,X]
    #         cur_val = tf.reshape(cur_val, [self.batch_size, self.n_particles, 1, input_size_i])
    #         #[P,X,X']->[B,P,X,X']
    #         W = tf.tile(W, [self.batch_size,1,1])
    #         W = tf.reshape(W, [self.batch_size, self.n_particles, input_size_i, output_size_i])

    #         #Forward Propagate  [B,P,1,X]*[B,P,X,X']->[B,P,1,X']
    #         if layer_i != len(net)-2:
    #             cur_val = self.act_func(tf.matmul(cur_val, W))
    #         else:
    #             cur_val = tf.matmul(cur_val, W)

    #         #[B,P,1,X']->[B,P,X']
    #         cur_val = tf.reshape(cur_val, [self.batch_size,self.n_particles,output_size_i])

    #     return cur_val, log_p_W_sum, log_q_W_sum














    def train(self, train_x, valid_x=[], display_step=5, path_to_load_variables='', path_to_save_variables='', epochs=10, batch_size=20, n_particles=3):
        '''
        Train.
        '''
        random_seed=1
        rs=npr.RandomState(random_seed)
        n_datapoints = len(train_x)
        arr = np.arange(n_datapoints)

        if path_to_load_variables == '':
            self.sess.run(self.init_vars)

        else:
            #Load variables
            self.saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables

        #start = time.time()
        for epoch in range(epochs):

            #shuffle the data
            rs.shuffle(arr)
            train_x = train_x[arr]
            # train_y = train_y[arr]

            data_index = 0
            for step in range(n_datapoints/batch_size):

                #Make batch
                batch = []
                # batch_y = []
                while len(batch) != batch_size:
                    batch.append(train_x[data_index]) 
                    # one_hot=np.zeros(10)
                    # one_hot[train_y[data_index]]=1.
                    # batch_y.append(one_hot)
                    data_index +=1

                # Fit training using batch data
                _ = self.sess.run((self.optimizer), feed_dict={self.x: batch, 
                                                        self.batch_size: batch_size,
                                                        self.n_z_particles: n_particles, 
                                                        self.n_W_particles: n_particles, 
                                                        self.batch_frac: 1./float(n_datapoints)})

                # Display logs per epoch step
                if step % display_step == 0:

                    # cost,logpy,logpW,logqW,pred = self.sess.run((self.elbo,self.logpy, self.logpW, self.logqW,self.prediction), 
                    #                                 feed_dict={self.x: batch, 
                    #                                     self.batch_size: batch_size, 
                    #                                     self.n_z_particles: n_particles, 
                    #                                     self.n_W_particles: n_particles, 
                    #                                     self.batch_fraction_of_dataset: 1./float(n_datapoints)})

                    cost,z_elbo,log_px,log_pz,log_qz,log_pW,log_qW = self.sess.run((self.elbo, self.z_elbo, self.log_px, self.log_pz, self.log_qz, self.log_pW, self.log_qW), 
                                                    feed_dict={self.x: batch, 
                                                        self.batch_size: batch_size, 
                                                        self.n_z_particles: n_particles, 
                                                        self.n_W_particles: n_particles, 
                                                        self.batch_frac: 1./float(n_datapoints)})



                    # cost = -cost #because I want to see the NLL
                    print "Epoch", str(epoch+1)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "elbo=", "{:.6f}".format(float(cost)),z_elbo,log_px,log_pz,log_qz,log_pW,log_qW#,logpy,logpW,logqW #, 'time', time.time() - start
                    # # print 'target'
                    # print ["{:.2f}".format(float(x)) for x in batch_y[0]] 
                    # # print 'prediciton'
                    # # print pred.shape
                    # print ["{:.2f}".format(float(x)) for x in pred[0][0]] 
                    # print


        if path_to_save_variables != '':
            self.saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables








if __name__ == '__main__':

    x_size = 784
    z_size = 10

    net = [[x_size,20,z_size*2],
            [z_size,20,x_size]]


    model = BVAE(net)

    print 'Loading data'
    with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
        mnist_data = pickle.load(f)

    train_x = mnist_data[0][0]
    train_y = mnist_data[0][1]
    valid_x = mnist_data[1][0]
    valid_y = mnist_data[1][1]
    test_x = mnist_data[2][0]
    test_y = mnist_data[2][1]

    # path_to_load_variables=home+'/Documents/tmp/vars.ckpt' 
    path_to_load_variables=''
    path_to_save_variables=home+'/Documents/tmp/vars2.ckpt'

    print 'Training'
    model.train(train_x=train_x,
                epochs=50, batch_size=20, n_particles=2, display_step=1000,
                path_to_load_variables=path_to_load_variables,
                path_to_save_variables=path_to_save_variables)


    print 'Done.'












