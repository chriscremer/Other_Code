

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
import matplotlib.pyplot as plt

# Normal = tf.contrib.distributions.Normal



class BNN(object):

    def __init__(self, network_architecture, eps_list):
        
        tf.reset_default_graph()

        #Model hyperparameters
        self.act_func = tf.nn.relu#tf.nn.softplus#tf.tanh #tf.nn.sigmoid### # ## #
        self.learning_rate = .001
        self.rs = 0
        self.input_size = network_architecture[0]
        self.output_size = network_architecture[-1]
        self.net = network_architecture

        #Placeholders - Inputs/Targets [B,X]
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.y = tf.placeholder(tf.float32, [None, self.output_size])

        self.n_particles_train = 2
        self.n_particles_test = 5
        self.n_particles_viz = 5

        self.batch_fraction_of_dataset = tf.placeholder(tf.float32, None)
        self.batch_size = tf.shape(self.x)[0]

        #Define variables
        self.W_means, self.W_logvars = self.init_weights()

        log_pWs = []
        log_qWs = []
        LLs = []
        for i in range(self.n_particles_train):
            #Sample weights _, [1], [1]
            Ws, log_p_W, log_q_W = self.sample_weights()
            #Feedforward: [B,Y]
            y_hat = self.feedforward(self.x, Ws)
            #Likelihood [B]
            log_p_y_hat = self.log_normal(y_hat, self.y, tf.zeros((self.batch_size, self.output_size))-10., type_=2)

            log_pWs.append(log_p_W)
            log_qWs.append(log_q_W)
            LLs.append(log_p_y_hat)

        log_pWs = tf.squeeze(tf.stack(log_pWs)) #[P]
        log_qWs = tf.squeeze(tf.stack(log_qWs)) #[P]
        LLs = tf.stack(LLs,1)  #[B,P]

        #Objective
        self.elbo = self.objective(LLs, log_pWs, log_qWs, self.batch_fraction_of_dataset)
        self.iwae_elbo_test = self.iwae_objective_test(LLs)

        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(-self.elbo)

        # #For evaluation
        # eps = []
        # for i in range(self.n_particles_viz):
        #     eps_part = []
        #     for layer_i in range(len(self.net)-1):
        #         input_size_i = net[layer_i]+1 #plus 1 for bias
        #         output_size_i = net[layer_i+1] #plus 1 because we want layer i+1
        #         eps_part.append(tf.random_normal((input_size_i, output_size_i), 0, 1, seed=self.rs))
        #     eps.append(eps_part)

        # Ws_, log_p_W_, log_q_W_ = self.sample_weights(eps_list)
        # self.y_hat_viz = self.feedforward(self.x, Ws_)

        self.outputs = []
        for i in range(self.n_particles_viz):
            #Sample weights _, [1], [1]
            Ws, log_p_W, log_q_W = self.sample_weights(eps_list[i])
            #Feedforward: [B,Y]
            y_hat = self.feedforward(self.x, Ws)
            self.outputs.append(y_hat)
        self.outputs = tf.stack(self.outputs)


        #To init variables
        self.init_vars = tf.global_variables_initializer()

        #For loadind/saving variables
        self.saver = tf.train.Saver()

        # #For debugging 
        # self.vars = tf.trainable_variables()
        # self.grads = tf.gradients(self.elbo, tf.trainable_variables())

        #to make sure im not adding nodes to the graph
        tf.get_default_graph().finalize()

        #Start session
        self.sess = tf.Session()


    def log_normal(self, position, mean, log_var, type_=1):
        '''
        Log of normal distribution

        type 1:
        position is [P, D]
        mean is [D]
        log_var is [D]
        output is [P]

        type 2:
        position is [P, D]
        mean is [P,D]
        log_var is [P,D]
        output is [P]
        '''

        n_D = tf.to_float(tf.shape(position)[1])
        term1 = n_D * tf.log(2*math.pi)

        if type_==1:
            term2 = tf.reduce_sum(log_var, 0) #sum over D [1]
            dif_cov = tf.square(position - mean) / tf.exp(log_var)
            term3 = tf.reduce_sum(dif_cov, 1) #sum over D [P]
            all_ = term1 + term2 + term3
            log_normal_ = -.5 * all_

        elif type_==2:
            term2 = tf.reduce_sum(log_var, 1) #sum over D [1]
            dif_cov = tf.square(position - mean) / tf.exp(log_var)
            term3 = tf.reduce_sum(dif_cov, 1) #sum over D [P]
            all_ = term1 + term2 + term3
            log_normal_ = -.5 * all_

        return log_normal_



    def init_weights(self):

        W_means = []
        W_logvars = []

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Define variables [IS,OS]
            W_means.append(tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1)))
            W_logvars.append(tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1))-5.)

        return W_means, W_logvars


    def sample_weights(self, eps_list=None):

        Ws = []
        log_p_W_sum = 0
        log_q_W_sum = 0

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Get vars [I,O]
            W_means = self.W_means[layer_i]
            W_logvars = self.W_logvars[layer_i]

            if eps_list==None:
                eps = tf.random_normal((input_size_i, output_size_i), 0, 1, seed=self.rs)
            else:
                eps = eps_list[layer_i]

            #Sample weights [IS,OS]*[IS,OS]=[IS,OS]
            W = tf.add(W_means, tf.multiply(tf.sqrt(tf.exp(W_logvars)), eps))
            # W = W_means

            #Compute probs of samples  [1]
            flat_w = tf.reshape(W,[1,input_size_i*output_size_i]) #[1,IS*OS]
            flat_W_means = tf.reshape(W_means, [input_size_i*output_size_i]) #[IS*OS]
            flat_W_logvars = tf.reshape(W_logvars, [input_size_i*output_size_i]) #[IS*OS]
            log_p_W_sum += self.log_normal(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))
            log_q_W_sum += self.log_normal(flat_w, flat_W_means, flat_W_logvars)

            Ws.append(W)

        return Ws, log_p_W_sum, log_q_W_sum



    def feedforward(self, x, W_list):
        '''
        x: [B,X]
        W: list of layers weights
        y: [B,Y]
        '''

        #[B,X]
        cur_val = x
        for layer_i in range(len(self.net)-1):
            W = W_list[layer_i] #[I,O]
            #Concat 1 to input for biases  [B,X]->[B,X+1]
            cur_val = tf.concat([cur_val,tf.ones([tf.shape(cur_val)[0], 1])], axis=1)
            #Forward Propagate  [B,X]*[X,X']->[B,X']
            if layer_i != len(self.net)-2:
                cur_val = self.act_func(tf.matmul(cur_val, W))
            else:
                cur_val = tf.matmul(cur_val, W)
        #[B,Y]
        y = cur_val

        return y




    def objective(self, logpy, logpW, logqW, batch_frac):
        '''
        logpy:[B,P]
        logpW,logqW:[P]
        elbo: [1]
        '''
        self.logpy=tf.reduce_mean(logpy)
        self.logpW = tf.reduce_mean(logpW)
        self.logqW = tf.reduce_mean(logqW)
        elbo = self.logpy #+ batch_frac*(self.logpW - self.logqW)
        return elbo

    def iwae_objective_test(self, logpy):
        '''
        logpy:[B,P]
        logpW,logqW:[P]
        elbo: [1]
        '''
        temp_elbo = tf.reduce_mean(logpy, axis=0)   #[P]
        max_ = tf.reduce_max(temp_elbo, axis=0)
        iwae_elbo = tf.log(tf.reduce_mean(tf.exp(temp_elbo-max_))) + max_  #[1]
        return iwae_elbo






def build_toy_dataset(n_data=40, noise_std=0.1):
    D = 1
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(-4, -2, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    inputs = np.concatenate([inputs,np.array([20]*10)])
    targets = np.cos(inputs) + rs.randn(n_data+10) * noise_std
    inputs = (inputs - 4.0) / 4.0
    inputs  = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets




if __name__ == '__main__':

    inputs, targets = build_toy_dataset()
    n_particles = 2
    net = [1,20,20,1]

    #epsilon for viewing constant weights
    eps = []
    rs = npr.RandomState(0)
    for p in range(5):
        eps_part =[]
        for layer_i in range(len(net)-1):
            input_size_i = net[layer_i]+1 #plus 1 for bias
            output_size_i = net[layer_i+1] #plus 1 because we want layer i+1
            eps_part.append(rs.randn(input_size_i,output_size_i))
        eps.append(eps_part)

    m = BNN(net,eps)
    m.sess.run(m.init_vars)

    # Set up figure.
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)



    for iter_ in range(50000):

        # Fit training using batch data
        _,elbo,logpy,logpW,logqW = m.sess.run((m.optimizer, m.elbo, m.logpy, m.logpW, m.logqW), 
                                            feed_dict={m.x: inputs, m.y: targets, 
                                                m.batch_fraction_of_dataset: 1.})



        # # Display logs per epoch step
        # if step % display_step[1] == 0 and epoch % display_step[0] ==0:

        #     cost,logpy,logpW,logqW,pred = self.sess.run((self.elbo,self.logpy, self.logpW, self.logqW,self.prediction), 
                            # feed_dict={self.x: batch, self.y: batch_y, 
        #                                         self.batch_size: batch_size, 
        #                                         self.n_particles:n_particles, 
        #                                         self.batch_fraction_of_dataset: 1./float(n_datapoints)})

        

        if iter_ % 50 == 0:
            print iter_, elbo, logpy,logpW,logqW

            # # Sample functions from posterior.

            plot_inputs = np.linspace(-8, 8, num=400)
            plot_inputs_ = np.reshape(plot_inputs, [-1,1])
            outputs = m.sess.run((m.outputs), feed_dict={m.x: plot_inputs_})
            outputs = np.reshape(outputs, [outputs.shape[0],400])
            # print outputs.shape  #[P,400,1]
            # fasdf



            # Plot data and functions.
            plt.cla()
            ax.plot(inputs.ravel(), targets.ravel(), 'bx')
            ax.plot(plot_inputs, outputs.T)
            ax.set_ylim([-2, 3])
            plt.draw()
            plt.pause(1.0/60.0)











