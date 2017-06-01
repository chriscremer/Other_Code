


#Neural Network

import numpy as np
import numpy.random as npr
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle




class NN(object):

    def __init__(self, network_architecture):
        
        tf.reset_default_graph()

        #Model hyperparameters
        self.act_func = tf.nn.softplus #tf.tanh
        self.learning_rate = .0001
        self.rs = 0
        self.input_size = network_architecture[0]
        self.output_size = network_architecture[-1]
        self.net = network_architecture

        #Placeholders - Inputs/Targets [B,X]
        self.batch_size = tf.placeholder(tf.int32, None)
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.y = tf.placeholder(tf.float32, [None, self.output_size])

        #Feedforward: [B,P,Y], [P], [P]
        y_hat = self.model(self.x)

        #Likelihood [B,P]
        log_p_y_hat = self.log_likelihood(self.y, y_hat)
        self.cost = -log_p_y_hat

        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost)

        #For evaluation
        self.prediction = tf.nn.softmax(y_hat) 

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




    def log_likelihood(self, y, y_hat):
        '''
        Calculate p(y|x,theta)
        y: [B,Y]
        y_hat: [B,P,Y]
        '''
        # self.y1 = y
        # #tile y for each particle [B,P,Y]
        # self.y2 = tf.tile(self.y1, [1,self.n_particles])
        # self.y3 = tf.reshape(self.y2, [self.batch_size,self.n_particles,self.output_size])

        # [B,P]
        log_likelihood = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))

        return -log_likelihood


    def model(self, x):
        '''
        x: [B,X]
        y_hat: [B,P,O]
        log_p_W: [P]
        log_q_W: [P]
        '''

        net = self.net

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        #[B,X]
        cur_val = x
        #tile x for each particle [B,P,X]
        # cur_val = tf.tile(cur_val, [1,self.n_particles])
        # cur_val = tf.reshape(cur_val, [self.batch_size,self.n_particles,self.input_size])


        # log_p_W_sum = 0
        # log_q_W_sum = 0

        for layer_i in range(len(net)-1):

            input_size_i = net[layer_i]+1 #plus 1 for bias
            output_size_i = net[layer_i+1] #plus 1 because we want layer i+1

            #Define variables [IS,OS]
            W = tf.Variable(xavier_init(input_size_i, output_size_i))
            # W_logvars = tf.Variable(xavier_init(input_size_i, output_size_i))

            #Sample weights [IS,OS]*[P,IS,OS]=[P,IS,OS]
            # eps = tf.random_normal((self.n_particles, input_size_i, output_size_i), 0, 1, seed=self.rs)
            # W = tf.add(W_means, tf.multiply(tf.sqrt(tf.exp(W_logvars)), eps))

            # #debug
            # W = tf.tile(W_means, [self.n_particles, 1])
            # W = tf.reshape(W, [self.n_particles, input_size_i, output_size_i])

            # #Compute probs of samples  [B,P,1]
            # flat_w = tf.reshape(W,[self.n_particles,input_size_i*output_size_i]) #[P,IS*OS]
            # flat_W_means = tf.reshape(W_means, [input_size_i*output_size_i]) #[IS*OS]
            # flat_W_logvars = tf.reshape(W_logvars, [input_size_i*output_size_i]) #[IS*OS]
            # log_p_W_sum += self.log_normal(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))
            # log_q_W_sum += self.log_normal(flat_w, flat_W_means, flat_W_logvars)

            # if layer_i==0:
            #     self.debug_pw = self.log_normal(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))

            #Concat 1 to input for biases  [B,P,X]->[B,P,X+1]
            cur_val = tf.concat([cur_val,tf.ones([self.batch_size, 1])], axis=1)
            # #[B,P,X]->[B,P,1,X]
            # cur_val = tf.reshape(cur_val, [self.batch_size, self.n_particles, 1, input_size_i])
            # #[P,X,X']->[B,P,X,X']
            # W = tf.tile(W, [self.batch_size,1,1])
            # W = tf.reshape(W, [self.batch_size, self.n_particles, input_size_i, output_size_i])

            #Forward Propagate  [B,P,1,X]*[B,P,X,X']->[B,P,1,X']
            if layer_i != len(net)-2:
                cur_val = self.act_func(tf.matmul(cur_val, W))
            else:
                cur_val = tf.matmul(cur_val, W)

            #[B,P,1,X']->[B,P,X']
            # cur_val = tf.reshape(cur_val, [self.batch_size,self.n_particles,output_size_i])

        return cur_val#, log_p_W_sum, log_q_W_sum



    def train(self, train_x, train_y, valid_x=[], valid_y=[], display_step=5, path_to_load_variables='', path_to_save_variables='', epochs=10, batch_size=20):
        '''
        Train.
        '''
        random_seed=1
        rs=npr.RandomState(random_seed)
        n_datapoints = len(train_y)
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
            train_y = train_y[arr]

            data_index = 0
            for step in range(n_datapoints/batch_size):

                #Make batch
                batch = []
                batch_y = []
                while len(batch) != batch_size:
                    batch.append(train_x[data_index]) 
                    one_hot=np.zeros(10)
                    one_hot[train_y[data_index]]=1.
                    batch_y.append(one_hot)
                    data_index +=1

                # Fit training using batch data
                _ = self.sess.run((self.optimizer), feed_dict={self.x: batch, self.y: batch_y, 
                                                        self.batch_size: batch_size})

                # Display logs per epoch step
                if step % display_step == 0:

                    cost,pred = self.sess.run((self.cost,self.prediction), feed_dict={self.x: batch, self.y: batch_y, 
                                                        self.batch_size: batch_size})
                    # cost = -cost #because I want to see the NLL
                    print "Epoch", str(epoch+1)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "elbo=", "{:.6f}".format(float(cost))#,logpy,logpW,logqW #, 'time', time.time() - start
                    # print 'target'
                    print ["{:.2f}".format(float(x)) for x in batch_y[0]] 
                    # print 'prediciton'
                    # print pred.shape
                    print ["{:.2f}".format(float(x)) for x in pred[0]] 
                    print


                    # debugpw = self.sess.run((self.debug_pw), feed_dict={self.x: batch, self.y: batch_y, 
                    #                                     self.batch_size: batch_size, 
                    #                                     self.n_particles:n_particles, 
                    #                                     self.batch_fraction_of_dataset: batch_size/float(n_datapoints)})
                    # print debugpw


                    # grads = self.sess.run((self.grads), feed_dict={self.x: batch, self.y: batch_y, 
                    #                                     self.batch_size: batch_size, 
                    #                                     self.n_particles:n_particles, 
                    #                                     self.batch_fraction_of_dataset: batch_size/float(n_datapoints)})
                    # print len(grads)
                    # print grads[0].shape
                    # print grads[0]
                    # # print vars_[0]
                    # print grads[1].shape
                    # print grads[1]
                    # print grads[2].shape
                    # print grads[2]
                    # print grads[3].shape
                    # print grads[3]


                    # vars_ = self.sess.run((self.vars), feed_dict={self.x: batch, self.y: batch_y, 
                    #                                     self.batch_size: batch_size, 
                    #                                     self.n_particles:n_particles, 
                    #                                     self.batch_fraction_of_dataset: batch_size/float(n_datapoints)})
                    # print len(vars_)
                    # print vars_[0].shape
                    # # print vars_[0]
                    # print vars_[1].shape
                    # print vars_[2].shape
                    # print vars_[2]
                    # print vars_[3].shape
                    

                    # y1,y2,y3 = self.sess.run((self.y1,self.y2,self.y3 ), feed_dict={self.x: batch, self.y: batch_y, 
                    #                                     self.batch_size: batch_size, 
                    #                                     self.n_particles:n_particles, 
                    #                                     self.batch_fraction_of_dataset: batch_size/float(n_datapoints)})
                    # print y1.shape
                    # for b in range(len(y1)):
                    #     print y1[b]
                    # print y2.shape
                    # print y3.shape
                    # for b in range(len(y3)):
                    #     print y3[b]
                    # print 
                    # fasd

        if path_to_save_variables != '':
            self.saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables








if __name__ == '__main__':

    net = [784,20,10] 

    model = NN(net)

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
    model.train(train_x=train_x, train_y=train_y, 
                epochs=50, batch_size=20, display_step=1000,
                path_to_load_variables=path_to_load_variables,
                path_to_save_variables=path_to_save_variables)


    print 'Done.'












