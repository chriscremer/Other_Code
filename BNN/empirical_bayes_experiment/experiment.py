





import numpy as np
import tensorflow as tf

from os.path import expanduser
home = expanduser("~")
import time
import pickle

import matplotlib.pyplot as plt


from BNN_given_prior_variance import BNN
from NN2 import NN

from utils import log_normal2 as log_norm


class BNN_MNIST(object):

    def __init__(self, n_classes, prior_variance):
        
        tf.reset_default_graph()

        #Model hyperparameters
        # self.act_func = tf.nn.softplus #tf.tanh
        self.learning_rate = .001
        self.rs = 0
        self.input_size = 784
        self.output_size = n_classes
        # self.net = network_architecture

        #Placeholders - Inputs/Targets [B,X]
        # self.batch_size = tf.placeholder(tf.int32, None)
        # self.n_particles = tf.placeholder(tf.int32, None)
        self.one_over_N = tf.placeholder(tf.float32, None)
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.y = tf.placeholder(tf.float32, [None, self.output_size])


        # first_half_NN = NN([784, 100, 100, 2], [tf.nn.tanh,tf.nn.tanh, None])
        # first_half_NN = NN([784, 100, 100, 2], [tf.nn.softplus,tf.nn.softplus, None])
        # second_half_BNN = BNN([self.input_size, 100, 100, n_classes], [tf.nn.softplus,tf.nn.softplus, None], prior_variance)
        # second_half_BNN = NN([self.input_size, 100, 100, n_classes], [tf.nn.softplus,tf.nn.softplus, None])
        second_half_BNN = NN([self.input_size, 200, 50, 200, n_classes], [tf.nn.softplus,tf.nn.softplus,tf.nn.softplus, None])




        # Ws, log_p_W_sum, log_q_W_sum = second_half_BNN.sample_weights()


        #Feedforward [B,2]
        # self.z = first_half_NN.feedforward(self.x)
        # log_pz = tf.reduce_mean(log_norm(self.z, tf.zeros([2]), tf.log(tf.ones([2]))))



        # self.pred = second_half_BNN.feedforward(self.x, Ws)
        self.pred = second_half_BNN.feedforward(self.x)

        # y_hat, log_p_W, log_q_W = self.model(self.x)

        #Likelihood [B,P]
        # log_p_y_hat = self.log_likelihood(self.y, y_hat)
        # softmax_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.pred))

        sigmoid_cross_entropy =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.pred))

        #Objective
        # self.elbo = self.objective(log_p_y_hat, log_p_W, log_q_W, self.batch_fraction_of_dataset)
        # self.cost = softmax_error + self.one_over_N*(-log_p_W_sum + log_q_W_sum) #+ .00001*first_half_NN.weight_decay()
        # self.cost = softmax_error #+ self.one_over_N*(log_q_W_sum) 
        self.cost = sigmoid_cross_entropy #+ self.one_over_N*(log_q_W_sum) 



        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost)






        #For evaluation
        # self.prediction = tf.nn.softmax(self.pred) 

        # correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.prediction,1))
        # self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #To init variables
        self.init_vars = tf.global_variables_initializer()

        #For loadind/saving variables
        self.saver = tf.train.Saver()

        #For debugging 
        # self.vars = tf.trainable_variables()
        # self.grads = tf.gradients(self.elbo, tf.trainable_variables())

        #to make sure im not adding nodes to the graph
        tf.get_default_graph().finalize()

        #Start session
        self.sess = tf.Session()













    def train(self, train_x, train_y, valid_x=[], valid_y=[], display_step=5, 
                    path_to_load_variables='', path_to_save_variables='', 
                    epochs=10, batch_size=20):
        '''
        Train.
        '''
        random_seed=1
        rs=np.random.RandomState(random_seed)
        n_datapoints = len(train_y)
        one_over_N = 1./float(n_datapoints)
        arr = np.arange(n_datapoints)

        if path_to_load_variables == '':
            self.sess.run(self.init_vars)

        else:
            #Load variables
            self.saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables

        #start = time.time()
        for epoch in range(1,epochs+1):

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
                    one_hot=np.zeros(self.output_size)
                    one_hot[train_y[data_index]]=1.
                    batch_y.append(one_hot)
                    data_index +=1

                # Fit training using batch data
                _ = self.sess.run((self.optimizer), feed_dict={self.x: batch, self.y: batch_y, 
                                                        self.one_over_N: one_over_N})

                # Display logs per epoch step
                if step % display_step == 0:

                    # cost,logpy,logpW,logqW,pred = self.sess.run((self.cost,self.logpy, self.logpW, self.logqW,self.prediction), feed_dict={self.x: batch, self.y: batch_y, 
                    #                                     self.batch_fraction_of_dataset: 1./float(n_datapoints)})

                    cost = self.sess.run((self.cost), feed_dict={self.x: batch, self.y: batch_y, 
                                                        self.one_over_N: one_over_N})



                    # cost = -cost #because I want to see the NLL
                    print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost))#, acc#,logpy,logpW,logqW #, 'time', time.time() - start
                    # print 'target'
                    # print ["{:.2f}".format(float(x)) for x in batch_y[0]] 
                    # # print 'prediciton'
                    # # print pred.shape
                    # print ["{:.2f}".format(float(x)) for x in pred[0][0]] 
                    # print


        if path_to_save_variables != '':
            self.saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables



    def load_vars(self, path_to_load_variables):

        #Load variables
        self.saver.restore(self.sess, path_to_load_variables)
        print 'loaded variables ' + path_to_load_variables



    # def accuracy(self, x, y):

    #     pred = self.sess.run((self.prediction), feed_dict={self.x: x})

    #     # print pred.shape

    #     count = 0.
    #     for i in range(len(pred)):

    #         if np.argmax(pred[i]) == y[i]:
    #             count+=1

    #     return count / len(pred)





































if __name__ == '__main__':


    #need to calc elbo

    train= 1
    eval_ = 0
    # viz_encodings_and_boundaries=1
    # viz_encodings_and_probabilities=1


    #zoom of viz
    # limit_value = 5

    n_all_classes = 784
    # n_limited_classes = 10


    print 'Loading data'
    with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
        mnist_data = pickle.load(f)

    train_x = mnist_data[0][0]
    train_y = mnist_data[0][1]
    valid_x = mnist_data[1][0]
    valid_y = mnist_data[1][1]
    test_x = mnist_data[2][0]
    test_y = mnist_data[2][1]

    print train_x.shape
    print train_y.shape


    prior_variances = [.001, .01, .1, 1., 10., 100., 1000.]
    for prior_var in range(len(prior_variances)):

        print 'Prior Variance', prior_variances[prior_var]


        model = BNN_MNIST(n_classes=n_all_classes, prior_variance=prior_variances[prior_var])

        # path_to_load_variables=home+'/Documents/tmp/vars_emp_bayes_'+str(int(prior_var)) +'.ckpt'
        path_to_load_variables=''
        path_to_save_variables=home+'/Documents/tmp/vars_emp_bayes_'+str(int(prior_var)) +'.ckpt'




        if train == 1:
            print 'Training'
            model.train(train_x=train_x, train_y=train_x, 
                        epochs=40, batch_size=50, display_step=2000,
                        path_to_load_variables=path_to_load_variables,
                        path_to_save_variables=path_to_save_variables)

            print 'Train Accuracy:', model.accuracy(train_x, train_y)
            print 'Test Accuracy:', model.accuracy(test_x, test_y)


        else:
            model.load_vars(path_to_load_variables=path_to_save_variables)

            print 'Train Accuracy:', model.accuracy(train_x, train_y)
            print 'Test Accuracy:', model.accuracy(test_x, test_y)







    print 'Done.'


