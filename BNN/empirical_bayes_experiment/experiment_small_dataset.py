







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
        # second_half_BNN = NN([self.input_size, 200, 200, 100, 200, 200, n_classes], [tf.nn.softplus,tf.nn.softplus,None,tf.nn.softplus,tf.nn.softplus, None])
        second_half_BNN = NN([self.input_size, 200, 200, n_classes], [tf.nn.softplus,tf.nn.softplus,None])





        # Ws, log_p_W_sum, log_q_W_sum = second_half_BNN.sample_weights()


        #Feedforward [B,2]
        # self.z = first_half_NN.feedforward(self.x)
        # log_pz = tf.reduce_mean(log_norm(self.z, tf.zeros([2]), tf.log(tf.ones([2]))))



        # self.pred = second_half_BNN.feedforward(self.x, Ws)
        self.pred = second_half_BNN.feedforward(self.x)

        # y_hat, log_p_W, log_q_W = self.model(self.x)

        #Likelihood [B,P]
        # log_p_y_hat = self.log_likelihood(self.y, y_hat)
        softmax_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.pred))

        # sigmoid_cross_entropy =  tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.pred))

        #Objective
        # self.elbo = self.objective(log_p_y_hat, log_p_W, log_q_W, self.batch_fraction_of_dataset)
        # self.cost = softmax_error + self.one_over_N*(-log_p_W_sum + log_q_W_sum) #+ .00001*first_half_NN.weight_decay()
        self.cost = softmax_error #+ self.one_over_N*(log_q_W_sum) 
        # self.cost = sigmoid_cross_entropy #+ self.one_over_N*(log_q_W_sum) 



        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost)






        #For evaluation
        self.prediction = tf.nn.softmax(self.pred) 

        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.prediction,1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
        train_costs = []
        test_costs = []
        train_accs = []
        test_accs = []
        epoch_list = []

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
                    # one_hot=np.zeros(self.output_size)
                    # one_hot[train_y[data_index]]=1.
                    # batch_y.append(one_hot)
                    batch_y.append(train_y[data_index])
                    data_index +=1

                # Fit training using batch data
                _ = self.sess.run((self.optimizer), feed_dict={self.x: batch, self.y: batch_y, 
                                                        self.one_over_N: one_over_N})

                # Display logs per epoch step
                if epoch % display_step == 0 and step ==0:

                    epoch_list.append(epoch)

                    # cost, acc = self.sess.run((self.cost, self.acc), feed_dict={self.x: batch, self.y: batch_y, 
                    #                                     self.one_over_N: one_over_N})
                    cost, acc = self.sess.run((self.cost, self.acc), feed_dict={self.x: train_x, self.y: train_y, 
                                                        self.one_over_N: one_over_N})
                    train_costs.append(cost)
                    train_accs.append(acc)
                    
                    if len(valid_x) > 0:
                        test_cost, test_acc = self.sess.run((self.cost, self.acc), feed_dict={self.x: valid_x, self.y: valid_y, 
                                                        self.one_over_N: one_over_N})
                        test_costs.append(test_cost)
                        test_accs.append(test_acc)

                    
                    print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost)), acc, 'test', test_cost, test_acc
                    # print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost)), acc#,logpy,logpW,logqW #, 'time', time.time() - start


        if path_to_save_variables != '':
            self.saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables


        return epoch_list, np.array(train_costs), np.array(train_accs), np.array(test_costs), np.array(test_accs)




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



    # def eval(self, x, y):




    #     onehoty=[]
    #     for i in range(len(y)):
    #         one_hot=np.zeros(self.output_size)
    #         one_hot[y[i]]=1.
    #         onehoty.append(one_hot)

    #     onehoty = np.array(onehoty)

    #     error = self.sess.run((self.cost), feed_dict={self.x: x, self.y: onehoty})

    #     return error































if __name__ == '__main__':


    #need to calc elbo

    train = 1
    plot_training_curves = 1
    eval_ = 0

    # viz_encodings_and_boundaries=1
    # viz_encodings_and_probabilities=1


    #zoom of viz
    # limit_value = 5

    n_all_classes = 10
    # n_limited_classes = 10

    epochs = 200
    display_step = epochs / 10
    batch_size = 50


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

    train_x = train_x[:1000]
    train_y = train_y[:1000]

    print train_x.shape
    print train_y.shape

    #Change y to one hot
    train_y_one_hot = []
    for i in range(len(train_y)):
        one_hot=np.zeros(n_all_classes)
        one_hot[train_y[i]]=1.
        train_y_one_hot.append(one_hot)
    train_y = np.array(train_y_one_hot)

    test_y_one_hot = []
    for i in range(len(test_y)):
        one_hot=np.zeros(n_all_classes)
        one_hot[test_y[i]]=1.
        test_y_one_hot.append(one_hot)
    test_y = np.array(test_y_one_hot)



    # prior_variances = [.001, .01, .1, 1., 10., 100., 1000.]
    prior_variances = [0.]

    for prior_var in range(len(prior_variances)):

        print 'Prior Variance', prior_variances[prior_var]


        model = BNN_MNIST(n_classes=n_all_classes, prior_variance=prior_variances[prior_var])

        # path_to_load_variables=home+'/Documents/tmp/vars_emp_bayes_'+str(int(prior_var)) +'.ckpt'
        path_to_load_variables=''
        path_to_save_variables=home+'/Documents/tmp/vars_emp_bayes_'+str(int(prior_var)) +'.ckpt'




        if train:
            print 'Training'
            epoch_list, train_costs, train_accs, test_costs, test_accs = model.train(
                        train_x=train_x, train_y=train_y, 
                        valid_x=test_x, valid_y=test_y, 
                        epochs=epochs, batch_size=batch_size, display_step=display_step,
                        path_to_load_variables=path_to_load_variables,
                        path_to_save_variables=path_to_save_variables)

            # print 'Train Accuracy:', model.accuracy(train_x, train_y)
            # print 'Test Accuracy:', model.accuracy(test_x, test_y)

            # print 'Train Error:', model.eval(train_x, train_y)
            # print 'Test Error:', model.eval(test_x, test_y)

        else:
            model.load_vars(path_to_load_variables=path_to_save_variables)

            # print 'Train Accuracy:', model.accuracy(train_x, train_y)
            # print 'Test Accuracy:', model.accuracy(test_x, test_y)

            # print 'Train Error:', model.eval(train_x, train_y)
            # print 'Test Error:', model.eval(test_x, test_y)



        if plot_training_curves:

            print '\nPlotting log probs over epochs'

            # plt.clf()
            # fig = plt.figure(figsize=(12,5), facecolor='white')
            fig = plt.figure(facecolor='white')


            ax1 = plt.subplot2grid((1, 2), (0, 0))#, colspan=3)

            ax1.plot(epoch_list, train_accs, label='Train')
            ax1.plot(epoch_list, test_accs, label='Test')

            # # print train_costs
            # train_costs = -train_costs
            # test_costs = -test_costs

            # train_costs = (train_costs - np.min(train_costs)) 
            # train_costs = train_costs / np.max(train_costs)

            # test_costs = (test_costs - np.min(test_costs)) 
            # test_costs = test_costs / np.max(test_costs)

            # # test_costs = test_costs / np.max(test_costs)

            # # (test_values - test_values.min(0)) / test_values.ptp(0)
            # # print train_costs

            # ax1.plot(epoch_list, train_costs, label='NLL Train')
            # ax1.plot(epoch_list, test_costs, label='NLL Test')

            #normalize values
            # x_normed = (values - values.min(0)) / values.ptp(0)

            # for vals in x_normed.T[1:]:
            #     ax1.plot(values.T[0], vals)
            #     # print vals

            ax1.legend(loc='best', fontsize=7)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Accuracy')
            ax1.tick_params(labelsize=6)
            ax1.set_ylim([None,1.])

            # ax1.set_title(exp_settings_name + 'train', fontsize=7)
            # # ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=7)
            # ax1.tick_params(labelsize=6)

            # plt.show()


            # #TEST
            ax2 = plt.subplot2grid((1, 2), (0, 1))#, colspan=3)
            train_costs = -train_costs
            test_costs = -test_costs
            ax2.plot(epoch_list, train_costs, label='Train')
            ax2.plot(epoch_list, test_costs, label='Test')

            ax2.legend(loc='best', fontsize=7)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('NLL')
            ax2.tick_params(labelsize=6)

            # #normalize values
            # x_normed = (test_values - test_values.min(0)) / test_values.ptp(0)

            # for vals in x_normed.T[1:]:
            #     ax2.plot(test_values.T[0], vals)
            #     # print vals

            # # ax1.set_grid('off')
            # ax2.legend(test_labels[1:], loc='best', fontsize=7)
            # ax2.set_xlabel('Epochs')
            # # ax2.set_ylabel('Normalized Values')
            # ax2.set_title(exp_settings_name + 'test', fontsize=7)
            # # ax2.set_xticklabels(fontsize=7)
            # # ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=7)
            # ax2.tick_params(labelsize=6)



            plt.savefig(home+'/Documents/tmp/vars_emp_bayes_.png')
            print 'saved fig to ' + home+'/Documents/tmp/vars_emp_bayes_.png'
            print 


    print 'Done.'


