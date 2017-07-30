



import numpy as np
import tensorflow as tf

from os.path import expanduser
home = expanduser("~")
import time
import pickle

import matplotlib.pyplot as plt


from BNN import BNN
from NN2 import NN




class BNN_bottleneck(object):

    def __init__(self, n_classes):
        
        tf.reset_default_graph()

        #Model hyperparameters
        # self.act_func = tf.nn.softplus #tf.tanh
        self.learning_rate = .0001
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


        first_half_NN = NN([784, 100, 100, 2], [tf.nn.softplus,tf.nn.softplus, None])
        second_half_BNN = BNN([2,100, 100, n_classes], [tf.nn.softplus,tf.nn.softplus, None])


        Ws, log_p_W_sum, log_q_W_sum = second_half_BNN.sample_weights()

        #Feedforward: [B,P,Y], [P], [P]
        self.y1 = first_half_NN.feedforward(self.x)
        self.y2 = second_half_BNN.feedforward(self.y1, Ws)
        # y_hat, log_p_W, log_q_W = self.model(self.x)

        #Likelihood [B,P]
        # log_p_y_hat = self.log_likelihood(self.y, y_hat)
        softmax_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y2))

        #Objective
        # self.elbo = self.objective(log_p_y_hat, log_p_W, log_q_W, self.batch_fraction_of_dataset)
        self.cost = softmax_error +  self.one_over_N*(-log_p_W_sum + log_q_W_sum) + .00001*first_half_NN.weight_decay()

        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost)






        #For evaluation
        self.prediction = tf.nn.softmax(self.y2) 

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
                    print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost))#,logpy,logpW,logqW #, 'time', time.time() - start
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


    # def get_min_max(self, train_x, batch_size):


    #     # #Load variables
    #     # self.saver.restore(self.sess, path_to_load_variables)
    #     # print 'loaded variables ' + path_to_load_variables

    #     n_datapoints = len(train_x)

    #     x_min_max = [0,0]
    #     y_min_max = [0,0]



    #     data_index = 0
    #     for step in range(n_datapoints/batch_size):

    #         #Make batch
    #         batch = []
    #         batch_y = []
    #         while len(batch) != batch_size:
    #             batch.append(train_x[data_index]) 
    #             data_index +=1

    #         # Fit training using batch data
    #         y1 = self.sess.run((self.y1), feed_dict={self.x: batch})

    #         max_x = np.max(y1, axis=0)
    #         max_y = np.max(y1, axis=1)
    #         min_x = np.min(y1, axis=0)
    #         min_y = np.min(y1, axis=1)

    #         if max_x > x_min_max[1]:
    #             x_min_max[1] = max_x
    #         if max_y > y_min_max[1]:
    #             y_min_max[1] = max_y
    #         if min_x < x_min_max[0]:
    #             x_min_max[0] = min_x
    #         if min_y < y_min_max[0]:
    #             y_min_max[0] = min_y


    #     return 





    def encode(self, train_x):

        y1 = self.sess.run((self.y1), feed_dict={self.x: train_x})


        return y1




    def predict(self, encodings):

        y2 = self.sess.run((self.prediction), feed_dict={self.y1: encodings})

        return y2




    def accuracy(self, x, y):

        pred = self.sess.run((self.prediction), feed_dict={self.x: x})

        # print pred.shape

        count = 0.
        for i in range(len(pred)):

            if np.argmax(pred[i]) == y[i]:
                count+=1

        return count / len(pred)





































if __name__ == '__main__':




    train= 0
    viz_encodings_and_boundaries=0
    viz_encodings_and_probabilities=1




    n_all_classes = 10
    n_limited_classes = 5


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


    model = BNN_bottleneck(n_classes=n_limited_classes)

    # path_to_load_variables=home+'/Documents/tmp/vars2.ckpt' 
    path_to_load_variables=''
    path_to_save_variables=home+'/Documents/tmp/vars_5_classes2.ckpt'


    # get n from each class
    n = 10
    ten_from_each = [[] for x in range(n_all_classes)]
    for i in range(len(train_y)):
        if len(ten_from_each[train_y[i]]) < n:
            ten_from_each[train_y[i]].append(train_x[i])

    #make limited dataset
    new_x = []
    new_y = []
    for i in range(len(train_y)):

        if train_y[i] < n_limited_classes:
            new_x.append(train_x[i])
            new_y.append(train_y[i])
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    print 'Limited Dataset'
    print new_x.shape
    print new_y.shape

    #same on test set
    new_x_test = []
    new_y_test = []
    for i in range(len(train_y)):

        if train_y[i] < n_limited_classes:
            new_x_test.append(train_x[i])
            new_y_test.append(train_y[i])
    new_x_test = np.array(new_x_test)
    new_y_test = np.array(new_y_test)




    if train == 1:
        print 'Training'
        model.train(train_x=new_x, train_y=new_y, 
                    epochs=50, batch_size=20, display_step=1000,
                    path_to_load_variables=path_to_load_variables,
                    path_to_save_variables=path_to_save_variables)

        print 'Train Accuracy:', model.accuracy(new_x, new_y)
        print 'Test Accuracy:', model.accuracy(new_x_test, new_y_test)


    else:
        model.load_vars(path_to_load_variables=path_to_save_variables)

        print 'Train Accuracy:', model.accuracy(new_x, new_y)
        print 'Test Accuracy:', model.accuracy(new_x_test, new_y_test)


    # if viz_encoding ==1:
    #     #get min and max of encoding.
    #     # model.get_min_max(train_x, batch_size=200, path_to_load_variables=path_to_save_variables)
    #     encoding = model.encode(train_x)
    #     print encoding.shape


    #     fig, ax = plt.subplots(1, 1)
    #     fig.patch.set_visible(False)



    #     # ax.axis('off')
    #     # ax.set_yticks([])
    #     # ax.set_xticks([])

    #     ax.scatter(encoding.T[0], encoding.T[1])
    #     plt.show()





    # if viz_boundaries ==1:

    #     numticks = 50
    #     x_min_max = [-15.,20.]
    #     y_min_max = [-15.,20.]
    #     x = np.linspace(*x_min_max, num=numticks)
    #     y = np.linspace(*y_min_max, num=numticks)
    #     X, Y = np.meshgrid(x, y)
    #     # print X.shape  [numticks, numticks]
    #     # print Y.shape [numticks, numticks]
    #     flat = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
    #     # print flat.shape  [numticks*numticks, 2]
    #     # fadsf

    #     # [N,C]
    #     predictions = model.predict(flat)

    #     #[N]
    #     predictions = np.argmax(predictions, axis=1)

    #     # print predictions.shape

    #     #  [numticks, numticks]
    #     predictions = predictions.reshape(X.shape)
        

    #     fig, ax = plt.subplots(1, 1)
    #     fig.patch.set_visible(False)
    #     # ax.axis('off')
    #     # ax.set_yticks([])
    #     # ax.set_xticks([])

    #     cs = ax.contourf(X, Y, predictions, levels=range(0,10))

    #     proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])  for pc in cs.collections]

    #     print len(proxy)

    #     plt.legend(proxy, range(0,10))

    #     plt.show()





    if viz_encodings_and_boundaries:


        #lets confirm the boundaries
        # get one class

        # samps = []
        # for i in range(len(train_y)):

        #     if train_y[i] == 9:
        #         samps.append(train_x[i])

        # samps =np.array(samps)
        # print samps.shape

        # encoding = model.encode(samps)


        fig = plt.figure(figsize=(15,8), facecolor='white')
        # fig.patch.set_visible(False)


        #BOUNDARIES
        numticks = 100
        limit_value= 15
        # x_min_max = [-15.,20.]
        # y_min_max = [-15.,20.]

        # x_min_max = [-40.,40.]
        # y_min_max = [-40.,40.]

        # x_min_max = [-20.,20.]
        # y_min_max = [-20.,20.]

        x_min_max = [-limit_value,limit_value]
        y_min_max = [-limit_value,limit_value]

        x = np.linspace(*x_min_max, num=numticks)
        y = np.linspace(*y_min_max, num=numticks)
        X, Y = np.meshgrid(x, y)
        # print X.shape  [numticks, numticks]
        # print Y.shape [numticks, numticks]
        flat = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
        # print flat.shape  [numticks*numticks, 2]
        # fadsf



        # print predictions.shape

        levels = [-1] + range(0,n_all_classes)


        n_plots = 3
        rows=1
        columns = n_plots
        # fig, ax = plt.subplots(rows, n_plots)
        # fig = plt.figure(figsize=(3+columns,3+rows), facecolor='white')



        ten_from_each_encoded = []
        for i in range(n_all_classes):
            ten_from_each_encoded.append(model.encode(ten_from_each[i]))



        for j in range(n_plots):

            ax = plt.subplot2grid((rows,columns), (0,j), frameon=False)#, colspan=3)
            # ax.axis('off')
            # ax.set_yticks([])
            # ax.set_xticks([])
            
            # [N,C]
            predictions = model.predict(flat)

            #[N]
            predictions = np.argmax(predictions, axis=1)
            #  [numticks, numticks]
            predictions = predictions.reshape(X.shape)
            



            cs = ax.contourf(X, Y, predictions, levels=levels)

            proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])  for pc in cs.collections]

            # print len(proxy)

            # plt.legend(proxy, range(0,n_all_classes), fontsize=5)


            # if j!=0:
            #     ax.scatter(encoding.T[0], encoding.T[1], alpha=.2)


            plt.gca().set_aspect('equal', adjustable='box')


            first_legend = plt.legend(proxy, range(0,n_limited_classes), fontsize=5)
            plt.gca().add_artist(first_legend)



            # if j != 0:
            if 1:

                # scats =[]
                # colors = plt.cm.rainbow(np.linspace(0, 1, n_all_classes))

                for c in range(n_all_classes):

                    # aaa = ax.scatter(ten_from_each_encoded[c].T[0], ten_from_each_encoded[c].T[1], alpha=.5, label=str(c), c=colors[c], marker='x', s=4)
                    aaa = ax.scatter(ten_from_each_encoded[c].T[0], ten_from_each_encoded[c].T[1], alpha=.5, label=str(c), c='black', marker='x', s=3)

                    # scats.append(aaa)
                    for ii in range(len(ten_from_each_encoded[c])):

                        plt.text(ten_from_each_encoded[c][ii][0], ten_from_each_encoded[c][ii][1], str(c), color="black", fontsize=6)


                # plt.legend(fontsize=5, loc=2)

        plt.show()



        


    if viz_encodings_and_probabilities:


        fig = plt.figure(figsize=(15,8), facecolor='white')

        numticks = 100
        limit_value= 15

        x_min_max = [-limit_value,limit_value]
        y_min_max = [-limit_value,limit_value]

        x = np.linspace(*x_min_max, num=numticks)
        y = np.linspace(*y_min_max, num=numticks)
        X, Y = np.meshgrid(x, y) # [numticks, numticks]
        flat = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T  #[numticks*numticks, 2]

        rows=2
        columns = 3


        ten_from_each_encoded = []
        for i in range(n_all_classes):
            ten_from_each_encoded.append(model.encode(ten_from_each[i]))


        for r in range(rows):
            for j in range(columns):

                ax = plt.subplot2grid((rows,columns), (r,j), frameon=False)#, colspan=3)
                # ax.axis('off')
                # ax.set_yticks([])
                # ax.set_xticks([])
                
                predictions = model.predict(flat)  # [N,C]
                predictions = np.max(predictions, axis=1)  #[N]
                predictions = predictions.reshape(X.shape) #  [numticks, numticks]

                cs = ax.contourf(X, Y, predictions)

                nm, lbl = cs.legend_elements()
                plt.legend(nm, lbl, fontsize=4) 

                for c in range(n_all_classes):

                    aaa = ax.scatter(ten_from_each_encoded[c].T[0], ten_from_each_encoded[c].T[1], alpha=.5, label=str(c), c='black', marker='x', s=3)

                    for ii in range(len(ten_from_each_encoded[c])):
                        plt.text(ten_from_each_encoded[c][ii][0], ten_from_each_encoded[c][ii][1], str(c), color="black", fontsize=6)

                plt.gca().set_aspect('equal', adjustable='box')


        plt.show()








    print 'Done.'


