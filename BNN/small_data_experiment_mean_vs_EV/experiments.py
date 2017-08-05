



import numpy as np
import tensorflow as tf

from os.path import expanduser
home = expanduser("~")
import time
import pickle

import matplotlib.pyplot as plt


from BNN import BNN





class BNN_all(object):

    def __init__(self, n_classes):
        
        tf.reset_default_graph()

        #Model hyperparameters
        self.learning_rate = .001
        self.rs = 0
        self.input_size = 784
        self.output_size = n_classes

        #Placeholders - Inputs/Targets [B,X]
        self.one_over_N = tf.placeholder(tf.float32, None)
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.y = tf.placeholder(tf.float32, [None, self.output_size])


        BNN_main = BNN([self.input_size, 200, 200, n_classes], [tf.nn.softplus,tf.nn.softplus, None])

        Ws, log_p_W_sum, log_q_W_sum = BNN_main.sample_weights()

        #Feedforward: [B,P,Y], [P], [P]
        self.y2 = BNN_main.feedforward(self.x, Ws)

        #Likelihood [B,P]
        self.softmax_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y2))

        #Objective
        self.cost = self.softmax_error +  self.one_over_N*(-log_p_W_sum + log_q_W_sum) 

        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost)



        #For evaluation
        self.prediction = tf.nn.softmax(self.y2) 

        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.prediction,1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        W_means = BNN_main.sample_weight_means()
        y2_mean = BNN_main.feedforward(self.x, W_means)
        self.prediction_using_mean = tf.nn.softmax(y2_mean) 

        correct_prediction2 = tf.equal(tf.argmax(self.y,1), tf.argmax(self.prediction_using_mean,1))
        self.acc_using_mean = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

        self.softmax_error_using_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y2_mean))



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
                    # one_hot=np.zeros(self.output_size)
                    # one_hot[train_y[data_index]]=1.
                    # batch_y.append(one_hot)
                    batch_y.append(train_y[data_index]) 
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



        if path_to_save_variables != '':
            self.saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables



    def load_vars(self, path_to_load_variables):

        #Load variables
        self.saver.restore(self.sess, path_to_load_variables)
        print 'loaded variables ' + path_to_load_variables





    # def predict(self, samples, using_mean=False):

    #     if not using_mean:
    #         y2 = self.sess.run((self.prediction), feed_dict={self.y1: samples})
    #     else:
    #         y2 = self.sess.run((self.prediction_using_mean), feed_dict={self.y1: samples})

    #     return y2




    def accuracy(self, x, y, using_mean=False):

        if using_mean:
            acc = self.sess.run((self.acc_using_mean), feed_dict={self.x: x, self.y: y})
        else:
            preds = []
            for i in range(20):
                pred = self.sess.run((self.prediction), feed_dict={self.x: x})
                preds.append(pred)
            pred = np.mean(preds, axis=0)

            acc = self.sess.run((self.acc), feed_dict={self.prediction: pred, self.y: y})



        return acc



    def NLL(self, x, y, using_mean=False):


        if using_mean:
            s = self.sess.run((self.softmax_error_using_mean), feed_dict={self.x: x, self.y: y})
        else:
            preds = []
            for i in range(20):
                pred = self.sess.run((self.y2), feed_dict={self.x: x})
                # s = self.sess.run((self.softmax_error), feed_dict={self.x: x, self.y: y})
                preds.append(pred)
            pred = np.mean(preds, axis=0)       
            s = self.sess.run((self.softmax_error), feed_dict={self.y2: pred, self.y: y})     

        return s






    def entropy(self, x, using_mean=False):



        if using_mean:
            pred = self.sess.run((self.prediction_using_mean), feed_dict={self.x: x})
        else:
            predictions = []
            for i in range(20):
                pred = self.sess.run((self.prediction), feed_dict={self.x: x})
                predictions.append(pred)
            pred = np.mean(predictions, axis=0)


        # print pred[:10]

        entropy = -np.sum(pred*np.log(pred),axis=1)

        # print entropy[:10]

        entropy = np.mean(entropy)

        return entropy

































if __name__ == '__main__':


    train = 1
    eval_ = 1


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


    train_x = train_x[:2000]
    train_y = train_y[:2000]

    print train_x.shape
    print train_y.shape





    model = BNN_all(n_classes=n_limited_classes)

    # path_to_load_variables=home+'/Documents/tmp/vars2.ckpt' 
    path_to_load_variables=''
    path_to_save_variables=home+'/Documents/tmp/vars_mean_vs_ev_small.ckpt'


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
    for i in range(len(test_y)):

        if test_y[i] < n_limited_classes:
            new_x_test.append(test_x[i])
            new_y_test.append(test_y[i])
    new_x_test = np.array(new_x_test)
    new_y_test = np.array(new_y_test)



    #Change y to one hot
    train_y_one_hot = []
    for i in range(len(new_y)):
        one_hot=np.zeros(n_limited_classes)
        one_hot[new_y[i]]=1.
        train_y_one_hot.append(one_hot)
    new_y = np.array(train_y_one_hot)

    test_y_one_hot = []
    for i in range(len(new_y_test)):
        one_hot=np.zeros(n_limited_classes)
        one_hot[new_y_test[i]]=1.
        test_y_one_hot.append(one_hot)
    new_y_test = np.array(test_y_one_hot)



    #other half
    new_xo = []
    new_yo = []
    for i in range(len(train_y)):

        if train_y[i] > n_limited_classes:
            new_xo.append(train_x[i])
            new_yo.append(train_y[i])
    new_xo = np.array(new_xo)
    new_yo = np.array(new_yo)
    print 'Limited Dataset other half'
    print new_xo.shape
    print new_yo.shape

    #same on test set
    new_x_testo = []
    new_y_testo = []
    for i in range(len(train_y)):

        if train_y[i] > n_limited_classes:
            new_x_testo.append(train_x[i])
            new_y_testo.append(train_y[i])
    new_x_testo = np.array(new_x_testo)
    new_y_testo = np.array(new_y_testo)


    # #Change y to one hot
    # train_y_one_hot = []
    # for i in range(len(new_yo)):
    #     one_hot=np.zeros(n_limited_classes)
    #     one_hot[new_yo[i]]=1.
    #     train_y_one_hot.append(one_hot)
    # new_yo = np.array(train_y_one_hot)

    # test_y_one_hot = []
    # for i in range(len(new_y_testo)):
    #     one_hot=np.zeros(n_limited_classes)
    #     one_hot[new_y_testo[i]]=1.
    #     test_y_one_hot.append(one_hot)
    # new_y_testo = np.array(test_y_one_hot)








    if train == 1:
        print 'Training'
        model.train(train_x=new_x, train_y=new_y, 
                    epochs=100, batch_size=50, display_step=3000,
                    path_to_load_variables=path_to_load_variables,
                    path_to_save_variables=path_to_save_variables)

        print 'Train Accuracy:', model.accuracy(new_x, new_y)
        print 'Test Accuracy:', model.accuracy(new_x_test, new_y_test)

        print 'Train Accuracy mean:', model.accuracy(new_x, new_y, using_mean=True)
        print 'Test Accuracy mean:', model.accuracy(new_x_test, new_y_test, using_mean=True)

        print 'Train NLL:', model.NLL(new_x, new_y)
        print 'Test NLL:', model.NLL(new_x_test, new_y_test)

        print 'Train NLL mean:', model.NLL(new_x, new_y, using_mean=True)
        print 'Test NLL mean:', model.NLL(new_x_test, new_y_test, using_mean=True)

        print 'Entropy held out:', model.entropy(new_xo)
        print 'Entropy held out mean:', model.entropy(new_xo, using_mean=True)


    else:
        model.load_vars(path_to_load_variables=path_to_save_variables)

        print 'Train Accuracy:', model.accuracy(new_x, new_y)
        print 'Test Accuracy:', model.accuracy(new_x_test, new_y_test)

        print 'Train Accuracy mean:', model.accuracy(new_x, new_y, using_mean=True)
        print 'Test Accuracy mean:', model.accuracy(new_x_test, new_y_test, using_mean=True)

        print 'Train NLL:', model.NLL(new_x, new_y)
        print 'Test NLL:', model.NLL(new_x_test, new_y_test)

        print 'Train NLL mean:', model.NLL(new_x, new_y, using_mean=True)
        print 'Test NLL mean:', model.NLL(new_x_test, new_y_test, using_mean=True)

        print 'Entropy held out:', model.entropy(new_xo)
        print 'Entropy held out mean:', model.entropy(new_xo, using_mean=True)




    print 'Done.'


