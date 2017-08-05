

import cPickle
import numpy as np
import tensorflow as tf
from os.path import expanduser
home = expanduser("~")

import matplotlib.pyplot as plt


from BNN import BNN


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict







class BNN_full(object):

    def __init__(self, n_classes):
        
        tf.reset_default_graph()

        #Model hyperparameters
        # self.act_func = tf.nn.softplus #tf.tanh
        self.learning_rate = .0001
        self.rs = 0
        self.input_size = 3072
        self.output_size = n_classes

        #Placeholders - Inputs/Targets [B,X]
        self.one_over_N = tf.placeholder(tf.float32, None)
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.y = tf.placeholder(tf.float32, [None, self.output_size])

        B = tf.shape(self.x)[0]

        out = tf.reshape(self.x, [B, 32,32,3])
        conv1_weights = tf.Variable(tf.truncated_normal([5, 5, 3, 20], stddev=0.1))
        conv1_biases = tf.Variable(tf.truncated_normal([20], stddev=0.1))
        out = tf.nn.conv2d(out,conv1_weights,strides=[1, 2, 2, 1],padding='VALID')
        out = tf.nn.relu(tf.nn.bias_add(out, conv1_biases))
        out = tf.nn.max_pool(out,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
        out = tf.nn.lrn(out)


        out = tf.reshape(out, [B, -1])
        # dim = out.get_shape()[1].value

        # BNN_model = BNN([self.input_size, 100, 100, n_classes], [tf.nn.softplus,tf.nn.softplus, None])
        BNN_model = BNN([980, 100, 100, n_classes], [tf.nn.softplus,tf.nn.softplus, None])



        Ws, log_p_W_sum, log_q_W_sum = BNN_model.sample_weights()


        #Feedforward: [B,P,Y], [P], [P]
        # self.y2 = BNN_model.feedforward(self.x, Ws)
        self.y2 = BNN_model.feedforward(out, Ws)


        #Likelihood [B,P]
        self.softmax_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y2))
        self.reg =  .01*self.one_over_N*(-log_p_W_sum*.0000001 + log_q_W_sum) 

        #Objective
        self.cost = self.softmax_error + self.reg


        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost)






        #For evaluation
        self.prediction = tf.nn.softmax(self.y2) 
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.prediction,1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        W_means,log_p_W_sum = BNN_model.sample_weight_means()
        # y2_mean = BNN_model.feedforward(self.x, W_means)
        y2_mean = BNN_model.feedforward(out, W_means)

        self.prediction_using_mean = tf.nn.softmax(y2_mean) 
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.prediction_using_mean,1))
        self.acc2 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.softmax_error2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y2_mean))


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
                    batch_y.append(train_y[data_index])
                    data_index +=1

                # Fit training using batch data
                _ = self.sess.run((self.optimizer), feed_dict={self.x: batch, self.y: batch_y, 
                                                        self.one_over_N: one_over_N})

                # Display logs per epoch step
                if step % display_step == 0:

                    # cost,logpy,logpW,logqW,pred = self.sess.run((self.cost,self.logpy, self.logpW, self.logqW,self.prediction), feed_dict={self.x: batch, self.y: batch_y, 
                    #                                     self.batch_fraction_of_dataset: 1./float(n_datapoints)})

                    cost, se, reg, acc = self.sess.run((self.cost, self.softmax_error, self.reg, self.acc), feed_dict={self.x: batch, self.y: batch_y, 
                                                        self.one_over_N: one_over_N})



                    # cost = -cost #because I want to see the NLL
                    print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost)), se, reg, acc#,logpy,logpW,logqW #, 'time', time.time() - start



        if path_to_save_variables != '':
            self.saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables



    def load_vars(self, path_to_load_variables):

        #Load variables
        self.saver.restore(self.sess, path_to_load_variables)
        print 'loaded variables ' + path_to_load_variables
























if __name__ == '__main__':


    train = 0

    eval_ = 1




    print 'Loading data'
    file_ = home+'/Documents/cifar-10-batches-py/data_batch_'

    for i in range(1,6):
        file__ = file_ + str(i)
        b1 = unpickle(file__)
        if i ==1:
            data_x = b1['data']
            data_y = b1['labels']
        else:
            data_x = np.concatenate([data_x, b1['data']], axis=0)
            data_y = np.concatenate([data_y, b1['labels']], axis=0)

    file__ = home+'/Documents/cifar-10-batches-py/test_batch'
    b1 = unpickle(file__)
    test_x = b1['data']
    test_y = b1['labels']



    n_limited_classes = 10

    #Change y to one hot
    train_y_one_hot = []
    for i in range(len(data_y)):
        one_hot=np.zeros(n_limited_classes)
        one_hot[data_y[i]]=1.
        train_y_one_hot.append(one_hot)
    data_y = np.array(train_y_one_hot)


    data_x = data_x[:500]
    data_y = data_y[:500]

    print data_x.shape
    print data_y.shape

    test_y_one_hot = []
    for i in range(len(test_y)):
        one_hot=np.zeros(n_limited_classes)
        one_hot[test_y[i]]=1.
        test_y_one_hot.append(one_hot)
    test_y = np.array(test_y_one_hot)

    print test_x.shape
    print test_y.shape


    # n = 100
    # data_x=data_x[:n]
    # X = data_x.reshape(n, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    # # Y = np.array(Y)

    # #Visualizing CIFAR 10
    # fig, axes1 = plt.subplots(5,5,figsize=(3,3))
    # for j in range(5):
    #     for k in range(5):
    #         i = np.random.choice(range(len(X)))
    #         axes1[j][k].set_axis_off()
    #         axes1[j][k].imshow(X[i:i+1][0])


    # plt.show()

    # print data_x.shape
    # print data_y.shape



    model = BNN_full(n_classes=n_limited_classes)

    # path_to_load_variables=home+'/Documents/tmp/vars_5_classes_back_to_normal.ckpt'
    path_to_load_variables=''
    # path_to_save_variables=home+'/Documents/tmp/vars_5_classes_relu.ckpt'
    path_to_save_variables=home+'/Documents/tmp/vars_cifar.ckpt'




    if train == 1:
        print 'Training'
        model.train(train_x=data_x, train_y=data_y, 
                    epochs=1000, batch_size=50, display_step=3000,
                    path_to_load_variables=path_to_load_variables,
                    path_to_save_variables=path_to_save_variables)


        acc = model.sess.run((model.acc), feed_dict={model.x: data_x, model.y: data_y})
        print acc

        # acc = model.sess.run((model.acc), feed_dict={model.x: data_x, model.y: data_y})
        # print acc


    #     print 'Train Accuracy:', model.accuracy(new_x, new_y)
    #     print 'Test Accuracy:', model.accuracy(new_x_test, new_y_test)


    else:
        model.load_vars(path_to_load_variables=path_to_save_variables)

        # print 'Train Accuracy:', model.accuracy(new_x, new_y)
        # print 'Test Accuracy:', model.accuracy(new_x_test, new_y_test)





    if eval_ == 1:

        preds = []
        for i in range(100):
            print i
            pred = model.sess.run((model.prediction), feed_dict={model.x: data_x})
            preds.append(pred)
        preds = np.array(preds)
        pred = np.mean(preds, axis=0)
        acc, se = model.sess.run((model.acc, model.softmax_error), feed_dict={model.x: data_x, model.prediction: pred, model.y: data_y})
        print 'Train', acc, se




        preds = []
        for i in range(100):
            print i
            pred = model.sess.run((model.prediction), feed_dict={model.x: test_x})
            preds.append(pred)
        preds = np.array(preds)
        pred = np.mean(preds, axis=0)
        acc, se = model.sess.run((model.acc, model.softmax_error), feed_dict={model.x: test_x, model.prediction: pred, model.y: test_y})
        print 'Test', acc, se



        acc, se = model.sess.run((model.acc2, model.softmax_error2), feed_dict={model.x: data_x, model.y: data_y})
        print 'mean Train', acc, se
        acc, se = model.sess.run((model.acc2, model.softmax_error2), feed_dict={model.x: test_x, model.y: test_y})
        print 'mean Test', acc, se








































