
#Neural Network

import numpy as np
import tensorflow as tf



class dropout_NN(object):

    def __init__(self, network_architecture, act_functions, lmba, keep_prob):
       
        tf.reset_default_graph()

        self.keep_prob = keep_prob

        self.rs = 0
        self.input_size = network_architecture[0]
        self.output_size = network_architecture[-1]
        self.act_functions = act_functions
        self.learning_rate = .0001

        self.net = network_architecture

        #Placeholders - Inputs/Targets [B,X]
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.y = tf.placeholder(tf.float32, [None, self.output_size])

        self.Ws = self.init_weights()

        #Feedforward
        self.y_ = self.feedforward(self.x)

        #Likelihood
        self.softmax_error_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_))
        self.reg = lmba*self.weight_decay()

        #Objective
        self.cost = self.softmax_error_train + self.reg


        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost)





        #For evaluation
        y_avg = self.avg_feedforward(self.x)
        self.softmax_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_avg))

        self.prediction = tf.nn.softmax(y_avg) 
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









    def init_weights(self):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        Ws = []

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Define variables [IS,OS]
            Ws.append(tf.Variable(xavier_init(input_size_i, output_size_i)))

        return Ws





    def weight_decay(self):

        l2 = 0
        for weight_layer in self.Ws:

            l2 += tf.reduce_sum(tf.square(weight_layer))

        return l2







    def feedforward(self, x):
        '''
        x: [B,X]
        y_hat: [B,O]
        '''

        B = tf.shape(x)[0]
        net = self.net

        #[B,X]
        cur_val = x

        for layer_i in range(len(net)-1):

            #Concat 1 to input for biases  [B,P,X]->[B,P,X+1]
            cur_val = tf.concat([cur_val,tf.ones([B, 1])], axis=1)

            cur_val = tf.matmul(cur_val, self.Ws[layer_i])


            if self.act_functions[layer_i] != None:
                cur_val = self.act_functions[layer_i](cur_val)

                cur_val = tf.nn.dropout(cur_val, self.keep_prob)

        return cur_val






    def avg_feedforward(self, x):

        preds = []

        for i in range(50):

            preds.append(self.feedforward(self.x))

        preds = tf.stack(preds)

        preds = tf.reduce_mean(preds, axis=0)

        return preds
































