


# MNF  Bayesian Neural Network

#step 1: add auxiliary variable z, complete.
#step 2: add flows to q(z) and r(z|W)


import numpy as np
import numpy.random as npr
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle




class MNF(object):

    def __init__(self, network_architecture):
        
        tf.reset_default_graph()

        #Model hyperparameters
        self.act_func = tf.nn.softplus #tf.tanh
        self.learning_rate = .001
        self.rs = 0
        self.input_size = network_architecture[0]
        self.output_size = network_architecture[-1]
        self.net = network_architecture

        #Placeholders - Inputs/Targets [B,X]
        # self.batch_size = tf.placeholder(tf.int32, None)
        self.n_particles = tf.placeholder(tf.int32, None)
        self.batch_fraction_of_dataset = tf.placeholder(tf.float32, None)
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.batch_size = tf.shape(self.x)[0]
        self.y = tf.placeholder(tf.float32, [None, self.output_size])

        #Feedforward: [B,P,Y], [P], [P]
        y_hat, log_p_W, log_q_W = self.model(self.x)

        #Likelihood [B,P]
        log_p_y_hat = self.log_likelihood(self.y, y_hat)

        #Objective
        self.elbo = self.objective(log_p_y_hat, log_p_W, log_q_W, self.batch_fraction_of_dataset)
        self.iwae_elbo_test = self.iwae_objective_test(log_p_y_hat)

        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(-self.elbo)

        #For evaluation
        self.prediction = tf.nn.softmax(y_hat) 

        #To init variables
        self.init_vars = tf.global_variables_initializer()

        #For loadind/saving variables
        self.saver = tf.train.Saver()

        #For debugging 
        self.vars = tf.trainable_variables()
        self.grads = tf.gradients(self.elbo, tf.trainable_variables())

        #to make sure im not adding nodes to the graph
        tf.get_default_graph().finalize()

        #Start session
        self.sess = tf.Session()


    def log_normal(self, position, mean, log_var):
        '''
        Log of normal distribution
        position is [P, D]
        mean is [D]
        log_var is [D]
        output is [P]
        '''

        n_D = tf.to_float(tf.shape(mean)[0])
        term1 = n_D * tf.log(2*math.pi)
        term2 = tf.reduce_sum(log_var, 0) #sum over D [1]
        dif_cov = tf.square(position - mean) / tf.exp(log_var)
        term3 = tf.reduce_sum(dif_cov, 1) #sum over D [P]
        all_ = term1 + term2 + term3
        log_normal_ = -.5 * all_

        return log_normal_


    def log_likelihood(self, y, y_hat):
        '''
        Calculate p(y|x,theta)
        y: [B,Y]
        y_hat: [B,P,Y]
        '''
        self.y1 = y
        #tile y for each particle [B,P,Y]
        self.y2 = tf.tile(self.y1, [1,self.n_particles])
        self.y3 = tf.reshape(self.y2, [self.batch_size,self.n_particles,self.output_size])

        # [B,P]
        log_likelihood = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.y3)



        return -log_likelihood


    def model(self, x):
        '''
        x: [B,X]
        y_hat: [B,P,O]
        log_p_W: [P]
        log_q_W: [P]
        '''

        net = self.net

        # def xavier_init(fan_in, fan_out, constant=1): 
        #     """ Xavier initialization of network weights"""
        #     # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
        #     low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
        #     high = constant*np.sqrt(6.0/(fan_in + fan_out))
        #     return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
        

        #[B,X]
        cur_val = x
        #tile x for each particle [B,P,X]
        cur_val = tf.tile(cur_val, [1,self.n_particles])
        cur_val = tf.reshape(cur_val, [self.batch_size,self.n_particles,self.input_size])


        log_p_W_sum = 0
        log_q_W_sum = 0
        log_q_z_sum = 0
        log_r_z_sum = 0

        for layer_i in range(len(net)-1):

            input_size_i = net[layer_i]+1 #plus 1 for bias
            output_size_i = net[layer_i+1] #plus 1 because we want layer i+1

            z_means = tf.Variable(tf.random_normal([input_size_i], stddev=0.1))
            z_logvars = tf.Variable(tf.random_normal([input_size_i], stddev=0.1) - 5.)

            #Sample z   [I]
            eps = tf.random_normal([input_size_i], 0, 1, seed=self.rs)
            z = tf.add(z_means, tf.multiply(tf.sqrt(tf.exp(z_logvars)), eps))
            z = tf.reshape(z, [input_size_i, 1])  #[I,1]

            #Define variables [IS,OS]
            # W_means = tf.Variable(xavier_init(input_size_i, output_size_i))
            W_means = tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1))
            #[I,O]
            W_means = W_means * z

            # W_logvars = tf.Variable(xavier_init(input_size_i, output_size_i) - 5.)
            W_logvars = tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1) - 5.)

            #Sample weights [IS,OS]*[P,IS,OS]=[P,IS,OS]
            eps = tf.random_normal((self.n_particles, input_size_i, output_size_i), 0, 1, seed=self.rs)
            W = tf.add(W_means, tf.multiply(tf.sqrt(tf.exp(W_logvars)), eps))

            #Compute probs of samples 

            flat_z = tf.reshape(z,[1,input_size_i]) #[1,I]
            #q(z)  [1]
            log_q_z_sum += self.log_normal(flat_z, z_means, z_logvars)
            # r(z|W)
            c = tf.Variable(tf.random_normal([1, 1, input_size_i], stddev=0.1)) #[1,1,I]
            c = tf.tile(c, [self.n_particles, 1, 1]) #[P,1,I]
            cW = tf.matmul(c, W) # [P,1,O]
            cW = tf.tanh(cW)
            b1 = tf.Variable(tf.random_normal([1,input_size_i,1], stddev=0.1)) #[1,I,1]
            b1 = tf.tile(b1, [self.n_particles, 1,1]) #[P,I,1]
            b1cW = tf.matmul(b1, cW) #[P,I,O] 
            ones = tf.ones([self.n_particles, output_size_i, 1]) / tf.to_float(output_size_i) #[P,O,1]
            b1cW = tf.matmul(b1cW, ones) #[P,I,1]
            b1cW = tf.reshape(b1cW, [self.n_particles*input_size_i]) #[P*I]

            b2 = tf.Variable(tf.random_normal([1,input_size_i,1], stddev=0.1)) #[1,I,1]
            b2 = tf.tile(b2, [self.n_particles, 1,1]) #[P,I,1]
            b2cW = tf.matmul(b2, cW) #[P,I,O] 
            b2cW = tf.matmul(b2cW, ones) #[P,I,1]
            b2cW = tf.reshape(b2cW, [self.n_particles*input_size_i]) #[P*I]


            flat_z = tf.tile(flat_z, [1,self.n_particles])
            print flat_z
            print b1cW
            print b2cW
            log_r_z_sum += self.log_normal(flat_z, b1cW, b2cW) #[1]



            flat_w = tf.reshape(W,[self.n_particles,input_size_i*output_size_i]) #[P,IS*OS]
            flat_W_means = tf.reshape(W_means, [input_size_i*output_size_i]) #[IS*OS]
            flat_W_logvars = tf.reshape(W_logvars, [input_size_i*output_size_i]) #[IS*OS]
            # p(W) q(W|z)  [P]
            log_p_W_sum += self.log_normal(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))
            log_q_W_sum += self.log_normal(flat_w, flat_W_means, flat_W_logvars)

            # if layer_i==0:
            #     self.debug_pw = self.log_normal(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])))

            #Concat 1 to input for biases  [B,P,X]->[B,P,X+1]
            cur_val = tf.concat([cur_val,tf.ones([self.batch_size,self.n_particles, 1])], axis=2)
            #[B,P,X]->[B,P,1,X]
            cur_val = tf.reshape(cur_val, [self.batch_size, self.n_particles, 1, input_size_i])
            #[P,X,X']->[B,P,X,X']
            W = tf.tile(W, [self.batch_size,1,1])
            W = tf.reshape(W, [self.batch_size, self.n_particles, input_size_i, output_size_i])

            #Forward Propagate  [B,P,1,X]*[B,P,X,X']->[B,P,1,X']
            if layer_i != len(net)-2:
                cur_val = self.act_func(tf.matmul(cur_val, W))
            else:
                cur_val = tf.matmul(cur_val, W)

            #[B,P,1,X']->[B,P,X']
            cur_val = tf.reshape(cur_val, [self.batch_size,self.n_particles,output_size_i])

        return cur_val, log_p_W_sum + log_r_z_sum - log_q_z_sum, log_q_W_sum


    def objective(self, logpy, logpW, logqW, batch_frac):
        '''
        logpy:[B,P]
        logpW:[P]
        logqW:[P]
        elbo: [1]
        '''

        # logpy = tf.reduce_mean(logpy,axis=1)
        # self.logpy = tf.reduce_sum(logpy,axis=0)

        self.logpy=tf.reduce_mean(logpy)
        self.logpW = tf.reduce_mean(logpW)
        self.logqW = tf.reduce_mean(logqW)

        # #Average over particles [B]
        # elbo = tf.reduce_mean(logpy + (batch_frac*logpW) - logqW, axis=1)
        # #Average over batch [1]
        # elbo = tf.reduce_mean(elbo, axis=0)

        elbo = self.logpy + batch_frac*(self.logpW - self.logqW)

        return elbo



    def iwae_objective_test(self, logpy):
        '''
        logpy:[B,P]
        logpW:[P]
        logqW:[P]
        elbo: [1]
        '''

        temp_elbo = tf.reduce_mean(logpy, axis=0)   #[P]
        max_ = tf.reduce_max(temp_elbo, axis=0)
        iwae_elbo = tf.log(tf.reduce_mean(tf.exp(temp_elbo-max_))) + max_  #[1]

        return iwae_elbo







    def train(self, train_x, train_y, valid_x=[], valid_y=[], display_step=[1,10], path_to_load_variables='', path_to_save_variables='', epochs=10, batch_size=20, n_particles=3):
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
                                                        self.batch_size: batch_size,
                                                        self.n_particles: n_particles, 
                                                        self.batch_fraction_of_dataset: 1./float(n_datapoints)})

                # Display logs per epoch step
                if step % display_step[1] == 0 and epoch % display_step[0] ==0:

                    cost,logpy,logpW,logqW,pred = self.sess.run((self.elbo,self.logpy, self.logpW, self.logqW,self.prediction), feed_dict={self.x: batch, self.y: batch_y, 
                                                        self.n_particles:n_particles, 
                                                        self.batch_fraction_of_dataset: 1./float(n_datapoints)})


                    # cost = -cost #because I want to see the NLL
                    print "Epoch", str(epoch+1)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "elbo=", "{:.6f}".format(float(cost)),logpy,logpW,logqW #, 'time', time.time() - start
                    # print 'target'
                    print ["{:.2f}".format(float(x)) for x in batch_y[0]] 
                    # print 'prediciton'
                    # print pred.shape
                    print ["{:.2f}".format(float(x)) for x in pred[0][0]] 
                    print


        if path_to_save_variables != '':
            self.saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables



    def eval(self, data_x, data_y, path_to_load_variables='', batch_size=20, n_particles=3):
        '''
        Test.
        '''
        # random_seed=1
        # rs=npr.RandomState(random_seed)
        n_datapoints = len(data_y)
        # arr = np.arange(n_datapoints)

        if path_to_load_variables == '':
            self.sess.run(self.init_vars)

        else:
            #Load variables
            self.saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables


        test_iwaes = []

        data_index = 0
        for step in range(n_datapoints/batch_size):

            #Make batch
            batch = []
            batch_y = []
            while len(batch) != batch_size:
                batch.append(data_x[data_index]) 
                one_hot=np.zeros(10)
                one_hot[data_y[data_index]]=1.
                batch_y.append(one_hot)
                data_index +=1

            # Fit training using batch data
            iwae = self.sess.run((self.iwae_elbo_test), feed_dict={self.x: batch, self.y: batch_y, 
                                                    self.n_particles: n_particles})

            test_iwaes.append(iwae)


        return np.mean(test_iwaes)










if __name__ == '__main__':

    net = [784,20,10] 

    model = MNF(net)

    print 'Loading data'
    with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
        mnist_data = pickle.load(f)

    train_x = mnist_data[0][0]
    train_y = mnist_data[0][1]
    valid_x = mnist_data[1][0]
    valid_y = mnist_data[1][1]
    test_x = mnist_data[2][0]
    test_y = mnist_data[2][1]

    # train_x = train_x[:100]
    # train_y = train_y[:100]

    # path_to_load_variables=home+'/Documents/tmp/vars.ckpt' 
    path_to_load_variables=''
    # path_to_save_variables=home+'/Documents/tmp/vars2.ckpt'
    path_to_save_variables=''


    print 'Training'
    model.train(train_x=train_x, train_y=train_y, 
                epochs=50, batch_size=20, n_particles=2, display_step=[1,1000],
                path_to_load_variables=path_to_load_variables,
                path_to_save_variables=path_to_save_variables)


    print 'Done.'












