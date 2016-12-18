


import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle







class MoG_VAE():

    def __init__(self, network_architecture, learning_rate=0.001, batch_size=5, n_particles=4, n_clusters=2):

        tf.reset_default_graph()


        self.network_architecture = network_architecture
        self.transfer_fct = tf.tanh
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_particles = n_particles
        self.n_z = network_architecture["n_z"]
        self.n_input = network_architecture["n_input"]
        self.n_clusters = n_clusters

        #Placeholders - Inputs
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        
        #Variables
        self.network_weights = self._initialize_weights(**self.network_architecture)
        
        #Encoder - Recognition model - p(z|x): recog_mean,z_log_std_sq=[batch_size, n_z*C]
        self.recog_means, self.recog_log_vars, self.weights = self._recognition_network(self.network_weights["weights_recog"], self.network_weights["biases_recog"])

        #Sample - the particles are divided among the clusters evenly [P,B,Z]
        eps = tf.random_normal((self.n_particles, self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        #Transform the N(0,I) samples with the cluster means and vars
        self.z = self.transform_eps(eps, self.recog_means, self.recog_log_vars) #[P,B,Z]

        #Decoder - Generative model - p(x|z) [P,B,D]
        self.x_reconstr_mean_no_sigmoid = self._generator_network(self.network_weights["weights_gener"], self.network_weights["biases_gener"]) #no sigmoid

        #Objective
        self.log_q = self._log_q_z_given_x(self.z, self.recog_means, self.recog_log_vars, self.weights)
        self.log_p_z = self._log_p_z(self.z)
        self.log_p = self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid)
        self.log_w_ = self.log_p  + self.log_p_z - self.log_q #[P,B]
        # [B,C]
        self.log_w = self.weight_by_cluster(self.log_w_, self.weights)
        self.log_w_batch = tf.reduce_mean(self.log_w, reduction_indices=1) #over clusters, so its [B]
        
        # vars1   = tf.trainable_variables() 
        # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars1])*0.1

        self.elbo = tf.reduce_mean(self.log_w_batch, reduction_indices=0) #- lossL2 #+ (.1*(tf.reduce_sum(tf.log(self.weights))))  #over batch, so its a scalar



        # Optimization
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-04).minimize(-self.elbo)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(-self.elbo)


        #For evaluation
        self.x_reconstr_mean = tf.nn.sigmoid(self.x_reconstr_mean_no_sigmoid)


    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        #Recognition model output changed to n_z*n_clusters

        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z*self.n_clusters)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z*self.n_clusters)),
            'out_log_weights': tf.Variable(xavier_init(n_hidden_recog_2, self.n_clusters))}

        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z*self.n_clusters], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z*self.n_clusters], dtype=tf.float32)),
            'out_log_weights': tf.Variable(tf.zeros([self.n_clusters], dtype=tf.float32))}

        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights


    def _recognition_network(self, weights, biases):

        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])) 

        z_mean_t = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        z_log_sigma_sq_t = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])
        z_weights= tf.nn.softmax(tf.add(tf.matmul(layer_2, weights['out_log_weights']), biases['out_log_weights']))

        return (z_mean_t, z_log_sigma_sq_t, z_weights)


    def _generator_network(self, weights, biases):

        z = tf.reshape(self.z, [self.n_particles*self.batch_size, self.n_z])

        layer_1 = self.transfer_fct(tf.add(tf.matmul(z, weights['h1']), 
                                           biases['b1'])) #shape is now [p*b,l1]

        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 

        x_reconstr_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                     biases['out_mean'])

        x_reconstr_mean = tf.reshape(x_reconstr_mean, [self.n_particles, self.batch_size, self.n_input])

        return x_reconstr_mean




    def transform_eps(self, eps, means, log_vars):
        '''
        Divides the samples among the mixture of Gaussians

        eps are samples from N(0,I)  [P, B, Z]
        means are [B,Z*C]
        log_vars are [B,Z*C]
        output are [P,B,Z]

        So need to split P by C 
        '''

        samps_per_cluster = int(np.ceil(self.n_particles / float(self.n_clusters)))

        # [P/C,C,B,Z] 
        eps_reshaped = tf.reshape(eps, [samps_per_cluster, self.n_clusters, self.batch_size, self.n_z])

        # # [1,C,B,Z]
        # means_reshaped = tf.reshape(means, [1, self.n_clusters, self.batch_size, self.n_z])
        # # [1,C,B,Z]
        # log_vars_reshaped = tf.reshape(log_vars, [1, self.n_clusters, self.batch_size, self.n_z])

        # [B,C,Z]
        means_reshaped = tf.reshape(means, [self.batch_size, self.n_clusters, self.n_z])
        # [1,B,C,Z]
        means_reshaped = tf.pack([means_reshaped])
        # [1,C,B,Z]
        means_reshaped = tf.transpose(means_reshaped, [0, 2, 1, 3])


        # [B,C,Z]
        log_vars_reshaped = tf.reshape(log_vars, [self.batch_size, self.n_clusters, self.n_z])
        # [1,B,C,Z]
        log_vars_reshaped = tf.pack([log_vars_reshaped])
        # [1,C,B,Z]
        log_vars_reshaped = tf.transpose(log_vars_reshaped, [0, 2, 1, 3])

        # [P/C,C,B,Z] 
        zs = tf.add(means_reshaped, tf.mul(tf.sqrt(tf.exp(log_vars_reshaped)), eps_reshaped))
        # [P,B,Z] 
        zs = tf.reshape(zs, [self.n_clusters*samps_per_cluster, self.batch_size, self.n_z])

        return zs


    # def _log_normal(self, x, mean, log_var):
    #     '''
    #     x is [P/C, C, B, Z]
    #     mean is [1,C,B,Z]
    #     log_var is [1,C,B,Z]
    #     return [P/C,C,B]
    #     '''

    #     # [1] scalar
    #     term1 = self.n_z * tf.log(2*math.pi)
    #     # [1,C,B,Z]
    #     # mean_reshape = tf.reshape(mean, [1,self.n_clusters, self.batch_size, self.n_z])
    #     # log_var_reshape = tf.reshape(log_var, [1,self.n_clusters, self.batch_size, self.n_z])
    #     # [1,C,B]
    #     term2 = tf.reduce_sum(log_var, reduction_indices=3) #sum over dimensions
    #     # [P/C, C, B, Z]
    #     term3 = tf.square(x - mean) / tf.exp(log_var)
    #     # [P/C, C, B]
    #     term3 = tf.reduce_sum(term3, 3)

    #     all_ = term1 + term2 + term3
    #     log_normal = -.5 * all_  

    #     return log_normal


    def _log_normal(self, x, means, log_vars):
        '''
        x is [P, B, Z]
        mean, log_var is [batch_size, n_z*C]
        return [P,C,B]
        '''

        # [P*C,B,Z]
        x = tf.tile(x, [self.n_clusters,1,1])
        # [P,C,B,Z]
        x = tf.reshape(x, [self.n_particles,self.n_clusters,self.batch_size, self.n_z])

        # [B,C,Z]
        means_reshaped = tf.reshape(means, [self.batch_size, self.n_clusters, self.n_z])
        # [1,B,C,Z]
        means_reshaped = tf.pack([means_reshaped])
        # [1,C,B,Z]
        means_reshaped = tf.transpose(means_reshaped, [0, 2, 1, 3])

        # [B,C,Z]
        log_vars_reshaped = tf.reshape(log_vars, [self.batch_size, self.n_clusters, self.n_z])
        # [1,B,C,Z]
        log_vars_reshaped = tf.pack([log_vars_reshaped])
        # [1,C,B,Z]
        log_vars_reshaped = tf.transpose(log_vars_reshaped, [0, 2, 1, 3])


        # [1] scalar
        term1 = self.n_z * tf.log(2*math.pi)
        # [1,C,B]
        term2 = tf.reduce_sum(log_vars_reshaped, reduction_indices=3) #sum over dimensions
        # [P, C, B, Z]
        term3 = tf.square(x - means_reshaped) / tf.exp(log_vars_reshaped)
        # [P, C, B]
        term3 = tf.reduce_sum(term3, 3)

        all_ = term1 + term2 + term3
        log_normal = -.5 * all_  

        return log_normal




    def _log_q_z_given_x(self, z, means, log_vars, mixture_weights):
        '''
        Log of normal distribution

        z is [n_particles, batch_size, n_z]
        means is [batch_size, n_z*C]
        log_vars is [batch_size, n_z*C]
        mixture_weights is [B,C]
        output is [n_particles, batch_size]
        '''

        #[P,C,B]
        log_q_z_per_cluster = self._log_normal(z, means, log_vars)
        self.log_q_z_per_cluster = log_q_z_per_cluster
        # print log_q_z_per_cluster
        # fasdf

        # [C,B]
        weights_reshaped = tf.transpose(mixture_weights, [1,0])
        # print weights_reshaped


        #THIS WAS THE ERROR, the reshaping screwed it up
        #Fix: pack instead of reshape to add a dimension

        # [1,C,B]
        # weights_reshaped = tf.reshape(mixture_weights, [1, self.n_clusters, self.batch_size])
        # [1,C,B]
        weights_reshaped = tf.pack([weights_reshaped])
        # print weights_reshaped
        self.weights_reshaped = weights_reshaped
        # [P,C,B]
        log_pi_qc_z = log_q_z_per_cluster + tf.log(weights_reshaped)
        self.log_pi_qc_z = log_pi_qc_z

        # log_pi_qc_z = tf.log(tf.exp(log_q_z_per_cluster) * weights_reshaped)


        #logsumexp over clusters [P,B]
        # log_q_z = tf.reduce_logsumexp(log_pi_qc_z, reduction_indices=1)
        max_ = tf.reduce_max(log_pi_qc_z, reduction_indices=1)
        max_2 = tf.reshape(max_, [self.n_particles, 1, self.batch_size])
        log_q_z = tf.log(tf.reduce_sum(tf.exp(log_pi_qc_z-max_2), reduction_indices=1)) + max_

        return log_q_z



    def _log_p_z(self, z):
        '''
        Log of normal distribution with zero mean and one var

        z is [n_particles, batch_size, n_z]
        output is [n_particles, batch_size]
        '''

        # term1 = 0
        term2 = self.n_z * tf.log(2*math.pi)
        term3 = tf.reduce_sum(tf.square(z), 2) #sum over dimensions n_z so now its [particles, batch]

        all_ = term2 + term3
        log_p_z = -.5 * all_

        return log_p_z


    def _log_likelihood(self, t, pred_no_sig):
        '''
        Log of bernoulli distribution

        target t is [batch_size, n_input]
        pred_no_sig is [n_particles, batch_size, n_input] 
        output is [n_particles, batch_size]
        '''

        reconstr_loss = \
                tf.reduce_sum(tf.maximum(pred_no_sig, 0) 
                            - pred_no_sig * t
                            + tf.log(1 + tf.exp(-tf.abs(pred_no_sig))),
                             2) #sum over dimensions

        #negative because the above calculated the NLL, so this is returning the LL
        return -reconstr_loss



    def weight_by_cluster(self, log_w, normalized_weights):
        '''
        log_w: [P,B]
        normalized_weights: [B,C]
        log_w output: [B,C]
        '''

        samps_per_cluster = int(np.ceil(self.n_particles / float(self.n_clusters)))

        #[P/C,C,B]
        log_w_reshaped = tf.reshape(log_w, [samps_per_cluster, self.n_clusters, self.batch_size])
        #[C,B]
        log_w_reshaped = tf.reduce_mean(log_w_reshaped, reduction_indices=0)
        #[B,C]
        log_w = tf.transpose(log_w_reshaped, [1,0])
        #[B,C]
        # weights_reshaped = tf.reshape(normalized_weights, [self.batch_size, self.n_clusters])
        #[B,C]
        # log_w = log_w + tf.log(normalized_weights)
        log_w = log_w * normalized_weights



        return log_w


    # def train(self, train_x, valid_x=[], display_step=5, path_to_load_variables='', path_to_save_variables='', starting_stage=0, ending_stage=5, path_to_save_training_info=''):
    #     '''
    #     Train.
    #     Use early stopping
    #     '''

    #     n_datapoints = len(train_x)
    #     max_epochs = 1000
    #     best_valid_elbo = -99999
    #     best_valid_epochs = 0
    #     early_stopping_count = 0


    #     #Load variables
    #     saver = tf.train.Saver()
    #     self.sess = tf.Session()
    #     if path_to_load_variables == '':
    #         self.sess.run(tf.initialize_all_variables())
    #     else:
    #         #Load variables
    #         saver.restore(self.sess, path_to_load_variables)
    #         print 'loaded variables ' + path_to_load_variables
        

    #     for epoch in range(max_epochs):

    #         #shuffle the data
    #         arr = np.arange(len(train_x))
    #         np.random.shuffle(arr)
    #         train_x = train_x[arr]

    #         data_index = 0
    #         for step in range(n_datapoints/self.batch_size):

    #             #Make batch
    #             batch = []
    #             while len(batch) != self.batch_size:
    #                 datapoint = train_x[data_index]
    #                 batch.append(datapoint)
    #                 data_index +=1

    #             # Fit training using batch data
    #             # _ = self.sess.run((self.optimizer), feed_dict={self.x: batch})
    #             _, cost_ = self.sess.run((self.optimizer, self.elbo), feed_dict={self.x: batch})

    #             if np.isnan(cost_).any():
    #                 q, p, p2 = self.sess.run((self._log_q_z_given_x(self.z, self.recog_means, self.recog_log_vars, self.weights), self._log_p_z(self.z), self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid)), feed_dict={self.x: batch})
    #                 print 'nan!'
    #                 print np.sum(q)
    #                 print np.sum(p)
    #                 print np.sum(p2)
    #                 fsafd
                
    #             # Display logs per epoch step
    #             if step % display_step == 0:

    #                 cost = self.sess.run((self.elbo), feed_dict={self.x: batch})
    #                 cost = -cost #because I want to see the NLL

    #                 print "Epoch", str(epoch), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/self.batch_size), "cost=", "{:.6f}".format(float(cost))#, 'time', time.time() - start

    #                 # lw= self.sess.run((self.log_weights), feed_dict={self.x: batch})
    #                 # print np.exp(lw)



    #         if epoch %20 ==0 and epoch != 0:

    #             valid_elbos =[]
    #             #calculate the validation score
    #             data_index = 0
    #             for step in range(len(valid_x)/self.batch_size):
    #                 batch = []
    #                 while len(batch) != self.batch_size:
    #                     datapoint = valid_x[data_index]
    #                     batch.append(datapoint)
    #                     data_index +=1

    #                 elbo = self.sess.run((self.elbo), feed_dict={self.x: batch})
    #                 valid_elbos.append(elbo)

    #             valid_elbo = np.mean(valid_elbos)
    #             print 'validation elbo', valid_elbo

    #             if valid_elbo > best_valid_elbo or best_valid_elbo == -99999:
    #                 best_valid_elbo = valid_elbo
    #                 early_stopping_count =0
    #                 best_valid_epochs = epoch

    #                 # save if its the best
    #                 if path_to_save_variables != '':
    #                     # print 'saving variables to ' + path_to_save_variables
    #                     saver.save(self.sess, path_to_save_variables)
    #                     print 'Saved variables to ' + path_to_save_variables

    #             else:
    #                 early_stopping_count += 1
    #                 print 'early stopping', early_stopping_count
    #                 if early_stopping_count == 2:
    #                     print 'early stopping break'
    #                     break

    #     return best_valid_epochs








    def encode(self, data):

        return self.sess.run([self.recog_means, self.recog_log_vars, self.weights], feed_dict={self.x:data})


    def decode(self, sample):

        return self.sess.run(self.x_reconstr_mean_no_sigmoid, feed_dict={self.z:sample})



    def load_parameters(self, path_to_load_variables):




        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            # self.sess.run(tf.initialize_all_variables())
            print 'No path tpo variables'
            error
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables






    def train(self, train_x, valid_x=[], display_step=5, path_to_load_variables='', path_to_save_variables='', starting_stage=0, ending_stage=5, path_to_save_training_info=''):
        '''
        Train.
        Use early stopping, actually no, because I want it to be equal for each model. Time? Epochs? 
        I'll do stages for now.
        '''

        n_datapoints = len(train_x)
        
        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            self.sess.run(tf.initialize_all_variables())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables

        #start = time.time()
        for stage in range(starting_stage,ending_stage+1):

            self.learning_rate = .001 * 10.**(-stage/float(ending_stage))
            print 'learning rate', self.learning_rate

            passes_over_data = 3**stage

            for pass_ in range(passes_over_data):

                #shuffle the data
                arr = np.arange(len(train_x))
                np.random.shuffle(arr)
                train_x = train_x[arr]

                data_index = 0
                for step in range(n_datapoints/self.batch_size):

                    #Make batch
                    batch = []
                    while len(batch) != self.batch_size:
                        datapoint = train_x[data_index]
                        batch.append(datapoint)
                        data_index +=1

                    
                    # if step==0:
                    #     w= self.sess.run((self.weights), feed_dict={self.x: batch})
                    #     print w
                    #     qz, pz, pxz = self.sess.run((self.log_q, self.log_p_z, self.log_p), feed_dict={self.x: batch})
                    #     print np.mean(qz), np.mean(pz), np.mean(pxz)

                    # Fit training using batch data
                    _ = self.sess.run((self.optimizer), feed_dict={self.x: batch})
                    # _, cost_ = self.sess.run((self.optimizer, self.elbo), feed_dict={self.x: batch})

                    qz, pz, pxz = self.sess.run((self.log_q, self.log_p_z, self.log_p), feed_dict={self.x: batch})
                    print np.mean(pxz), np.mean(pz), np.mean(qz)
                    print '1'
                    log_q_z_per_cluster= self.sess.run((self.log_q_z_per_cluster), feed_dict={self.x: batch})
                    print log_q_z_per_cluster

                    print '1.5'
                    weights_reshaped = self.sess.run((self.weights_reshaped), feed_dict={self.x: batch})
                    print weights_reshaped

                    print '2'
                    log_pi_qc_z = self.sess.run((self.log_pi_qc_z), feed_dict={self.x: batch})
                    print log_pi_qc_z

                    print '3'
                    print qz
                    print 

                    if np.isnan(qz).any():
                        print 'nan!'
                        fsafd     
                    

                    # if step %5==0:
                    #     w= self.sess.run((self.weights), feed_dict={self.x: batch})
                    #     print w



                        # if step%333==0 and step!=0:
                        #     fadsf
                    # Display logs per epoch step
                    if step % display_step == 0:

                        cost = self.sess.run((self.elbo), feed_dict={self.x: batch})
                        cost = -cost #because I want to see the NLL

                        print "Stage:" + str(stage)+'/' + str(ending_stage), "Pass", str(pass_)+'/'+str(passes_over_data-1), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/self.batch_size), "cost=", "{:.6f}".format(float(cost))#, 'time', time.time() - start
                        
                        qz, pz, pxz = self.sess.run((self.log_q, self.log_p_z, self.log_p), feed_dict={self.x: batch})
                        print np.mean(pxz), np.mean(pz), np.mean(qz)                        
                        w= self.sess.run((self.weights), feed_dict={self.x: batch})
                        print w

                        if np.isnan(cost).any():

                            q, p, p2 = self.sess.run((self._log_q_z_given_x(self.z, self.recog_means, self.recog_log_vars, self.weights), self._log_p_z(self.z), self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid)), feed_dict={self.x: batch})
                            print 'nan!'
                            print np.sum(q)
                            print np.sum(p)
                            print np.sum(p2)
                            fsafd

        if path_to_save_variables != '':
            # print 'saving variables to ' + path_to_save_variables
            saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables










