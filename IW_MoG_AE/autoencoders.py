
#Generative Autoencoder classes
import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle




class VAE():

    def __init__(self, network_architecture, learning_rate=0.001, batch_size=5, n_particles=3):
        self.network_architecture = network_architecture
        self.transfer_fct = tf.tanh
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_particles = n_particles
        self.n_z = network_architecture["n_z"]
        self.n_input = network_architecture["n_input"]

        #Placeholders - Inputs
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        
        #Variables
        self.network_weights = self._initialize_weights(**self.network_architecture)
        
        #Encoder - Recognition model - p(z|x): recog_mean,z_log_std_sq=[batch_size, n_z]
        self.recog_mean, self.recog_log_std_sq = self._recognition_network(self.network_weights["weights_recog"], self.network_weights["biases_recog"])
        
        #Sample
        eps = tf.random_normal((self.n_particles, self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.recog_mean, tf.mul(tf.sqrt(tf.exp(self.recog_log_std_sq)), eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]
        
        #Decoder - Generative model - p(x|z)
        self.x_reconstr_mean_no_sigmoid = self._generator_network(self.network_weights["weights_gener"], self.network_weights["biases_gener"]) #no sigmoid

        #Objective
        self.elbo = self.elbo(self.x, self.x_reconstr_mean_no_sigmoid, self.z, self.recog_mean, self.recog_log_std_sq)

        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-04).minimize(-self.elbo)

        #For evaluation
        self.log_w = self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid) + self._log_p_z(self.z) - self._log_q_z_given_x(self.z, self.recog_mean, self.recog_log_std_sq)
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

        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
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

        return (z_mean_t, z_log_sigma_sq_t)


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


    def _log_q_z_given_x(self, z, mean, log_var):
        '''
        Log of normal distribution

        z is [n_particles, batch_size, n_z]
        mean is [batch_size, n_z]
        log_var is [batch_size, n_z]
        output is [n_particles, batch_size]
        '''

        # term1 = tf.log(tf.reduce_prod(tf.exp(log_var_sq), reduction_indices=1))
        term1 = tf.reduce_sum(log_var, reduction_indices=1) #sum over dimensions n_z so now its [batch]

        term2 = self.n_z * tf.log(2*math.pi)
        dif = tf.square(z - mean)
        dif_cov = dif / tf.exp(log_var)
        # term3 = tf.reduce_sum(dif_cov * dif, 1) 
        term3 = tf.reduce_sum(dif_cov, 2) #sum over dimensions n_z so now its [particles, batch]

        all_ = term1 + term2 + term3
        log_p_z_given_x = -.5 * all_

        return log_p_z_given_x


    def _log_likelihood(self, t, pred_no_sig):
        '''
        Log of bernoulli distribution

        t is [batch_size, n_input]
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



    def elbo(self, x, x_recon, z, mean, log_var):

        elbo = self._log_likelihood(x, x_recon) + self._log_p_z(z) - self._log_q_z_given_x(z, mean, log_var)

        elbo = tf.reduce_mean(elbo, 1) #average over batch
        elbo = tf.reduce_mean(elbo) #average over particles

        return elbo



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

                    # Fit training using batch data
                    _ = self.sess.run((self.optimizer), feed_dict={self.x: batch})
                    
                    # Display logs per epoch step
                    if step % display_step == 0:

                        cost = self.sess.run((self.elbo), feed_dict={self.x: batch})
                        cost = -cost #because I want to see the NLL

                        print "Stage:" + str(stage)+'/' + str(ending_stage), "Pass", str(pass_)+'/'+str(passes_over_data-1), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/self.batch_size), "cost=", "{:.6f}".format(float(cost))#, 'time', time.time() - start

                        # lw= self.sess.run((self.log_weights), feed_dict={self.x: batch})
                        # print np.exp(lw)

        if path_to_save_variables != '':
            # print 'saving variables to ' + path_to_save_variables
            saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables




    def evaluate(self, datapoints, n_samples, path_to_load_variables=''):
        '''
        Evaluate like the IWAE paper

        Make sure to initilize the model with the right number of particles, nevermind, I just iterate

        For each test datapoint, sample 5000 times, which will give me 5000 weights. 
        Get the average of the weights and log it. This is the NLL of that datapoint.
        Repeat for each datapoint, and average them.
        '''


        saver = tf.train.Saver()
        self.sess = tf.Session()

        saver.restore(self.sess, path_to_load_variables)
        print 'loaded variables ' + path_to_load_variables

        datapoint_log_avg_w = []
        datapoint_index = 0

        n_iters = len(datapoints)/self.batch_size
        for i in range(n_iters):

            print i, '/', n_iters

            #Make batch
            batch = []
            while len(batch) != self.batch_size:

                datapoint = datapoints[datapoint_index]
                batch.append(datapoint)
                datapoint_index +=1

            #for this batch, get n_samples ws. average the ws, and log them.
            log_ws =[]
            n_iters2 = n_samples/self.n_particles
            for j in range(n_iters2):

                log_w = self.sess.run((self.log_w), feed_dict={self.x: batch}) #[n_particles, batch]
                log_ws.append(log_w)

            log_ws = np.array(log_ws) # [iters, n_particles, batch]

            # for each datapoint, average over the iterations and particles
            for j in range(self.batch_size):

                datapoint_log_ws = log_ws[:,:,j]
                #logmeanexp
                max_ = np.max(datapoint_log_ws)
                datapoint_ws = np.exp(datapoint_log_ws - max_)
                mean_datapoint_ws = np.mean(datapoint_ws)
                log_mean_datapoint_ws = np.log(mean_datapoint_ws) + max_

                datapoint_log_avg_w.append(log_mean_datapoint_ws)

        avg = np.mean(datapoint_log_avg_w)

        return avg

















class IWAE(VAE):

    def elbo(self, x, x_recon, z, mean, log_var):

        # [P, B]
        temp_elbo = self._log_likelihood(x, x_recon) + self._log_p_z(z) - self._log_q_z_given_x(z, mean, log_var)

        max_ = tf.reduce_max(temp_elbo, reduction_indices=0) #over particles? so its [B]

        elbo = tf.log(tf.reduce_mean(tf.exp(temp_elbo-max_), 0)) + max_  #mean over particles so its [B]

        elbo = tf.reduce_mean(elbo) #over batch

        return elbo






















class MoG_VAE(VAE):

    def __init__(self, network_architecture, learning_rate=0.001, batch_size=5, n_particles=3, n_clusters=2):
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
        self.log_weights_unormalized = tf.Variable(tf.ones([n_clusters]))
        self.log_weights = self.log_weights_unormalized - tf.reduce_logsumexp(self.log_weights_unormalized, reduction_indices=0, keep_dims=True)
        
        #Encoder - Recognition model - p(z|x): recog_mean,z_log_std_sq=[batch_size, n_z]
        self.recog_means, self.recog_log_vars = self._recognition_network(self.network_weights["weights_recog"], self.network_weights["biases_recog"])
        
        #Sample - the particles are divided amond the clusters evenly
        eps = tf.random_normal((self.n_particles, self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        # self.z = tf.add(self.recog_mean, tf.mul(tf.sqrt(tf.exp(self.recog_log_std_sq)), eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]
        #Transform the N(0,I) samples with the cluster means and vars
        self.z = self.transform_eps(eps, self.recog_means, self.recog_log_vars) #[P,B,Z]

        #Decoder - Generative model - p(x|z)
        self.x_reconstr_mean_no_sigmoid = self._generator_network(self.network_weights["weights_gener"], self.network_weights["biases_gener"]) #no sigmoid

        #Objective
        self.log_q = self._log_q_z_given_x(self.z, self.recog_means, self.recog_log_vars, self.log_weights)
        self.log_p = self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid) + self._log_p_z(self.z)
        self.log_w = self.log_p - self.log_q #[P,B]
        # [C,B]
        self.weighted_log_w = self.weight_by_cluster(self.log_w, self.log_weights)

        # self.mean_log_w = tf.reduce_mean(self.log_w, reduction_indices=0) #over particles, so its [B]
        # log_weights_reshaped = tf.reshape(self.log_weights, [self.n_clusters, 1]) #[C,1]

        temp = tf.reduce_mean(self.weighted_log_w, reduction_indices=0) #over clusters, so its [B]
        self.elbo = tf.reduce_mean(temp, reduction_indices=0) #over batch, so its a scalar

        # self.elbo = self.elbo(self.x, self.x_reconstr_mean_no_sigmoid, self.z, self.recog_mean, self.recog_log_var)

        # Optimization
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-04).minimize(-self.elbo)

        #For evaluation
        # self.log_w = self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid) + self._log_p_z(self.z) - self._log_q_z_given_x(self.z, self.recog_mean, self.recog_log_std_sq)



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
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z*self.n_clusters))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z*self.n_clusters], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z*self.n_clusters], dtype=tf.float32))}
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



    def transform_eps(self, eps, means, log_vars):
        '''
        Divides the samples among the mixture of Gaussians

        eps are samples from N(0,I)  [P, B, Z]
        means are [B,Z*C]
        log_vars are [B,Z*C]
        output are [P,B,Z]

        So need to split P by C 
        '''

        # split, slice,
        # or I could reshape it to [B,C,Z]

        samps_per_cluster = self.n_particles / self.n_clusters

        zs = []
        for cluster in range(self.n_clusters):

            # [P/C,B,Z]
            this_cluster_samps = tf.slice(eps, [samps_per_cluster*cluster,0,0], [samps_per_cluster, self.batch_size, self.n_z])
            # [B,Z]
            this_mean = tf.slice(means, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            this_log_var = tf.slice(log_vars, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            # [P/C,B,Z]
            this_z = tf.add(this_mean, tf.mul(tf.sqrt(tf.exp(this_log_var)), this_cluster_samps)) #uses broadcasting, z=[n_parts, n_batches, n_z]

            zs.append(this_z)

        # [n_clusters, samps_per_cluster, batch, n_z]
        zs = tf.pack(zs)
        # [n_clusters*samps_per_cluster, batch, n_z] = [P,B,Z]
        zs = tf.reshape(zs, [self.n_clusters*samps_per_cluster, self.batch_size, self.n_z])

        return zs


    def _log_normal(self, x, mean, log_var):
        '''
        x is [P, B, D]
        mean is [B,D]
        log_var is [B,D]
        '''

        term1 = self.n_z * tf.log(2*math.pi)
        term2 = tf.reduce_sum(log_var, reduction_indices=1) #sum over dimensions, now its [B]

        term3 = tf.square(x - mean) / tf.exp(log_var)
        term3 = tf.reduce_sum(term3, 2) #sum over dimensions so now its [particles, batch]

        all_ = term1 + term2 + term3
        log_normal = -.5 * all_  

        return log_normal



    def _log_q_z_given_x(self, z, means, log_vars, log_mixture_weights):
        '''
        Log of normal distribution

        z is [n_particles, batch_size, n_z]
        mean is [batch_size, n_z]
        log_var is [batch_size, n_z]
        output is [n_particles, batch_size]
        '''

        # samps_per_cluster = self.n_particles / self.n_clusters

        #Get log_qi_z for each z for each cluster, which becomes becomes [C,P,B]

        log_q_z__ = []
        for cluster in range(self.n_clusters):

            # this_weight = tf.slice(mixture_weights, [cluster,0], [1, self.batch_size])
            this_mean = tf.slice(means, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            this_log_var = tf.slice(log_vars, [0,cluster*self.n_z], [self.batch_size, self.n_z])

            #[P,B]
            log_q_z = self._log_normal(z, this_mean, this_log_var)

            # max_1 = tf.reduce_max(log_q_z, 0) #over particles
            # q_z = tf.exp(log_q_z - max_1)
            log_q_z__.append(log_q_z)


        # [C,P,B]
        log_q_z__ = tf.pack(log_q_z__)
        # prob_sum = tf.reduce_sum(prob_sum, 0) #over clusters so it should be [P,B]

        log_cluster_ws = tf.reshape(log_mixture_weights, [self.n_clusters, 1, 1])

        log_weighted_q = log_q_z__ + log_cluster_ws

        log_weighted_q = tf.reduce_logsumexp(log_weighted_q, reduction_indices=0, keep_dims=True) #over clusters, so its [P,B]
        log_weighted_q = tf.reshape(log_weighted_q, [self.n_particles, self.batch_size])

        return log_weighted_q




    def weight_by_cluster(self, log_w, log_weights):

        samps_per_cluster = self.n_particles / self.n_clusters

        weighted_log_w = []
        for cluster in range(self.n_clusters):

            samples_from_this_cluster = tf.slice(log_w, [cluster*samps_per_cluster, 0], [samps_per_cluster, self.batch_size])
            avg = tf.reduce_mean(samples_from_this_cluster, reduction_indices=0) #over particles, so its [B]

            this_weight = tf.slice(log_weights, [cluster], [1])

            #add weight, which is weighting these samples, so if it have low weight, the samples shouldnt matter much
            this_weighted_samps = avg + this_weight

            weighted_log_w.append(this_weighted_samps)
            
        # [C,B]
        weighted_log_w = tf.pack(weighted_log_w)

        return weighted_log_w























class MoG_VAE2(VAE):

    def __init__(self, network_architecture, learning_rate=0.001, batch_size=5, n_particles=3, n_clusters=2):
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
        self.log_weights_unormalized = tf.Variable(tf.ones([n_clusters]))
        self.log_weights = self.log_weights_unormalized - tf.reduce_logsumexp(self.log_weights_unormalized, reduction_indices=0, keep_dims=True)
        
        #Encoder - Recognition model - p(z|x): recog_mean,z_log_std_sq=[batch_size, n_z]
        self.recog_means, self.recog_log_vars = self._recognition_network(self.network_weights["weights_recog"], self.network_weights["biases_recog"])
        
        #Sample - the particles are divided amond the clusters evenly
        eps = tf.random_normal((self.n_particles, self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        # self.z = tf.add(self.recog_mean, tf.mul(tf.sqrt(tf.exp(self.recog_log_std_sq)), eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]
        #Transform the N(0,I) samples with the cluster means and vars
        self.z = self.transform_eps(eps, self.recog_means, self.recog_log_vars) #[P,B,Z]

        #Decoder - Generative model - p(x|z)
        self.x_reconstr_mean_no_sigmoid = self._generator_network(self.network_weights["weights_gener"], self.network_weights["biases_gener"]) #no sigmoid

        #Objective
        self.log_q = self._log_q_z_given_x(self.z, self.recog_means, self.recog_log_vars, self.log_weights)
        self.log_p = self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid) + self._log_p_z(self.z)
        self.log_w_ = self.log_p - self.log_q #[P,B]
        # [P,B]
        self.log_w = self.weight_by_cluster(self.log_w_, self.log_weights)

        self.log_w_batch = tf.reduce_mean(self.log_w, reduction_indices=0) #over particles, so its [B]
        # log_weights_reshaped = tf.reshape(self.log_weights, [self.n_clusters, 1]) #[C,1]

        # temp = tf.reduce_mean(self.weighted_log_w, reduction_indices=0) #over clusters, so its [B]
        self.elbo = tf.reduce_mean(self.log_w_batch, reduction_indices=0) #over batch, so its a scalar

        # self.elbo = self.elbo(self.x, self.x_reconstr_mean_no_sigmoid, self.z, self.recog_mean, self.recog_log_var)

        # Optimization
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-04).minimize(-self.elbo)

        #For evaluation
        # self.log_w = self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid) + self._log_p_z(self.z) - self._log_q_z_given_x(self.z, self.recog_mean, self.recog_log_std_sq)



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
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z*self.n_clusters))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z*self.n_clusters], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z*self.n_clusters], dtype=tf.float32))}
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



    def transform_eps(self, eps, means, log_vars):
        '''
        Divides the samples among the mixture of Gaussians

        eps are samples from N(0,I)  [P, B, Z]
        means are [B,Z*C]
        log_vars are [B,Z*C]
        output are [P,B,Z]

        So need to split P by C 
        '''

        # split, slice,
        # or I could reshape it to [B,C,Z]

        samps_per_cluster = self.n_particles / self.n_clusters

        zs = []
        for cluster in range(self.n_clusters):

            # [P/C,B,Z]
            this_cluster_samps = tf.slice(eps, [samps_per_cluster*cluster,0,0], [samps_per_cluster, self.batch_size, self.n_z])
            # [B,Z]
            this_mean = tf.slice(means, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            this_log_var = tf.slice(log_vars, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            # [P/C,B,Z]
            this_z = tf.add(this_mean, tf.mul(tf.sqrt(tf.exp(this_log_var)), this_cluster_samps)) #uses broadcasting, z=[n_parts, n_batches, n_z]

            zs.append(this_z)

        # [n_clusters, samps_per_cluster, batch, n_z]
        zs = tf.pack(zs)
        # [n_clusters*samps_per_cluster, batch, n_z] = [P,B,Z]
        zs = tf.reshape(zs, [self.n_clusters*samps_per_cluster, self.batch_size, self.n_z])

        return zs


    def _log_normal(self, x, mean, log_var):
        '''
        x is [P, B, D]
        mean is [B,D]
        log_var is [B,D]
        '''

        term1 = self.n_z * tf.log(2*math.pi)
        term2 = tf.reduce_sum(log_var, reduction_indices=1) #sum over dimensions, now its [B]

        term3 = tf.square(x - mean) / tf.exp(log_var)
        term3 = tf.reduce_sum(term3, 2) #sum over dimensions so now its [particles, batch]

        all_ = term1 + term2 + term3
        log_normal = -.5 * all_  

        return log_normal



    def _log_q_z_given_x(self, z, means, log_vars, log_mixture_weights):
        '''
        Log of normal distribution

        z is [n_particles, batch_size, n_z]
        mean is [batch_size, n_z]
        log_var is [batch_size, n_z]
        output is [n_particles, batch_size]
        '''

        # samps_per_cluster = self.n_particles / self.n_clusters

        #Get log_qi_z for each z for each cluster, which becomes becomes [C,P,B]

        log_q_z__ = []
        for cluster in range(self.n_clusters):

            # this_weight = tf.slice(mixture_weights, [cluster,0], [1, self.batch_size])
            this_mean = tf.slice(means, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            this_log_var = tf.slice(log_vars, [0,cluster*self.n_z], [self.batch_size, self.n_z])

            #[P,B]
            log_q_z = self._log_normal(z, this_mean, this_log_var)

            # max_1 = tf.reduce_max(log_q_z, 0) #over particles
            # q_z = tf.exp(log_q_z - max_1)
            log_q_z__.append(log_q_z)


        # [C,P,B]
        log_q_z__ = tf.pack(log_q_z__)
        # prob_sum = tf.reduce_sum(prob_sum, 0) #over clusters so it should be [P,B]

        log_cluster_ws = tf.reshape(log_mixture_weights, [self.n_clusters, 1, 1])

        log_weighted_q = log_q_z__ + log_cluster_ws

        log_weighted_q = tf.reduce_logsumexp(log_weighted_q, reduction_indices=0, keep_dims=True) #over clusters, so its [P,B]
        log_weighted_q = tf.reshape(log_weighted_q, [self.n_particles, self.batch_size])

        return log_weighted_q




    def weight_by_cluster(self, log_w, log_weights):

        samps_per_cluster = self.n_particles / self.n_clusters

        weights= []
        for cluster in range(self.n_clusters):

            # samples_from_this_cluster = tf.slice(log_w, [cluster*samps_per_cluster, 0], [samps_per_cluster, self.batch_size])
            # avg = tf.reduce_mean(samples_from_this_cluster, reduction_indices=0) #over particles, so its [B]

            this_weight = tf.exp(tf.slice(log_weights, [cluster], [1]))

            tiled = tf.tile(this_weight, [samps_per_cluster])

            #add weight, which is weighting these samples, so if it have low weight, the samples shouldnt matter much
            # this_weighted_samps = avg + this_weight

            weights.append(tiled)
            
        # [C,samps_per_cluster]
        weights = tf.pack(weights)
        weights = tf.reshape(weights, [self.n_particles,1]) * self.n_clusters


        log_w = weights * log_w

        return log_w



    def encode(self, data):

        return self.sess.run([self.recog_mean, self.recog_log_std_sq], feed_dict={self.x=data})


    def decode(self, sample):

        return self.sess.run(self.x_reconstr_mean, feed_dict={self.z=samples})



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



# class MoG_IWAE(MoG_VAE):

#     def elbo(self, x, x_recon, z, mean, log_var):

#         # [P, B]
#         temp_elbo = self._log_likelihood(x, x_recon) + self._log_p_z(z) - self._log_q_z_given_x(z, mean, log_var)

#         max_ = tf.reduce_max(temp_elbo, reduction_indices=0) #over particles? so its [B]

#         elbo = tf.log(tf.reduce_mean(tf.exp(temp_elbo-max_), 0)) + max_  #mean over particles so its [B]

#         elbo = tf.reduce_mean(elbo) #over batch

#         return elbo




























class IW_MoG_AE(IWAE):

    def __init__(self, network_architecture, learning_rate=0.001, batch_size=5, n_particles=3, n_clusters=3):
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
        
        #Encoder - Recognition model - p(z|x): recog_mean,z_log_std_sq=[batch_size, n_z*n_clusters]
        self.recog_means, self.recog_log_vars = self._recognition_network(self.network_weights["weights_recog"], self.network_weights["biases_recog"])

        #Sample
        eps = tf.random_normal((self.n_particles, self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        
        #Transform the N(0,I) samples with the cluster means and vars
        self.z = self.transform_eps(eps, self.recog_means, self.recog_log_vars)
        
        #Decoder - Generative model - p(x|z)
        self.x_reconstr_mean_no_sigmoid = self._generator_network(self.network_weights["weights_gener"], self.network_weights["biases_gener"]) #no sigmoid

        #Objective
        self.elbo = self.elbo(self.x, self.x_reconstr_mean_no_sigmoid, self.z, self.recog_means, self.recog_log_vars)

        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-04).minimize(-self.elbo)

        #For evaluation
        self.log_w = self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid) + self._log_p_z(self.z) - self._log_q_z_given_x(self.z, self.recog_means, self.recog_log_vars)

        #FOR debugging
        self.log_q_z_g_x = self._log_q_z_given_x(self.z, self.recog_means, self.recog_log_vars)
        self.log_p_z = self._log_p_z(self.z)
        self.ll = self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid)



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
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z*self.n_clusters))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z*self.n_clusters], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z*self.n_clusters], dtype=tf.float32))}
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



    def transform_eps(self, eps, means, log_vars):
        '''
        Divides the samples among the mixture of Gaussians

        eps are samples from N(0,I)  [P, B, Z]
        means are [B,Z*C]
        log_vars are [B,Z*C]
        output are [P,B,Z]

        So need to split P by C 
        '''

        # split, slice,
        # or I could reshape it to [B,C,Z]

        samps_per_cluster = self.n_particles / self.n_clusters

        zs = []
        for cluster in range(self.n_clusters):

            # [P/C,B,Z]
            this_cluster_samps = tf.slice(eps, [samps_per_cluster*cluster,0,0], [samps_per_cluster, self.batch_size, self.n_z])
            # [B,Z]
            this_mean = tf.slice(means, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            this_log_var = tf.slice(log_vars, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            # [P/C,B,Z]
            this_z = tf.add(this_mean, tf.mul(tf.sqrt(tf.exp(this_log_var)), this_cluster_samps)) #uses broadcasting, z=[n_parts, n_batches, n_z]

            zs.append(this_z)

        # [n_clusters, samps_per_cluster, batch, n_z]
        zs = tf.pack(zs)
        # [n_clusters*samps_per_cluster, batch, n_z] = [P,B,Z]
        zs = tf.reshape(zs, [self.n_clusters*samps_per_cluster, self.batch_size, self.n_z])

        return zs



    def _log_normal(self, x, mean, log_var):
        '''
        x is [P, B, D]
        mean is [B,D]
        log_var is [B,D]
        '''

        term1 = self.n_z * tf.log(2*math.pi)
        term2 = tf.reduce_sum(log_var, reduction_indices=1) #sum over dimensions, now its [B]

        term3 = tf.square(x - mean) / tf.exp(log_var)
        term3 = tf.reduce_sum(term3, 2) #sum over dimensions so now its [particles, batch]

        all_ = term1 + term2 + term3
        log_normal = -.5 * all_  

        return log_normal




    def _log_q_z_given_x(self, z, means, log_vars):
        '''
        Log of normal distribution

        z is [n_particles, batch_size, n_z]
        mean is [batch_size, n_z]
        log_var is [batch_size, n_z]
        output is [n_particles, batch_size]
        '''


        #First get the qi(z|x) -> the probability of sampling of coming from its own cluster
        #Now get the cluster weights
        samps_per_cluster = self.n_particles / self.n_clusters

        # log_qi_zs = []
        cluster_weights = []
        for cluster in range(self.n_clusters):

            # [P/C,B,Z]
            this_cluster_samps = tf.slice(z, [samps_per_cluster*cluster,0,0], [samps_per_cluster, self.batch_size, self.n_z])
            # [B,Z]
            this_mean = tf.slice(means, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            this_log_var = tf.slice(log_vars, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            # [P/C,B,Z]
            log_qi_z = self._log_normal(this_cluster_samps, this_mean, this_log_var)

            this_reconstruction = tf.slice(self.x_reconstr_mean_no_sigmoid, [samps_per_cluster*cluster,0,0], [samps_per_cluster, self.batch_size, self.n_input])

            log_p_x_z = self._log_likelihood(self.x, this_reconstruction) + self._log_p_z(this_cluster_samps)

            log_w = log_p_x_z-log_qi_z #[P,B]

            #do this so if they all have low prob, we wont devide by zero
            # max_ = tf.reduce_max(log_w, 0) # over smaples I think so its [B]

            #cant minus max, since the max will be different for different clusters
            # unnormalized_cluster_weight = tf.reduce_sum(tf.exp(log_w - max_), 0) #over samples so its [B]

            cluster_weights.append(log_w)

        # [n_clusters, particles, batch]
        cluster_weights = tf.pack(cluster_weights)

        max_ = tf.reduce_max(cluster_weights, 1, keep_dims=True)
        cluster_weights = cluster_weights - max_
        cluster_weights = tf.reduce_sum(cluster_weights, 1) # over samples [clusters, B]

        # batch sum of weights
        total_weight = tf.reduce_sum(cluster_weights, 0)  #over clusters now its [B]

        # self.normalized_cluster_weights = cluster_weights / total_weight
        normalized_cluster_weights = cluster_weights / total_weight #[C,B]


        # prob_sum = []
        # for cluster in range(self.n_clusters):

        #     this_weight = tf.slice(normalized_cluster_weights, [cluster,0], [1, self.batch_size])
        #     this_mean = tf.slice(means, [0,cluster*self.n_z], [self.batch_size, self.n_z])
        #     this_log_var = tf.slice(log_vars, [0,cluster*self.n_z], [self.batch_size, self.n_z])

        #     log_q_z = self._log_normal(z, this_mean, this_log_var)
        #     # max_1 = tf.reduce_max(log_q_z, 0) #over particles
        #     # q_z = tf.exp(log_q_z - max_1)
        #     prob_sum.append(log_q_z)

        # # [C,P,B]
        # prob_sum = tf.pack(prob_sum)

        # max_1 = tf.reduce_max(prob_sum, 1, keep_dims=True) #over particles
        # q_z = tf.exp(prob_sum - max_1)# [C,P,B]


        # normalized_cluster_weights = tf.reshape(normalized_cluster_weights, [self.n_clusters, 1, self.batch_size])

        # weighted_q_z = q_z* normalized_cluster_weights

        # prob_sum = max_1+ tf.reduce_sum(prob_sum, 0) #over clusters so it should be [P,B]


        prob_sum = []
        for cluster in range(self.n_clusters):

            this_weight = tf.slice(normalized_cluster_weights, [cluster,0], [1, self.batch_size])
            this_mean = tf.slice(means, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            this_log_var = tf.slice(log_vars, [0,cluster*self.n_z], [self.batch_size, self.n_z])

            log_q_z = self._log_normal(z, this_mean, this_log_var)
            max_1 = tf.reduce_max(log_q_z, 0) #over particles
            q_z = tf.exp(log_q_z - max_1)
            prob_sum.append(this_weight * q_z)


        prob_sum = tf.pack(prob_sum)
        prob_sum = tf.reduce_sum(prob_sum, 0) #over clusters so it should be [P,B]


        return tf.log(prob_sum)




    def q_z_given_x(self, z, means, log_vars):
        '''
        Log of normal distribution

        z is [n_particles, batch_size, n_z]
        mean is [batch_size, n_z]
        log_var is [batch_size, n_z]
        output is [n_particles, batch_size]
        '''


        #First get the qi(z|x) -> the probability of sampling of coming from its own cluster
        #Now get the cluster weights
        samps_per_cluster = self.n_particles / self.n_clusters

        # log_qi_zs = []
        cluster_weights = []
        for cluster in range(self.n_clusters):

            # [P/C,B,Z]
            this_cluster_samps = tf.slice(z, [samps_per_cluster*cluster,0,0], [samps_per_cluster, self.batch_size, self.n_z])
            # [B,Z]
            this_mean = tf.slice(means, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            this_log_var = tf.slice(log_vars, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            # [P/C,B,Z]
            log_qi_z = self._log_normal(this_cluster_samps, this_mean, this_log_var)

            this_reconstruction = tf.slice(self.x_reconstr_mean_no_sigmoid, [samps_per_cluster*cluster,0,0], [samps_per_cluster, self.batch_size, self.n_input])

            log_p_x_z = self._log_likelihood(self.x, this_reconstruction) + self._log_p_z(this_cluster_samps)

            log_w = log_p_x_z-log_qi_z #[P,B]

            #do this so if they all have low prob, we wont devide by zero
            # max_ = tf.reduce_max(log_w, 0) # over smaples I think so its [B]

            #cant minus max, since the max will be different for different clusters
            # unnormalized_cluster_weight = tf.reduce_sum(tf.exp(log_w - max_), 0) #over samples so its [B]

            cluster_weights.append(log_w)

        # [n_clusters, particles, batch]
        cluster_weights = tf.pack(cluster_weights)

        max_ = tf.reduce_max(cluster_weights, 1, keep_dims=True)
        cluster_weights = cluster_weights - max_
        cluster_weights = tf.reduce_sum(cluster_weights, 1) # over samples [clusters, B]

        # batch sum of weights
        total_weight = tf.reduce_sum(cluster_weights, 0)  #over clusters now its [B]

        # self.normalized_cluster_weights = cluster_weights / total_weight
        normalized_cluster_weights = cluster_weights / total_weight


        prob_sum = []
        for cluster in range(self.n_clusters):

            this_weight = tf.slice(normalized_cluster_weights, [cluster,0], [1, self.batch_size])
            this_mean = tf.slice(means, [0,cluster*self.n_z], [self.batch_size, self.n_z])
            this_log_var = tf.slice(log_vars, [0,cluster*self.n_z], [self.batch_size, self.n_z])

            log_q_z = self._log_normal(z, this_mean, this_log_var)
            max_1 = tf.reduce_max(log_q_z, 0) #over particles
            q_z = tf.exp(log_q_z - max_1)
            prob_sum.append(this_weight * q_z)


        prob_sum = tf.pack(prob_sum)
        prob_sum = tf.reduce_sum(prob_sum, 0) #over clusters so it should be [P,B]



        return prob_sum



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

            self.learning_rate = .001 * 10.**(-stage/7.)
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

                    # Fit training using batch data
                    # _ = self.sess.run((self.optimizer), feed_dict={self.x: batch})

                    _, a, b, c, d, e, f, g, h = self.sess.run([self.optimizer, self.log_q_z_g_x, self.log_p_z, self.ll, self.recog_means, self.recog_log_vars, self.z, self.x_reconstr_mean_no_sigmoid,self.elbo], feed_dict={self.x: batch})


                    if np.isinf(a).any():
                        print 'aaaqqq'
                    if np.isinf(b).any():
                        print 'bbbqqq'
                    if np.isinf(c).any():
                        print 'cccqqq'
                    if np.isinf(d).any():
                        print 'dddqqq'
                    if np.isinf(e).any():
                        print 'eeqeqqeq'
                    if np.isinf(f).any():
                        print 'fffqqq'
                    if np.isinf(g).any():
                        print 'ggqeqqeq'
                    if np.isinf(h).any():
                        print 'gghhhqeqqeq'


                    if np.isnan(a).any():
                        print 'aaa'
                    if np.isnan(b).any():
                        print 'bbb'
                    if np.isnan(c).any():
                        print 'ccc'
                    if np.isnan(d).any():
                        print 'ddd'
                    if np.isnan(e).any():
                        print 'eee'
                    if np.isnan(f).any():
                        print 'fff'
                    if np.isnan(g).any():
                        print 'ggg'
                    if np.isnan(h).any():
                        print 'hhh'
                        fadsfa
                          
                    # Display logs per epoch step
                    if step % display_step == 0:

                        cost = self.sess.run((self.elbo), feed_dict={self.x: batch})
                        cost = -cost #because I want to see the NLL

                        print "Stage:" + str(stage)+'/' + str(ending_stage), "Pass", str(pass_)+'/'+str(passes_over_data-1), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/self.batch_size), "cost=", "{:.6f}".format(cost)#, 'time', time.time() - start


        if path_to_save_variables != '':
            # print 'saving variables to ' + path_to_save_variables
            saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables


    # def elbo(self, x, x_recon, z, mean, log_var):

    #     # [P, B]
    #     temp_elbo = tf.exp(self._log_likelihood(x, x_recon) + self._log_p_z(z)) - self.q_z_given_x(z, mean, log_var)

    #     elbo = tf.log(tf.reduce_mean(temp_elbo, 0))

    #     # max_ = tf.reduce_max(temp_elbo, reduction_indices=0) #over particles? so its [B]

    #     # elbo = tf.log(tf.reduce_mean(tf.exp(temp_elbo-max_), 0)) + max_  #mean over particles so its [B]

    #     elbo = tf.reduce_mean(elbo) #over batch

    #     return elbo




