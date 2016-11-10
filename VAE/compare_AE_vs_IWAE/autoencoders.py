

#Generative Autoencoder classes


import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
# import imageio



class VAE():

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=5, n_particles=3):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_particles = n_particles
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # Create autoencoder network
        # self._create_network()
        self.network_weights = self._initialize_weights(**self.network_architecture)

        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss())
       
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)



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
            


    def _recognition_network(self, x_t, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.

        layer_1 = self.transfer_fct(tf.add(tf.matmul(x_t, weights['h1']), biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])) 

        z_mean_t = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        z_log_sigma_sq_t = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])

        return (z_mean_t, z_log_sigma_sq_t)

    def _generator_network(self, z, weights, biases, sigmoid=True):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # Used for reconstruction

        layer_1 = self.transfer_fct(tf.add(tf.matmul(z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 

        if sigmoid:
            x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                     biases['out_mean']))

        else:
            x_reconstr_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                     biases['out_mean'])

        return x_reconstr_mean
    

    def _log_p_z(self, z):

        #Get log(p(z))
        #This is just exp of standard normal

        term1 = 0
        term2 = self.network_architecture["n_z"] * tf.log(2*math.pi)
        dif = tf.square(z)
        # dif_cov = dif 
        term3 = tf.reduce_sum(dif, 1)

        all_ = term1 + term2 + term3
        log_p_z = -.5 * all_

        return log_p_z

    def _log_p_z_given_x(self, z, mean, log_var_sq):

        #Get log(p(z|x))
        #This is just exp of a normal with some mean and var

        # term1 = tf.log(tf.reduce_prod(tf.exp(log_var_sq), reduction_indices=1))
        term1 = tf.reduce_sum(log_var_sq, reduction_indices=1)

        term2 = self.network_architecture["n_z"] * tf.log(2*math.pi)
        dif = tf.square(z - mean)
        dif_cov = dif / tf.exp(log_var_sq)
        # term3 = tf.reduce_sum(dif_cov * dif, 1) 
        term3 = tf.reduce_sum(dif_cov, 1) 


        all_ = term1 + term2 + term3
        log_p_z_given_x = -.5 * all_


        return log_p_z_given_x



    def loss(self):
        '''
        Negative Log Likelihood Lower Bound : -ELBO
        '''

        recog_mean, recog_log_sigma_sq = self._recognition_network(self.x, self.network_weights["weights_recog"], self.network_weights["biases_recog"])

        reconstr_loss_list = []
        prior_loss_list = []
        recognition_loss_list = []
        for particle in range(self.n_particles):

            eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), 0, 1, dtype=tf.float32)
            z = tf.add(recog_mean, tf.mul(tf.sqrt(tf.exp(recog_log_sigma_sq)), eps))

            prior_loss = self._log_p_z(z)

            recognition_loss = self._log_p_z_given_x(z, recog_mean, recog_log_sigma_sq)

            #Calculate reconstruction error
            reconstructed_mean = self._generator_network(z, self.network_weights["weights_gener"], self.network_weights["biases_gener"], sigmoid=False)
            #this sum is over the dimensions 
            reconstr_loss = \
                    tf.reduce_sum(tf.maximum(reconstructed_mean, 0) 
                                - reconstructed_mean * self.x
                                + tf.log(1 + tf.exp(-abs(reconstructed_mean))),
                                 1)

            prior_loss_list.append(prior_loss)
            recognition_loss_list.append(recognition_loss)
            reconstr_loss_list.append(reconstr_loss)

        prior_loss_tensor = tf.pack(prior_loss_list, axis=1)
        recognition_loss_tensor = tf.pack(recognition_loss_list, axis=1)
        reconstr_loss_tensor = tf.pack(reconstr_loss_list, axis=1)


        log_w = tf.sub(tf.add(reconstr_loss_tensor, recognition_loss_tensor),prior_loss_tensor) 
        mean_log_w = tf.reduce_mean(log_w, 1) #average over particles
        cost = tf.reduce_mean(mean_log_w) #average over batch

        return cost


    def loss2(self):
        '''
        IWAE -ELBO
        '''

        recog_mean, recog_log_sigma_sq = self._recognition_network(self.x, self.network_weights["weights_recog"], self.network_weights["biases_recog"])

        reconstr_loss_list = []
        prior_loss_list = []
        recognition_loss_list = []
        # print self.n_particles, 'particles'
        for particle in range(self.n_particles):

            eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), 0, 1, dtype=tf.float32)
            z = tf.add(recog_mean, tf.mul(tf.sqrt(tf.exp(recog_log_sigma_sq)), eps))

            prior_loss = self._log_p_z(z)

            recognition_loss = self._log_p_z_given_x(z, recog_mean, recog_log_sigma_sq)

            #Calculate reconstruction error
            reconstructed_mean = self._generator_network(z, self.network_weights["weights_gener"], self.network_weights["biases_gener"], sigmoid=False)
            #this sum is over the dimensions 
            reconstr_loss = \
                    tf.reduce_sum(tf.maximum(reconstructed_mean, 0) 
                                - reconstructed_mean * self.x
                                + tf.log(1 + tf.exp(-abs(reconstructed_mean))),
                                 1)

            prior_loss_list.append(prior_loss)
            recognition_loss_list.append(recognition_loss)
            reconstr_loss_list.append(reconstr_loss)
            # print len(reconstr_loss_list), 'list'

        prior_loss_tensor = tf.pack(prior_loss_list, axis=1)
        recognition_loss_tensor = tf.pack(recognition_loss_list, axis=1)
        reconstr_loss_tensor = tf.pack(reconstr_loss_list, axis=1)
        # print reconstr_loss_tensor

        log_w = tf.sub(tf.add(reconstr_loss_tensor, recognition_loss_tensor),prior_loss_tensor)
        # print log_w
        max_ = tf.reduce_max(log_w, reduction_indices=1, keep_dims=True)
        # print max_
        # max_ = tf.tile(log_w, multiples=tf.constant([self.n_particles]))

        log_mean_w = tf.log(tf.reduce_mean(tf.exp(log_w-max_), 1)) + max_#average over particles
        cost = tf.reduce_mean(log_mean_w) #average over batch

        return cost


    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.loss()), feed_dict={self.x: X})

        return cost
    

    def generate(self):
        """ 
        Generate data by sampling from the latent space.       
        """

        z = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), 0, 1, dtype=tf.float32)
        reconstructed_mean = self._generator_network(z, self.network_weights["weights_gener"], self.network_weights["biases_gener"], sigmoid=True)

        reconstructed_mean1 = self.sess.run(reconstructed_mean)

        return reconstructed_mean1


    def evaluate(self, datapoints, n_samples, n_datapoints=None):
        '''
        Negative Log Likelihood Lower Bound
        '''

        normal_n_particles = self.n_particles
        self.n_particles = n_samples
        sum_ = 0
        datapoint_index = 0
        use_all = False
        if n_datapoints == None:
            use_all = True
            n_datapoints=len(datapoints)
        for i in range(n_datapoints/self.batch_size):

            #Make batch
            batch = []
            while len(batch) != self.batch_size:
                if use_all:
                    datapoint = datapoints[datapoint_index]
                else:
                    datapoint = datapoints[random.randint(0,n_datapoints-1)]

                batch.append(datapoint)
                datapoint_index +=1

            nll = self.sess.run((self.loss2()), feed_dict={self.x: batch})
            sum_ += nll

        avg = sum_ / (n_datapoints/self.batch_size)

        self.n_particles = normal_n_particles

        return avg





    def train(self, train_x, valid_x=[], timelimit=60, max_steps=999, display_step=5, valid_step=10000, path_to_load_variables='', path_to_save_variables=''):

        n_datapoints = len(train_x)

        #Load variables
        saver = tf.train.Saver()
        if path_to_load_variables != '':
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables


        start = time.time()
        for step in range(max_steps):

            #Make batch
            batch = []
            while len(batch) != self.batch_size:
                datapoint = train_x[random.randint(0,n_datapoints-1)]
                batch.append(datapoint)

            # Fit training using batch data
            cost = self.partial_fit(batch)
            
            # Display logs per epoch step
            if step % display_step == 0:
                print "Step:", '%04d' % (step+1), "t{:.1f},".format(time.time() - start), "cost=", "{:.9f}".format(cost)

            # Get validation NLL
            if step % valid_step == 0 and len(valid_x)!= 0:
                print "Step:", '%04d' % (step+1), "validation NLL=", "{:.9f}".format(self.evaluate(valid_x, 10, 100)), 'timelimit', str(timelimit)

            #Check if time is up
            if time.time() - start > timelimit:
                print 'times up', timelimit
                break

        if path_to_save_variables != '':
            print 'saving variables to ' + path_to_save_variables
            saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables


    def train2(self, train_x, valid_x=[], display_step=5, path_to_load_variables='', path_to_save_variables='', starting_stage=0):
        '''
        This training method is the IWAE one where they do many passes over the data with decreasing LR
        One difference is that I look at the validation NLL after each stage and save the variables
        '''

        n_datapoints = len(train_x)

        #Load variables
        saver = tf.train.Saver()
        if path_to_load_variables != '':
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables

        total_stages= 7
        start = time.time()
        for stage in range(starting_stage,total_stages+1):

            self.learning_rate = .001 * 10.**(-stage/float(total_stages))
            print 'learning rate', self.learning_rate

            passes_over_data = 3**stage

            for pass_ in range(passes_over_data):

                #I should shuffle the data

                data_index = 0
                for step in range(n_datapoints/self.batch_size):

                    #Make batch
                    batch = []
                    while len(batch) != self.batch_size:
                        datapoint = train_x[data_index]
                        batch.append(datapoint)

                    # Fit training using batch data
                    cost = self.partial_fit(batch)
                    
                    # Display logs per epoch step
                    if step % display_step == 0:
                        # print "Step:", '%04d' % (step+1), "t{:.1f},".format(time.time() - start), "cost=", "{:.9f}".format(cost)
                        print "Stage:", str(stage)+'/7', "Pass", str(pass_)+'/'+str(passes_over_data-1), "Step:%04d' % (step+1) + str(n_datapoints/self.batch_size), "cost=", "{:.9f}".format(cost)


                    # #Check if time is up
                    # if time.time() - start > timelimit:
                    #     print 'times up', timelimit
                    #     break


            # Get validation NLL
            if step % valid_step == 0 and len(valid_x)!= 0:
                print 'Calculating validation NLL'
                print "Validation NLL=", "{:.9f}".format(self.evaluate(valid_x, 100, 300))

            if path_to_save_variables != '':
                print 'saving variables to ' + path_to_save_variables
                saver.save(self.sess, path_to_save_variables)
                print 'Saved variables to ' + path_to_save_variables








class IWAE(VAE):


    def loss(self):

        recog_mean, recog_log_sigma_sq = self._recognition_network(self.x, self.network_weights["weights_recog"], self.network_weights["biases_recog"])

        reconstr_loss_list = []
        prior_loss_list = []
        recognition_loss_list = []
        for particle in range(self.n_particles):

            eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), 0, 1, dtype=tf.float32)
            z = tf.add(recog_mean, tf.mul(tf.sqrt(tf.exp(recog_log_sigma_sq)), eps))

            prior_loss = self._log_p_z(z)

            recognition_loss = self._log_p_z_given_x(z, recog_mean, recog_log_sigma_sq)

            #Generate frame x_t and calc reconstruction error
            reconstructed_mean = self._generator_network(z, self.network_weights["weights_gener"], self.network_weights["biases_gener"], sigmoid=False)
            #this sum is over the dimensions 
            reconstr_loss = \
                    tf.reduce_sum(tf.maximum(reconstructed_mean, 0) 
                                - reconstructed_mean * self.x
                                + tf.log(1 + tf.exp(-abs(reconstructed_mean))),
                                 1)

            prior_loss_list.append(prior_loss)
            recognition_loss_list.append(recognition_loss)
            reconstr_loss_list.append(reconstr_loss)

        prior_loss_tensor = tf.pack(prior_loss_list, axis=1)
        recognition_loss_tensor = tf.pack(recognition_loss_list, axis=1)
        reconstr_loss_tensor = tf.pack(reconstr_loss_list, axis=1)

        log_w = tf.sub(tf.add(reconstr_loss_tensor, recognition_loss_tensor),prior_loss_tensor)

        #THIS IS THE ONLY DIFFERENCE FROM VAE
        max_ = tf.reduce_max(log_w, reduction_indices=1, keep_dims=True)
        log_mean_w = tf.log(tf.reduce_mean(tf.exp(log_w-max_), 1)) + max_#average over particles
        cost = tf.reduce_mean(log_mean_w) #average over batch

        # log_w = log_w * tf.nn.softmax(log_w)
        # mean_log_w = tf.reduce_sum(log_w, 1) #sum over particles
        # cost = tf.reduce_mean(mean_log_w)  #average over batch

        return cost


    def evaluate(self, datapoints, n_samples, n_datapoints=None):

        normal_n_particles = self.n_particles
        self.n_particles = n_samples
        sum_ = 0
        datapoint_index = 0
        use_all = False
        if n_datapoints == None:
            use_all = True
            n_datapoints=len(datapoints)
        for i in range(n_datapoints/self.batch_size):

            #Make batch
            batch = []
            while len(batch) != self.batch_size:
                if use_all:
                    datapoint = datapoints[datapoint_index]
                else:
                    datapoint = datapoints[random.randint(0,n_datapoints-1)]

                batch.append(datapoint)
                datapoint_index +=1

            #THIS IS THE CHANGE
            nll = self.sess.run((super.loss2()), feed_dict={self.x: batch})
            sum_ += nll

        avg = sum_ / (n_datapoints/self.batch_size)

        self.n_particles = normal_n_particles

        return avg






class VAE_MoG(VAE):


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
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z*3)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z*3))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z*3], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z*3], dtype=tf.float32))}
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




    def loss(self):
        '''
        Negative Log Likelihood Lower Bound
        '''

        recog_mean, recog_log_sigma_sq = self._recognition_network(self.x, self.network_weights["weights_recog"], self.network_weights["biases_recog"])

        reconstr_loss_list = []
        prior_loss_list = []
        recognition_loss_list = []
        for particle in range(self.n_particles):

            #which normal to sample from
            # selection = tf.multinomial(tf.ones([1, 3]), 1)
            # selection = np.random.multinomial(1, [1/3.]*3, size=1)
            selection = particle % 3
            # selection = tf.unpack(selection)[0]
            # print selection

            this_normal_mean = tf.slice(recog_mean, [0,selection], [self.batch_size,self.network_architecture["n_z"]])
            this_normal_sigma = tf.slice(recog_log_sigma_sq, [0, selection], [self.batch_size,self.network_architecture["n_z"]])

            eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), 0, 1, dtype=tf.float32)
            z = tf.add(this_normal_mean, tf.mul(tf.sqrt(tf.exp(this_normal_sigma)), eps))

            prior_loss = self._log_p_z(z)

            recognition_loss = self._log_p_z_given_x(z, this_normal_mean, this_normal_sigma)

            #Calculate reconstruction error
            reconstructed_mean = self._generator_network(z, self.network_weights["weights_gener"], self.network_weights["biases_gener"], sigmoid=False)
            #this sum is over the dimensions 
            reconstr_loss = \
                    tf.reduce_sum(tf.maximum(reconstructed_mean, 0) 
                                - reconstructed_mean * self.x
                                + tf.log(1 + tf.exp(-abs(reconstructed_mean))),
                                 1)

            prior_loss_list.append(prior_loss)
            recognition_loss_list.append(recognition_loss)
            reconstr_loss_list.append(reconstr_loss)

        prior_loss_tensor = tf.pack(prior_loss_list, axis=1)
        recognition_loss_tensor = tf.pack(recognition_loss_list, axis=1)
        reconstr_loss_tensor = tf.pack(reconstr_loss_list, axis=1)


        log_w = tf.sub(tf.add(reconstr_loss_tensor, recognition_loss_tensor),prior_loss_tensor) 
        mean_log_w = tf.reduce_mean(log_w, 1) #average over particles
        cost = tf.reduce_mean(mean_log_w) #average over batch

        return cost

















