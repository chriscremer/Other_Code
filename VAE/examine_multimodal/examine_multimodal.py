

import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import imageio
import time
import matplotlib.pyplot as plt



def make_ball_gif(f_height=28, f_width=28, ball_size=5, max_1=False, vector_output=False):

    n_frames = 2
    sequence = []
    action_list = []
    blank_frame = np.zeros([f_height,f_width])
    # pos = [f_height/2, f_width/2] #top_left_of_ball
    # init_frame[pos[0]:pos[0]+ball_size, pos[1]:pos[1]+ball_size] = 255.

    rand_int = random.randint(0,3)
    if rand_int == 0:
        pos = [7,7]
    elif rand_int == 1:
        pos = [18,18]
    elif rand_int == 2:
        pos = [7,18]
    elif rand_int == 3:
        pos = [18,7]


    #For each time step
    for i in range(n_frames):
        #Actions: up, down, right, left
        action = np.random.choice(4, 1, p=[.25, .25, .25, .25])[0]

        if action == 0:
            if pos[0] + 1 < f_height-1:
                pos[0] = pos[0] + 3
        elif action == 1:
            if pos[0] - 1 >= 0:
                pos[0] = pos[0] - 3
        elif action == 2:
            if pos[1] + 1 < f_width-1:
                pos[1] = pos[1] + 3
        elif action == 3:
            if pos[1] - 1 >= 0:
                pos[1] = pos[1] - 3

        new_frame = np.zeros([f_height,f_width])
        new_frame[pos[0]:pos[0]+ball_size, pos[1]:pos[1]+ball_size] = 255.
        sequence.append(new_frame)
        action_array = np.zeros([4])
        action_array[action] = 1
        action_list.append(action_array)

    sequence = np.array(sequence)

    if max_1:
        for i in range(len(sequence)):
            sequence[i] = sequence[i] / np.max(sequence[i])

    if vector_output:
        if vector_output:
            sequence = np.reshape(sequence, [n_frames,f_height*f_width])

    return np.array(sequence), action_list





#Generative Autoencoder class

class VAE():

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=5, n_particles=3):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_particles = n_particles
        self.n_z = network_architecture["n_z"]
        self.n_input = network_architecture["n_input"]

        #Placeholders - Inputs
        self.x1 = tf.placeholder(tf.float32, [None, self.n_input])
        self.x2 = tf.placeholder(tf.float32, [None, self.n_input])

        #Variables
        self.network_weights = self._initialize_weights(**self.network_architecture)
        #Encoder - Recognition model - q(z|x1,x2): recog_mean,z_log_std_sq=[batch_size, n_z]
        self.recog_mean, self.recog_log_std_sq = self._recognition_network(self.network_weights["weights_recog"], self.network_weights["biases_recog"])
        #Sample
        self.eps = tf.random_normal((self.n_particles, self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.recog_mean, tf.mul(tf.sqrt(tf.exp(self.recog_log_std_sq)), self.eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]
        #Decoder - Generative model - p(x|z)
        self.x_reconstr_mean_no_sigmoid = self._generator_network2(self.network_weights["weights_gener2"], self.network_weights["biases_gener2"]) #no sigmoid
        # self.x_reconstr_mean = tf.nn.sigmoid(self.x_reconstr_mean_no_sigmoid) #shape=[n_particles, n_batch, n_input]
        self.z_generative_mean, self.z_generative_log_std_sq = self._generator_network(self.network_weights["weights_gener"], self.network_weights["biases_gener"]) #no sigmoid

        #Objective
        self.elbo = self._log_p_x2_given_z_x1() + self._log_p_z_given_x1() - self._log_q_z_given_x1_x2()

        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-04).minimize(-self.elbo)


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

        #q(z|x1,x2)
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input+n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}

        #p(z|x1)
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_z))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}

        #p(x2|z,x1)
        all_weights['weights_gener2'] = {
            'h1': tf.Variable(xavier_init(n_z+n_input, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
            # 'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener2'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
            # 'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights
            

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.

        x = concatenate_state_and_frame_and_action = tf.concat(1, [self.x1, self.x2])

        layer_1 = self.transfer_fct(tf.add(tf.matmul(x, weights['h1']), biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])) 

        z_mean_t = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        z_log_sigma_sq_t = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])

        return (z_mean_t, z_log_sigma_sq_t)

    def _generator_network(self, weights, biases):
        # p(z|x1)

        # z = tf.reshape(self.z, [self.n_particles*self.batch_size, self.n_z])

        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x1, weights['h1']), biases['b1'])) #shape is now [p*b,l1]
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])) 

        z_reconstr_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        # z_reconstr_mean = tf.reshape(z_reconstr_mean, [self.n_particles, self.batch_size, self.n_z])

        z_log_sigma_sq_t = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])
        # z_log_sigma_sq_t = tf.reshape(z_log_sigma_sq_t, [self.n_particles, self.batch_size, self.n_z])

        return z_reconstr_mean, z_log_sigma_sq_t
    

    def _generator_network2(self, weights, biases):
        # p(x2|z,x1)

        z = tf.reshape(self.z, [self.n_particles*self.batch_size, self.n_z])
        x = tf.reshape(self.x1, [1, self.batch_size, self.n_input])
        x = tf.tile(x, [self.n_particles, 1, 1])
        x = tf.reshape(x, [self.n_particles*self.batch_size, self.n_input])
        input_ = concatenate_state_and_frame_and_action = tf.concat(1, [z, x])

        layer_1 = self.transfer_fct(tf.add(tf.matmul(input_, weights['h1']), biases['b1'])) #shape is now [p*b,l1]
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])) 

        x_reconstr_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        x_reconstr_mean = tf.reshape(x_reconstr_mean, [self.n_particles, self.batch_size, self.n_input])

        return x_reconstr_mean
    

    def _log_q_z_given_x1_x2(self):
        #Get log(p(z))
        #This is just exp of standard normal

        # term1 = 0
        # term2 = self.n_z * tf.log(2*math.pi)
        # term3 = tf.reduce_sum(tf.square(self.z), 2) #sum over dimensions n_z so now its [particles, batch]

        # all_ = term1 + term2 + term3
        # log_p_z = -.5 * all_

        # log_p_z = tf.reduce_mean(log_p_z, 1) #average over batch
        # log_p_z = tf.reduce_mean(log_p_z) #average over particles

        # return log_p_z

        term1 = tf.reduce_sum(self.recog_log_std_sq, reduction_indices=1) #sum over dimensions n_z so now its [batch]

        term2 = self.n_z * tf.log(2*math.pi)
        dif = tf.square(self.z - self.recog_mean)
        dif_cov = dif / tf.exp(self.recog_log_std_sq)
        # term3 = tf.reduce_sum(dif_cov * dif, 1) 
        term3 = tf.reduce_sum(dif_cov, 2) #sum over dimensions n_z so now its [particles, batch]

        all_ = term1 + term2 + term3
        log_p_z_given_x = -.5 * all_

        log_p_z_given_x = tf.reduce_mean(log_p_z_given_x, 1) #average over batch
        log_p_z_given_x = tf.reduce_mean(log_p_z_given_x) #average over particles

        return log_p_z_given_x

    def _log_p_z_given_x1(self):

        #Get log(p(z|x))
        #This is just exp of a normal with some mean and var

        # term1 = tf.log(tf.reduce_prod(tf.exp(log_var_sq), reduction_indices=1))
        # print self.z_generative_log_std_sq
        term1 = tf.reduce_sum(self.z_generative_log_std_sq, reduction_indices=1) #sum over dimensions n_z so now its [batch]

        term2 = self.n_z * tf.log(2*math.pi)
        dif = tf.square(self.z - self.z_generative_mean)
        dif_cov = dif / tf.exp(self.z_generative_log_std_sq)
        # term3 = tf.reduce_sum(dif_cov * dif, 1) 
        term3 = tf.reduce_sum(dif_cov, 2) #sum over dimensions n_z so now its [particles, batch]

        # print term1
        # print term2
        # print term3
        all_ = term1 + term2 + term3
        log_p_z_given_x = -.5 * all_

        log_p_z_given_x = tf.reduce_mean(log_p_z_given_x, 1) #average over batch
        log_p_z_given_x = tf.reduce_mean(log_p_z_given_x) #average over particles

        return log_p_z_given_x

    def _log_p_x2_given_z_x1(self):
        # log p(x|z) for bernoulli distribution
        # recontruction mean has shape=[n_particles, n_batch, n_input]
        # x has shape [batch, n_input] 
        # option 1: tile the x to the same size as reconstruciton.. but thats a waste of space
        # I could probably do some broadcasting 

        reconstr_loss = \
                tf.reduce_sum(tf.maximum(self.x_reconstr_mean_no_sigmoid, 0) 
                            - self.x_reconstr_mean_no_sigmoid * self.x2
                            + tf.log(1 + tf.exp(-abs(self.x_reconstr_mean_no_sigmoid))),
                             2) #sum over dimensions

        reconstr_loss = tf.reduce_mean(reconstr_loss, 1) #average over batch
        reconstr_loss = tf.reduce_mean(reconstr_loss) #average over particles

        #negative because the above calculated the NLL, so this is returning the LL
        return -reconstr_loss



    def partial_fit(self, batch1, batch2):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        # opt, cost = self.sess.run((self.optimizer, self.elbo), feed_dict={self.x: X})
        _ = self.sess.run((self.optimizer), feed_dict={self.x1: batch1, self.x2: batch2})

        return 0
    


    def generate(self, get_data_function, n_samples, path_to_load_variables):
        """ 
        Generate data by sampling from the latent space.       
        """

        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            self.sess.run(tf.initialize_all_variables())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables

        batch1 = []
        batch2 = []

        while len(batch1) != self.batch_size:
            frames = get_data_function()
            batch1.append(frames[0][0])
            batch2.append(frames[0][1])


        z_generative_mean, z_generative_log_std_sq = self.sess.run(self._generator_network(self.network_weights["weights_gener"], self.network_weights["biases_gener"]),
                                                        feed_dict={self.x1: batch1})


        frame2_predictions = []
        for i in range(n_samples):
            x_mean = self.sess.run(tf.nn.sigmoid(self._generator_network2(self.network_weights["weights_gener2"], self.network_weights["biases_gener2"])),
                                    feed_dict={self.x1: batch1, self.recog_mean: z_generative_mean, self.recog_log_std_sq: z_generative_log_std_sq})
            frame2_predictions.append(x_mean)


        return np.array(batch1), np.array(batch2), np.array(frame2_predictions)




    def generate2(self, first_frame, z_mu, path_to_load_variables, sess=None):
        """ 
        This one is meant for the 2 dimensions visualizetion   
        """

        if sess is None:
            saver = tf.train.Saver()
            self.sess = tf.Session()

            if path_to_load_variables == '':
                self.sess.run(tf.initialize_all_variables())
            else:
                #Load variables
                saver.restore(self.sess, path_to_load_variables)
                print 'loaded variables ' + path_to_load_variables

        else:
            self.sess = sess

        batch1 = []
        z_mus = []

        while len(batch1) != self.batch_size:
            batch1.append(first_frame)
            z_mus.append(z_mu)
        #for particles
        z_mus = [z_mus]

        z_generative_mean, z_generative_log_std_sq = self.sess.run(self._generator_network(self.network_weights["weights_gener"], self.network_weights["biases_gener"]),
                                                        feed_dict={self.x1: batch1})

        x_mean = self.sess.run(tf.nn.sigmoid(self._generator_network2(self.network_weights["weights_gener2"], self.network_weights["biases_gener2"])),
                                    feed_dict={self.x1: batch1, self.recog_mean: z_generative_mean, self.recog_log_std_sq: z_generative_log_std_sq, self.eps: z_mus})


        return x_mean[0][0], self.sess




    def generate3(self, first_frame, z_mu, path_to_load_variables, sess=None):
        """ 
        This is meant for >2 dimension visualization of the latent space
            - sample latent space N times
            - get 2D PCs
            - get mean and var of PCs
            - transform z_mu using mean and var then transform it to original space 
            - use that as the z to reconstruct

            OR

            - just look at many x reconstructed samples 
                - because any error in the reconstruction might be due to the PCs, not the latent space
                - so for this, just dont use the z_mu
        """

        if sess is None:
            saver = tf.train.Saver()
            self.sess = tf.Session()

            if path_to_load_variables == '':
                self.sess.run(tf.initialize_all_variables())
            else:
                #Load variables
                saver.restore(self.sess, path_to_load_variables)
                print 'loaded variables ' + path_to_load_variables

        else:
            self.sess = sess

        batch1 = []
        # z_mus = []

        while len(batch1) != self.batch_size:
            batch1.append(first_frame)
            # z_mus.append(z_mu)
        #for particles
        # z_mus = [z_mus]

        z_generative_mean, z_generative_log_std_sq = self.sess.run(self._generator_network(self.network_weights["weights_gener"], self.network_weights["biases_gener"]),
                                                        feed_dict={self.x1: batch1})

        x_mean = self.sess.run(tf.nn.sigmoid(self._generator_network2(self.network_weights["weights_gener2"], self.network_weights["biases_gener2"])),
                                    feed_dict={self.x1: batch1, self.recog_mean: z_generative_mean, self.recog_log_std_sq: z_generative_log_std_sq})


        return x_mean[0][0], self.sess


    # def evaluate(self, datapoints, n_samples, n_datapoints=None):
    #     '''
    #     Negative Log Likelihood Lower Bound
    #     '''
    #     # normal_n_particles = self.n_particles
    #     # self.n_particles = n_samples
    #     sum_ = 0
    #     datapoint_index = 0
    #     use_all = False
    #     if n_datapoints == None:
    #         use_all = True
    #         n_datapoints=len(datapoints)
    #     for i in range(n_datapoints/self.batch_size):

    #         #Make batch
    #         batch = []
    #         while len(batch) != self.batch_size:
    #             if use_all:
    #                 datapoint = datapoints[datapoint_index]
    #             else:
    #                 datapoint = datapoints[random.randint(0,n_datapoints-1)]

    #             batch.append(datapoint)
    #             datapoint_index +=1

    #         # print np.array(batch).shape

    #         negative_elbo = -self.sess.run((self.elbo), feed_dict={self.x: batch})

    #         sum_ += negative_elbo

    #     avg = sum_ / (n_datapoints/float(self.batch_size))

    #     # self.n_particles = normal_n_particles

    #     return avg




    def train2(self, get_data_function, valid_x=[], display_step=5, path_to_load_variables='', path_to_save_variables='', starting_stage=0, timelimit=100):
        '''
        This training method is the IWAE one where they do many passes over the data with decreasing LR
        One difference is that I look at the validation NLL after each stage and save the variables
        '''

        # n_datapoints = len(train_x)
        
        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            self.sess.run(tf.initialize_all_variables())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables

        total_stages= 7
        start = time.time()
        # for stage in range(starting_stage,total_stages+1):
        step = 0
        while(1):

            # self.learning_rate = .001 * 10.**(-stage/float(total_stages))
            # print 'learning rate', self.learning_rate

            # passes_over_data = 3**stage

            # for pass_ in range(passes_over_data):

            #shuffle the data
            # arr = np.arange(len(train_x))
            # print arr.shape
            # print train_x.shape
            # train_x = np.reshape(train_x[np.random.shuffle(arr)], [50000,784])
            # print train_x.shape

            # data_index = 0
            # for step in range(n_datapoints/self.batch_size):

            #Make batch
            batch1 = []
            batch2 = []

            while len(batch1) != self.batch_size:
                # datapoint = train_x[data_index]
                frames = get_data_function()
                # print frames[0].shape
                batch1.append(frames[0][0])
                batch2.append(frames[0][1])

                # data_index +=1

            # Fit training using batch data
            nothing = self.partial_fit(batch1, batch2)
            
            # Display logs per epoch step
            if step % display_step == 0:
                # print np.array(batch).shape
                cost = -self.sess.run((self.elbo), feed_dict={self.x1: batch1, self.x2: batch2})

                # print "Stage:" + str(stage)+'/7', "Pass", str(pass_)+'/'+str(passes_over_data-1), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/self.batch_size), "cost=", "{:.6f}".format(cost)#, 'time', time.time() - start
                print 'Step:%04d' % (step) , "cost=", "{:.6f}".format(cost), 'time', time.time() - start, '/'+str(timelimit)


            #Check if time is up
            if time.time() - start > timelimit:
                print 'times up', timelimit
                break
            step+=1


        # print 'Calculating validation NLL'
        # print "Validation NLL=", "{:.9f}".format(self.evaluate(train_x, 1, 300))

        #TODO: save what stage the variables are
        if path_to_save_variables != '':
            print 'saving variables to ' + path_to_save_variables
            saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables


















if __name__ == "__main__":

   
    f_height=28
    f_width=28
    ball_size=3
    batch_size = 20
    n_particles = 1
    timelimit=500
    path_to_load_variables=home+'/Documents/tmp/examine_mulitmodal.ckpt'
    # path_to_load_variables=''
    path_to_save_variables=home+'/Documents/tmp/examine_mulitmodal.ckpt'

    network_architecture = \
        dict(n_hidden_recog_1=200, # 1st layer encoder neurons
             n_hidden_recog_2=200, # 2nd layer encoder neurons
             n_hidden_gener_1=200, # 1st layer decoder neurons
             n_hidden_gener_2=200, # 2nd layer decoder neurons
             n_input=f_height*f_width, # 784 image
             n_z=20)  # dimensionality of latent space
    

    def get_ball():
        return make_ball_gif(f_height=f_height, f_width=f_width, ball_size=ball_size, max_1=True, vector_output=True)


    vae = VAE(network_architecture, transfer_fct=tf.tanh, learning_rate=0.001, batch_size=batch_size, n_particles=n_particles)
    
    # Train model
    # print 'Training VAE'
    # vae.train2(get_ball, valid_x=[], display_step=200, path_to_load_variables=path_to_load_variables, path_to_save_variables=path_to_save_variables, starting_stage=7, timelimit=timelimit)

    # Sample the model
    n_samples=3
    batch1, batch2, frame2_predictions = vae.generate(get_data_function=get_ball, n_samples=n_samples, path_to_load_variables=path_to_save_variables)
    print batch1.shape
    print batch2.shape
    print frame2_predictions.shape


    # for i in range(batch_size):

    #     frame = np.reshape(batch1[i], [28,28])
    #     frame2 = np.reshape(batch2[i], [28,28])

    #     print 'Frame1', i
    #     fig = plt.figure(figsize=(6, 3.2))
    #     ax = fig.add_subplot(111)
    #     plt.imshow(frame, cmap=plt.cm.binary)
    #     plt.show()

    #     print 'Frame2'
    #     fig = plt.figure(figsize=(6, 3.2))
    #     ax = fig.add_subplot(111)
    #     plt.imshow(frame2, cmap=plt.cm.binary)
    #     plt.show()

    #     for j in range(n_samples):

    #         pred = np.reshape(frame2_predictions[j][0][i], [28,28])

    #         print 'Sample', j
    #         fig = plt.figure(figsize=(6, 3.2))
    #         ax = fig.add_subplot(111)
    #         plt.imshow(pred, cmap=plt.cm.binary)
    #         plt.show()

    #     if i ==0:
    #         break



    #TODO:
        #-show multiple images at the same time. 

    # frame = np.reshape(batch1[0], [28,28])
    frame = batch1[0]

    nx = ny = 5
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)


    sess = None
    canvas = np.empty((28*ny, 28*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):

            print i,j

            # z_mu = np.array([[xi, yi]])
            z_mu = [xi,yi]

            x_mean, sess = vae.generate3(first_frame=frame, z_mu=z_mu, path_to_load_variables=path_to_save_variables, sess=sess)
            # print x_mean.shape

            x_mean = x_mean + (frame*3)


            x_mean = np.reshape(x_mean, [28,28])

            x_mean[:,-1] = 4
            x_mean[-1,:] = 4


            # x_mean = frame
            # canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean


    plt.figure(figsize=(8, 10))        
    # Xi, Yi = np.meshgrid(x_values, y_values)
    # plt.imshow(canvas, origin="upper", cmap=plt.cm.binary)
    plt.imshow(canvas, origin="upper")

    # plt.tight_layout()
    plt.show()


    print 'Done'


    fasfa
































