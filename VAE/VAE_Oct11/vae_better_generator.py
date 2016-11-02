

import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import imageio

######################################################################
#DATA
######################################################################

def make_ball_gif(n_frames=3, f_height=14, f_width=14, ball_size=2, max_1=True, vector_output=True):
    
    row = random.randint(0,f_height-ball_size-1)
    # speed = random.randint(1,9)
    speed = random.randint(1,3)
    # speed= 1
    
    gif = []
    for i in range(n_frames):

        hot = np.zeros([f_height,f_width])
        if i*speed+ball_size >= f_width:
            hot[row:row+ball_size:1,f_width-ball_size:f_width+ball_size:1] = 255.
        else:
            hot[row:row+ball_size:1,i*speed:i*speed+ball_size:1] = 255.
        gif.append(hot.astype('uint8'))


    gif = np.array(gif)

    if max_1:
        for i in range(len(gif)):
            gif[i] = gif[i] / np.max(gif[i])


    if n_frames == 1:
        if vector_output:
            gif = np.reshape(gif, [f_height*f_width])
    else:
        if vector_output:
            gif = np.reshape(gif, [n_frames,f_height*f_width])

    return gif

######################################################################
#OTHER
######################################################################

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

######################################################################
#MODEL
######################################################################

class VAE():

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=5, n_time_steps=2):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_time_steps = n_time_steps
        
        # tf Graph input
        #[batch, n_frames, vector_length]
        self.x = tf.placeholder(tf.float32, [None, None, network_architecture["n_input"]])
        
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    
    def _create_network(self):
        # Initialize autoencode network weights and biases
        self.network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(self.network_weights["weights_recog"], 
                                      self.network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, self.n_time_steps, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(self.network_weights["weights_gener"],
                                    self.network_weights["biases_gener"])
            
        self.x_reconstr_mean_no_sig = \
            self._generator_network_no_sigmoid(self.network_weights["weights_gener"],
                                    self.network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2,
                            n_hidden_trans_1,  n_hidden_trans_2, 
                            n_input, n_z):
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
        all_weights['weights_trans'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_trans_1)),
            'h2': tf.Variable(xavier_init(n_hidden_trans_1, n_hidden_trans_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_trans_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_trans_2, n_z))}
        all_weights['biases_trans'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_trans_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_trans_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        # layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
        #                                    biases['b1'])) 
        # layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
        #                                    biases['b2'])) 
        # z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
        #                 biases['out_mean'])
        # z_log_sigma_sq = \
        #     tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
        #            biases['out_log_sigma'])
        # return (z_mean, z_log_sigma_sq)


        #Get distribution of each z_t
        #for now, im just making each distribution dependent on x_t
        z_mean_list = []
        z_log_sigma_sq_list = []
        for time_step in range(self.n_time_steps):

            #slice: where to start, how many to get
            batch_at_time_step = tf.slice(self.x, [0,time_step,0], [self.batch_size, 1, self.network_architecture["n_input"]])
            batch_at_time_step = tf.reshape(batch_at_time_step, [self.batch_size,self.network_architecture["n_input"]])

            layer_1 = self.transfer_fct(tf.add(tf.matmul(batch_at_time_step, weights['h1']), biases['b1'])) 
            layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])) 

            z_mean_t = tf.add(tf.matmul(layer_2, weights['out_mean']),biases['out_mean'])
            z_log_sigma_sq_t = \
                tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                       biases['out_log_sigma'])

            z_mean_list.append(z_mean_t)
            z_log_sigma_sq_list.append(z_log_sigma_sq_t)

        z_mean = tf.pack(z_mean_list, axis=1)
        z_log_sigma_sq = tf.pack(z_log_sigma_sq_list, axis=1)

        return (z_mean, z_log_sigma_sq)



    def _transition_network(self, prev_z, weights, biases):

        layer_1 = self.transfer_fct(tf.add(tf.matmul(prev_z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])

        #FOR NOW, ill make it deterministic, since Im not how to change the loss atm

        # z_log_sigma_sq = \
        #     tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
        #            biases['out_log_sigma'])
        # return (z_mean, z_log_sigma_sq)

        return z_mean



    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.

        # layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
        #                                    biases['b1'])) 
        # layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
        #                                    biases['b2'])) 
        # x_reconstr_mean = \
        #     tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
        #                          biases['out_mean']))
        # return x_reconstr_mean

        x_reconstr_mean_list = []

        for time_step in range(self.n_time_steps):

            #slice: where to start, how many to get
            if time_step == 0:
                z_given_prev_z = tf.slice(self.z, [0,time_step,0], [self.batch_size, 1, self.network_architecture["n_z"]])
                z_given_prev_z = tf.reshape(z_given_prev_z, [self.batch_size,self.network_architecture["n_z"]])
            else:
                z_given_prev_z = self._transition_network(prev_z, self.network_weights["weights_trans"], self.network_weights["biases_trans"])

            layer_1 = self.transfer_fct(tf.add(tf.matmul(z_given_prev_z, weights['h1']), 
                                               biases['b1'])) 
            layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                               biases['b2'])) 

            x_reconstr_mean = \
                tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                     biases['out_mean']))

            x_reconstr_mean_list.append(x_reconstr_mean)

            prev_z = z_given_prev_z

        x_reconstr_mean_tensor = tf.pack(x_reconstr_mean_list, axis=1)

        return x_reconstr_mean_tensor



    def _generator_network_no_sigmoid(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        x_reconstr_mean_list = []

        for time_step in range(self.n_time_steps):

            #slice: where to start, how many to get
            if time_step == 0:
                z_given_prev_z = tf.slice(self.z, [0,time_step,0], [self.batch_size, 1, self.network_architecture["n_z"]])
                z_given_prev_z = tf.reshape(z_given_prev_z, [self.batch_size,self.network_architecture["n_z"]])
            else:
                z_given_prev_z = self._transition_network(prev_z, self.network_weights["weights_trans"], self.network_weights["biases_trans"])

            layer_1 = self.transfer_fct(tf.add(tf.matmul(z_given_prev_z, weights['h1']), 
                                               biases['b1'])) 
            layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                               biases['b2'])) 

            x_reconstr_mean = \
                tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                     biases['out_mean'])

            x_reconstr_mean_list.append(x_reconstr_mean)

            prev_z = z_given_prev_z

        x_reconstr_mean_tensor = tf.pack(x_reconstr_mean_list, axis=1)

        return x_reconstr_mean_tensor


            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)

        # reconstr_loss = \
        #     -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
        #                    + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
        #                    1)

        reconstr_loss_list = []

        for time_step in range(self.n_time_steps):

            batch_at_time_step = tf.slice(self.x_reconstr_mean_no_sig, [0,time_step,0], [self.batch_size, 1, self.network_architecture["n_input"]])
            batch_at_time_step = tf.reshape(batch_at_time_step, [self.batch_size,self.network_architecture["n_input"]])

            batch_at_time_step2 = tf.slice(self.x, [0,time_step,0], [self.batch_size, 1, self.network_architecture["n_input"]])
            batch_at_time_step2 = tf.reshape(batch_at_time_step2, [self.batch_size,self.network_architecture["n_input"]])

            #this sum is over the dimensions 
            reconstr_loss = \
                    tf.reduce_sum(tf.maximum(batch_at_time_step, 0) 
                                - batch_at_time_step * batch_at_time_step2
                                + tf.log(1 + tf.exp(-abs(batch_at_time_step))),
                                 1)

            reconstr_loss_list.append(reconstr_loss)

        reconstr_loss_tensor = tf.pack(reconstr_loss_list, axis=1)

        #Sum over time steps
        reconstr_loss_for_each_sequence = tf.reduce_sum(reconstr_loss_tensor, reduction_indices=1)



        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.

        # latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
        #                                    - tf.square(self.z_mean) 
        #                                    - tf.exp(self.z_log_sigma_sq), 1)



        latent_loss_list = []

        for time_step in range(self.n_time_steps):

            batch_at_time_step = tf.slice(self.z_log_sigma_sq, [0,time_step,0], [self.batch_size, 1, self.network_architecture["n_z"]])
            batch_at_time_step = tf.reshape(batch_at_time_step, [self.batch_size,self.network_architecture["n_z"]])

            batch_at_time_step2 = tf.slice(self.z_mean, [0,time_step,0], [self.batch_size, 1, self.network_architecture["n_z"]])
            batch_at_time_step2 = tf.reshape(batch_at_time_step2, [self.batch_size,self.network_architecture["n_z"]])

            #this sum is over the dimensions 
            latent_loss = -0.5 * tf.reduce_sum(1 + batch_at_time_step 
                                               - tf.square(batch_at_time_step2) 
                                               - tf.exp(batch_at_time_step), 1)

            latent_loss_list.append(latent_loss)

        latent_loss_tensor = tf.pack(latent_loss_list, axis=1)

        #Sum over time steps
        latent_loss_for_each_sequence = tf.reduce_sum(latent_loss_tensor, reduction_indices=1)



 
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})
        return cost
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            # z_mu = np.random.normal(size=self.network_architecture["n_z"])
            z_mu = np.random.normal(size=(self.batch_size, self.n_time_steps, self.network_architecture["n_z"]))

        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution

        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})

        # self.z = z_mu
        # return self.sess.run(self.x_reconstr_mean)
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})


    def train(self, get_data, steps=1000, display_step=10, path_to_load_variables='', path_to_save_variables=''):


        # if self.path_to_load_variables == '':
        #     self.sess.run(init)
        # else:
        saver = tf.train.Saver()

        if path_to_load_variables != '':
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables


        # Training cycle
        for step in range(steps):
            # avg_cost = 0.
            # total_batch = int(n_samples / self.batch_size)
            # Loop over all batches
            # for i in range(total_batch):
            batch = []
            while len(batch) != self.batch_size:
                # frame=make_ball_gif(n_frames=1, f_height=14, f_width=14, ball_size=2, max_1=True, vector_output=True)
                frame=get_data()
                # batch_xs, _ = mnist.train.next_batch(batch_size)
                batch.append(frame)

            # Fit training using batch data
            cost = vae.partial_fit(batch)
            

            # Compute average loss
            # avg_cost += cost / n_samples * self.batch_size
            if cost != cost:
                print 'here'
                print cost
                # rec_loss = self.sess.run(self.reconstr_loss_, feed_dict={self.x: batch})
                # print rec_loss
                # rec_loss = self.sess.run(self.another_test, feed_dict={self.x: batch})
                # print rec_loss
                rec_loss = self.sess.run(self.x_reconstr_mean, feed_dict={self.x: batch})
                print rec_loss
                
                fasdfas
            # Display logs per epoch step
            if step % display_step == 0:
                print "Step:", '%04d' % (step+1), \
                      "cost=", "{:.5f}".format(cost)


        if path_to_save_variables != '':
            saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables

        return vae



if __name__ == "__main__":

    f_height=30
    f_width=30
    ball_size=5
    n_time_steps = 2
    batch_size = 5
    path_to_load_variables=home+'/Documents/tmp/vae.ckpt'
    # path_to_load_variables=''
    path_to_save_variables=home+'/Documents/tmp/vae2.ckpt'

    def get_ball():
        return make_ball_gif(n_frames=n_time_steps, f_height=f_height, f_width=f_width, ball_size=ball_size, max_1=True, vector_output=True)


    network_architecture = \
        dict(n_hidden_recog_1=100, # 1st layer encoder neurons
             n_hidden_recog_2=100, # 2nd layer encoder neurons
             n_hidden_gener_1=100, # 1st layer decoder neurons
             n_hidden_gener_2=100, # 2nd layer decoder neurons
             n_hidden_trans_1=100,
             n_hidden_trans_2=100,
             n_input=f_height*f_width, # image as vector
             n_z=20)  # dimensionality of latent space

    vae = VAE(network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=batch_size, n_time_steps=n_time_steps)
    vae.train(get_data=get_ball, steps=5000, display_step=5, path_to_load_variables=path_to_load_variables, path_to_save_variables=path_to_save_variables)

    #RECONSTRUCT 
    batch = []
    while len(batch) != ball_size:
        frame=get_ball()
        batch.append(frame)
    x_reconstruct = vae.reconstruct(batch)
    print x_reconstruct.shape

    batch = np.array(batch)
    batch = np.reshape(batch, [batch_size*n_time_steps, f_height,f_width,1])

    for i in range(len(batch)):
        # batch[i] = np.reshape(batch[i], [f_height,f_width,1])
        batch[i] = batch[i] * (255. / np.max(batch[i]))
        batch[i] = batch[i].astype('uint8')
    # kargs = { 'duration': .8 }
    # imageio.mimsave(home+"/Downloads/inputs_gif.gif", batch, 'GIF', **kargs)

    x_reconstruct = np.reshape(x_reconstruct, [batch_size*n_time_steps, f_height,f_width,1])

    for i in range(len(x_reconstruct)):
        # reconstruct_batch.append(np.reshape(x_reconstruct[i], [f_height,f_width,1]))
        x_reconstruct[i] = x_reconstruct[i] * (255. / np.max(x_reconstruct[i]))
        x_reconstruct[i] = x_reconstruct[i].astype('uint8')
    # kargs = { 'duration': .5 }
    # imageio.mimsave(home+"/Downloads/inputs_rec_gif.gif", reconstruct_batch, 'GIF', **kargs)

    combined = []
    for i in range(0,len(batch),2):
        combined.append(batch[i])
        combined.append(batch[i+1])

        combined.append(x_reconstruct[i])
        combined.append(x_reconstruct[i+1])

    kargs = { 'duration': 1.2 }
    imageio.mimsave(home+"/Downloads/comb_gif.gif", combined, 'GIF', **kargs)



    #GENERATE 
    # generated = []
    # # for i in range(5):
    # generated.append(vae.generate())
    generated = vae.generate()
    print generated.shape
    generated = np.reshape(generated, [batch_size*n_time_steps, f_height,f_width,1])


    for i in range(len(generated)):
        # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
        generated[i] = generated[i] * (255. / np.max(generated[i]))
        generated[i] = generated[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/gen_gif.gif", generated, 'GIF', **kargs)

    print 'DONE'









