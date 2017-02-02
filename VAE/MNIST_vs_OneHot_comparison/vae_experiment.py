


import numpy as np
import tensorflow as tf
import math
from os.path import expanduser
home = expanduser("~")



class VAE(object):

    def __init__(self, network_architecture, batch_size=5):
        
        tf.reset_default_graph()

        self.transfer_fct = tf.tanh
        self.learning_rate = .001
        self.reg_param = .001
        self.batch_size = batch_size

        self.z_size = network_architecture["n_z"]
        self.x_size = network_architecture["n_x"]
        
        # Graph Input: [B,X], [B,A]
        self.x = tf.placeholder(tf.float32, [None, self.x_size])
        
        #Variables
        self.network_weights = self._initialize_weights(network_architecture)

        #Objective
        self.elbo = self.calc_elbo(self.x)
        self.cost = -self.elbo + (self.reg_param * self.l2_regularization())

        #Optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost)


    def _initialize_weights(self, network_architecture):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


        all_weights = dict()

        #Recognition/Inference net q(z|z-1,x-1,u,x)
        all_weights['encoder_weights'] = {}
        all_weights['encoder_biases'] = {}

        for layer_i in range(len(network_architecture['encoder_net'])):
            if layer_i == 0:
                all_weights['encoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.x_size, network_architecture['encoder_net'][layer_i]))
                all_weights['encoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['encoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['encoder_net'][layer_i-1], network_architecture['encoder_net'][layer_i]))
                all_weights['encoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

        all_weights['encoder_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.z_size))
        all_weights['encoder_weights']['out_log_var'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.z_size))
        all_weights['encoder_biases']['out_mean'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))
        all_weights['encoder_biases']['out_log_var'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))


        #Generator net p(x|x-1,z)
        all_weights['decoder_weights'] = {}
        all_weights['decoder_biases'] = {}

        for layer_i in range(len(network_architecture['decoder_net'])):
            if layer_i == 0:
                all_weights['decoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.z_size, network_architecture['decoder_net'][layer_i]))
                all_weights['decoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['decoder_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['decoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['decoder_net'][layer_i-1], network_architecture['decoder_net'][layer_i]))
                all_weights['decoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

        all_weights['decoder_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.x_size))
        all_weights['decoder_biases']['out_mean'] = tf.Variable(tf.zeros([self.x_size], dtype=tf.float32))
        # all_weights['decoder_weights']['out_log_var'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.x_size))
        # all_weights['decoder_biases']['out_log_var'] = tf.Variable(tf.zeros([self.x_size], dtype=tf.float32))

        return all_weights


    def l2_regularization(self):
        sum_ = 0
        for net in self.network_weights:
            for weights_biases in self.network_weights[net]:
                sum_ += tf.reduce_sum(tf.square(self.network_weights[net][weights_biases]))
        return sum_


    def log_normal(self, sample, mean, log_var):
        '''
        Log of normal distribution
        sample is [B, D]
        mean is [B, D]
        log_var is [B, D]
        output is [B]
        '''
        term1 = tf.reduce_sum(log_var, reduction_indices=1) #sum over dimensions D so now its [B]
        term2 = tf.cast(tf.shape(sample)[1], tf.float32) * tf.log(2*math.pi) #scalar, will be broadcast across batches
        dif_cov = tf.square(sample - mean) / tf.exp(log_var)
        term3 = tf.reduce_sum(dif_cov, 1) #sum over dimensions so its [B]
        log_norm = -.5 * (term1 + term2 + term3)
        return log_norm

    def log_bernoulli(self, sample, mean):
        '''
        Log of bernoulli distribution
        sample is [B, D]
        mean is [B, D]
        output is [B]
        '''
        negative_log_p =  \
            tf.reduce_sum(tf.maximum(mean, 0) 
                        - mean * sample
                        + tf.log(1 + tf.exp(-abs(mean))),
                         1) #sum over D
        log_p = -negative_log_p
        return log_p



    def recognition_net(self, input_):
        # input:[B,2X+A+Z]

        n_layers = len(self.network_weights['encoder_weights']) - 2 #minus 2 for the mean and var outputs
        weights = self.network_weights['encoder_weights']
        biases = self.network_weights['encoder_biases']

        for layer_i in range(n_layers):

            input_ = self.transfer_fct(tf.add(tf.matmul(input_, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 
            #add batch norm here

        z_mean = tf.add(tf.matmul(input_, weights['out_mean']), biases['out_mean'])
        z_log_var = tf.add(tf.matmul(input_, weights['out_log_var']), biases['out_log_var'])

        return z_mean, z_log_var


    def observation_net(self, input_):
        # input:[B,X+Z]

        n_layers = len(self.network_weights['decoder_weights']) - 1 #minus 1 for the mean 
        weights = self.network_weights['decoder_weights']
        biases = self.network_weights['decoder_biases']

        for layer_i in range(n_layers):

            input_ = self.transfer_fct(tf.add(tf.matmul(input_, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 
            #add batch norm here

        x_mean = tf.add(tf.matmul(input_, weights['out_mean']), biases['out_mean'])

        return x_mean





    def calc_elbo(self, x):
        '''
        x: [B,X]
        output: elbo scalar
        '''

        #Infer z distribution
        z_mean, z_log_var = self.recognition_net(x)

        #Sample it 
        eps = tf.random_normal((self.batch_size, self.z_size), 0, 1, dtype=tf.float32)
        self.particles = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_var)), eps))

        #Reconstruct z->x
        self.x_mean = self.observation_net(self.particles)

        #Calc log_px, log_pz, log_qz [B]
        log_qz = self.log_normal(self.particles, z_mean, z_log_var)
        log_pz = self.log_normal(self.particles, tf.zeros([self.batch_size, self.z_size]), tf.log(tf.ones([self.batch_size, self.z_size])))
        log_px = self.log_bernoulli(x, self.x_mean)
        #Average over batch [1]
        self.log_qz = tf.reduce_mean(log_qz, axis=0)
        self.log_pz = tf.reduce_mean(log_pz, axis=0)
        self.log_px = tf.reduce_mean(log_px, axis=0)

        #Calc elbo
        elbo = self.log_px + self.log_pz - self.log_qz
        # elbo = self.log_px - self.log_qz


        return elbo

                


    def train(self, get_data, steps=1000, display_step=10, path_to_load_variables='', path_to_save_variables=''):

        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            self.sess.run(tf.global_variables_initializer())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables


        # Training cycle
        for step in range(steps):

            batch = []
            while len(batch) != self.batch_size:
                sequence=get_data()
                batch.append(sequence)

            _ = self.sess.run(self.optimizer, feed_dict={self.x: batch})

            # Display
            if step % display_step == 0:

                el,p1,p2,p3,reg = self.sess.run([self.elbo, self.log_px, self.log_pz, self.log_qz,self.l2_regularization()], feed_dict={self.x: batch})
                # cost = -el #because I want to see the NLL
                print "Step:", '%04d' % (step), "elbo=", "{:.5f}".format(el), 'logprobs', p1, '+', p2, '-', p3, 'reg', reg*self.learning_rate

        if path_to_save_variables != '':
            saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables

        print 'Done training'
        return self.sess



    def generate(self, load=False, path_to_load_variables='', z_mu=None):
        if load:
            saver = tf.train.Saver()
            self.sess = tf.Session()
            if path_to_load_variables == '':
                self.sess.run(tf.global_variables_initializer())
            else:
                saver.restore(self.sess, path_to_load_variables)
                print 'loaded variables ' + path_to_load_variables

        if z_mu is None:
            z_mu = np.random.normal(size=(self.batch_size, self.z_size))

        return self.sess.run(tf.sigmoid(self.x_mean), feed_dict={self.particles: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """

        return self.sess.run(tf.sigmoid(self.x_mean), feed_dict={self.x: X})








if __name__ == "__main__":

    save_to = home + '/data/' #for boltz
    # save_to = home + '/Documents/tmp/' # for mac
    # path_to_load_variables=save_to + 'vae_vector.ckpt'
    path_to_load_variables=''
    path_to_save_variables=save_to + 'vae_mnist.ckpt'
    # path_to_save_variables=''

    training_steps = 40000
    n_x = 784
    n_z = 50
    n_batch = 50


    # To see the data
    # batch = []
    # for i in range(20):
    #     seq = get_sequence(n_timesteps=1,vector_height=10)
    #     batch.append(seq)
    # batch = np.array(batch)
    # import scipy.misc
    # scipy.misc.imsave(save_to +'outfile.jpg', batch)
    # fdsafafd

    import get_data
    mnist_x = get_data.load_binarized_mnist()
    def get_datapoint():
        return mnist_x[np.random.randint(0, 50000)]


    network_architecture = \
        dict(   encoder_net=[100,100],
                decoder_net=[100,100],
                n_x=n_x,
                n_z=n_z) 
    
    print 'Initializing model..'
    vae = VAE(network_architecture, batch_size=n_batch)


    # print 'Training'
    # sess = vae.train(get_data=get_datapoint, steps=training_steps, display_step=500, path_to_load_variables=path_to_load_variables, path_to_save_variables=path_to_save_variables)

    print 'Generating'
    generated = vae.generate(load=True, path_to_load_variables=path_to_save_variables, z_mu=None)


    print 'Reconstructing'
    batch = []
    for i in range(n_batch):
        seq = get_datapoint()
        batch.append(seq)
    batch = np.array(batch)
    reconstructed = vae.reconstruct(batch)

    batch= np.reshape(batch, [n_batch, 28,28])
    generated= np.reshape(generated, [n_batch, 28,28])
    reconstructed= np.reshape(reconstructed, [n_batch, 28,28])


    import scipy.misc

    # scipy.misc.imsave(save_to +'outfile.jpg', generated)
    scipy.misc.toimage(batch[0], cmin=0., cmax=1.).save(save_to +'original.jpg')
    scipy.misc.toimage(generated[0], cmin=0., cmax=1.).save(save_to +'generated.jpg')
    scipy.misc.toimage(reconstructed[0], cmin=0., cmax=1.).save(save_to +'reconstructed.jpg')

    print 'saved stuff'











