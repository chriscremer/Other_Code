


# Variational Autoencoder

import numpy as np
import tensorflow as tf
import pickle
from os.path import expanduser
home = expanduser("~")

from utils import log_normal as log_norm
from utils import log_bernoulli as log_bern
from utils import split_mean_logvar

from NN import NN







class VAE(object):


    def __init__(self, hyperparams):

        tf.reset_default_graph()

        #Model hyperparameters
        self.learning_rate = hyperparams['learning_rate']
        self.encoder_net = hyperparams['encoder_net']
        self.decoder_net = hyperparams['decoder_net']
        self.z_size = hyperparams['z_size']  #Z
        self.x_size = hyperparams['x_size']  #X
        self.rs = 0


        #Placeholders - Inputs/Targets
        self.x = tf.placeholder(tf.float32, [None, self.x_size])
        self.batch_size = tf.shape(self.x)[0]   #B        
        self.k = tf.placeholder(tf.int32, None)  #P

        encoder = NN(self.encoder_net, tf.nn.softplus)
        decoder = NN(self.decoder_net, tf.nn.softplus)
        

        #Objective
        logpx, logpz, logqz = self.log_probs(self.x, encoder, decoder) #[P,B]

        self.log_px = tf.reduce_mean(logpx)
        self.log_pz = tf.reduce_mean(logpz)
        self.log_qz = tf.reduce_mean(logqz)
        temp_elbo = logpx + logpz - logqz

        self.elbo = tf.reduce_mean(temp_elbo)

        max_ = tf.reduce_max(temp_elbo, axis=0) #[B]
        self.iwae_elbo = tf.reduce_mean(tf.log(tf.reduce_mean(tf.exp(temp_elbo-max_))) + max_)

        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                epsilon=1e-02).minimize(-self.elbo)


        #Finalize Initilization
        self.init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()
        # self.sess = tf.Session()



    def sample_z(self, x, encoder, decoder):
        '''
        z: [P,B,Z]
        log_pz: [P,B]
        log_qz: [P,B]
        '''

        #Encode
        z_mean_logvar = encoder.feedforward(x) #[B,Z*2]
        z_mean, z_logvar = split_mean_logvar(z_mean_logvar)

        #Sample z  [P,B,Z]
        eps = tf.random_normal((self.k, self.batch_size, self.z_size), 0, 1, seed=self.rs) 
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps)) #broadcast, [P,B,Z]

        # Calc log probs [P,B]
        log_pz = log_norm(z, tf.zeros([self.batch_size, self.z_size]), tf.zeros([self.batch_size, self.z_size]))
        log_qz = log_norm(z, z_mean, z_logvar)

        return z, log_pz, log_qz




    def log_probs(self, x, encoder, decoder):

        # Sample z   [P,B,Z], [P,B], [P,B]
        z, log_pz, log_qz = self.sample_z(x, encoder, decoder)
        z = tf.reshape(z, [self.k*self.batch_size, self.z_size])  #[PB,Z]

        # Decode [PB,X]
        y = decoder.feedforward(z)
        y = tf.reshape(y, [self.k, self.batch_size, self.x_size])  #[P,B,X]

        # Likelihood p(x|z)  [P,B]
        log_px = log_bern(x,y)

        return log_px, log_pz, log_qz

















    def train(self, train_x, valid_x=[], display_step=[1,100], 
                path_to_load_variables='', path_to_save_variables='', 
                epochs=10, batch_size=20, n_z_particles=1):
        '''
        Train.
        '''
        with tf.Session() as self.sess:
            # self.sess = tf.Session()
            random_seed=1
            rs=np.random.RandomState(random_seed)
            n_datapoints = len(train_x)
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

                data_index = 0
                for step in range(n_datapoints/batch_size):
                    #Make batch
                    batch = []
                    while len(batch) != batch_size:
                        batch.append(train_x[data_index]) 
                        data_index +=1
                    # Fit training using batch data
                    _ = self.sess.run((self.optimizer), feed_dict={self.x: batch, self.k: n_z_particles})
                    # Display logs per epoch step
                    if step % display_step[1] == 0 and epoch % display_step[0] == 0:
                        
                        elbo,log_px,log_pz,log_qz = self.sess.run((self.elbo, self.log_px, 
                                                                    self.log_pz, self.log_qz), 
                                                                    feed_dict={self.x: batch, self.k: n_z_particles})

                        if len(valid_x) > 0:
                            iwae_elbos = []
                            data_index2 = 0
                            for step in range(len(valid_x)/batch_size):

                                #Make batch
                                batch = []
                                while len(batch) != batch_size:
                                    batch.append(valid_x[data_index2]) 
                                    data_index2 +=1
                                # Calc iwae elbo on test set
                                iwae_elbo = self.sess.run((self.iwae_elbo), feed_dict={self.x: batch, self.k: 50})
                                iwae_elbos.append(iwae_elbo)

                            # print 'Valid elbo', np.mean(iwae_elbos)

                            print "Epoch", str(epoch+1)+'/'+str(epochs), \
                                    'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), \
                                    "elbo={:.4f}".format(float(elbo)), \
                                    log_px,log_pz,log_qz, 'valid elbo', np.mean(iwae_elbos)


                        else:
                            print "Epoch", str(epoch+1)+'/'+str(epochs), \
                                    'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), \
                                    "elbo={:.4f}".format(float(elbo)), \
                                    log_px,log_pz,log_qz, 'valid elbo'


            if path_to_save_variables != '':
                self.saver.save(self.sess, path_to_save_variables)
                print 'Saved variables to ' + path_to_save_variables





    def eval_iw(self, data, path_to_load_variables='', batch_size=20):
        '''
        Evaluate.
        '''
        with tf.Session() as self.sess:

            if path_to_load_variables == '':
                self.sess.run(self.init_vars)
            else:
                #Load variables
                self.saver.restore(self.sess, path_to_load_variables)
                print 'loaded variables ' + path_to_load_variables





            n_datapoints = len(data)
            # n_datapoints_for_frac = len(data2)

            iwae_elbos = []
            logpxs=[]
            logpzs=[]
            logqzs=[]

            data_index = 0
            for step in range(n_datapoints/batch_size):

                #Make batch
                batch = []
                while len(batch) != batch_size:
                    batch.append(data[data_index]) 
                    data_index +=1
                # Calc iwae elbo on test set
                iwae_elbo, log_px,log_pz,log_qz = self.sess.run((self.iwae_elbo,
                                                                            self.log_px, self.log_pz, 
                                                                            self.log_qz), 
                                                        feed_dict={self.x: batch, self.k: 50})
                iwae_elbos.append(iwae_elbo)
                logpxs.append(log_px)
                logpzs.append(log_pz)
                logqzs.append(log_qz)

        return [np.mean(iwae_elbos), np.mean(logpxs), np.mean(logpzs), np.mean(logqzs)]












if __name__ == '__main__':

    train_ = 1
    eval_ = 1

    x_size = 784
    z_size = 10
    batch_size = 50
    epochs = 20

    hyperparams = {
        'learning_rate': .0001,
        'x_size': x_size,
        'z_size': z_size,
        'encoder_net': [x_size, 20, z_size*2],
        'decoder_net': [z_size, 20, x_size]}

    model = VAE(hyperparams)

    print 'Loading data'
    with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
        mnist_data = pickle.load(f)

    train_x = mnist_data[0][0]
    train_y = mnist_data[0][1]
    valid_x = mnist_data[1][0]
    valid_y = mnist_data[1][1]
    test_x = mnist_data[2][0]
    test_y = mnist_data[2][1]

    # path_to_load_variables=home+'/Documents/tmp/vars.ckpt' 
    path_to_load_variables=''
    path_to_save_variables=home+'/Documents/tmp/vars2.ckpt'
    # path_to_save_variables=''

    if train_:
        print 'Training'
        model.train(train_x=train_x, valid_x=valid_x,
                    epochs=epochs, batch_size=batch_size, n_z_particles=3, 
                    display_step=[1,100000],
                    path_to_load_variables=path_to_load_variables,
                    path_to_save_variables=path_to_save_variables)

    if eval_:
        print 'Eval'
        iwae_elbo = model.eval_iw(data=test_x, batch_size=batch_size, path_to_load_variables=path_to_save_variables)
        print 'Test:', iwae_elbo
        iwae_elbo = model.eval_iw(data=train_x, batch_size=batch_size, path_to_load_variables=path_to_save_variables)
        print 'Train:', iwae_elbo



    print 'Done.'




# TODO:
# - make train, test, and ais, seperate from the class
# - make loading variables seperate from the train and eval
# - make sess part of model, not the functions
# - implement ais
























