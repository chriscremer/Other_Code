

# this didnt get completed



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



# class distribution(object):

#     def __init__(self, eval_=None, sample_=None):

#         self.eval_ = eval_
#         self.sample_ = sample_

#     def sample(self):

#         return self.sample_

#     def eval(self, z):

#         return self.eval_(x)





class Normal_distribution(object):

    def __init__(self, mean, logvar):

        self.mean = mean      #[B,D]
        self.logvar = logvar  #[B,D]

        self.batch_size = tf.shape(self.mean)[0]
        self.z_size = tf.shape(self.mean)[1]

        self.rs = 0


    def sample(self, k):
        #Sample z  [P,B,Z]
        eps = tf.random_normal((k, self.batch_size, self.z_size), 0, 1, seed=self.rs) 
        z = tf.add(self.mean, tf.multiply(tf.sqrt(tf.exp(self.logvar)), eps)) #broadcast, [P,B,Z]
        return z

    def eval(self, z):
        '''
        Log of normal distribution, log pdf

        z is [P, B, D]
        mean is [B, D]
        log_var is [B, D]
        output is [P,B]
        '''

        D = tf.to_float(tf.shape(self.mean)[1])
        term1 = D * tf.log(2*np.pi) #[1]
        term2 = tf.reduce_sum(self.log_var, axis=1) #sum over D, [B]
        dif_cov = tf.square(z - self.mean) / tf.exp(self.log_var) #[P,B,D]
        term3 = tf.reduce_sum(dif_cov, axis=2) #sum over D, [P,B]
        log_N = -.5 * (term1 + term2 + term3) #[P,B]
        return log_N









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


        # create_ais()

        self.z = tf.placeholder(tf.float32, [None, self.z_size], name='z')
        self.t = tf.placeholder(tf.float32, [], name='t')

        self.energy_func = (1.-self.t)*log_pz + self.t*(log_px+log_pz)



        init_z, _, _ = self.sample_z(x, encoder, decoder)

        # RIght ,so I dont want to recompute q every step of AIS, so I need to store it somewhere.

        approx_posterior = Normal_distribution(split_mean_logvar(encoder.feedforward(x)))
        prior_dist = Normal_distribution(tf.zeros([self.batch_size, self.z_size]), tf.zeros([self.batch_size, self.z_size]))
        # I dont think this will work, itll want x everytime I want to compute q. ...





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
















    def create_ais(self):

        # self.z = tf.placeholder(tf.float32, [None, self.generator.input_dim], name='z')
        self.t = tf.placeholder(tf.float32, [], name='t')
        self.energy_func = (1.-self.t)*log_pz + self.t*(log_px+log_pz)



        stepsize=0.01
        n_steps=10
                 
        target_acceptance_rate=.65
        avg_acceptance_slowness=0.9
        stepsize_min=0.0001
        stepsize_max=0.5
        stepsize_dec=0.98
        stepsize_inc=1.02

        self.zv = None
        self.batch_size = tf.shape(self.x)[0]
        self.num_samples = num_samples
        self.sigma = sigma


    mu = self.generator(z)
    mu = tf.reshape(mu, [self.num_samples, self.batch_size, self.generator.output_dim])

    # logpz + t*logpx  , id like to prove again why its only t, and not t-1...because second term includes first too, so it cancels out
    e = self.prior.logpdf(z) + self.t * tf.reshape(self.kernel.logpdf(self.x, mu, self.sigma), [self.num_samples * self.batch_size])
    return -e


        self.lld = tf.reshape(-self.energy_fn(self.z), [num_samples, self.batch_size])

        self.stepsize = tf.Variable(stepsize)
        self.avg_acceptance_rate = tf.Variable(target_acceptance_rate)

        self.accept, self.final_pos, self.final_vel = hmc_move(
            self.z,
            self.energy_fn,
            stepsize,
            n_steps
        )

        # Applies accept step to samples, and changes step size
        self.new_z, self.updates = hmc_updates(
            self.z,
            self.stepsize,
            avg_acceptance_rate=self.avg_acceptance_rate,
            final_pos=self.final_pos,
            accept=self.accept,
            stepsize_min=stepsize_min,
            stepsize_max=stepsize_max,
            stepsize_dec=stepsize_dec,
            stepsize_inc=stepsize_inc,
            target_acceptance_rate=target_acceptance_rate,
            avg_acceptance_slowness=avg_acceptance_slowness
        )









































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










    def ais(self, data, path_to_load_variables='', batch_size=20):
        '''
        Evaluate using ais.
        data: [B,X]
        '''
        with tf.Session() as self.sess:

            if path_to_load_variables == '':
                self.sess.run(self.init_vars)
            else:
                #Load variables
                self.saver.restore(self.sess, path_to_load_variables)
                print 'loaded variables ' + path_to_load_variables


            logws = []

            data_index = 0
            for step in range(len(data)/batch_size):
                #Make batch
                batch = []
                while len(batch) != batch_size:
                    batch.append(data[data_index]) 
                    data_index +=1


                #AIS

                logw = 0.0
                #Init sample  [B*P,D],  Sampled from the init distribution
                self.zv = np.random.normal(0.0, 1.0, [self.batch_size * self.k, self.x_size])

                for (t0, t1) in zip(schedule[:-1], schedule[1:]):

                    # pt / pt-1
                    new_u = self.sess.run(self.lld, feed_dict={self.t: t, self.x: batch, self.z: self.zv})  #[P,D]
                    prev_u = self.sess.run(self.lld, feed_dict={self.t: t, self.x: batch, self.z: self.zv})  #[P,D]
                    logw += new_u - prev_u  # [P,B]

                    accept = self.step(x, t1)  # [P*B]

                    print(np.mean(accept), t0)

                lld = np.squeeze(log_mean_exp(logw, axis=0), axis=0)  #[B]

                logws.append(lld)


        return np.mean(lld)










#         #Config
#         n_independent_chains = 16
#         n_LF_steps = 10
#         step_size = .1
#         n_intermediate_dists = 100 #10000

#         #Distributions
#         z_mean_logvar = encoder.feedforward(x) #[B,Z*2]
#         z_mean, z_logvar = split_mean_logvar(z_mean_logvar)

#         log_pz = log_norm(z, tf.zeros([self.batch_size, self.z_size]), tf.zeros([self.batch_size, self.z_size]))
#         log_qz = log_norm(z, z_mean, z_logvar)
#         y = decoder.feedforward(z)
#         y = tf.reshape(y, [self.k, self.batch_size, self.x_size])  #[P,B,X]
#         log_px = log_bern(x,y)


#         log_joint = log_px + log_pz
#         log_init_dist = log_qz

#         ws = []
#         for i in range(n_independent_chains):

#             init_sample = 
#             init_prob = 

#             w = self.compute_w(n_LF_steps=n_LF_steps, step_size=step_size, 
#                     n_intermediate_dists=n_intermediate_dists,
#                     log_joint=log_joint,
#                     log_init_dist=log_init_dist)

#             ws.append(w)

#         return tf.reduce_mean(tf.stack(ws))


#     def compute_w(self, n_LF_steps, step_size, n_intermediate_dists,log_joint,log_init_dist):

#         # pt / pt-1

#         return w

#     def HMC(self, proposal_dist, target_dist):

#         #sample momentum

#         #do LF

#         #accept prob  

#         return z

#     def LF(self):

#         return z

# ############NEED TO READ AIS PAPER AGAIN
# # Im confused by the two levels of ratios and when to sample momentum
# # since HMC needs to take many smaples, to approx dist, but here its like Im taking one smaple..









if __name__ == '__main__':

    train_ = 1
    eval_ = 1

    x_size = 784
    z_size = 10
    batch_size = 50
    epochs = 50

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

    path_to_load_variables=home+'/Documents/tmp/vars2.ckpt' 
    # path_to_load_variables=''
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
























