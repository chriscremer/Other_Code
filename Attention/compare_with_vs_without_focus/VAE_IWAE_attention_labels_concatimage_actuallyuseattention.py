




#Generative Autoencoder classes
import numpy as np
from numpy import random as npr
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle





class VAE(object):

    def __init__(self, batch_size, focus_bool=False):
        
        tf.reset_default_graph()

        self.focus_bool = focus_bool

        self.network_architecture = dict(n_input=784*2, # 784 image
                                         encoder_net=[100, 100], 
                                         n_z=5,  # dimensionality of latent space
                                         decoder_net=[100, 100]) 

        self.transfer_fct = tf.nn.relu #tf.nn.softplus #tf.tanh #
        self.learning_rate = 0.0001
        self.batch_size = batch_size
        self.n_particles = 1
        self.n_z = self.network_architecture["n_z"]
        self.z_size = self.n_z
        self.n_input = self.network_architecture["n_input"]
        self.input_size = self.n_input
        self.reg_param = .0001
        self.n_classes = 10

        with tf.name_scope('model_input'):
            #Placeholders - Inputs
            self.x = tf.placeholder(tf.float32, [None, self.n_input])
            self.y = tf.placeholder(tf.float32, [None, self.n_classes])
            self.focus = tf.placeholder(tf.float32, [self.n_input])

        
        
        #Variables
        self.params_dict, self.params_list, self.attention_params_list, self.model_params_list = self._initialize_weights(self.network_architecture)
        
        #Encoder - Recognition model - q(z|x): recog_mean,z_log_std_sq=[batch_size, n_z]
        self.recog_means, self.recog_log_vars = self._recognition_network(self.x)
        
        #Sample
        # eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        # self.z = tf.add(self.recog_means, tf.mul(tf.sqrt(tf.exp(self.recog_log_vars)), eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]
        self.z = self.recog_means

        #Decoder - Generative model - p(x|z)
        self.x_reconstr_mean_no_sigmoid, self.y_reconstr_mean_no_sigmoid = self._generator_network(self.z) #no sigmoid

        #Objective
        self.elbo = self.elbo(self.x, self.x_reconstr_mean_no_sigmoid, self.z, self.recog_means, self.recog_log_vars, self.y, self.y_reconstr_mean_no_sigmoid)
        self.cost = -self.elbo + (self.reg_param * self.l2_regularization())

        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost, var_list=self.model_params_list)

        # self.grad = tf.gradients(self.cost, self.model_params_list)
        # self.attn_cost = self.predict_grad_error()
        # self.attn_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.attn_cost, var_list=self.attention_params_list)

        #For evaluation
        # self.log_w = self._log_likelihood(self.x, self.x_reconstr_mean_no_sigmoid) + self._log_p_z(self.z) - self._log_q_z_given_x(self.z, self.recog_means, self.recog_log_vars)
        self.x_reconstr_mean = tf.nn.sigmoid(self.x_reconstr_mean_no_sigmoid)
        # self.y_reconstr_mean = tf.nn.sigmoid(self.y_reconstr_mean_no_sigmoid)

        self.y_reconstr_mean = tf.nn.softmax(self.y_reconstr_mean_no_sigmoid)



        # self.get_focus = tf.nn.softmax(tf.square(tf.gradients(self.y_reconstr_mean_no_sigmoid, self.x)))
        # self.get_focus = tf.gradients(self.y_reconstr_mean_no_sigmoid, self.x)

        # print self.y_reconstr_mean_no_sigmoid
        # fsada

        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = tf.split(self.y_reconstr_mean_no_sigmoid, num_or_size_splits=10, axis=1)


        # one_var = tf.slice(self.y_reconstr_mean_no_sigmoid, [self.batch_size, 0], [self.batch_size, 1])
        # self.get_focus = tf.gradients(self.y_reconstr_mean_no_sigmoid, self.x)


        a0 = tf.square(tf.gradients(a0, self.x))
        a1 = tf.square(tf.gradients(a1, self.x))
        a2 = tf.square(tf.gradients(a2, self.x))
        a3 = tf.square(tf.gradients(a3, self.x))
        a4 = tf.square(tf.gradients(a4, self.x))
        a5 = tf.square(tf.gradients(a5, self.x))
        a6 = tf.square(tf.gradients(a6, self.x))
        a7 = tf.square(tf.gradients(a7, self.x))
        a8 = tf.square(tf.gradients(a8, self.x))
        a9 = tf.square(tf.gradients(a9, self.x))

        self.get_focus = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 
        # print self.get_focus [1,B,X]

        self.get_focus = tf.reduce_mean(self.get_focus, 0)  #[B,X]
        self.get_focus = tf.reduce_mean(self.get_focus, 0)  #[X]




        # print self.get_focus
        # fasdf


        # generate_samples, predicted_y = self._generator_network(eps)
        # self.generate_samples = tf.nn.sigmoid(generate_samples)
        # self.predicted_y = tf.nn.sigmoid(predicted_y)



    def _initialize_weights(self, network_architecture):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


        params_dict = dict()

        #Recognition/Inference net q(z|z-1,u,x)
        with tf.name_scope('encoder_vars'):

            # params_dict['conv_weights'] = tf.Variable(tf.truncated_normal([self.filter_height, self.filter_width, self.n_channels, self.filter_out_channels1], stddev=0.1))
            # params_dict['conv_biases'] = tf.Variable(tf.truncated_normal([self.filter_out_channels1], stddev=0.1))

            for layer_i in range(len(network_architecture['encoder_net'])):
                if layer_i == 0:
                    params_dict['encoder_weights_l'+str(layer_i)] = tf.Variable(xavier_init(self.input_size, network_architecture['encoder_net'][layer_i]))
                    params_dict['encoder_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))
                else:
                    params_dict['encoder_weights_l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['encoder_net'][layer_i-1], network_architecture['encoder_net'][layer_i]))
                    params_dict['encoder_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

            params_dict['encoder_weights_out_mean'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.z_size))
            params_dict['encoder_weights_out_log_var'] = tf.Variable(xavier_init(network_architecture['encoder_net'][-1], self.z_size))
            params_dict['encoder_biases_out_mean'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))
            params_dict['encoder_biases_out_log_var'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))


        #Generator net p(x|z)
        with tf.name_scope('decoder_vars'):

            for layer_i in range(len(network_architecture['decoder_net'])):
                if layer_i == 0:
                    params_dict['decoder_weights_l'+str(layer_i)] = tf.Variable(xavier_init(self.z_size, network_architecture['decoder_net'][layer_i]))
                    params_dict['decoder_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['decoder_net'][layer_i]], dtype=tf.float32))
                else:
                    params_dict['decoder_weights_l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['decoder_net'][layer_i-1], network_architecture['decoder_net'][layer_i]))
                    params_dict['decoder_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['decoder_net'][layer_i]], dtype=tf.float32))

            params_dict['decoder_weights_out_mean'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.input_size))
            params_dict['decoder_biases_out_mean'] = tf.Variable(tf.zeros([self.input_size], dtype=tf.float32))


            params_dict['decoder_weights_out_y'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.n_classes))
            params_dict['decoder_biases_out_y'] = tf.Variable(tf.zeros([self.n_classes], dtype=tf.float32))


        # #Attention
        # with tf.name_scope('attention_vars'):

        #     for layer_i in range(len(network_architecture['attention'])):
        #         if layer_i == 0:
        #             params_dict['attention_weights_l'+str(layer_i)] = tf.Variable(xavier_init(self.input_size, network_architecture['attention'][layer_i]))
        #             params_dict['attention_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['attention'][layer_i]], dtype=tf.float32))
        #         else:
        #             params_dict['attention_weights_l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['attention'][layer_i-1], network_architecture['attention'][layer_i]))
        #             params_dict['attention_biases_l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['attention'][layer_i]], dtype=tf.float32))

        #     params_dict['attention_weights_out_mean'] = tf.Variable(xavier_init(network_architecture['attention'][-1], self.input_size))
        #     params_dict['attention_biases_out_mean'] = tf.Variable(tf.zeros([self.input_size], dtype=tf.float32))


        params_list = []
        model_params_list = []
        attention_params_list = []
        for layer in params_dict:

            params_list.append(params_dict[layer])
            if 'attention' in layer:
                attention_params_list.append(params_dict[layer])
            else:
                model_params_list.append(params_dict[layer])



        return params_dict, params_list, attention_params_list, model_params_list



    def l2_regularization(self):


        with tf.name_scope('L2_reg'):

            sum_ = 0
            for layer in self.params_dict:

                sum_ += tf.reduce_sum(tf.square(self.params_dict[layer]))

        return sum_



    # def grad_sum(self):


    #     with tf.name_scope('grad_sum_'):

    #         grad = tf.gradients(self.y_reconstr_mean_no_sigmoid, self.x)
    #         sum_ = tf.reduce_sum(tf.square(grad))


    #         # sum_ = 0
    #         # for layer in self.model_params_list:

    #         #   grad = tf.gradients(self.cost, layer)
    #         #     sum_ += tf.reduce_sum(tf.square(grad))

    #     return sum_




    # def predict_grad_error(self):

    #     prediction = self._attention_network(self.x)

    #     # print prediction

    #     grad = tf.gradients(self.y_reconstr_mean_no_sigmoid, self.x)

    #     # print grad


    #     sum_ = tf.reduce_sum(tf.square(grad-prediction))

    #     return sum_






    def _recognition_network(self, x):


        with tf.name_scope('recognition_net'):

            input_ = x

            # attention = self._attention_network(input_) #[B,X]
            # attention = tf.reshape(attention, [self.batch_size,784])
            # input_ = input_ * attention

            n_layers = len(self.network_architecture['encoder_net'])

            for layer_i in range(n_layers):

                # input_ = self.transfer_fct(tf.contrib.layers.layer_norm(tf.add(tf.matmul(input_, self.params_dict['encoder_weights_l'+str(layer_i)]), self.params_dict['encoder_biases_l'+str(layer_i)])))
                input_ = self.transfer_fct(tf.add(tf.matmul(input_, self.params_dict['encoder_weights_l'+str(layer_i)]), self.params_dict['encoder_biases_l'+str(layer_i)]))
                
            z_mean = tf.add(tf.matmul(input_, self.params_dict['encoder_weights_out_mean']), self.params_dict['encoder_biases_out_mean'])
            z_log_var = tf.add(tf.matmul(input_, self.params_dict['encoder_weights_out_log_var']), self.params_dict['encoder_biases_out_log_var'])

        return z_mean, z_log_var




    def _generator_network(self, z):

        z = tf.reshape(z, [self.n_particles*self.batch_size, self.n_z])


        with tf.name_scope('observation_net'):

            input_ = z

            n_layers = len(self.network_architecture['decoder_net'])

            for layer_i in range(n_layers):

                # input_ = self.transfer_fct(tf.contrib.layers.layer_norm(tf.add(tf.matmul(input_, self.params_dict['decoder_weights_l'+str(layer_i)]), self.params_dict['decoder_biases_l'+str(layer_i)])))
                input_ = self.transfer_fct(tf.add(tf.matmul(input_, self.params_dict['decoder_weights_l'+str(layer_i)]), self.params_dict['decoder_biases_l'+str(layer_i)]))

            x_mean = tf.add(tf.matmul(input_, self.params_dict['decoder_weights_out_mean']), self.params_dict['decoder_biases_out_mean'])
            y_mean = tf.add(tf.matmul(input_, self.params_dict['decoder_weights_out_y']), self.params_dict['decoder_biases_out_y'])

        return x_mean, y_mean





    # def _attention_network(self, x):


    #     with tf.name_scope('attention_net'):

    #         input_ = x
    #         n_layers = len(self.network_architecture['attention'])

    #         for layer_i in range(n_layers):

    #             input_ = self.transfer_fct(tf.contrib.layers.layer_norm(tf.add(tf.matmul(input_, self.params_dict['attention_weights_l'+str(layer_i)]), self.params_dict['attention_biases_l'+str(layer_i)])))
                
    #         # attn = tf.nn.softmax(tf.add(tf.matmul(input_, self.params_dict['attention_weights_out_mean']), self.params_dict['attention_biases_out_mean']))
    #         attn = tf.add(tf.matmul(input_, self.params_dict['attention_weights_out_mean']), self.params_dict['attention_biases_out_mean'])
            
    #         # z_log_var = tf.add(tf.matmul(input_, self.params_dict['attention_weights_out_log_var']), self.params_dict['attention_biases_out_log_var'])


    #     return attn




    def _log_p_z(self, z):
        '''
        Log of normal distribution with zero mean and one var

        z is [n_particles, batch_size, n_z]
        output is [n_particles, batch_size]
        '''

        # term1 = 0
        term2 = self.n_z * tf.log(2*math.pi)
        term3 = tf.reduce_mean(tf.square(z), 1) #sum over dimensions n_z so now its [particles, batch]

        all_ = term2 + term3
        log_p_z = -.5 * all_

        log_p_z = tf.reduce_mean(log_p_z)

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
        term3 = tf.reduce_mean(dif_cov, 1) #sum over dimensions n_z so now its [particles, batch]

        all_ = term1 + term2 + term3
        log_p_z_given_x = -.5 * all_

        log_p_z_given_x = tf.reduce_mean(log_p_z_given_x)


        return log_p_z_given_x


    def _log_likelihood(self, t, pred_no_sig):
        '''
        Log of bernoulli distribution

        t is [batch_size, n_input]
        pred_no_sig is [n_particles, batch_size, n_input] 
        output is [n_particles, batch_size]
        '''


        # reconstr_loss = \
        #         tf.reduce_sum(tf.maximum(pred_no_sig, 0) 
        #                     - pred_no_sig * t
        #                     + tf.log(1 + tf.exp(-tf.abs(pred_no_sig))),
        #                      2) #sum over dimensions



        # [B,X]
        reconstr_loss = tf.maximum(pred_no_sig, 0) - pred_no_sig * t + tf.log(1 + tf.exp(-tf.abs(pred_no_sig)))


        if self.focus_bool:
            reconstr_loss = reconstr_loss * self.focus  #[B,X] * [X]
        # print reconstr_loss
        # fasdf

        # attention = np.concatenate( (np.ones([392]), np.zeros([392])), axis=0)
        # attention = self._attention_network(t) #[B,X]
        # attention = self.focus

        # tf.nn.softmax(tf.square(tf.gradients(self.y_reconstr_mean_no_sigmoid, self.x)))

        # attention = tf.nn.softmax(tf.reshape(attention, [1,self.batch_size,784*2]))

        # reconstr_loss = reconstr_loss * attention
        reconstr_loss =  tf.reduce_mean(reconstr_loss) #*784 #sum over dimensions

        #negative because the above calculated the NLL, so this is returning the LL
        return -reconstr_loss




    def _log_likelihood_y(self, t, pred_no_sig):
        '''
        Log of bernoulli distribution

        t is [batch_size, n_input]
        pred_no_sig is [n_particles, batch_size, n_input] 
        output is [n_particles, batch_size]
        '''


        # reconstr_loss = \
        #         tf.reduce_sum(tf.maximum(pred_no_sig, 0) 
        #                     - pred_no_sig * t
        #                     + tf.log(1 + tf.exp(-tf.abs(pred_no_sig))),
        #                      2) #sum over dimensions


        # [P,B,X]
        # reconstr_loss = tf.maximum(pred_no_sig, 0) - pred_no_sig * t + tf.log(1 + tf.exp(-tf.abs(pred_no_sig)))

        # print pred_no_sig
        # print t
        # reconstr_loss = tf.contrib.losses.softmax_cross_entropy(logits=pred_no_sig, onehot_labels=t)
        reconstr_loss = tf.losses.softmax_cross_entropy(logits=pred_no_sig, onehot_labels=t)
        
        

        # reconstr_loss = tf.reduce_mean(tf.square(tf.nn.sigmoid(pred_no_sig) - t), 1)
        # reconstr_loss = tf.reduce_mean(reconstr_loss)

        # cross_entropy = tf.reduce_mean(-tf.reduce_mean(t * tf.log(tf.nn.softmax(pred_no_sig)), 1))
        # reconstr_loss = cross_entropy

        


        # print reconstr_loss

        # attention = np.concatenate( (np.ones([392]), np.zeros([392])), axis=0)
        # attention = self._attention_network(t) #[B,X]
        # attention = tf.reshape(attention, [1,self.batch_size,784])

        # reconstr_loss = reconstr_loss * attention
        # reconstr_loss = tf.reduce_sum(reconstr_loss, 1) #sum over dimensions

        #negative because the above calculated the NLL, so this is returning the LL
        return -reconstr_loss


    def elbo(self, x, x_recon, z, mean, log_var, y, y_recon):

        self.px = self._log_likelihood(x, x_recon)
        self.py = self._log_likelihood_y(y, y_recon)
        self.pz = self._log_p_z(z)
        self.qz = self._log_q_z_given_x(z, mean, log_var)

        elbo = self.px + self.py #+ self.pz - self.qz

        # elbo = self._log_likelihood(x, x_recon) + self._log_likelihood_y(y, y_recon) + self._log_p_z(z) - self._log_q_z_given_x(z, mean, log_var)
        # elbo = self._log_likelihood_y(y, y_recon) 

        # elbo = tf.reduce_mean(elbo, 1) #average over batch
        # elbo = tf.reduce_mean(elbo) #average over particles

        return elbo



    def train(self, train_x, valid_x=[], display_step=100, path_to_load_variables='', path_to_save_variables='', epochs=10, train_y=[]):
        '''
        Train.
        Use early stopping, actually no, because I want it to be equal for each model. Time? Epochs? 
        I'll do stages for now.
        '''

        n_datapoints = len(train_x)
        
        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            # self.sess.run(tf.initialize_all_variables())
            self.sess.run(tf.global_variables_initializer())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables

        start = time.time()
        # for stage in range(starting_stage,ending_stage+1):

            # self.learning_rate = .001 * 10.**(-stage/float(ending_stage))
            # print 'learning rate', self.learning_rate
            # print 'stage', stage

            # passes_over_data = 3**stage

        for epoch in range(epochs):

            #shuffle the data
            # arr = np.arange(len(train_x))
            # np.random.shuffle(arr)
            # train_x = train_x[arr]


            data_index = 0
            for step in range(n_datapoints/self.batch_size):

                #Make batch
                batch = []
                batch_y = []
                while len(batch) != self.batch_size:

                    img1 = np.reshape(train_x[data_index], [28,28])
                    img2 = np.reshape(train_x[np.random.randint(0,len(train_x)-1)], [28,28])

                    input_concat = np.concatenate([img1, img2], axis=1)
                    input_concat = np.reshape(input_concat, [784*2])

                    # batch.append(train_x[data_index])
                    batch.append(input_concat) #for concatenated


                    # print train_y[data_index]
                    one_hot = np.zeros([10])
                    one_hot[train_y[data_index]] = 1.
                    batch_y.append(one_hot)

                    data_index +=1

                # Fit training using batch data
                # _, _ = self.sess.run((self.optimizer, self.attn_opt), feed_dict={self.x: batch, self.y: batch_y})

                focus = self.sess.run(self.get_focus, feed_dict={self.x: batch, self.y: batch_y})

                _ = self.sess.run((self.optimizer), feed_dict={self.x: batch, self.y: batch_y, self.focus: focus})


                # print self.sess.run((self.asdf), feed_dict={self.x: batch})
                # fasdfa
                
                # Display logs per epoch step
                if step % display_step == 0:

                    elbo, cost, px, py, pz, qz = self.sess.run((self.elbo, self.cost, self.px, self.py, self.pz, self.qz), feed_dict={self.x: batch, self.y: batch_y, self.focus: focus})
                    elbo = -elbo #because I want to see the NLL

                    print ("Epoch", str(epoch+1)+'/'+str(epochs), 'Step:%04d' % (step) +'/'+ str(n_datapoints/self.batch_size), 
                                "cost=", "{:.3f}".format(float(cost)), 
                                "elbo=", "{:.3f}".format(float(elbo)), 
                                "px=", "{:.3f}".format(float(px)),
                                "py=", "{:.3f}".format(float(py)), 
                                "pz=", "{:.2f}".format(float(pz)), 
                                "qz=", "{:.2f}".format(float(qz)))#, 'time', time.time() - start

                    # lw= self.sess.run((self.log_weights), feed_dict={self.x: batch})
                    # print np.exp(lw)

            print 'time per epoch', time.time()-start
            start = time.time()

            if epoch % 50 == 0 and epoch != 0:
                if path_to_save_variables != '':
                    # print 'saving variables to ' + path_to_save_variables
                    save_to = path_to_save_variables+'_'+str(epoch)+'.ckpt'
                    saver.save(self.sess, save_to)
                    print 'Saved variables to ' + save_to

        if path_to_save_variables != '':
            # print 'saving variables to ' + path_to_save_variables
            save_to = path_to_save_variables+'_'+str(epochs)+'.ckpt'
            saver.save(self.sess, save_to)
            print 'Saved variables to ' + save_to








    def encode(self, data):

        return self.sess.run([self.recog_means, self.recog_log_vars], feed_dict={self.x:data})


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



    def reconstruct(self, sampling, data, labels, rs):


        # batch = []
        # while len(batch) != self.batch_size:
        #     datapoint = data[np.random.randint(0,len(data))]
        #     batch.append(datapoint)


        batch = []
        batch_y = []
        while len(batch) != self.batch_size:

            ind1 = rs.randint(0,len(data))
            # ind1 = np.random.randint(0,len(data))
            # ind2 = np.random.randint(0,len(data))

            img1 = np.reshape(data[ind1], [28,28])
            img2 = np.reshape(data[ind1], [28,28])

            input_concat = np.concatenate([img1, img2], axis=1)
            input_concat = np.reshape(input_concat, [784*2])

            # batch.append(data[ind1])
            batch.append(input_concat)


            one_hot = np.zeros([10])
            one_hot[labels[ind1]] = 1.
            batch_y.append(one_hot)


        if sampling == 'vae':

            #Encode and get p and q
            predictions, recons, focus = self.sess.run((self.y_reconstr_mean, self.x_reconstr_mean, self.get_focus), feed_dict={self.x: batch})

            for b in range(len(predictions)):
                # for c in range(len(predictions[b])):

                    print predictions[b]
                    print batch_y[b]
                    print
            # print log_ws.shape
            # print recons.shape



            # for b in range(len(focus)):
            #     # for c in range(len(predictions[b])):
            #     print 'b', b
            #     for d in range(len(focus[b])):
            #         print focus[b][d]



            # return recons, np.array(batch), np.abs(np.array(focus))
            return recons, np.array(batch), np.array(focus)


        if sampling == 'iwae':

            recons_resampled = []
            for i in range(self.n_particles):

                #Encode and get p and q.. log_ws [K,B,1], reons [K,B,X]
                log_ws, recons = self.sess.run((self.log_w, self.x_reconstr_mean), feed_dict={self.x: batch})

                #log normalize
                max_ = np.max(log_ws, axis=0)
                lse = np.log(np.sum(np.exp(log_ws-max_), axis=0)) + max_
                log_norm_ws = log_ws - lse

                # ws = np.exp(log_ws)
                # sums = np.sum(ws, axis=0)
                # norm_ws = ws / sums


                # print log_ws
                # print
                # print lse
                # print
                # print log_norm_ws
                # print 
                # print np.exp(log_norm_ws)
                # fsdfa

                #sample one based on cat(w)

                samps = []
                for j in range(self.batch_size):

                    samp = np.argmax(np.random.multinomial(1, np.exp(log_norm_ws.T[j])-.000001))
                    samps.append(recons[samp][j])
                    # print samp

                # print samps
                # print samps.shape
                # fasdf
                recons_resampled.append(np.array(samps))

            recons_resampled = np.array(recons_resampled)
            # print recons_resampled.shape


            return recons_resampled, batch


    def generate(self):

        samps = self.sess.run(self.generate_samples)

        # print log_ws.shape
        # print recons.shape

        return samps






class IWAE(VAE):

    def elbo(self, x, x_recon, z, mean, log_var):

        # [P, B]
        temp_elbo = self._log_likelihood(x, x_recon) + self._log_p_z(z) - self._log_q_z_given_x(z, mean, log_var)

        max_ = tf.reduce_max(temp_elbo, reduction_indices=0) #over particles? so its [B]

        elbo = tf.log(tf.reduce_mean(tf.exp(temp_elbo-max_), 0)) + max_  #mean over particles so its [B]

        elbo = tf.reduce_mean(elbo) #over batch

        return elbo



























