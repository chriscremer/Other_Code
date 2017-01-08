



import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import imageio

from ball_sequence import make_ball_gif


class DKF():

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=5, n_time_steps=2, n_particles=3):
        
        tf.reset_default_graph()

        self.transfer_fct = tf.tanh
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_time_steps = n_time_steps
        self.n_particles = n_particles
        self.z_size = network_architecture["n_z"]
        self.input_size = network_architecture["n_input"]
        self.action_size = network_architecture["n_actions"]


        # Graph Input: [B,T,X], [B,T,A]
        self.x = tf.placeholder(tf.float32, [None, None, self.input_size])
        self.actions = tf.placeholder(tf.float32, [None, None, self.action_size])
        
        #Variables
        self.network_weights = self._initialize_weights(network_architecture)

        #Objective
        self.elbo = self.Model(self.x, self.actions, self.network_weights)

        #Optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(-self.elbo)

        #Evaluation
        self.generate = self.generate(self.x, self.actions)


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
                all_weights['encoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.input_size+self.input_size+self.action_size+self.z_size, network_architecture['encoder_net'][layer_i]))
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
                all_weights['decoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.z_size+self.input_size, network_architecture['decoder_net'][layer_i]))
                all_weights['decoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['decoder_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['decoder_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['decoder_net'][layer_i-1], network_architecture['decoder_net'][layer_i]))
                all_weights['decoder_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['encoder_net'][layer_i]], dtype=tf.float32))

        all_weights['decoder_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['decoder_net'][-1], self.input_size))
        all_weights['decoder_biases']['out_mean'] = tf.Variable(tf.zeros([self.input_size], dtype=tf.float32))


        #Generator/Transition net q(z|z-1,x-1,u)
        all_weights['trans_weights'] = {}
        all_weights['trans_biases'] = {}

        for layer_i in range(len(network_architecture['trans_net'])):
            if layer_i == 0:
                all_weights['trans_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(self.input_size+self.action_size+self.z_size, network_architecture['trans_net'][layer_i]))
                all_weights['trans_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['trans_net'][layer_i]], dtype=tf.float32))
            else:
                all_weights['trans_weights']['l'+str(layer_i)] = tf.Variable(xavier_init(network_architecture['trans_net'][layer_i-1], network_architecture['trans_net'][layer_i]))
                all_weights['trans_biases']['l'+str(layer_i)] = tf.Variable(tf.zeros([network_architecture['trans_net'][layer_i]], dtype=tf.float32))

        all_weights['trans_weights']['out_mean'] = tf.Variable(xavier_init(network_architecture['trans_net'][-1], self.z_size))
        all_weights['trans_weights']['out_log_var'] = tf.Variable(xavier_init(network_architecture['trans_net'][-1], self.z_size))
        all_weights['trans_biases']['out_mean'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))
        all_weights['trans_biases']['out_log_var'] = tf.Variable(tf.zeros([self.z_size], dtype=tf.float32))

        return all_weights



    def log_normal(self, z, mean, log_var):
        '''
        Log of normal distribution

        z is [B, Z]
        mean is [B, Z]
        log_var is [B, Z]
        output is [B]
        '''

        # term1 = tf.log(tf.reduce_prod(tf.exp(log_var_sq), reduction_indices=1))
        term1 = tf.reduce_sum(log_var, reduction_indices=1) #sum over dimensions n_z so now its [B]

        # [1]
        term2 = self.z_size * tf.log(2*math.pi)

        dif = tf.square(z - mean)
        dif_cov = dif / tf.exp(log_var)
        term3 = tf.reduce_sum(dif_cov, 1) #sum over dimensions so its [B]

        all_ = term1 + term2 + term3
        log_norm = -.5 * all_

        return log_norm


    def recognition_net(self, input_, network_weights):
        # input:[B,2X+A+Z]

        n_layers = len(network_weights['encoder_weights']) - 2 #minus 2 for the mean and var outputs
        weights = network_weights['encoder_weights']
        biases = network_weights['encoder_biases']

        for layer_i in range(n_layers):

            input_ = self.transfer_fct(tf.add(tf.matmul(input_, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 

        z_mean = tf.add(tf.matmul(input_, weights['out_mean']), biases['out_mean'])
        z_log_var = tf.add(tf.matmul(input_, weights['out_log_var']), biases['out_log_var'])

        return z_mean, z_log_var


    def observation_net(self, input_, network_weights):
        # input:[B,X+Z]

        n_layers = len(network_weights['decoder_weights']) - 1 #minus 1 for the mean outputs
        weights = network_weights['decoder_weights']
        biases = network_weights['decoder_biases']

        for layer_i in range(n_layers):

            input_ = self.transfer_fct(tf.add(tf.matmul(input_, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 

        x_mean = tf.add(tf.matmul(input_, weights['out_mean']), biases['out_mean'])

        return x_mean


    def transition_net(self, input_, network_weights):
        # input:[B,X+Z+A]

        n_layers = len(network_weights['trans_weights'])  - 2 #minus 2 for the mean and var outputs
        weights = network_weights['trans_weights']
        biases = network_weights['trans_biases']

        for layer_i in range(n_layers):

            input_ = self.transfer_fct(tf.add(tf.matmul(input_, weights['l'+str(layer_i)]), biases['l'+str(layer_i)])) 

        z_mean = tf.add(tf.matmul(input_, weights['out_mean']), biases['out_mean'])
        z_log_var = tf.add(tf.matmul(input_, weights['out_log_var']), biases['out_log_var'])

        return z_mean, z_log_var

            





    def Model(self, x, actions, network_weights):
        '''
        x: [B,T,X]
        actions: [B,T,A]

        q(z|z-1,x-1,u,x) for each t

        output particles: [B,T,P,Z]
        and their probs: [B,T,P]
        '''

        elbo_list = []

        prev_x = tf.zeros([self.batch_size, self.input_size])
        prev_z = tf.zeros([self.batch_size, self.n_particles, self.z_size])

        for t in range(self.n_time_steps):

            #slice current x, current action: [B,1,X] and [B,1,A]
            current_x = tf.slice(x, [0, t, 0], [self.batch_size, 1, self.input_size])
            current_a = tf.slice(actions, [0, t, 0], [self.batch_size, 1, self.action_size])
            #reshape [B,X] and [B,A]
            current_x = tf.reshape(current_x, [self.batch_size, self.input_size])
            current_a = tf.reshape(current_a, [self.batch_size, self.action_size])

            
            log_q_z_list = []
            log_p_z_list = []
            log_p_x_list = []
            particles = []
            for k in range(self.n_particles):

                #slice out one z [B,1,Z]
                z_k = tf.slice(prev_z, [0, k, 0], [self.batch_size, 1, self.z_size])
                #reshape [B,Z]
                z_k = tf.reshape(z_k, [self.batch_size, self.z_size])

                #Concatenate current x, current action, prev x, z_k: [B,2X+A+Z]
                concatenate_all = tf.concat(1, [tf.concat(1, [tf.concat(1, [current_x, prev_x]), z_k]), current_a])
                #Predict q(z|z-1,x-1,u,x): [B,Z]
                z_mean, z_log_var = self.recognition_net(concatenate_all, network_weights)

                #Concatenate current action, prev x, z_k: [B,X+A+Z]
                concatenate_all = tf.concat(1, [tf.concat(1, [current_a, prev_x]), z_k])
                #Predict p(z|z-1,x-1,u): [B,Z]
                prior_mean, prior_log_var = self.transition_net(concatenate_all, network_weights)

                #Sample from q(z|z-1,x-1,u,x)  [B,Z]
                eps = tf.random_normal((self.batch_size, self.z_size), 0, 1, dtype=tf.float32)
                this_particle = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_var)), eps))

                #Concatenate prev x, this_particle: [B,X+A]
                concatenate_all = tf.concat(1, [this_particle, prev_x])
                #Predict p(x|z,x-1): [B,X]
                x_mean = self.observation_net(concatenate_all, network_weights)


                #Compute log q(z|z-1,x-1,u,x) [B]
                log_q_z = self.log_normal(this_particle, z_mean, z_log_var)

                #Compute log p(z|z-1,x-1,u)   [B]
                log_p_z = self.log_normal(this_particle, prior_mean, prior_log_var)

                #Compute log p(x|z,x-1)   [B]
                negative_log_p_x =  \
                    tf.reduce_sum(tf.maximum(x_mean, 0) 
                                - x_mean * current_x
                                + tf.log(1 + tf.exp(-abs(x_mean))),
                                 1)

                # [B]
                log_p_x = -negative_log_p_x

                log_q_z_list.append(log_q_z)
                log_p_z_list.append(log_p_z)
                log_p_x_list.append(log_p_x)
                particles.append(this_particle)

            # [B,X]
            prev_x = current_x
            # [B,K,Z]
            prev_z = tf.pack(particles, axis=1)

            #[B,K]
            log_q_z_list = tf.pack(log_q_z_list, axis=1)
            log_p_z_list = tf.pack(log_p_z_list, axis=1)
            log_p_x_list = tf.pack(log_p_x_list, axis=1)

            #[B]
            elbo_t = tf.reduce_mean(log_p_x_list + log_p_z_list - log_q_z_list, axis=1) #over particles

            self.log_p_x = tf.reduce_mean(log_p_x_list)
            self.log_q_z = tf.reduce_mean(log_q_z_list)
            self.log_p_z = tf.reduce_mean(log_p_z_list)

            elbo_list.append(elbo_t)

        #[B,T]
        elbo_list = tf.pack(elbo_list, axis=1)

        #[B]
        elbo_over_time = tf.reduce_sum(elbo_list, axis=1) #over timesteps

        elbo = tf.reduce_mean(elbo_over_time) #over batch

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
            batch_actions = []
            while len(batch) != self.batch_size:

                sequence, actions=get_data()
                batch.append(sequence)
                batch_actions.append(actions)

            _ = self.sess.run(self.optimizer, feed_dict={self.x: batch, self.actions: batch_actions})

            # Display
            if step % display_step == 0:

                cost = self.sess.run(self.elbo, feed_dict={self.x: batch, self.actions: batch_actions})
                cost = -cost #because I want to see the NLL

                p1,p2,p3 = self.sess.run([self.log_p_x, self.log_p_z, self.log_q_z ], feed_dict={self.x: batch, self.actions: batch_actions})


                print "Step:", '%04d' % (step+1), "cost=", "{:.5f}".format(cost), p1, p2, p3


        if path_to_save_variables != '':
            saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables

        print 'Done training'


    def generate(self, x, actions):

        #give actions, show what the resulting frame is
        # ill provide the correct frame at each timestep

        prev_x = tf.zeros([self.batch_size, self.input_size])
        prev_z = tf.zeros([self.batch_size, self.n_particles, self.z_size])

        frames = []
        for t in range(self.n_time_steps):

            # #slice current x, current action: [B,1,X] and [B,1,A]
            current_x = tf.slice(x, [0, t, 0], [self.batch_size, 1, self.input_size])
            current_a = tf.slice(actions, [0, t, 0], [self.batch_size, 1, self.action_size])
            #reshape [B,X] and [B,A]
            current_x = tf.reshape(current_x, [self.batch_size, self.input_size])
            current_a = tf.reshape(current_a, [self.batch_size, self.action_size])

            x_means = []
            particles = []
            for k in range(self.n_particles):

                #slice out one z [B,1,Z]
                z_k = tf.slice(prev_z, [0, k, 0], [self.batch_size, 1, self.z_size])
                #reshape [B,Z]
                z_k = tf.reshape(z_k, [self.batch_size, self.z_size])

                #dont need the inference net

                # #Concatenate current x, current action, prev x, z_k: [B,2X+A+Z]
                # concatenate_all = tf.concat(1, [tf.concat(1, [tf.concat(1, [current_x, prev_x]), z_k]), current_a])
                # #Predict q(z|z-1,x-1,u,x): [B,Z]
                # z_mean, z_log_var = self.recognition_net(concatenate_all, network_weights)

                #Concatenate current action, prev x, z_k: [B,X+A+Z]
                concatenate_all = tf.concat(1, [tf.concat(1, [current_a, prev_x]), z_k])
                #Predict p(z|z-1,x-1,u): [B,Z]
                prior_mean, prior_log_var = self.transition_net(concatenate_all, self.network_weights)

                #Sample from p_z  [B,Z]
                eps = tf.random_normal((self.batch_size, self.z_size), 0, 1, dtype=tf.float32)
                this_particle = tf.add(prior_mean, tf.mul(tf.sqrt(tf.exp(prior_log_var)), eps))

                #Concatenate prev x, this_particle: [B,X+A]
                concatenate_all = tf.concat(1, [this_particle, prev_x])
                #Predict p(x|z,x-1): [B,X]
                x_mean = self.observation_net(concatenate_all, self.network_weights)

                x_mean_sigmoid = tf.sigmoid(x_mean)

                x_means.append(x_mean_sigmoid)

                particles.append(this_particle)


            # [B,X]
            prev_x = current_x
            # [B,K,Z]
            # prev_z = tf.reshape(this_particle, [self.batch_size, self.n_particles, self.z_size])
            prev_z = tf.pack(particles, axis=1)

            frames.append(x_means)


        return frames


    def run_generate(self, get_data, path_to_load_variables=''):

        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            self.sess.run(tf.global_variables_initializer())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables


        batch = []
        batch_actions = []
        while len(batch) != self.batch_size:

            sequence, actions=get_data()
            batch.append(sequence)
            batch_actions.append(actions)

        gen_frames = self.sess.run(self.generate, feed_dict={self.x: batch, self.actions: batch_actions})

        gen_frames = np.array(gen_frames)
        batch = np.array(batch)


        return batch, gen_frames











if __name__ == "__main__":

    steps = 40
    f_height=30
    f_width=30
    ball_size=5
    n_time_steps = 10
    n_particles = 3
    batch_size = 4
    path_to_load_variables=home+'/Documents/tmp/dkf_ball4.ckpt'
    # path_to_load_variables=''
    path_to_save_variables=home+'/Documents/tmp/dkf_ball5.ckpt'
    # path_to_save_variables=''

    train = 1
    generate = 1


    def get_ball():
        return make_ball_gif(n_frames=n_time_steps, f_height=f_height, f_width=f_width, ball_size=ball_size, max_1=True, vector_output=True)


    network_architecture = \
        dict(   encoder_net=[100,100],
                decoder_net=[100,100],
                trans_net=[100,100],
                n_input=f_height*f_width, # image as vector
                n_z=20,  # dimensionality of latent space
                n_actions=4) #4 possible actions
    
    print 'Initializing model..'
    dkf = DKF(network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=batch_size, n_time_steps=n_time_steps, n_particles=n_particles)


    if train:
        print 'Training'
        dkf.train(get_data=get_ball, steps=steps, display_step=20, path_to_load_variables=path_to_load_variables, path_to_save_variables=path_to_save_variables)


    #GENERATE - Give it a sequence of actions and make it generate the frames
    if generate:
        print 'Generating'
        
        # real: [B,T,X] gen:[T,K,B,X]
        real_frames, gen_frames = dkf.run_generate(get_data=get_ball, path_to_load_variables=path_to_save_variables)


        real_gif = []
        gen_gif = []
        for t in range(n_time_steps):

            real_frame = real_frames[0][t]
            real_frame = real_frame * (255. / np.max(real_frame))
            real_frame = real_frame.astype('uint8')
            real_gif.append(np.reshape(real_frame, [f_height,f_width, 1]))

            gen_frame = gen_frames[t][0][0]
            gen_frame = gen_frame * (255. / np.max(gen_frame))
            gen_frame = gen_frame.astype('uint8')
            gen_gif.append(np.reshape(gen_frame, [f_height,f_width, 1]))


        real_gif = np.array(real_gif)
        gen_gif = np.array(gen_gif)


        kargs = { 'duration': .6 }
        imageio.mimsave(home+"/Downloads/gen_gif.gif", gen_gif, 'GIF', **kargs)
        print 'saved gif: /Downloads/gen_gif.gif'

        kargs = { 'duration': .6 }
        imageio.mimsave(home+"/Downloads/real_gif.gif", real_gif, 'GIF', **kargs)
        print 'saved gif: /Downloads/real_gif.gif'

    donedonedone

























    generated = vae.call_generate()
    generated = np.array(generated)
    print 'generated', generated.shape #[timestep, batch, image_vector]
    # generated = np.reshape(generated, [batch_size*n_time_steps, f_height,f_width,1])
    one_sequence = []
    for t in range(len(generated)):
        one_sequence.append(generated[t][0])
    one_sequence = np.array(one_sequence)
    print one_sequence.shape
    one_sequence = np.reshape(one_sequence, [n_time_steps, f_height,f_width, 1])

    # print 'generated'
    # print one_sequence[0,:,0]

    for i in range(len(one_sequence)): 
        # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
        one_sequence[i] = one_sequence[i] * (255. / np.max(one_sequence[i]))
        one_sequence[i] = one_sequence[i].astype('uint8')

    # for i in range(len(generated)): 
    #     # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
    #     generated[i] = generated[i] * (255. / np.max(generated[i]))
    #     generated[i] = generated[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/gen_gif.gif", one_sequence, 'GIF', **kargs)



    one_sequence = []
    for t in range(len(generated)):
        one_sequence.append(generated[t][1])
    one_sequence = np.array(one_sequence)
    print one_sequence.shape
    one_sequence = np.reshape(one_sequence, [n_time_steps, f_height,f_width, 1])

    for i in range(len(one_sequence)): 
        # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
        one_sequence[i] = one_sequence[i] * (255. / np.max(one_sequence[i]))
        one_sequence[i] = one_sequence[i].astype('uint8')

    # for i in range(len(generated)): 
    #     # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
    #     generated[i] = generated[i] * (255. / np.max(generated[i]))
    #     generated[i] = generated[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/gen_gif1.gif", one_sequence, 'GIF', **kargs)



    one_sequence = []
    for t in range(len(generated)):
        one_sequence.append(generated[t][2])
    one_sequence = np.array(one_sequence)
    print one_sequence.shape
    one_sequence = np.reshape(one_sequence, [n_time_steps, f_height,f_width, 1])

    for i in range(len(one_sequence)): 
        # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
        one_sequence[i] = one_sequence[i] * (255. / np.max(one_sequence[i]))
        one_sequence[i] = one_sequence[i].astype('uint8')

    # for i in range(len(generated)): 
    #     # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
    #     generated[i] = generated[i] * (255. / np.max(generated[i]))
    #     generated[i] = generated[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/gen_gif2.gif", one_sequence, 'GIF', **kargs)

    print 'Generated 3 sequences'
    fasdfadf


    # #GENERATE 2

    # gen1, gen2, gen3, gen4, gen5 = vae.call_generate2()
    # # gen1 = gen1.eval()
    # # gen2 = gen2.eval()
    # # gen3 = gen3.eval()
    # # gen4 = gen4.eval()
    # # gen5 = gen5.eval()
    # print 'gen', gen1.shape, gen2.shape, gen3.shape, gen4.shape, gen5.shape

    # together = [gen1[0], gen2[0], gen3[0], gen4[0], gen5[0]]
    # together = np.array(together)
    # together = np.reshape(together, [5, f_height,f_width, 1])
    # for i in range(len(together)): 
    #     together[i] = together[i] * (255. / np.max(together[i]))
    #     together[i] = together[i].astype('uint8')

    # kargs = { 'duration': .6 }
    # imageio.mimsave(home+"/Downloads/generate2_gif.gif", together, 'GIF', **kargs)




    # #GENERATE 3

    # gen1, gen2, gen3, gen4, gen5 = vae.generate3()
    # gen1 = gen1.eval()
    # gen2 = gen2.eval()
    # gen3 = gen3.eval()
    # gen4 = gen4.eval()
    # gen5 = gen5.eval()
    # print 'gen', gen1.shape, gen2.shape, gen3.shape, gen4.shape, gen5.shape

    # together = [gen1[0], gen2[0], gen3[0], gen4[0], gen5[0]]
    # together = np.array(together)
    # together = np.reshape(together, [5, f_height,f_width, 1])
    # for i in range(len(together)): 
    #     together[i] = together[i] * (255. / np.max(together[i]))
    #     together[i] = together[i].astype('uint8')

    # kargs = { 'duration': .6 }
    # imageio.mimsave(home+"/Downloads/generate3_gif.gif", together, 'GIF', **kargs)




    #RECONSTRUCT 
    batch = []
    batch_actions = []
    while len(batch) != batch_size:
        sequence, actions =get_ball()
        batch.append(sequence)
        batch_actions.append(actions)

    batch = np.array(batch)
    batch_actions = np.array(batch_actions)
    print batch.shape
    #REMEMBER THE BATCH SHAPE AND RECONSTRUCTION ARENT THE SAME
    # batch is [bathc, timestep, frame]
    # recon is [timestep, batch, frame]

    #look at one batch sequence
    one_sequence = []
    for t in range(len(batch[0])):
        one_sequence.append(batch[0][t])
    one_sequence = np.array(one_sequence)
    print one_sequence.shape
    one_sequence = np.reshape(one_sequence, [n_time_steps, f_height,f_width, 1])

    for i in range(len(one_sequence)): 
        # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
        one_sequence[i] = one_sequence[i] * (255. / np.max(one_sequence[i]))
        one_sequence[i] = one_sequence[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/reconstr_input_gif.gif", one_sequence, 'GIF', **kargs)
    print 'saved', home+"/Downloads/reconstr_input_gif.gif"




    x_reconstruct = vae.call_reconstruct(batch)
    x_reconstruct = np.array(x_reconstruct)
    print x_reconstruct.shape

    one_sequence = []
    for t in range(len(x_reconstruct)):
        one_sequence.append(x_reconstruct[t][0])
    one_sequence = np.array(one_sequence)
    print one_sequence.shape
    one_sequence = np.reshape(one_sequence, [n_time_steps, f_height,f_width, 1])

    # print 'reconstructed'
    # print one_sequence[0,:,0]

    for i in range(len(one_sequence)): 
        # generated.append(np.reshape(generated[i], [f_height,f_width,1]))
        one_sequence[i] = one_sequence[i] * (255. / np.max(one_sequence[i]))
        one_sequence[i] = one_sequence[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/reconstr_gif.gif", one_sequence, 'GIF', **kargs)
    print 'saved', home+"/Downloads/reconstr_gif.gif"



    # #RECONSRUCT 2
    # batch = []
    # while len(batch) != batch_size:
    #     frame=make_ball_gif(n_frames=1, f_height=f_height, f_width=f_width, ball_size=ball_size, max_1=True, vector_output=True)
    #     batch.append(frame)

    # batch = np.array(batch)
    # batch = np.reshape(batch, [batch_size, 1 , f_height*f_width])
    # print batch.shape

    # recon1, recon2, recon3 = vae.call_reconstruct2(batch)
    # # recon1 = recon1.eval()
    # # recon2 = recon2.eval()
    # # recon3 = recon3.eval()
    # print 'recon2', recon1.shape, recon2.shape, recon3.shape

    # # x_reconstruct = np.array(x_reconstruct)
    # # print x_reconstruct.shape

    # together = [batch[0][0], recon1[0], recon2[0], recon3[0]]
    # together = np.array(together)
    # together = np.reshape(together, [4, f_height,f_width, 1])
    # for i in range(len(together)): 
    #     together[i] = together[i] * (255. / np.max(together[i]))
    #     together[i] = together[i].astype('uint8')

    # kargs = { 'duration': .6 }
    # imageio.mimsave(home+"/Downloads/reconstruct2_gif.gif", together, 'GIF', **kargs)




    # batch = np.array(batch)
    # batch = np.reshape(batch, [batch_size*n_time_steps, f_height,f_width,1])

    # for i in range(len(batch)):
    #     # batch[i] = np.reshape(batch[i], [f_height,f_width,1])
    #     batch[i] = batch[i] * (255. / np.max(batch[i]))
    #     batch[i] = batch[i].astype('uint8')
    # # kargs = { 'duration': .8 }
    # # imageio.mimsave(home+"/Downloads/inputs_gif.gif", batch, 'GIF', **kargs)

    # x_reconstruct = np.reshape(x_reconstruct, [batch_size*n_time_steps, f_height,f_width,1])

    # for i in range(len(x_reconstruct)):
    #     # reconstruct_batch.append(np.reshape(x_reconstruct[i], [f_height,f_width,1]))
    #     x_reconstruct[i] = x_reconstruct[i] * (255. / np.max(x_reconstruct[i]))
    #     x_reconstruct[i] = x_reconstruct[i].astype('uint8')
    # # kargs = { 'duration': .5 }
    # # imageio.mimsave(home+"/Downloads/inputs_rec_gif.gif", reconstruct_batch, 'GIF', **kargs)

    # combined = []
    # for i in range(0,len(batch),3):
    #     combined.append(batch[i])
    #     combined.append(batch[i+1])
    #     combined.append(batch[i+2])

    #     combined.append(x_reconstruct[i])
    #     combined.append(x_reconstruct[i+1])
    #     combined.append(x_reconstruct[i+2])

    # kargs = { 'duration': .8 }
    # imageio.mimsave(home+"/Downloads/comb_gif.gif", combined, 'GIF', **kargs)





    #PREDICT

    batch = []
    while len(batch) != batch_size:
        frame=make_ball_gif(n_frames=1, f_height=f_height, f_width=f_width, ball_size=ball_size, max_1=True, vector_output=True)
        batch.append(frame)

    batch = np.array(batch)
    batch = np.reshape(batch, [batch_size, 1 , f_height*f_width])
    print batch.shape

    # predicted = vae.call_predict_next(batch)

    # print predicted.shape

    # together = [batch[0][0], predicted[0]]
    # together = np.array(together)
    # together = np.reshape(together, [2, f_height,f_width, 1])
    # for i in range(len(together)): 
    #     together[i] = together[i] * (255. / np.max(together[i]))
    #     together[i] = together[i].astype('uint8')

    # kargs = { 'duration': .6 }
    # imageio.mimsave(home+"/Downloads/pred_gif.gif", together, 'GIF', **kargs)



    predicted2, predicted3 = vae.call_predict_next2(batch)

    print predicted.shape

    together = [batch[0][0], predicted2[0], predicted3[0]]
    together = np.array(together)
    together = np.reshape(together, [3, f_height,f_width, 1])
    for i in range(len(together)): 
        together[i] = together[i] * (255. / np.max(together[i]))
        together[i] = together[i].astype('uint8')

    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/pred2_gif.gif", together, 'GIF', **kargs)

    print 'DONE'









