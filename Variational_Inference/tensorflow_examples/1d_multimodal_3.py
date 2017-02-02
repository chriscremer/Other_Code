






import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import imageio
import time
import matplotlib.pyplot as plt
from scipy.stats import norm



class Univariate_Gaussian_SVI():

    def __init__(self, learning_rate=0.001, n_particles=1):

        self.learning_rate = learning_rate
        self.n_particles = n_particles
        self.D = 1

        #Recogntiion model
        # self.mean = tf.Variable([-1.])
        # self.log_var = tf.Variable([1.])

        self.rnn_layer1 = tf.Variable(tf.truncated_normal([6, 6],stddev=0.1))
        self.rnn_layer1_biases = tf.Variable(tf.truncated_normal([6], stddev=0.1))

        self.rnn_layer2 = tf.Variable(tf.truncated_normal([6, self.D],stddev=0.1))
        self.rnn_layer2_biases = tf.Variable(tf.truncated_normal([self.D], stddev=0.1))

        self.rnn_layer2_2 = tf.Variable(tf.truncated_normal([6, self.D],stddev=0.1))
        self.rnn_layer2_biases_2 = tf.Variable(tf.truncated_normal([self.D], stddev=0.1))

        self.rnn_layer2_3 = tf.Variable(tf.truncated_normal([6, 5],stddev=0.1))
        self.rnn_layer2_biases_3 = tf.Variable(tf.truncated_normal([5], stddev=0.1))

        #Sample
        # self.eps = tf.random_normal((self.n_particles, self.D), 0, 1, dtype=tf.float32)
        # self.z = tf.add(self.mean, tf.mul(tf.sqrt(tf.exp(self.log_var)), self.eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]
        self.z, self.log_q_z, self.means, self.log_vars = self.RNN()

        self.log_p_z = self.log_p_z()
        # self.log_p_z = tf.clip_by_value(aaaa, clip_value_min=-8, clip_value_max=8)

        # self.log_q_z  = self.log_q_z()

        # self.log_q_z = tf.clip_by_value(bbbb, clip_value_min=-8, clip_value_max=8)

        self.log_w = self.log_p_z - self.log_q_z

        #SVI Objective
        self.elbo = tf.reduce_mean(self.log_w) #average over particles

        #W-SVI Objective
        # max_ = tf.reduce_max(self.log_w) #max over particles
        # min_ = tf.reduce_min(self.log_w) #max over particles

        # #IWAE Code
        # log_ws_minus_max = self.log_w - min_
        # ws = tf.exp(log_ws_minus_max)
        # ws_normalized = ws / tf.reduce_sum(ws)
        # self.elbo = tf.reduce_sum(ws_normalized * self.log_w) #average over particles


        #THIS KINDA WORKS
        # max_ = tf.reduce_max(self.log_w) #max over particles
        # self.elbo = tf.log(tf.reduce_mean(tf.exp(self.log_w - max_))) + max_ #average over particles


        # self.elbo = tf.log(tf.reduce_mean(tf.exp(self.log_w))) #average over particles


        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-04).minimize(-self.elbo)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=.8).minimize(-self.elbo)

        # self.opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=.8)
        # grads_and_vars = self.opt.compute_gradients(-self.elbo)
        # capped_grads_and_vars = [(tf.clip_by_value(gv[0], -1, 1), gv[1]) for gv in grads_and_vars]
        # self.optimizer = self.opt.apply_gradients(capped_grads_and_vars)


    def init_model(self):

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        return sess

    def log_normal(self, x, mean, log_var):
        '''
        x is [P, B, D]
        mean is [D]
        log_var is [D]
        '''

        term1 = self.D * tf.log(2*math.pi)
        term2 = tf.reduce_sum(log_var) #sum over dimensions, now its a scalar [1]

        term3 = tf.square(x - mean) / tf.exp(log_var)
        term3 = tf.reduce_sum(term3, 1) #sum over dimensions so now its [particles]

        all_ = term1 + term2 + term3
        log_normal = -.5 * all_  

        return log_normal

    def log_p_z(self):

        #Mixture of Gaussians
        log_p_z = tf.log(
            (.3*tf.exp(self.log_normal(self.z, -2., tf.log(.5)))) 
          + (.3*tf.exp(self.log_normal(self.z, 0., tf.log(1.)))) 
          + (.4*tf.exp(self.log_normal(self.z, 4., tf.log(.5)))))

        return log_p_z

    def log_q_z(self):

        log_q_z = self.log_normal(self.z, self.mean, self.log_var) 

        return log_q_z



    def RNN(self):


        state = tf.zeros([1,5])
        prev_z = tf.zeros([1,self.D])

        samples = []
        log_q_zs = []
        means = []
        log_vars = []
        for i in range(self.n_particles):

            input_ = tf.reshape(tf.concat(1, [state, prev_z]), [1,6])

            layer_1 = tf.tanh(tf.add(tf.matmul(input_, self.rnn_layer1), self.rnn_layer1_biases)) 

            state = tf.add(tf.matmul(layer_1, self.rnn_layer2_3), self.rnn_layer2_biases_3)

            z_mean = tf.add(tf.matmul(layer_1, self.rnn_layer2), self.rnn_layer2_biases)

            z_log_var = tf.add(tf.matmul(layer_1, self.rnn_layer2_2), self.rnn_layer2_biases_2)

            z = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_var)), tf.random_normal([self.D], 0, 1, dtype=tf.float32))) 

            log_q_z = self.log_normal(z, z_mean, z_log_var)

            samples.append(z)
            log_q_zs.append(log_q_z)
            means.append(z_mean)
            log_vars.append(z_log_var)
            prev_z = z



        samples = tf.pack(samples)
        log_q_zs = tf.pack(log_q_zs)
        means = tf.pack(means)
        log_vars = tf.pack(log_vars)

        return samples, log_q_zs, means, log_vars




    # def importance_weight(self):


    #     log_w = self.log_p_z - self.log_q_z

    #     return log_w



    # def iwae_sampling(self, k=1):

    #     return stuff



def p_x(x):

    # x = np.linspace(-4, 4, 100)

    #square root scale to make var
    y = (.3*norm.pdf(x, loc=-2, scale=.5**(.5)) + (.3*norm.pdf(x, loc=0, scale=1**(.5))) + (.4*norm.pdf(x, loc=4, scale=.5**(.5))))

    # y = np.array(y)
    # y = y / np.sum(y)

    return y



def p_x_distribution():

    x = np.linspace(-7, 7, 100)
    # y = [norm.pdf(x_i, loc=0, scale=1)+norm.pdf(x_i, loc=1, scale=.5)+norm.pdf(x_i, loc=-2, scale=.5) for x_i in x]

    y = [p_x(x_i) for x_i in x]

    # y = np.array(y)
    # y = y / np.sum(y)

    return x, y


def recognition_distribution(mean, var):


    def rec(x, mean, var):

        return norm.pdf(x, loc=mean, scale=var**(.5))


    x = np.linspace(-7, 7, 100)

    y = [rec(x_i,mean,var) for x_i in x]

    return x, y




# def p_x_distribution():

#     x = np.linspace(-6, 6, 100)
#     # y = [norm.pdf(x_i, loc=0, scale=1)+norm.pdf(x_i, loc=1, scale=.5)+norm.pdf(x_i, loc=-2, scale=.5) for x_i in x]

#     y = [sess.run(tf.exp(SVI.log_p_z()), feed_dict={SVI.z: [[x_i]]})[0] for x_i in x]

#     # y = np.array(y)
#     # y = y / np.sum(y)

#     return x, y




def iwae_sampling(SVI, sess, k):

    #Sample from the recognition model, k times, calculate their weights, sample one from a categorical dist
    samples, log_ws = sess.run([SVI.z, SVI.log_w])
    samples = np.reshape(samples, [k,1])
    log_ws = np.reshape(log_ws, [k])

    # print log_ws.shape
    # print samples.shape

    # print samples.shape
    # print log_ws.shape
    # samples = [x[0] for x in samples]
    # log_ws = [x[0] for x in log_ws]
    # max_ = np.max(log_ws)
    min_ = np.min(log_ws)

    aaa = np.exp(log_ws - min_)
    normalized_ws = aaa / np.sum(aaa)



    if sum(normalized_ws) > 1.:
        # print normalized_ws
        # normalized_ws[-1] = normalized_ws[-1] - (sum(normalized_ws)-1)
        # normalized_ws = normalized_ws / np.sum(normalized_ws)
        # normalized_ws[-1] = 0
        # print normalized_ws
        normalized_ws = normalized_ws - .0001


        # print sum(normalized_ws)
        # print normalized_ws
        # print log_ws

        # fsadasd

    # print log_ws
    # print normalized_ws
    normalized_ws = np.reshape(normalized_ws, [k])
    # print normalized_ws.shape, 'sahpe'

    sampled = np.random.multinomial(k, normalized_ws)

    sampled = np.reshape(sampled, [k,1])

    # weighted_sample = np.argmax(weighted_sample)
    # print sampled.shape

    weighted_samples = []
    for i in range(len(sampled)):
        for j in range(sampled[i]):
            weighted_samples.append(samples[i])

    return weighted_samples






if __name__ == "__main__":

    n_particles = 10

    SVI = Univariate_Gaussian_SVI(n_particles=n_particles)
    sess = SVI.init_model()

    # print np.log(p_x(0))
    # print norm.pdf(0, loc=-2, scale=.5**(.5))

    # print sess.run(SVI.log_p_z(), feed_dict={SVI.z: [[0]]})

    # print sess.run(SVI.elbo)
    # sess.run(SVI.optimizer)
    # print sess.run(SVI.elbo)

    for i in range(20000):
    # for i in range(20):


        # _, elbo = sess.run([SVI.optimizer, SVI.elbo])

        _, elbo, log_w, log_p_z, log_q_z = sess.run([SVI.optimizer, SVI.elbo, SVI.log_w, SVI.log_p_z, SVI.log_q_z])



        # print elbo
        if i%500 == 0:
            # m, v = sess.run([SVI.mean, SVI.log_var])
            print i, elbo #m, v
            # print log_p_z
            # print log_q_z
            # print log_w


        # elif i>6000:
        #     m, v = sess.run([SVI.mean, SVI.log_var])
        #     print i, elbo, m, v, log_w

    # print sess.run([SVI.mean, SVI.log_var])



    means, log_vars = sess.run([SVI.means, SVI.log_vars])

    print means
    print log_vars
    fasdfa

    print 'sampling'
    samps = []
    for i in range(10000):

        # samples = sess.run(SVI.z)
        samples = iwae_sampling(SVI, sess, k=n_particles)

        samples = [x[0] for x in samples]

        for j in samples:
            samps.append(j)

        # samps.append(sess.run(SVI.z)[0][0])


    fig, ax = plt.subplots(1, 1)

    x, y = p_x_distribution()
    ax.plot(x, y, linewidth=2, label="True Distribution")

    # m, v = sess.run([SVI.mean, SVI.log_var])
    # mean = m[0]
    # var = np.exp(v[0])
    # x, y = recognition_distribution(mean, var)
    # ax.plot(x, y, linewidth=2, label="Recognition Distribution")

    ax.hist(samps, bins=100, normed=True, range=[-7,7], alpha=.6, label='Approx. Distribution, k='+str(n_particles))

    plt.legend(fontsize=6)
    plt.show()
    print 'DOne'



















