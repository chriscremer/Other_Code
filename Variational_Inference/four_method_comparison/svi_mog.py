






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



class Univariate_Gaussian_SVI_MoG():

    def __init__(self, learning_rate=0.001, n_particles=1):

        self.learning_rate = learning_rate
        self.n_particles = n_particles
        self.D = 1

        #Recogntiion model
        # self.mean = tf.Variable([-1.])
        # self.log_var = tf.Variable([1.])

        self.mean1 = tf.Variable([-5.])
        self.log_var1 = tf.Variable([1.])
        self.pi1_unnormal = tf.Variable([.5])

        self.mean2 = tf.Variable([1.])
        self.log_var2 = tf.Variable([1.])
        self.pi2_unnormal = tf.Variable([.5])

        self.pi1 = tf.exp(self.pi1_unnormal) / (tf.exp(self.pi1_unnormal) + tf.exp(self.pi2_unnormal))
        self.pi2 = tf.exp(self.pi2_unnormal) / (tf.exp(self.pi1_unnormal) + tf.exp(self.pi2_unnormal))

        self.log_pi1 = tf.log(self.pi1)
        self.log_pi2 = tf.log(self.pi2)

        #Sample
        # self.eps = tf.random_normal((self.n_particles, self.D), 0, 1, dtype=tf.float32)
        # self.z = tf.add(self.mean, tf.mul(tf.sqrt(tf.exp(self.log_var)), self.eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]

        # self.p_z = tf.maximum(self.p_z(self.z), 0.000000000000001) #prevent -inf 
        # self.log_p_z = tf.log(self.p_z)

        # self.log_q_z = self.log_q_z()
        # self.log_w = self.log_p_z - self.log_q_z

        self.eps1 = tf.random_normal((self.n_particles, self.D), 0, 1, dtype=tf.float32)
        self.eps2 = tf.random_normal((self.n_particles, self.D), 0, 1, dtype=tf.float32)

        self.z1 = tf.add(self.mean1, tf.mul(tf.sqrt(tf.exp(self.log_var1)), self.eps1)) 
        self.z2 = tf.add(self.mean2, tf.mul(tf.sqrt(tf.exp(self.log_var2)), self.eps2))
        self.z_all = tf.concat(0, [self.z1, self.z2])

        # self.log_p_z = self.log_p_z(self.z_all)
        # self.log_p_z1 = self.log_p_z(self.z1)
        # self.log_p_z2 = self.log_p_z(self.z2)

        # self.log_q_z1 = self.log_q_z(self.z1, self.mean1, self.log_var1) + self.log_pi1 
        # self.log_q_z2 = self.log_q_z(self.z2, self.mean2, self.log_var2) + self.log_pi2

        # component_log_densitites = tf.concat(1, [self.log_q_z1, self.log_q_z2])


        component_log_densitites_1 = tf.transpose(tf.pack([self.log_q_z(self.z1, self.mean1, self.log_var1), self.log_q_z(self.z1, self.mean2, self.log_var2)]))
        component_log_densitites_2 = tf.transpose(tf.pack([self.log_q_z(self.z2, self.mean1, self.log_var1), self.log_q_z(self.z2, self.mean2, self.log_var2)]))


        # component_log_densitites_1 = tf.transpose(self.log_q_z1)
        # component_log_densitites_2 = tf.transpose(self.log_q_z2)

        # component_log_densitites_1 = self.log_q_z1
        # component_log_densitites_2 = self.log_q_z2



        log_weights = tf.reshape(tf.pack([self.log_pi1, self.log_pi2]), [2])

        # print component_log_densitites
        # print log_weights

        #[n_samps]
        log_qs_1 = tf.log(tf.reduce_sum(tf.exp(component_log_densitites_1 + log_weights), reduction_indices=1, keep_dims=False))
        log_qs_2 = tf.log(tf.reduce_sum(tf.exp(component_log_densitites_2 + log_weights), reduction_indices=1, keep_dims=False))

        #[n_samps]
        log_ps_1 = tf.reshape(self.log_p_z(self.z1), [n_particles, 1])
        log_ps_2 = tf.reshape(self.log_p_z(self.z2), [n_particles, 1])

        log_qs = tf.concat(0, [log_qs_1, log_qs_2])

        log_ps = tf.reshape(tf.concat(0, [log_ps_1, log_ps_2]), [2*n_particles])


        self.w_all = tf.exp(log_ps - log_qs)



        # print aaa

        component_elbos_1 = tf.reduce_mean(log_ps_1 - log_qs_1) #over particles
        component_elbos_2 = tf.reduce_mean(log_ps_2 - log_qs_2)

        # [n_clusters, 1]
        component_elbos = tf.reshape(tf.pack([component_elbos_1,component_elbos_2]), [2,1])
        # print component_elbos

        self.elbo = tf.reduce_sum(component_elbos + log_weights) #over components

        # print elbo
        # dfas

        # # self.log_q_z = tf.concat(0, [self.log_q_z1,self.log_q_z2])

        # self.log_w1 = tf.reduce_mean(self.log_p_z1 - self.log_q_z1) #over samples
        # self.log_w2 = tf.reduce_mean(self.log_p_z2 - self.log_q_z2)
        # # self.w_all = tf.concat(0, [self.log_w1, self.log_w2])


        # # elbo = tf.reduce_sum(component_elbos + log_weights)
        # self.elbo = self.log_w1 + self.log_w2 + self.log_pi1 + self.log_pi2



        #Objective
        # self.elbo = tf.reduce_mean(self.log_w) #average over particles

        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-04).minimize(-self.elbo)


        # self.grad = tf.gradients(-self.elbo, self.log_w)


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



    def log_p_z(self, z):

        #Mixture of Gaussians
        # log_p_z = tf.log(
        #     (.3*tf.exp(self.log_normal(self.z, -2., tf.log(.5)))) 
        #   + (.3*tf.exp(self.log_normal(self.z, 0., tf.log(1.)))) 
        #   + (.4*tf.exp(self.log_normal(self.z, 4., tf.log(.5)))))

        log_p_z = tf.log(
            (.33*tf.exp(self.log_normal(z, -9., tf.log(.5)))) 
          # + (.25*tf.exp(self.log_normal(self.z, -3., tf.log(.5)))) 
          + (.33*tf.exp(self.log_normal(z, 3., tf.log(.5))))
          + (.33*tf.exp(self.log_normal(z, 9., tf.log(.5)))))

        return log_p_z


    def p_z(self, z):
        #Mixture of Gaussians
        p_z = (
            (.33*tf.exp(self.log_normal(z, -9., tf.log(.5)))) 
          # + (.25*tf.exp(self.log_normal(z, -3., tf.log(.5)))) 
          + (.33*tf.exp(self.log_normal(z, 3., tf.log(.5))))
          + (.33*tf.exp(self.log_normal(z, 9., tf.log(.5)))))

        return p_z



    # def log_q_z(self):

    #     log_q_z = self.log_normal(self.z, self.mean, self.log_var) 

    #     return log_q_z

    def log_q_z(self, z, mean, log_var):

        log_q_z = self.log_normal(z, mean, log_var) 

        return log_q_z



def p_x(x):

    # x = np.linspace(-4, 4, 100)

    #square root scale to make var into std
    y = (.33*norm.pdf(x, loc=-9, scale=.5**(.5)) + 
        # (.25*norm.pdf(x, loc=-3, scale=.5**(.5))) + 
        (.33*norm.pdf(x, loc=3, scale=.5**(.5))) +
        (.33*norm.pdf(x, loc=9, scale=.5**(.5))) )

    # y = np.array(y)
    # y = y / np.sum(y)

    return y



def p_x_distribution():

    x = np.linspace(-12, 12, 200)
    # y = [norm.pdf(x_i, loc=0, scale=1)+norm.pdf(x_i, loc=1, scale=.5)+norm.pdf(x_i, loc=-2, scale=.5) for x_i in x]

    y = [p_x(x_i) for x_i in x]

    # y = np.array(y)
    # y = y / np.sum(y)

    return x, y


def recognition_distribution(mean, var):


    def rec(x, mean, var):

        return norm.pdf(x, loc=mean, scale=var**(.5))


    x = np.linspace(-12, 12, 200)

    y = [rec(x_i,mean,var) for x_i in x]

    return x, y




# def p_x_distribution():

#     x = np.linspace(-6, 6, 100)
#     # y = [norm.pdf(x_i, loc=0, scale=1)+norm.pdf(x_i, loc=1, scale=.5)+norm.pdf(x_i, loc=-2, scale=.5) for x_i in x]

#     y = [sess.run(tf.exp(SVI.log_p_z()), feed_dict={SVI.z: [[x_i]]})[0] for x_i in x]

#     # y = np.array(y)
#     # y = y / np.sum(y)

#     return x, y




# def iwae_sampling(SVI, sess, k):

#     #Sample from the recognition model, k times, calculate their weights, sample one from a categorical dist
#     samples, log_ws = sess.run([SVI.z, SVI.log_w])
#     # print samples.shape
#     # print log_ws.shape
#     # samples = [x[0] for x in samples]
#     # log_ws = [x[0] for x in log_ws]
#     max_ = np.max(log_ws)
#     aaa = np.exp(log_ws - max_)
#     normalized_ws = aaa / np.sum(aaa)

#     if sum(normalized_ws) > 1.:
#         normalized_ws = normalized_ws - .000001

#         # normalized_ws[-1] = normalized_ws[-1] - (sum(normalized_ws)-1)
#         # print sum(normalized_ws)
#         # print normalized_ws
#         # print log_ws

#         # fsadasd

#     # print log_ws
#     # print normalized_ws
#     sampled = np.random.multinomial(k, normalized_ws)#, size=1)

#     # weighted_sample = np.argmax(weighted_sample)
#     # print sampled.shape

#     weighted_samples = []
#     for i in range(len(sampled)):
#         for j in range(sampled[i]):
#             weighted_samples.append(samples[i])

#     return weighted_samples



def iwae_sampling(SVI, sess, k):

    #Sample from the recognition model, k times, calculate their weights, sample one from a categorical dist
    samples, ws = sess.run([SVI.z_all, SVI.w_all])
    # print samples.shape
    # print log_ws.shape
    # samples = [x[0] for x in samples]
    # log_ws = [x[0] for x in log_ws]
    # max_ = np.max(log_ws)
    # min_ = np.min(log_ws)

    # print samples.shape
    # print ws.shape

    # ws = np.reshape(ws, [k])



    # aaa = np.exp(log_ws - min_)
    normalized_ws = ws / np.sum(ws)

    if sum(normalized_ws) > 1.:
        # print normalized_ws
        # normalized_ws[-1] = normalized_ws[-1] - (sum(normalized_ws)-1)
        # normalized_ws = normalized_ws / np.sum(normalized_ws)
        # normalized_ws[-1] = 0
        normalized_ws = normalized_ws - .000001


        # print sum(normalized_ws)
        # print normalized_ws
        # print log_ws

        # fsadasd

    # print log_ws
    # print normalized_ws
    sampled = np.random.multinomial(k, normalized_ws)#, size=1)

    # weighted_sample = np.argmax(weighted_sample)
    # print sampled.shape

    weighted_samples = []
    for i in range(len(sampled)):
        for j in range(sampled[i]):
            weighted_samples.append(samples[i])

    return weighted_samples






if __name__ == "__main__":

    n_particles = 3

    SVI = Univariate_Gaussian_SVI_MoG(n_particles=n_particles)
    sess = SVI.init_model()

    # print np.log(p_x(0))
    # print norm.pdf(0, loc=-2, scale=.5**(.5))

    # print sess.run(SVI.log_p_z(), feed_dict={SVI.z: [[0]]})

    # print sess.run(SVI.elbo)
    # sess.run(SVI.optimizer)
    # print sess.run(SVI.elbo)
    print 'training'
    for i in range(12000):
        # print 
        # m, v = sess.run([SVI.mean, SVI.log_var])
        # print m,v

        _, elbo = sess.run([SVI.optimizer, SVI.elbo])
        # _, elbo, m, v, p, q, z = sess.run([SVI.optimizer, SVI.elbo, SVI.mean, SVI.log_var, SVI.log_p_z, SVI.log_q_z, SVI.z])


        # if i ==1000:
        #     print sess.run([SVI.grad])
        #     fasdffa

        # if np.isnan(elbo).any() or np.isnan(m).any() or np.isnan(v).any() or np.isnan(p).any() or np.isnan(q).any():
        #     print elbo
        #     # m, v, p, q = sess.run([SVI.mean, SVI.log_var, SVI.log_p_z, SVI.log_q_z])
        #     print m, v
        #     print p
        #     print q
        #     erorrororor

        # print elbo
        if i%500 == 0:
            # m, v = sess.run([SVI.mean, SVI.log_var])
            print i, elbo#, m, v
            # print log_p_z
            # print log_q_z
            # print log_w
            print sess.run([SVI.mean1, SVI.log_var1, SVI.pi1])
            print sess.run([SVI.mean2, SVI.log_var2, SVI.pi2])

            if np.isnan(elbo).any():
                erorrororor




    # m, v =  sess.run([SVI.mean, SVI.log_var])
    # print 'mean:' + str(m) + ' log_var:' + str(v)


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
    fig.patch.set_visible(False)

    x, y = p_x_distribution()
    ax.plot(x, y, linewidth=2, label="True Distribution")

    m, v = sess.run([SVI.mean1, SVI.log_var1])
    mean = m[0]
    var = np.exp(v[0])
    x, y = recognition_distribution(mean, var)
    ax.plot(x, y, linewidth=2, label="Recognition Distribution 1")



    ax.hist(samps, bins=200, normed=True, range=[-12,12], alpha=.6, label='Approx. Distribution, k='+str(n_particles))

    m, v = sess.run([SVI.mean2, SVI.log_var2])
    mean = m[0]
    var = np.exp(v[0])
    x, y = recognition_distribution(mean, var)
    ax.plot(x, y, linewidth=2, label="Recognition Distribution 2")

    plt.ylim([0,.6])
    plt.legend(fontsize=6)
    plt.show()
    fasdf



















