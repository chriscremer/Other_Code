




import numpy as np
# import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
# import imageio
# import time
import matplotlib.pyplot as plt
from scipy.stats import norm



# class Univariate_Gaussian_SVI():

#     def __init__(self, learning_rate=0.001, n_particles=1):

#         self.learning_rate = learning_rate
#         self.n_particles = n_particles
#         self.D = 1

#         #Recogntiion model
#         self.mean = tf.Variable([-1.])
#         self.log_var = tf.Variable([1.])

#         #Sample
#         self.eps = tf.random_normal((self.n_particles, self.D), 0, 1, dtype=tf.float32)
#         self.z = tf.add(self.mean, tf.multiply(tf.sqrt(tf.exp(self.log_var)), self.eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]


#         # aaa = tf.stop_gradient(tf.identity(self.z))

#         # self.p_x = tf.placeholder("int32", shape=[self.batch_size])

#         # aaaa = self.log_p_z()
#         # self.log_p_z = tf.clip_by_value(aaaa, clip_value_min=-20, clip_value_max=20)

#         self.p_z = tf.maximum(self.p_z(self.z), 0.000000000000001) #prevent -inf 
#         # self.log_p_z = tf.stop_gradient(tf.log(self.p_z))
#         self.log_p_z = tf.log(self.p_z)

#         # self.log_p_z = tf.Variable(self.log_p_z, trainable=False)

#         self.log_q_z = self.log_q_z()
#         self.log_w = self.log_p_z - self.log_q_z

#         #Objective
#         self.elbo = tf.reduce_mean(self.log_w) #average over particles

#         # Use ADAM optimizer
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-04).minimize(-self.elbo)


#         self.grad = tf.gradients(-self.elbo, self.log_w)


#     def init_model(self):

#         sess = tf.Session()
#         sess.run(tf.initialize_all_variables())

#         return sess

#     def log_normal(self, x, mean, log_var):
#         '''
#         x is [P, B, D]
#         mean is [D]
#         log_var is [D]
#         '''

#         term1 = self.D * tf.log(2*math.pi)
#         term2 = tf.reduce_sum(log_var) #sum over dimensions, now its a scalar [1]

#         term3 = tf.square(x - mean) / tf.exp(log_var)
#         term3 = tf.reduce_sum(term3, 1) #sum over dimensions so now its [particles]

#         all_ = term1 + term2 + term3
#         log_normal = -.5 * all_  

#         return log_normal

#     # def log_p_z(self):

#     #     #Mixture of Gaussians
#     #     # log_p_z = tf.log(
#     #     #     (.3*tf.exp(self.log_normal(self.z, -2., tf.log(.5)))) 
#     #     #   + (.3*tf.exp(self.log_normal(self.z, 0., tf.log(1.)))) 
#     #     #   + (.4*tf.exp(self.log_normal(self.z, 4., tf.log(.5)))))

#     #     log_p_z = tf.log(
#     #         (.25*tf.exp(self.log_normal(self.z, -9., tf.log(.5)))) 
#     #       + (.25*tf.exp(self.log_normal(self.z, -3., tf.log(.5)))) 
#     #       + (.25*tf.exp(self.log_normal(self.z, 3., tf.log(.5))))
#     #       + (.25*tf.exp(self.log_normal(self.z, 9., tf.log(.5)))))

#     #     return log_p_z


#     def p_z(self, z):
#         #Mixture of Gaussians
#         p_z = (
#             (.33*tf.exp(self.log_normal(z, -9., tf.log(.5)))) 
#           # + (.25*tf.exp(self.log_normal(z, -3., tf.log(.5)))) 
#           + (.33*tf.exp(self.log_normal(z, 3., tf.log(.5))))
#           + (.33*tf.exp(self.log_normal(z, 9., tf.log(.5)))))

#         return p_z



#     def log_q_z(self):

#         log_q_z = self.log_normal(self.z, self.mean, self.log_var) 

#         return log_q_z




# def p_x(x):

#     # x = np.linspace(-4, 4, 100)

#     #square root scale to make var into std
#     y = (.33*norm.pdf(x, loc=-9, scale=.5**(.5)) + 
#         # (.25*norm.pdf(x, loc=-3, scale=.5**(.5))) + 
#         (.33*norm.pdf(x, loc=3, scale=.5**(.5))) +
#         (.33*norm.pdf(x, loc=9, scale=.5**(.5))) )

#     # y = np.array(y)
#     # y = y / np.sum(y)

#     return y



def q_x(x):

    y = norm.pdf(x, loc=20, scale=10**(.5)) 

    # y = np.array(y)
    # y = y / np.sum(y)

    return y


def q_x_distribution():

    x = np.linspace(-10, 10, 200)
    # y = [norm.pdf(x_i, loc=0, scale=1)+norm.pdf(x_i, loc=1, scale=.5)+norm.pdf(x_i, loc=-2, scale=.5) for x_i in x]

    y = [q_x(x_i) for x_i in x]

    # y = np.array(y)
    # y = y / np.sum(y)

    return x, y


def p_x(x):

    # x = np.linspace(-4, 4, 100)

    # #square root scale to make var into std
    # y = (.7*norm.pdf(x, loc=-2, scale=1.2**(.5)) + 
    #     # (.25*norm.pdf(x, loc=-3, scale=.5**(.5))) + 
    #     (.3*norm.pdf(x, loc=3, scale=1.2**(.5))) )



    # #square root scale to make var into std
    # y = (.3*norm.pdf(x, loc=-2, scale=1.2**(.5)) + 
    #     # (.25*norm.pdf(x, loc=-3, scale=.5**(.5))) + 
    #     (.1*norm.pdf(x, loc=3, scale=1.2**(.5))) )



    #square root scale to make var into std
    y = (.1*norm.pdf(x, loc=-2, scale=1.2**(.5)) + 
        # (.25*norm.pdf(x, loc=-3, scale=.5**(.5))) + 
        (.03*norm.pdf(x, loc=3, scale=1.2**(.5))) )










    # y = np.array(y)
    # y = y / np.sum(y)

    return y



def p_x_distribution():

    x = np.linspace(-10, 10, 200)
    # y = [norm.pdf(x_i, loc=0, scale=1)+norm.pdf(x_i, loc=1, scale=.5)+norm.pdf(x_i, loc=-2, scale=.5) for x_i in x]

    y = [p_x(x_i) for x_i in x]

    # y = np.array(y)
    # y = y / np.sum(y)

    return x, y







def qIW_x(x, A, k):

    px = p_x(x)
    qx = q_x(x)

    dem = (1./k)*((px / qx) + A)

    qiw = px / dem

    # x = np.linspace(-4, 4, 100)

    # #square root scale to make var into std
    # y = (.7*norm.pdf(x, loc=-2, scale=1.2**(.5)) + 
    #     # (.25*norm.pdf(x, loc=-3, scale=.5**(.5))) + 
    #     (.3*norm.pdf(x, loc=3, scale=1.2**(.5))) )

    # y = np.array(y)
    # y = y / np.sum(y)

    return qiw



def qIW_x_distribution(A, k):

    x = np.linspace(-50, 120, 200)
    # y = [norm.pdf(x_i, loc=0, scale=1)+norm.pdf(x_i, loc=1, scale=.5)+norm.pdf(x_i, loc=-2, scale=.5) for x_i in x]

    y = [qIW_x(x_i, A, k) for x_i in x]

    # y = np.array(y)
    # y = y / np.sum(y)

    return x, y



# def recognition_distribution(mean, var):


#     def rec(x, mean, var):

#         return norm.pdf(x, loc=mean, scale=var**(.5))


#     x = np.linspace(-12, 12, 200)

#     y = [rec(x_i,mean,var) for x_i in x]

#     return x, y




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






if __name__ == "__main__":



    # n_rows = 3
    # n_cols = 3
    # fig = plt.figure(figsize=(n_cols+6,n_rows+4), facecolor='white')


    rv = norm()

    n_samps = 1
        

    # for r in range(n_rows):
    #     for c in range(n_cols):

            

            # ax = plt.subplot2grid((n_rows,n_cols), (r,c), frameon=False)#, colspan=3)
            # ax.set_yticks([])
            # ax.set_xticks([])
            # ax.set_ylim([-.02,.5])

            # if r==0 and c==0:


            #     x, y = p_x_distribution()
                

            #     width = x[1] - x[0]
            #     int_ = width*np.sum(y)
            #     print 'integral true', int_


            #     ax.plot(x, y, linewidth=2, label='Z_p: '+("%.2f" % int_))


            #     x, y = q_x_distribution()


            #     width = x[1] - x[0]
            #     int_ = width*np.sum(y)
            #     print 'integral q', int_


            #     ax.plot(x, y, linewidth=2, label='Z_q: '+("%.2f" % int_))

            #     plt.legend(fontsize=6)


            # else:

                # x, y = p_x_distribution()
                # ax.plot(x, y, linewidth=2)

    iters = 10
    sum__ = 0
    for j in range(iters):

        if j %10 == 0:
            print j


        sum_ = 0
        for i in range(n_samps):
            x = norm.rvs(loc=20, scale=10**(.5), size=1)#, random_state=r*c+r)
            # ax.plot([x,x], [-.02,0], 'k-', lw=1)#, label='frozen pdf')

            # print p_x(x), q_x(x), p_x(x) / q_x(x)
            sum_ += p_x(x) / q_x(x)
        # print r, c, sum_



        x, y = qIW_x_distribution(A=sum_, k=1+n_samps)

        #qIW integral
        width = x[1] - x[0]
        int_ = width*np.sum(y)
        # print 'integral qIW', int_

        sum__ += int_


    print sum__ / iters


        # ax.plot(x, y, linewidth=2, label='Z_qIW: '+("%.2f" % int_))




        # ax.plot(0, 0, label='A: '+("%.2f" % (sum_/n_samps)))

        # plt.legend(fontsize=6)






    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()

    plt.show()

    fdafa

















    # for r in range()



    fig, ax = plt.subplots(1, 1)
    fig.patch.set_visible(False)

    x, y = p_x_distribution()
    ax.plot(x, y, linewidth=2, label="True Distribution")

    # plt.ylim([0,.3])
    plt.legend(fontsize=6)
    plt.show()

    fsadfasd




    alpha=.2
    rows = len(posteriors)
    columns = len(models) +1 #+1 for posteriors

    fig = plt.figure(figsize=(6+columns,4+rows), facecolor='white')

    for p_i in range(len(posteriors)):

        print '\nPosterior', p_i, posterior_names[p_i]

        posterior = ttp.posterior_class(posteriors[p_i])


        ax = plt.subplot2grid((rows,columns), (p_i,0), frameon=False)#, colspan=3)
        plot_isocontours(ax, posterior.run_log_post, cmap='Blues')
        if p_i == 0: ax.annotate('Posterior', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

        for q_i in range(len(models)):

            print model_names[q_i]
            ax = plt.subplot2grid((rows,columns), (p_i,q_i+1), frameon=False)#, colspan=3)
            model = models[q_i](posteriors[p_i])
            # model.train(10000, save_to=home+'/Documents/tmp/vars.ckpt')
            model.train(10000, save_to='')
            samps = model.sample(1000)
            plot_kde(ax, samps, cmap='Reds')
            plot_isocontours(ax, posterior.run_log_post, cmap='Blues', alpha=alpha)
            if p_i == 0: ax.annotate(model_names[q_i], xytext=(.38, 1.1), xy=(0, 1), textcoords='axes fraction')

    # plt.show()
    plt.savefig(home+'/Documents/tmp/plots.png')
    print 'saved'

    fasdfd



    n_particles = 1

    SVI = Univariate_Gaussian_SVI(n_particles=n_particles)
    sess = SVI.init_model()

    # print np.log(p_x(0))
    # print norm.pdf(0, loc=-2, scale=.5**(.5))

    # print sess.run(SVI.log_p_z(), feed_dict={SVI.z: [[0]]})

    # print sess.run(SVI.elbo)
    # sess.run(SVI.optimizer)
    # print sess.run(SVI.elbo)
    print 'training'
    for i in range(10000):
        # print 
        # m, v = sess.run([SVI.mean, SVI.log_var])
        # print m,v

        # _, elbo = sess.run([SVI.optimizer, SVI.elbo])
        _, elbo, m, v, p, q, z = sess.run([SVI.optimizer, SVI.elbo, SVI.mean, SVI.log_var, SVI.log_p_z, SVI.log_q_z, SVI.z])


        # if i ==1000:
        #     print sess.run([SVI.grad])
        #     fasdffa

        if np.isnan(elbo).any() or np.isnan(m).any() or np.isnan(v).any() or np.isnan(p).any() or np.isnan(q).any():
            print elbo
            # m, v, p, q = sess.run([SVI.mean, SVI.log_var, SVI.log_p_z, SVI.log_q_z])
            print m, v
            print p
            print q
            erorrororor

        # print elbo
        if i%500 == 0:
            # m, v, p, q = sess.run([SVI.mean, SVI.log_var, SVI.log_p_z, SVI.log_q_z])
            print i, elbo, m, v
            print 'samp', z
            print np.exp(p)
            print np.exp(q)
            print p
            print q
        # print sess.run([SVI.grad])

        # m, v = sess.run([SVI.mean, SVI.log_var])
        # print m,v



    m, v =  sess.run([SVI.mean, SVI.log_var])
    print 'mean:' + str(m) + ' log_var:' + str(v)


    print 'sampling'
    samps = []
    for i in range(1000):

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



    m, v = sess.run([SVI.mean, SVI.log_var])
    mean = m[0]
    var = np.exp(v[0])
    x, y = recognition_distribution(mean, var)
    ax.plot(x, y, linewidth=2, label="Recognition Distribution")

    ax.hist(samps, bins=200, normed=True, range=[-12,12], alpha=.6, label='Approx. Distribution, k='+str(n_particles))


    plt.ylim([0,.6])
    plt.legend(fontsize=6)
    plt.show()

    fasdf



















