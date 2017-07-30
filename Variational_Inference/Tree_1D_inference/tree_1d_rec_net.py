







#this isnt working


import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
# import imageio
# import time
import matplotlib.pyplot as plt
from scipy.stats import norm


RelaxedOneHotCategorical = tf.contrib.distributions.RelaxedOneHotCategorical


from NN2 import NN


def log_normal(x, mean, log_var):
    '''
    x is [P, D]
    mean is [D]
    log_var is [D]
    '''
    D = tf.cast(tf.shape(mean)[0], tf.float32)
    term1 = D * tf.log(2*math.pi)
    term2 = tf.reduce_sum(log_var) #sum over dimensions, [1]
    term3 = tf.square(x - mean) / tf.exp(log_var)
    term3 = tf.reduce_sum(term3, 1) #sum over dimensions, [P]
    all_ = term1 + term2 + term3
    log_normal = -.5 * all_  
    return log_normal



def log_normal_new(x, mean, log_var):
    '''
    x is [B, D]
    mean is [K, D]
    log_var is [K, D]
    '''
    D = tf.cast(tf.shape(mean)[1], tf.float32)
    D_int = tf.shape(mean)[1]
    B = tf.shape(x)[0]
    K = tf.shape(mean)[0]

    term1 = D * tf.log(2*math.pi) #[1]
    term2 = tf.reduce_sum(log_var, axis=1) #sum over D, [K]
    term2 = tf.reshape(term2, [1,K])  #[1,K]


    x = tf.reshape(x, [B, 1, D_int])  #[B,1,D]
    mean = tf.reshape(mean, [1, K, D_int])
    mean = tf.tile(mean, [B, 1, 1])  #[B,K,D]
    log_var = tf.reshape(log_var, [1, K, D_int])
    log_var = tf.tile(log_var, [B, 1, 1])  #[B,K,D]


    term3 = tf.square(x - mean) / tf.exp(log_var)   #[B,K,D]
    term3 = tf.reduce_sum(term3, axis=2) #sum over D, [B,K]
    all_ = term1 + term2 + term3  #[B,K]
    log_normal = -.5 * all_  
    return log_normal




def sample_Gaussian(mean, logvar, rs=1):
    '''
    mean, logvar: [Z]
    outpit: [Z]
    '''

    shape = tf.shape(mean)
    # B = shape[0]
    Z = shape[0]

    eps = tf.random_normal([Z], 0, 1, seed=rs) 
    z = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(logvar)), eps)) # [P,B,Z]

    return z



class Gaus_class_1D(object):

    def __init__(self, mean, logvar):

        tf.reset_default_graph()

        self.mean = mean

        self.logvar = logvar


        self.x = tf.placeholder(tf.float32, [1,None])
        self.logp = log_normal(tf.transpose(self.x), self.mean, self.logvar)

        self.sampled_x = sample_Gaussian(self.mean, self.logvar)


        tf.get_default_graph().finalize()

        self.sess = tf.Session()


    def run_log_post(self, x):

        return self.sess.run(self.logp, feed_dict={self.x: x})

    def run_sample_post(self):

        return self.sess.run(self.sampled_x)






def return_1d_distribution(distribution, xlimits, numticks):

    x = np.linspace(*xlimits, num=numticks)

    px = np.exp(distribution.run_log_post([x]))

    return x, px








# class qIW_class_1D(object):

#     def __init__(self, p, q, sum_, k):

#         self.p = p
#         self.q = q
#         self.sum_ = sum_
#         self.k = k


#     def run_log_post(self, x):

#         px = np.exp(self.p.run_log_post(x))
#         px = np.maximum(px, np.exp(np.ones((len(x)))*-30.))

#         qx = np.exp(self.q.run_log_post(x))
#         qx = np.maximum(qx, np.exp(np.ones((len(x)))*-30.))
#         p_q = px / qx
#         dem = (1./self.k) * (p_q + self.sum_)
#         aaa = px / dem

#         aaa = np.maximum(aaa, np.exp(np.ones((len(x)))*-30.))

#         return np.log(aaa)




class Recognition_MoG():

    def __init__(self, n_clusters):

        tf.reset_default_graph()
        self.rs = 0
        self.input_size = 1

        #Model hyperparameters
        self.learning_rate = .001
        self.n_clusters = n_clusters

        #Placeholders - Inputs/Targets [B,X]
        self.one_over_N = tf.placeholder(tf.float32, None)
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.tau = tf.placeholder(tf.float32, None)


        #q(z|x)
        f_z_given_x = NN([self.input_size, self.n_clusters], [None])
        self.logits = f_z_given_x.feedforward(self.x)  #[B,K]
        q_z = RelaxedOneHotCategorical(self.tau, logits=self.logits) 
        self.z = q_z.sample()  #[B,K] 
        self.log_qz = q_z.log_prob(self.z) #[B]

        #q(T)
        q_tree_logits = tf.Variable(tf.random_normal([self.n_clusters,self.n_clusters], stddev=0.35)) 
        # q_tree = RelaxedOneHotCategorical(self.tau, logits=q_tree_logits) 
        # self.tree = q_tree.sample()
        # self.log_qT = tf.reduce_sum(q_tree.log_prob(self.tree))
        self.tree = tf.nn.softmax(q_tree_logits)
        self.log_qT = tf.zeros([1])

        # q(pi)
        q_pi_pre_softmax = tf.Variable(tf.random_normal([self.n_clusters], stddev=0.35))  #[K]
        cluster_mixture_weight = tf.nn.softmax(q_pi_pre_softmax)  #[K]
        log_qp =  tf.zeros([1])  #[1]
        self.cluster_mixture_weight = tf.reshape(cluster_mixture_weight, [1,self.n_clusters])

        # q(means)
        q_mu = tf.Variable(tf.random_normal([self.n_clusters, self.input_size], stddev=0.35))+25  #[K,X]
        self.cluster_means = q_mu   #[K,X]
        self.log_qm = tf.zeros([self.n_clusters])  #[K]

        # q(variance)
        q_logvar = tf.Variable(tf.random_normal([self.n_clusters, self.input_size], stddev=0.35))  #[K,X]
        self.cluster_logvars = q_logvar   #[K,X]
        log_qv = tf.zeros([self.n_clusters])  #[K]


        #Compute prob of joint

        #p(x|z,mu,sigma)  get prob of x under each gaussian, then sum weighted by z
        log_px = log_normal_new(self.x, self.cluster_means, self.cluster_logvars) #[B,K]
        log_px = tf.log(tf.maximum(tf.reduce_sum(tf.exp(log_px) * self.z, axis=1), tf.exp(-30.)))  #[B]


        #log_p(z|pi)
        # self.log_pz = 0.  #[B]
        self.log_pz = tf.log(tf.maximum(tf.reduce_sum(self.cluster_mixture_weight * tf.stop_gradient(self.z), axis=1), tf.exp(-30.)))  #[B]


        #log p(mu|T)

        self.log_pm = tf.zeros([self.n_clusters])  #[K]

        # log_pm = log_normal(self.cluster_logvars, tf.ones([self.input_size])*20., tf.ones([self.input_size])*10.)
        
        # c_m = tf.transpose(self.cluster_means) #[X,K]
        # c_m = tf.matmul(c_m, self.tree) #[X,K]
        # c_m = c_m - tf.transpose(self.cluster_means) #[X,K]
        # prob_mean = 1./(1.+tf.exp(-2.*c_m))
        # self.log_pm = tf.reduce_sum(tf.log(tf.maximum(prob_mean, tf.exp(-30.))), axis=0) #[K]


        #log p(sigma)
        log_pv = tf.zeros([self.n_clusters])  #[K]
        # log_pv = log_normal(self.cluster_logvars, tf.ones([self.input_size])*3., tf.ones([self.input_size])*-.5)


        #log p(pi)
        log_pp = tf.zeros([1])  #[1]
        # log_pp =  10* tf.log(1. / tf.reduce_max(cluster_mixture_weight))  #[1]

        #log p(T)
        log_pT = tf.zeros([1]) 


        # elbo = log_px + log_pz + log_pm + log_pv + log_pp - log_qz - log_qm - log_qv - log_qp

        self.elbo = (tf.reduce_mean(log_px + self.log_pz - self.log_qz, axis=0) #over batch 
                + self.one_over_N*( tf.reduce_sum(self.log_pm - self.log_qm + log_pv - log_qv, axis=0) + log_pp - log_qp +log_pT - self.log_qT))

        # Minimize negative ELBO
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,epsilon=1e-02).minimize(-self.elbo)


        #To init variables
        self.init_vars = tf.global_variables_initializer()

        #For loadind/saving variables
        self.saver = tf.train.Saver()

        #For debugging 
        # self.vars = tf.trainable_variables()
        # self.grads = tf.gradients(self.elbo, tf.trainable_variables())
        self.grads = tf.gradients(self.elbo, self.cluster_means)


        #to make sure im not adding nodes to the graph
        tf.get_default_graph().finalize()

        #Start session
        self.sess = tf.Session()




    def train(self, train_x, valid_x=[], display_step=5, 
                    path_to_load_variables='', path_to_save_variables='', 
                    epochs=10, batch_size=20, tau=1.):
        '''
        Train.
        '''
        random_seed=1
        rs=np.random.RandomState(random_seed)
        n_datapoints = len(train_x)
        one_over_N = 1./float(n_datapoints)
        arr = np.arange(n_datapoints)

        if path_to_load_variables == '':
            self.sess.run(self.init_vars)

        else:
            #Load variables
            self.saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables

        #start = time.time()
        for epoch in range(1,epochs+1):

            #shuffle the data
            rs.shuffle(arr)
            train_x = train_x[arr]

            data_index = 0
            for step in range(n_datapoints/batch_size):

                #Make batch
                batch = []
                batch_y = []
                while len(batch) != batch_size:
                    batch.append(train_x[data_index]) 
                    data_index +=1

                # Fit training using batch data
                _ = self.sess.run((self.optimizer), feed_dict={self.x: batch,
                                                        self.one_over_N: one_over_N,
                                                        self.tau: tau})

                # Display logs per epoch step
                if step % display_step == 0:



                    # cost = self.sess.run((self.elbo), feed_dict={self.x: batch, 
                    #                                     self.one_over_N: one_over_N})

                    # print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost))




                    # cost, mw = self.sess.run((self.elbo, self.cluster_mixture_weight), feed_dict={self.x: batch, 
                    #                                     self.one_over_N: one_over_N, self.tau: tau})

                    # print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost)), mw



                    cost, mw, qT, qm, pm = self.sess.run((self.elbo, self.cluster_mixture_weight, self.log_qT, self.log_qm, self.log_pm), feed_dict={self.x: batch, 
                                                        self.one_over_N: one_over_N, self.tau: tau})

                    print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost)), mw, pm






                    # batch=[[x] for x in range(-20,50)]

                    # cost,z = self.sess.run((self.elbo, self.z), feed_dict={self.x: batch,
                    #                                     self.one_over_N: one_over_N, self.tau: tau})

                    # print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost))
                    
                    # for i in range(len(batch)):
                    #     print batch[i]
                    #     print z[i]
                    # # print z

                    # print 
                    # fdad




                    # cost,z = self.sess.run((self.elbo, self.z), feed_dict={self.x: batch,
                    #                                     self.one_over_N: one_over_N})

                    # print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost))
                    # print z

                    # print 




                    # cost,z = self.sess.run((self.elbo, self.z), feed_dict={self.x: batch, self.logits: [[[2.,0.,0.,0.,0.]]],
                    #                                     self.one_over_N: one_over_N})

                    # print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost))
                    # print z

                    # print 






                    # cost, z, g = self.sess.run((self.elbo, self.z, self.grads), feed_dict={self.x: batch, 
                    #                                     self.one_over_N: one_over_N})




                    # pz, qz = self.sess.run((self.log_pz, self.log_qz), feed_dict={self.x: [[-9.],[-10.],[-.11]], 
                    #                                     self.one_over_N: one_over_N})
                    # print pz
                    # print qz

                    # fasdfasd


                    # cost, z, g, l, pz, qz = self.sess.run((self.elbo, self.z, self.grads, self.logits, self.log_pz, self.log_qz), feed_dict={self.x: [[-9.],[-10.],[-.11]], 
                    #                                     self.one_over_N: one_over_N})

                    # print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost))
                    # print z
                    # print g
                    # print l
                    # print pz
                    # print qz



                    # fas
                    #,logpy,logpW,logqW #, 'time', time.time() - start


        if path_to_save_variables != '':
            self.saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables





    def load_vars(self, path_to_load_variables):

        #Load variables
        self.saver.restore(self.sess, path_to_load_variables)
        print 'loaded variables ' + path_to_load_variables




    def get_vars(self):

        a,b,c,d = self.sess.run((self.cluster_mixture_weight, self.cluster_means, self.cluster_logvars, self.tree))

        print a.shape
        print b.shape
        print c.shape

        return a, b, c, d


    def get_cluster_assigments(self, batch):

        tau= .001
        z = self.sess.run((self.z), feed_dict={self.x: batch, self.tau: tau})

        return z







if __name__ == "__main__":

    mixture_weights = [.4, .3, .2, .1]


    means = [[50.], [40.], [20.], [10.]]
    logvars = [[3.], [3.5], [2.], [3.]]

    gaussians =[]
    for i in range(len(mixture_weights)):
        gaussians.append(Gaus_class_1D(means[i], logvars[i]))




    make_dataset = 1
    train = 1
    see_model = 1
    viz = 0




    path_to_load_variables=home+'/Documents/tmp/vars_mog.ckpt' 
    # path_to_load_variables=''
    path_to_save_variables=home+'/Documents/tmp/vars_mog.ckpt'



    n_clusters = 5
    batch_size = 50
    epochs = 2000
    display_step =1000
    tau = 1.


    model = Recognition_MoG(n_clusters)




    if make_dataset:


        N = 1000

        # sample cluster index
        counts = np.random.multinomial(N, mixture_weights)
        print counts

        # sample from gaussian
        samps = []
        for i in range(len(mixture_weights)):

            for j in range(counts[i]):

                samps.append(gaussians[i].run_sample_post())

        samps = np.array(samps)
        # print samps.shape
        # samps = np.reshape(samps, [N])


        # plt.hist(samps, 100, normed=1, alpha=.6)

        # plt.show()








    if train:

        model.train(train_x=samps,
                    epochs=epochs, batch_size=batch_size, display_step=display_step,
                    path_to_load_variables=path_to_load_variables,
                    path_to_save_variables=path_to_save_variables,
                    tau=tau)








    if see_model:

        viz_range = [-30,70]
        numticks = 200

        fig, ax = plt.subplots(1, 1)
        fig.patch.set_visible(False)


        # REAL DISTRIBUTION
        for i in range(len(mixture_weights)):

            x, y_i = return_1d_distribution(distribution=gaussians[i], xlimits=viz_range, numticks=numticks)

            if i ==0:
                y = y_i*mixture_weights[i]
            else:
                y += y_i*mixture_weights[i]

        ax.plot(x, y, linewidth=2, label="P(x)")




        model.load_vars(path_to_load_variables=path_to_save_variables)
        q_mixture, q_means, q_logvars, tree = model.get_vars()  #[1,K], [K,D], [K,D]

        print q_mixture
        print q_means
        print q_logvars
        print tree



        for i in range(len(q_mixture[0])):

            aa = Gaus_class_1D(q_means[i], q_logvars[i])

            x, y_i = return_1d_distribution(distribution=aa, xlimits=viz_range, numticks=numticks)

            if i ==0:
                y = y_i*q_mixture[0][i]
            else:
                y += y_i*q_mixture[0][i]

        ax.plot(x, y, linewidth=2, label="Q(x)")


        #histograms of assignments
        z = model.get_cluster_assigments(samps)
        for i in range(n_clusters):
            cluster_samps = []
            for j in range(len(z)):
                if np.argmax(z[j]) == i:
                    cluster_samps.append(samps[j][0])

            cluster_samps = np.array(cluster_samps)
            # print cluster_samps.shape
            # fdsaf

            weights = np.ones_like(cluster_samps)/float(len(cluster_samps))*q_mixture[0][i] #/ (np.max(cluster_samps)-np.min(cluster_samps))
            # weights = np.ones_like(cluster_samps)/np.sum(*q_mixture[0][i]

            # plt.hist(cluster_samps, 40, normed=1, weights=weights, alpha=.6)
            plt.hist(cluster_samps, bins=np.array(range(-20,140))/2.,  alpha=.6, weights=weights*2., label=str(i))



        plt.legend(fontsize=6)
        plt.show()
















    if viz:

        viz_range = [-1,65]
        numticks = 50

        fig, ax = plt.subplots(1, 1)
        fig.patch.set_visible(False)


        for i in range(len(mixture_weights)):

            x, y_i = return_1d_distribution(distribution=gaussians[i], xlimits=viz_range, numticks=numticks)

            if i ==0:
                y = y_i*mixture_weights[i]
            else:
                y += y_i*mixture_weights[i]

        # x, y2 = return_1d_distribution(distribution=p2, xlimits=viz_range, numticks=numticks)

        # y = y1*mixture_weights[0] + y2*mixture_weights[1]
        # y = y /np.sum(y)


        ax.plot(x, y, linewidth=2, label="P(x)")


        # plt.ylim([0,.3])
        plt.legend(fontsize=6)
        plt.show()






    print 'Done'


















