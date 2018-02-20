







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








class qIW_class_1D(object):

    def __init__(self, p, q, sum_, k):

        self.p = p
        self.q = q
        self.sum_ = sum_
        self.k = k


    def run_log_post(self, x):

        px = np.exp(self.p.run_log_post(x))
        px = np.maximum(px, np.exp(np.ones((len(x)))*-30.))

        qx = np.exp(self.q.run_log_post(x))
        qx = np.maximum(qx, np.exp(np.ones((len(x)))*-30.))
        p_q = px / qx
        dem = (1./self.k) * (p_q + self.sum_)
        aaa = px / dem

        aaa = np.maximum(aaa, np.exp(np.ones((len(x)))*-30.))

        return np.log(aaa)




























if __name__ == "__main__":

    # Define 2 distribution p and q
    # Define function to get integral of distribution
    # Sample q, get qIW , take integral
    # Average the integrals


    p_mean = [-2.5]
    p_logvar = [1.]

    q_mean = [4.5]
    q_logvar = [.5]

    p_x = Gaus_class_1D(p_mean, p_logvar)
    q_x = Gaus_class_1D(q_mean, q_logvar)


    viz = 0
    integral = 0
    viz_qIWs = 0
    viz_all = 0
    integral_all = 0
    viz_integrate_all = 1
    viz_qEIW = 0



    if viz:

        viz_range = [-10,10]
        numticks = 50

        fig, ax = plt.subplots(1, 1)
        fig.patch.set_visible(False)

        x, y = return_1d_distribution(distribution=p_x, xlimits=viz_range, numticks=numticks)
        ax.plot(x, y, linewidth=2, label="P(x)")

        x, y = return_1d_distribution(distribution=q_x, xlimits=viz_range, numticks=numticks)
        ax.plot(x, y, linewidth=2, label="Q(x)")


        # plt.ylim([0,.3])
        plt.legend(fontsize=6)
        plt.show()




    if integral:

        integral_range = [-30,30]
        numticks = 100

        x, y = return_1d_distribution(distribution=p_x, xlimits=integral_range, numticks=numticks)
        width = x[1] - x[0]
        int_ = width*np.sum(y)

        print 'Integral p(x):', int_

        x, y = return_1d_distribution(distribution=q_x, xlimits=integral_range, numticks=numticks)
        width = x[1] - x[0]
        int_ = width*np.sum(y)

        print 'Integral q(x):', int_





    if viz_qIWs:

        k = 2
        n_dists = 3

        viz_range = [-10,10]
        numticks = 50


        fig, ax = plt.subplots(1, 1)
        fig.patch.set_visible(False)

        for j in range(n_dists):

            if k>1:
                z2_zk = [q_x.run_sample_post() for i in range(k-1)]

            sum_ = np.sum([np.exp(p_x.run_log_post([z]))/np.exp(q_x.run_log_post([z])) for z in z2_zk])

            print sum_

            qIW = qIW_class_1D(p_x, q_x, sum_, k)

            x, y = return_1d_distribution(distribution=qIW, xlimits=viz_range, numticks=numticks)
            ax.plot(x, y, linewidth=2, label="qIW_"+str(j))


        plt.legend(fontsize=6)
        plt.show()







    if viz_all:

        k = 2
        n_dists = 3

        viz_range = [-10,10]
        numticks = 200


        fig, ax = plt.subplots(1, 1)
        fig.patch.set_visible(False)


        x, y = return_1d_distribution(distribution=p_x, xlimits=viz_range, numticks=numticks)
        ax.plot(x, y, linewidth=2, label="P(x)")

        x, y = return_1d_distribution(distribution=q_x, xlimits=viz_range, numticks=numticks)
        ax.plot(x, y, linewidth=2, label="Q(x)")


        qIWs = []
        for j in range(n_dists):

            if k>1:
                z2_zk = [q_x.run_sample_post() for i in range(k-1)]

            sum_ = np.sum([np.exp(p_x.run_log_post([z]))/np.exp(q_x.run_log_post([z])) for z in z2_zk])

            print sum_

            qIW = qIW_class_1D(p_x, q_x, sum_, k)

            x, y = return_1d_distribution(distribution=qIW, xlimits=viz_range, numticks=numticks)
            ax.plot(x, y, linewidth=2, label="qIW_"+str(j))

            qIWs.append(qIW)


        plt.legend(fontsize=6)
        plt.show()






    if integral_all:

        k = 2
        n_dists = 100000


        integral_range = [-30,30]
        numticks = 100

        x, y = return_1d_distribution(distribution=p_x, xlimits=integral_range, numticks=numticks)
        width = x[1] - x[0]
        int_ = width*np.sum(y)

        print 'Integral p(x):', int_

        x, y = return_1d_distribution(distribution=q_x, xlimits=integral_range, numticks=numticks)
        width = x[1] - x[0]
        int_ = width*np.sum(y)

        print 'Integral q(x):', int_

        qIW_ints = []
        sums = []
        for j in range(n_dists):

            if k>1:
                z2_zk = [q_x.run_sample_post() for i in range(k-1)]

            sum_ = np.sum([np.exp(p_x.run_log_post([z]))/np.exp(q_x.run_log_post([z])) for z in z2_zk])
            qIW = qIW_class_1D(p_x, q_x, sum_, k)

            x, y = return_1d_distribution(distribution=qIW, xlimits=integral_range, numticks=numticks)
            width = x[1] - x[0]
            int_ = width*np.sum(y)

            # print 'Integral qIW_' +str(j) +':', int_

            qIW_ints.append(int_)
            sums.append(sum_)


            if j % 10 ==0:

                print 'avg qIW', np.mean(qIW_ints), 'avg sum', np.mean(sums), j



        print 'avg qIW', np.mean(qIW_ints)
        print 'avg sum', np.mean(sums)









    if viz_integrate_all:

        k = 2
        n_dists = 3

        viz_range = [-10,10]
        numticks = 200

        integral_range = [-30,30]
        integrate_numticks = 100

        # fig = plt.figure(facecolor='white')
        fig, ax = plt.subplots(1, 1, facecolor='white')
        # fig.patch.set_visible(False)
        ax.axis('off')
        ax.set_yticks([])
        ax.set_xticks([])

        x, y = return_1d_distribution(distribution=p_x, xlimits=viz_range, numticks=numticks)
        ax.plot(x, y, linewidth=2, label=r'$p(z)$')

        x, y = return_1d_distribution(distribution=q_x, xlimits=viz_range, numticks=numticks)
        ax.plot(x, y, linewidth=2, label=r'$q(z)$')


        qIWs = []
        for j in range(n_dists):

            if k>1:
                z2_zk = [q_x.run_sample_post() for i in range(k-1)]

            sum_ = np.sum([np.exp(p_x.run_log_post([z]))/np.exp(q_x.run_log_post([z])) for z in z2_zk])

            print sum_

            qIW = qIW_class_1D(p_x, q_x, sum_, k)

            x, y = return_1d_distribution(distribution=qIW, xlimits=viz_range, numticks=numticks)
            ax.plot(x, y, linewidth=2, label=r'$\tilde{q}_{IW}(z|x,z_2)$')

            qIWs.append(qIW)


        x, y = return_1d_distribution(distribution=p_x, xlimits=integral_range, numticks=numticks)
        width = x[1] - x[0]
        int_ = width*np.sum(y)

        print 'Integral p(x):', int_

        x, y = return_1d_distribution(distribution=q_x, xlimits=integral_range, numticks=numticks)
        width = x[1] - x[0]
        int_ = width*np.sum(y)

        print 'Integral q(x):', int_

        for j in range(len(qIWs)):

            x, y = return_1d_distribution(distribution=qIWs[j], xlimits=integral_range, numticks=numticks)
            width = x[1] - x[0]
            int_ = width*np.sum(y)

            print 'Integral qIW_' +str(j) +':', int_



        #Expected q_IW
        n_dists = 30
        qIWs = []
        ys = []
        for j in range(n_dists):

            if k>1:
                z2_zk = [q_x.run_sample_post() for i in range(k-1)]

            sum_ = np.sum([np.exp(p_x.run_log_post([z]))/np.exp(q_x.run_log_post([z])) for z in z2_zk])

            qIW = qIW_class_1D(p_x, q_x, sum_, k)

            x, y = return_1d_distribution(distribution=qIW, xlimits=viz_range, numticks=numticks)
            # ax.plot(x, y, linewidth=2, label="qIW_"+str(j))

            qIWs.append(qIW)
            ys.append(y)

        ax.plot(x, np.mean(ys, axis=0), linewidth=2, label=r'$q_{EW}(z|x)$')
        width = x[1] - x[0]
        int_ = width*np.sum(np.mean(ys, axis=0))
        print 'Integral qIW' +':', int_



        plt.legend(fontsize=9, loc=2)
        plt.show()








    if viz_qEIW:

        k = 2
        n_dists = 30

        viz_range = [-10,10]
        numticks = 50


        fig, ax = plt.subplots(1, 1)
        fig.patch.set_visible(False)


        x, y = return_1d_distribution(distribution=p_x, xlimits=viz_range, numticks=numticks)
        ax.plot(x, y, linewidth=2, label="P(x)")

        x, y = return_1d_distribution(distribution=q_x, xlimits=viz_range, numticks=numticks)
        ax.plot(x, y, linewidth=2, label="Q(x)")


        qIWs = []
        ys = []
        for j in range(n_dists):

            if k>1:
                z2_zk = [q_x.run_sample_post() for i in range(k-1)]

            sum_ = np.sum([np.exp(p_x.run_log_post([z]))/np.exp(q_x.run_log_post([z])) for z in z2_zk])

            print sum_

            qIW = qIW_class_1D(p_x, q_x, sum_, k)

            x, y = return_1d_distribution(distribution=qIW, xlimits=viz_range, numticks=numticks)
            # ax.plot(x, y, linewidth=2, label="qIW_"+str(j))

            qIWs.append(qIW)
            ys.append(y)

        ax.plot(x, np.mean(ys, axis=0), linewidth=2, label="qEIW")

        plt.legend(fontsize=6)
        plt.show()










