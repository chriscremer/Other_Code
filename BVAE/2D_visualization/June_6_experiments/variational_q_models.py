



import tensorflow as tf
slim=tf.contrib.slim
import math
import numpy as np



def log_normal(x, mean, log_var):
    '''
    x is [P, D]
    mean is [D]
    log_var is [D]
    return [P]
    '''
    term1 = 2 * tf.log(2*math.pi)
    term2 = tf.reduce_sum(log_var) #sum over dimensions, [1]
    term3 = tf.square(x - mean) / tf.exp(log_var)
    term3 = tf.reduce_sum(term3, 1) #sum over dimensions, [P]
    all_ = term1 + term2 + term3
    log_normal = -.5 * all_  
    return log_normal





class Factorized_Gaussian_model(object):

    def __init__(self, log_posterior):

        tf.reset_default_graph()
    
        mean = tf.Variable([0.,0.])
        logvar = tf.Variable([.1,.1])

        #Sample
        eps = tf.random_normal((1,2), 0, 1, dtype=tf.float32) 
        self.z = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(logvar)), eps)) 

        log_q_z = log_normal(self.z, mean, logvar) 
        log_p_z = log_posterior(self.z)

        self.elbo = log_p_z - log_q_z

        self.optimizer = tf.train.AdamOptimizer(learning_rate=.001, 
                                                epsilon=1e-02).minimize(-self.elbo)

        init_vars = tf.global_variables_initializer()
        tf.get_default_graph().finalize()

        self.sess = tf.Session()
        self.sess.run(init_vars)



    def train(self, iters):

        best_100_elbo = -1
        worse_count = 0
        elbos = []
        for i in range(iters):

            # stop if the last 100 hasnt improved
            if i % 200 == 0 and i != 0:
                elbo_100 = np.mean(elbos)
                if elbo_100 < best_100_elbo and best_100_elbo != -1:
                    worse_count +=1
                    print i, elbo_100,worse_count
                    if worse_count == 3:
                        print 'done training'
                        break
                else:
                    best_100_elbo = elbo_100
                    worse_count = 0
                    elbos = []
                    print i, elbo_100

            _, elbo = self.sess.run((self.optimizer, self.elbo))
            elbos.append(elbo)



    def sample(self, n_samples):

        samps = []
        for i in range(n_samples):
            samps.append(self.sess.run(self.z))

        samps = np.array(samps)
        samps = np.reshape(samps, [n_samples, 2])

        return samps






class IW_model(Factorized_Gaussian_model):

    def __init__(self, log_posterior, n_samples=3):

        tf.reset_default_graph()
    
        mean = tf.Variable([0.,0.])
        logvar = tf.Variable([.1,.1])

        #Sample
        eps = tf.random_normal((n_samples,2), 0, 1, dtype=tf.float32) 
        self.z = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(logvar)), eps)) 

        log_q_z = log_normal(self.z, mean, logvar)  #[P]
        log_p_z = log_posterior(self.z)  #[P]

        self.log_w = log_p_z - log_q_z
        max_ = tf.reduce_max(self.log_w)
        self.elbo = tf.log(tf.reduce_mean(tf.exp(self.log_w - max_))) +max_

        self.optimizer = tf.train.AdamOptimizer(learning_rate=.001, 
                                                epsilon=1e-02).minimize(-self.elbo)

        init_vars = tf.global_variables_initializer()
        tf.get_default_graph().finalize()

        self.sess = tf.Session()
        self.sess.run(init_vars)



    def sample(self, n_samples):

        samps = []
        for i in range(n_samples):
            z, log_w = self.sess.run((self.z, self.log_w))

            max_ = np.max(log_w)
            aaa = np.exp(log_w - max_)
            normalized_ws = aaa / np.sum(aaa)

            if sum(normalized_ws) > 1.:
                normalized_ws = normalized_ws - .000001

            sampled = np.random.multinomial(1, normalized_ws)#, size=1)
            sampled = z[np.argmax(sampled)]

            samps.append(sampled)

        samps = np.array(samps)
        samps = np.reshape(samps, [n_samples, 2])

        return samps








class AV_model(Factorized_Gaussian_model):

    def __init__(self, log_posterior):

        tf.reset_default_graph()

        v_size = 10
        z_size = 2
    
        qv_mean = tf.Variable(tf.zeros([v_size]))
        qv_logvar = tf.Variable(tf.ones([v_size])-3.)

        #Sample v
        eps = tf.random_normal((1,v_size), 0, 1, dtype=tf.float32) 
        v = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(logvar)), eps)) 

        net = slim.stack(v,slim.fully_connected,[30])
        net = slim.fully_connected(net,z_size*2,activation_fn=None) #[1,4]

        print net


        fasf


        #Sample v
        eps = tf.random_normal((1,2), 0, 1, dtype=tf.float32) 
        self.z = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(logvar)), eps)) 

        log_q_z = log_normal(self.z, mean, logvar) 
        log_p_z = log_posterior(self.z)

        self.elbo = log_p_z - log_q_z

        self.optimizer = tf.train.AdamOptimizer(learning_rate=.001, 
                                                epsilon=1e-02).minimize(-self.elbo)

        init_vars = tf.global_variables_initializer()
        tf.get_default_graph().finalize()

        self.sess = tf.Session()
        self.sess.run(init_vars)







