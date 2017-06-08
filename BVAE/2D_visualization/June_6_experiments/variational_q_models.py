



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
        self.saver = tf.train.Saver()

        tf.get_default_graph().finalize()

        self.sess = tf.Session()
        self.sess.run(init_vars)



    def train(self, iters, save_to=''):

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
                    if worse_count == 10:
                        print 'done training'
                        break
                else:
                    best_100_elbo = elbo_100
                    worse_count = 0
                    elbos = []
                    print i, elbo_100

                    if save_to != '':
                        self.saver.save(self.sess, save_to)
                        print 'Saved variables to ' + save_to

            _, elbo = self.sess.run((self.optimizer, self.elbo))
            elbos.append(elbo)

        if save_to != '':
            self.saver.restore(self.sess, save_to)
            print 'loaded variables ' + save_to



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
        self.saver = tf.train.Saver()
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
    
        #q(v)
        qv_mean = tf.Variable(tf.zeros([v_size]))
        qv_logvar = tf.Variable(tf.ones([v_size])-3.)

        #Sample v
        eps = tf.random_normal((1,v_size), 0, 1, dtype=tf.float32) 
        v = tf.add(qv_mean, tf.multiply(tf.sqrt(tf.exp(qv_logvar)), eps)) 

        #q(z|v)
        net = slim.stack(v,slim.fully_connected,[30,30])
        net = slim.fully_connected(net,z_size*2,activation_fn=None) #[1,4]
        qz_mean = tf.slice(net, [0,0], [1,2])
        qz_logvar = tf.slice(net, [0,1], [1,2])

        #Sample z
        eps = tf.random_normal((1,z_size), 0, 1, dtype=tf.float32) 
        self.z = tf.add(qz_mean, tf.multiply(tf.sqrt(tf.exp(qz_logvar)), eps)) 

        #r(v|z)
        net = slim.stack(self.z,slim.fully_connected,[30,30])
        net = slim.fully_connected(net,v_size*2,activation_fn=None) #[1,20]
        rv_mean = tf.slice(net, [0,0], [1,v_size])
        rv_logvar = tf.slice(net, [0,1], [1,v_size])        

        #logprobs
        log_q_v = log_normal(v, qv_mean, qv_logvar) 
        log_q_z = log_normal(self.z, qz_mean, qz_logvar) 
        log_r_v = log_normal(v, rv_mean, rv_logvar) 
        log_p_z = log_posterior(self.z)

        self.elbo = log_p_z + log_r_v - log_q_z - log_q_v

        self.optimizer = tf.train.AdamOptimizer(learning_rate=.001, 
                                                epsilon=1e-02).minimize(-self.elbo)

        init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()

        self.sess = tf.Session()
        self.sess.run(init_vars)




class Norm_Flow_model(Factorized_Gaussian_model):

    def __init__(self, log_posterior, n_flows=3):

        tf.reset_default_graph()

        z_size = 2
    
        qz0_mean = tf.Variable(tf.zeros([z_size]))
        qz0_logvar = tf.Variable(tf.ones([z_size])-3.)

        #Sample
        eps = tf.random_normal((1,z_size), 0, 1, dtype=tf.float32) 
        z0 = tf.add(qz0_mean, tf.multiply(tf.sqrt(tf.exp(qz0_logvar)), eps)) 

        z = z0
        logdetsum = 0.
        for i in range(n_flows):

            z, logdet = self.norm_flow(z)
            logdetsum += logdet

        self.z = z

        log_q_z0 = log_normal(z0, qz0_mean, qz0_logvar) 
        log_p_zT = log_posterior(self.z)

        self.elbo = log_p_zT - log_q_z0 + logdetsum

        self.optimizer = tf.train.AdamOptimizer(learning_rate=.001, 
                                                epsilon=1e-02).minimize(-self.elbo)

        init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()

        self.sess = tf.Session()
        self.sess.run(init_vars)



    def random_bernoulli(self, shape, p=0.5):
        if isinstance(shape, (list, tuple)):
            shape = tf.stack(shape)
        return tf.where(tf.random_uniform(shape) < p, tf.ones(shape), tf.zeros(shape))


    def norm_flow(self, z):
        '''
        z: [1,Z]
        '''

        z_size = 2

        #Flows z0 -> zT
        mask = self.random_bernoulli(tf.shape(z), p=0.5)
        h = slim.stack(mask*z,slim.fully_connected,[30])
        mew_ = slim.fully_connected(h,z_size,activation_fn=None) 
        sig_ = slim.fully_connected(h,z_size,activation_fn=tf.nn.sigmoid) 

        z = (mask * z) + (1-mask)*(z*sig_ + (1-sig_)*mew_)

        logdet = tf.reduce_sum((1-mask)*tf.log(sig_), axis=1)

        return z, logdet






class Hamiltonian_Variational_model(Factorized_Gaussian_model):

    def __init__(self, log_posterior, n_flows=3):

        tf.reset_default_graph()

        z_size = 2
        v_size = 2
        step_size = tf.Variable([.1])

    
        #q(v)
        qv0_mean = tf.Variable(tf.zeros([v_size]))
        qv0_logvar = tf.Variable(tf.ones([v_size])-3.)

        #Sample v
        eps = tf.random_normal((1,v_size), 0, 1, dtype=tf.float32) 
        v0 = tf.add(qv0_mean, tf.multiply(tf.sqrt(tf.exp(qv0_logvar)), eps)) 

        #q(z|v)
        net = slim.stack(v0,slim.fully_connected,[30,30])
        net = slim.fully_connected(net,z_size*2,activation_fn=None) #[1,4]
        qz0_mean = tf.slice(net, [0,0], [1,2])
        qz0_logvar = tf.slice(net, [0,1], [1,2])

        #Sample z
        eps = tf.random_normal((1,z_size), 0, 1, dtype=tf.float32) 
        z0 = tf.add(qz0_mean, tf.multiply(tf.sqrt(tf.exp(qz0_logvar)), eps)) 

        z = z0
        v = v0
        for i in range(n_flows):

            z, v = self.leapfrog_step2(z, v, log_posterior, step_size)

        self.z = z

        #r(vT|zT)
        net = slim.stack(self.z,slim.fully_connected,[30,30])
        net = slim.fully_connected(net,v_size*2,activation_fn=None)
        rvT_mean = tf.slice(net, [0,0], [1,v_size])
        rvT_logvar = tf.slice(net, [0,1], [1,v_size])   

        #logprobs
        log_q_v0 = log_normal(v0, qv0_mean, qv0_logvar) 
        log_q_z0 = log_normal(z0, qz0_mean, qz0_logvar) 
        log_r_vT = log_normal(v, rvT_mean, rvT_logvar) 
        log_p_zT = log_posterior(self.z)

        self.elbo = log_p_zT + log_r_vT - log_q_z0 - log_q_v0

        self.optimizer = tf.train.AdamOptimizer(learning_rate=.001, 
                                                epsilon=1e-02).minimize(-self.elbo)

        init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()

        self.sess = tf.Session()
        self.sess.run(init_vars)


    def leapfrog_step(self, z, v, log_posterior, step_size):
        '''
        z: [1,2]
        v: [1,2]
        '''

        grad = tf.gradients(-log_posterior(z), [z])[0]
        v = v - ((.5*step_size) * grad)

        z = z + (step_size * v)

        grad = tf.gradients(-log_posterior(z), [z])[0]
        v = v - ((.5*step_size) * grad)

        # v = v * friction_for_LF

        return z, v


    def leapfrog_step2(self, z, v, log_posterior, step_size):
        '''
        z: [1,2]
        v: [1,2]
        '''

        z = z + (.5*step_size * v)

        grad = tf.gradients(-log_posterior(z), [z])[0]
        v = v - (step_size * grad)

        z = z + (.5*step_size * v)

        # v = v * friction_for_LF

        return z, v








class Auxiliary_Flow_model(Factorized_Gaussian_model):

    def __init__(self, log_posterior, n_flows=3):

        tf.reset_default_graph()

        z_size = 2
        v_size = 2

    
        #q(v)
        qv0_mean = tf.Variable(tf.zeros([v_size]))
        qv0_logvar = tf.Variable(tf.ones([v_size])-3.)

        #Sample v
        eps = tf.random_normal((1,v_size), 0, 1, dtype=tf.float32) 
        v0 = tf.add(qv0_mean, tf.multiply(tf.sqrt(tf.exp(qv0_logvar)), eps)) 

        #q(z|v)
        net = slim.stack(v0,slim.fully_connected,[30,30])
        net = slim.fully_connected(net,z_size*2,activation_fn=None) #[1,4]
        qz0_mean = tf.slice(net, [0,0], [1,2])
        qz0_logvar = tf.slice(net, [0,1], [1,2])

        #Sample z
        eps = tf.random_normal((1,z_size), 0, 1, dtype=tf.float32) 
        z0 = tf.add(qz0_mean, tf.multiply(tf.sqrt(tf.exp(qz0_logvar)), eps)) 

        z = z0
        v = v0
        logdetsum = 0.
        for i in range(n_flows):

            z, v, logdet = self.flow_step(z, v)
            logdetsum += logdet

        self.z = z

        #r(vT|zT)
        net = slim.stack(self.z,slim.fully_connected,[30,30])
        net = slim.fully_connected(net,v_size*2,activation_fn=None)
        rvT_mean = tf.slice(net, [0,0], [1,v_size])
        rvT_logvar = tf.slice(net, [0,1], [1,v_size])   

        #logprobs
        log_q_v0 = log_normal(v0, qv0_mean, qv0_logvar) 
        log_q_z0 = log_normal(z0, qz0_mean, qz0_logvar) 
        log_r_vT = log_normal(v, rvT_mean, rvT_logvar) 
        log_p_zT = log_posterior(self.z)

        self.elbo = log_p_zT + log_r_vT - log_q_z0 - log_q_v0 + logdetsum

        self.optimizer = tf.train.AdamOptimizer(learning_rate=.001, 
                                                epsilon=1e-02).minimize(-self.elbo)

        init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()

        self.sess = tf.Session()
        self.sess.run(init_vars)





    def flow_step(self, z, v):
        '''
        z: [1,2]
        v: [1,2]
        '''

        z_size = 2
        v_size = 2

        h = slim.stack(z,slim.fully_connected,[30])
        mew_ = slim.fully_connected(h,v_size,activation_fn=None) 
        sig_ = slim.fully_connected(h,v_size,activation_fn=tf.nn.sigmoid) 

        v = v*sig_ + mew_

        h2 = slim.stack(v,slim.fully_connected,[30])
        mew_2 = slim.fully_connected(h2,z_size,activation_fn=None) 
        sig_2 = slim.fully_connected(h2,z_size,activation_fn=tf.nn.sigmoid) 

        z = z*sig_2 + mew_2

        logdet = tf.reduce_sum(tf.log(sig_)) + tf.reduce_sum(tf.log(sig_2))

        return z, v, logdet








class Hamiltonian_Flow_model(Factorized_Gaussian_model):

    def __init__(self, log_posterior, n_flows=3):

        tf.reset_default_graph()

        z_size = 2
        v_size = 2

        # step_size = tf.Variable([.1])
        step_size = 1

    
        #q(v)
        qv0_mean = tf.Variable(tf.zeros([v_size]))
        qv0_logvar = tf.Variable(tf.ones([v_size])-3.)

        #Sample v
        eps = tf.random_normal((1,v_size), 0, 1, dtype=tf.float32) 
        v0 = tf.add(qv0_mean, tf.multiply(tf.sqrt(tf.exp(qv0_logvar)), eps)) 

        #q(z|v)
        net = slim.stack(v0,slim.fully_connected,[30,30])
        net = slim.fully_connected(net,z_size*2,activation_fn=None) #[1,4]
        qz0_mean = tf.slice(net, [0,0], [1,2])
        qz0_logvar = tf.slice(net, [0,1], [1,2])

        #Sample z
        eps = tf.random_normal((1,z_size), 0, 1, dtype=tf.float32) 
        z0 = tf.add(qz0_mean, tf.multiply(tf.sqrt(tf.exp(qz0_logvar)), eps)) 

        z = z0
        v = v0
        logdetsum = 0.
        for i in range(n_flows):

            z, v, logdet = self.flow_step(z, v, log_posterior, step_size)
            logdetsum += logdet

        self.z = z

        #r(vT|zT)
        net = slim.stack(self.z,slim.fully_connected,[30,30])
        net = slim.fully_connected(net,v_size*2,activation_fn=None)
        rvT_mean = tf.slice(net, [0,0], [1,v_size])
        rvT_logvar = tf.slice(net, [0,1], [1,v_size])   

        #logprobs
        log_q_v0 = log_normal(v0, qv0_mean, qv0_logvar) 
        log_q_z0 = log_normal(z0, qz0_mean, qz0_logvar) 
        log_r_vT = log_normal(v, rvT_mean, rvT_logvar) 
        log_p_zT = log_posterior(self.z)

        self.elbo = log_p_zT + log_r_vT - log_q_z0 - log_q_v0 + logdetsum

        self.optimizer = tf.train.AdamOptimizer(learning_rate=.001, 
                                                epsilon=1e-02).minimize(-self.elbo)

        init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()

        self.sess = tf.Session()
        self.sess.run(init_vars)





    def flow_step(self, z, v, log_posterior, step_size):
        '''
        z: [1,2]
        v: [1,2]
        '''

        z_size = 2
        v_size = 2

        grad = tf.gradients(-log_posterior(z), [z])[0]

        h = slim.stack(z,slim.fully_connected,[30])
        # mew_ = slim.fully_connected(h,v_size,activation_fn=None) 
        sig_ = slim.fully_connected(h,v_size,activation_fn=tf.nn.sigmoid) 

        mew_ = slim.fully_connected(h,v_size,activation_fn=tf.nn.sigmoid) 


        v = v*sig_ + (mew_* -grad)

        h2 = slim.stack(v,slim.fully_connected,[30])
        # mew_2 = slim.fully_connected(h2,z_size,activation_fn=None) 
        # sig_2 = slim.fully_connected(h2,z_size,activation_fn=tf.nn.sigmoid) 

        mew_2 = slim.fully_connected(h2,z_size,activation_fn=tf.nn.sigmoid) 

        # z = z*sig_2 + (step_size*mew_2*v)
        z = z + (mew_2*v)


        logdet = tf.reduce_sum(tf.log(sig_)) #+ tf.reduce_sum(tf.log(sig_2))

        return z, v, logdet





class Hamiltonian_Flow_model2(Factorized_Gaussian_model):

    def __init__(self, log_posterior, n_flows=3):

        tf.reset_default_graph()

        z_size = 2
        v_size = 2

        # step_size = tf.Variable([.1])
        step_size = 1

    
        #q(v)
        qv0_mean = tf.Variable(tf.zeros([v_size]))
        qv0_logvar = tf.Variable(tf.ones([v_size])-3.)

        #Sample v
        eps = tf.random_normal((1,v_size), 0, 1, dtype=tf.float32) 
        v0 = tf.add(qv0_mean, tf.multiply(tf.sqrt(tf.exp(qv0_logvar)), eps)) 

        #q(z|v)
        net = slim.stack(v0,slim.fully_connected,[30,30])
        net = slim.fully_connected(net,z_size*2,activation_fn=None) #[1,4]
        qz0_mean = tf.slice(net, [0,0], [1,2])
        qz0_logvar = tf.slice(net, [0,1], [1,2])

        #Sample z
        eps = tf.random_normal((1,z_size), 0, 1, dtype=tf.float32) 
        z0 = tf.add(qz0_mean, tf.multiply(tf.sqrt(tf.exp(qz0_logvar)), eps)) 

        z = z0
        v = v0
        logdetsum = 0.
        for i in range(n_flows):

            z, v, logdet = self.flow_step(z, v, log_posterior, step_size)
            logdetsum += logdet

        self.z = z

        #r(vT|zT)
        net = slim.stack(self.z,slim.fully_connected,[30,30])
        net = slim.fully_connected(net,v_size*2,activation_fn=None)
        rvT_mean = tf.slice(net, [0,0], [1,v_size])
        rvT_logvar = tf.slice(net, [0,1], [1,v_size])   

        #logprobs
        log_q_v0 = log_normal(v0, qv0_mean, qv0_logvar) 
        log_q_z0 = log_normal(z0, qz0_mean, qz0_logvar) 
        log_r_vT = log_normal(v, rvT_mean, rvT_logvar) 
        log_p_zT = log_posterior(self.z)

        self.elbo = log_p_zT + log_r_vT - log_q_z0 - log_q_v0 + logdetsum

        self.optimizer = tf.train.AdamOptimizer(learning_rate=.001, 
                                                epsilon=1e-02).minimize(-self.elbo)

        init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()

        self.sess = tf.Session()
        self.sess.run(init_vars)





    def flow_step(self, z, v, log_posterior, step_size):
        '''
        z: [1,2]
        v: [1,2]
        '''

        z_size = 2
        v_size = 2

        grad = tf.gradients(-log_posterior(z), [z])[0]

        h = slim.stack(tf.concat([z, grad], axis=1),slim.fully_connected,[30])
        mew_ = slim.fully_connected(h,v_size,activation_fn=None) 
        sig_ = slim.fully_connected(h,v_size,activation_fn=tf.nn.sigmoid) 

        # mew_ = slim.fully_connected(h,v_size,activation_fn=tf.nn.sigmoid) 


        v = v*sig_ + (mew_)#* -grad)

        h2 = slim.stack(v,slim.fully_connected,[30])
        mew_2 = slim.fully_connected(h2,z_size,activation_fn=None) 
        sig_2 = slim.fully_connected(h2,z_size,activation_fn=tf.nn.sigmoid) 

        # mew_2 = slim.fully_connected(h2,z_size,activation_fn=tf.nn.sigmoid) 

        z = z*sig_2 + mew_2
        # z = z + (mew_2*v)


        logdet = tf.reduce_sum(tf.log(sig_)) + tf.reduce_sum(tf.log(sig_2))

        return z, v, logdet




class Hamiltonian_Flow_model3(Factorized_Gaussian_model):

    def __init__(self, log_posterior, n_flows=3):

        tf.reset_default_graph()

        z_size = 2
        v_size = 2

        # step_size = tf.Variable([.1])
        step_size = 1

    
        #q(v)
        qv0_mean = tf.Variable(tf.zeros([v_size]))
        qv0_logvar = tf.Variable(tf.ones([v_size])-3.)

        #Sample v
        eps = tf.random_normal((1,v_size), 0, 1, dtype=tf.float32) 
        v0 = tf.add(qv0_mean, tf.multiply(tf.sqrt(tf.exp(qv0_logvar)), eps)) 

        #q(z|v)
        net = slim.stack(v0,slim.fully_connected,[30,30])
        net = slim.fully_connected(net,z_size*2,activation_fn=None) #[1,4]
        qz0_mean = tf.slice(net, [0,0], [1,2])
        qz0_logvar = tf.slice(net, [0,1], [1,2])

        #Sample z
        eps = tf.random_normal((1,z_size), 0, 1, dtype=tf.float32) 
        z0 = tf.add(qz0_mean, tf.multiply(tf.sqrt(tf.exp(qz0_logvar)), eps)) 

        z = z0
        v = v0
        logdetsum = 0.
        for i in range(n_flows):

            z, v, logdet = self.flow_step(z, v, log_posterior, step_size)
            logdetsum += logdet

        self.z = z

        #r(vT|zT)
        net = slim.stack(self.z,slim.fully_connected,[30,30])
        net = slim.fully_connected(net,v_size*2,activation_fn=None)
        rvT_mean = tf.slice(net, [0,0], [1,v_size])
        rvT_logvar = tf.slice(net, [0,1], [1,v_size])   

        #logprobs
        log_q_v0 = log_normal(v0, qv0_mean, qv0_logvar) 
        log_q_z0 = log_normal(z0, qz0_mean, qz0_logvar) 
        log_r_vT = log_normal(v, rvT_mean, rvT_logvar) 
        log_p_zT = log_posterior(self.z)

        self.elbo = log_p_zT + log_r_vT - log_q_z0 - log_q_v0 + logdetsum

        self.optimizer = tf.train.AdamOptimizer(learning_rate=.001, 
                                                epsilon=1e-02).minimize(-self.elbo)

        init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()

        self.sess = tf.Session()
        self.sess.run(init_vars)





    def flow_step(self, z, v, log_posterior, step_size):
        '''
        z: [1,2]
        v: [1,2]
        '''

        z_size = 2
        v_size = 2

        grad = tf.gradients(-log_posterior(z), [z])[0]

        h = slim.stack(tf.concat([z, grad], axis=1),slim.fully_connected,[30])
        # mew_ = slim.fully_connected(h,v_size,activation_fn=None) 
        sig_ = slim.fully_connected(h,v_size,activation_fn=tf.nn.sigmoid) 

        mew_ = slim.fully_connected(h,v_size,activation_fn=tf.nn.sigmoid) 


        v = v*sig_ + (mew_* -grad)

        h2 = slim.stack(v,slim.fully_connected,[30])
        # mew_2 = slim.fully_connected(h2,z_size,activation_fn=None) 
        # sig_2 = slim.fully_connected(h2,z_size,activation_fn=tf.nn.sigmoid) 

        mew_2 = slim.fully_connected(h2,z_size,activation_fn=None) 

        # z = z*sig_2 + (step_size*mew_2*v)
        z = z + (mew_2*v)


        logdet = tf.reduce_sum(tf.log(sig_)) #+ tf.reduce_sum(tf.log(sig_2))

        return z, v, logdet











