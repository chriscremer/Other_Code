
import tensorflow as tf
import math




def log_normal(x, mean, log_var):
    '''
    x is [P, D]
    mean is [D]
    log_var is [D]
    '''
    term1 = 2 * tf.log(2*math.pi)
    term2 = tf.reduce_sum(log_var) #sum over dimensions, [1]
    term3 = tf.square(x - mean) / tf.exp(log_var)
    term3 = tf.reduce_sum(term3, 1) #sum over dimensions, [P]
    all_ = term1 + term2 + term3
    log_normal = -.5 * all_  
    return log_normal




def log_posterior_0(x):
    '''
    x: [P,D]
    '''
    prob = (tf.exp(log_normal(x, [1,3], tf.ones(2)/100))
                + tf.exp(log_normal(x, [3,0], tf.ones(2)/100))
                + tf.exp(log_normal(x, [1,1], tf.ones(2)/100))
                + tf.exp(log_normal(x, [-3,-1], tf.ones(2)/100))
                + tf.exp(log_normal(x, [-1,-3], tf.ones(2)/100))
                + tf.exp(log_normal(x, [2,2], tf.ones(2)))
                + tf.exp(log_normal(x, [-2,-2], tf.ones(2)))
                + tf.exp(log_normal(x, [0,0], tf.ones(2))))
    prob = tf.maximum(prob, tf.exp(-40.))
    return tf.log(prob)

def log_posterior_1(x):
    '''
    x: [P,D]
    '''
    prob = (tf.exp(log_normal(x, [0.,4.], [2., -1.]))
                + tf.exp(log_normal(x, [0.,-4.], [2., -1.])))
    prob = tf.maximum(prob, tf.exp(-40.))
    return tf.log(prob)

def log_posterior_2(x):
    '''
    x: [P,D]
    '''
    prob = (tf.exp(log_normal(x, [-2.,2], [-2., 1.5]))
                    + tf.exp(log_normal(x, [2.,-1], [1.5, -2.])))
    prob = tf.maximum(prob, tf.exp(-40.))
    return tf.log(prob)

def log_posterior_3(x):
    '''
    x: [P,D]
    '''
    prob = (tf.exp(log_normal(x, [-4.,4], [0., 0.])))
    prob = tf.maximum(prob, tf.exp(-40.))
    return tf.log(prob)


def log_proposal(x):
    return log_normal(x, [0.,0.], [1., 1.])



class posterior_class(object):

	def __init__(self, log_posterior):

		tf.reset_default_graph()


		self.z = tf.placeholder(tf.float32, [None, 2])
		self.logp = log_posterior(self.z)


		tf.get_default_graph().finalize()

		self.sess = tf.Session()

	def run_log_post(self, z):

		return self.sess.run(self.logp, feed_dict={self.z: z})





















