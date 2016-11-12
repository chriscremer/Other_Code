

import numpy as np
from scipy.stats import multivariate_normal

import math



def _log_p_z_given_x(x, mean, log_std_sq):
    #Get log(p(z|x))
    #This is just exp of a normal with some mean and var

    # term1 = tf.log(tf.reduce_prod(tf.exp(log_var_sq), reduction_indices=1))
    term1 = np.sum(log_std_sq) #sum over dimensions n_z so now its [particles, batch]

    term2 = len(x) * np.log(2*math.pi)
    dif = np.square(x - mean)
    dif_cov = dif / np.exp(log_std_sq)
    # term3 = tf.reduce_sum(dif_cov * dif, 1) 
    term3 = np.sum(dif_cov) #sum over dimensions n_z so now its [particles, batch]

    all_ = term1 + term2 + term3
    log_p_z_given_x = -.5 * all_

    # log_p_z_given_x = tf.reduce_mean(log_p_z_given_x, 1) #average over batch
    # log_p_z_given_x = tf.reduce_mean(log_p_z_given_x) #average over particles

    return log_p_z_given_x







mean = np.array([0,0])
cov = np.array([.2, .3])
x = np.array([.5,.5])
y = multivariate_normal.pdf(x, mean=mean, cov=cov)
print y
print np.log(y)

cov = np.log(cov)


print _log_p_z_given_x(x, mean, cov)






