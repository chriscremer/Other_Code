
import tensorflow as tf
import math


def log_normal(z, mean, log_var):
    '''
    Log of normal distribution

    z is [B, Z]
    mean is [B, Z]
    log_var is [B, Z]
    output is [B]
    '''

    n_D = tf.to_float(tf.shape(mean)[0])
    term1 = n_D * tf.log(2*math.pi) #[1]
    term2 = tf.reduce_sum(log_var, axis=1) #sum over Z, [B]
    dif_cov = tf.square(z - mean) / tf.exp(log_var)
    term3 = tf.reduce_sum(dif_cov, axis=1) #sum over Z, [B]
    all_ = term1 + term2 + term3
    log_N = -.5 * all_
    return log_N





def log_bernoulli(t, pred_no_sig):
    '''
    Log of bernoulli distribution
    t is [B, X]
    pred_no_sig is [B, P, X] 
    output is [B, P]
    '''

    B = tf.shape(t)[0]
    X = tf.shape(t)[1]

    #[B,1,X]
    t = tf.reshape(t, [B, 1, X])

    reconstr_loss = \
            tf.reduce_sum(tf.maximum(pred_no_sig, 0) 
                        - pred_no_sig * t
                        + tf.log(1 + tf.exp(-tf.abs(pred_no_sig))),
                         2) #sum over dimensions

    #negative because the above calculated the NLL, so this is returning the LL
    return -reconstr_loss
