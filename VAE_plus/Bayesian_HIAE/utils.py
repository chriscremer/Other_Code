


import tensorflow as tf
import math


def log_normal(z, mean, log_var):
    '''
    Log of normal distribution

    z is [B, D]
    mean is [B, D]
    log_var is [B, D]
    output is [B]
    '''

    D = tf.to_float(tf.shape(mean)[1])
    term1 = D * tf.log(2*math.pi) #[1]
    term2 = tf.reduce_sum(log_var, axis=1) #sum over D, [B]
    dif_cov = tf.square(z - mean) / tf.exp(log_var)
    term3 = tf.reduce_sum(dif_cov, axis=1) #sum over D, [B]
    all_ = term1 + term2 + term3
    log_N = -.5 * all_
    return log_N


def log_normal2(position, mean, log_var):
    '''
    Log of normal distribution
    position is [P, D]
    mean is [D]
    log_var is [D]
    output is [P]
    '''

    n_D = tf.to_float(tf.shape(mean)[0])
    term1 = n_D * tf.log(2*math.pi)
    term2 = tf.reduce_sum(log_var, 0) #sum over D [1]
    dif_cov = tf.square(position - mean) / tf.exp(log_var)
    term3 = tf.reduce_sum(dif_cov, 1) #sum over D [P]
    all_ = term1 + term2 + term3
    log_normal_ = -.5 * all_

    return log_normal_


def log_normal3(x, mean, log_var):
    '''
    Log of normal distribution

    x is [D]
    mean is [D]
    log_var is [D]
    output is [1]
    '''

    D = tf.to_float(tf.shape(mean)[0])
    term1 = D * tf.log(2*math.pi) #[1]
    term2 = tf.reduce_sum(log_var) #[1]
    dif_cov = tf.square(x - mean) / tf.exp(log_var)
    term3 = tf.reduce_sum(dif_cov) #[1]
    all_ = term1 + term2 + term3
    log_N = -.5 * all_
    return log_N


# def log_normal4(x, mean, log_var):
#     '''
#     Log of normal distribution

#     x is [B,P,D]
#     mean is [B,D]
#     log_var is [B,D]
#     output is [B,P]
#     '''

#     B = tf.shape(mean)[0]
#     D = tf.shape(mean)[1]
#     # P = tf.shape(x)[1]
#     # D_float = tf.to_float(tf.shape(mean)[1])


#     # term1 = D_float * tf.log(2*math.pi) #[1]
#     # term2 = tf.reduce_sum(log_var, axis=1) #sum over D, [B]


#     # term2 = tf.reshape(term2, [B,1])
#     # # term2 = tf.tile(term2, [1,P])

#     # mean = tf.reshape(mean, [B,1,D])
#     # log_var = tf.reshape(log_var, [B,1,D])


#     # dif_cov = tf.square(x - mean) / tf.exp(log_var)
#     # term3 = tf.reduce_sum(dif_cov, axis=2) #sum over D, [B,P]
#     # all_ = term1 + term2 + term3
#     # log_N = -.5 * all_
#     # return log_N

#     # [P,B,D]
#     x = tf.transpose(x, [1,0,2])
#     n_D = tf.to_float(tf.shape(mean)[1])
#     mean = tf.reshape(mean, [1,B,D])
#     log_var = tf.reshape(log_var, [1,B,D])

#     term1 = n_D * tf.log(2*math.pi) #[1]
#     term2 = tf.reduce_sum(log_var, 2) #sum over D [1,B]
#     dif_cov = tf.square(x - mean) / tf.exp(log_var)
#     term3 = tf.reduce_sum(dif_cov, 2) #sum over D [P,B]
#     all_ = term1 + term2 + term3
#     log_normal_ = -.5 * all_

#     return log_normal_



def log_normal4(x, mean, log_var):
    '''
    Log of normal distribution

    x is [B,P,D]
    mean is [B,D]
    log_var is [B,D]
    output is [B,P]
    '''

    P = tf.shape(x)[1]

    mean=tf.tile(mean, [P,1])
    log_var=tf.tile(log_var, [P,1])
    mean=tf.reshape(mean, [-1])
    log_var=tf.reshape(log_var, [-1])

    x=tf.reshape(x, [-1])


    dist = tf.contrib.distributions.Normal(mu=mean, sigma=tf.sqrt(tf.exp(log_var)))

    return dist.log_pdf(x)





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













