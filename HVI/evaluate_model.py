

import numpy as np
import tensorflow as tf
# import argparse
# import os
from os.path import expanduser
home = expanduser("~")
import pickle
# from scipy.stats import multivariate_normal as norm
import time
import datetime

from VAE_IWAE import VAE
from VAE_IWAE import IWAE





if __name__ == '__main__':

    data_location = home + '/Documents/MNIST_Data/'
    path_for_variables = home + 'Documents/tmp/'


    #Load data
    print 'Loading data'    
    with open(data_location + 'binarized_mnist.pkl', 'rb') as f:
        train_x, valid_x, test_x = pickle.load(f)
    print 'Train', train_x.shape
    print 'Valid', valid_x.shape
    print 'Test', test_x.shape



    # Define model
    f_height=28
    f_width=28
    n_batch = 10
    k = 3
    n_a0 = \
        dict(n_input=f_height*f_width, # 784 image
             encoder_net=[200,200], 
             n_z=20, 
             decoder_net=[200,200]) 



    # Eval 
    k_evaluation = 1000

    path_to_load_variables = path_for_variables + 'model.pkl'
    # path_to_save_variables = path_for_variables + 'model.pkl'


    model = VAE(n_a0, batch_size=n_batch, n_particles=k)
    # model = IWAE(list_of_archs[arch], batch_size=n_batch, n_particles=k)

    model.eval(train_x=train_x, valid_x=valid_x, display_step=9000, 
                path_to_load_variables='', 
                path_to_save_variables=path_to_save_variables, 
                epochs=epochs)


    print 'Done.'
    fasd




















