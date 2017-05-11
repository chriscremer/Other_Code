


import numpy as np
import tensorflow as tf
import pickle
import time
import datetime
from os.path import expanduser
home = expanduser("~")

# import argparse
# import os
# from scipy.stats import multivariate_normal as norm

from IWAE import VAE
from IWAE import IWAE





def load_binarized_mnist(location):

    with open(location, 'rb') as f:
        train_x, valid_x, test_x = pickle.load(f)
    return train_x, valid_x, test_x


def load_mnist(location):

    with open(location,'rb') as f:
        mnist_data = pickle.load(f)
    train_x = mnist_data[0][0]
    train_y = mnist_data[0][1]
    valid_x = mnist_data[1][0]
    valid_y = mnist_data[1][1]
    test_x = mnist_data[2][0]
    test_y = mnist_data[2][1]
    return train_x, valid_x, test_x



if __name__ == '__main__':

    # Paths
    mnist_path = home+'/Documents/MNIST_data/mnist.pkl'
    binarized_mnist_path = home+'/Documents/MNIST_data/binarized_mnist.pkl'
    parameter_path = home+'/Documents/tmp/'
    experiment_log_path = home+'/Documents/tmp/'

    #Load data
    train_x, valid_x, test_x = load_binarized_mnist(location=binarized_mnist_path)
    print 'Train', train_x.shape
    print 'Valid', valid_x.shape
    print 'Test', test_x.shape


    # Training settings
    f_height=28
    f_width=28
    n_batch = 100
    epochs = 1000
    list_of_models = ['bvae', 'biwae']
    list_of_k_samples = [1,10,50]
    n_a0 = \
        dict(n_input=f_height*f_width, # 784 image
             encoder_net=[200,200], 
             n_z=50,  # dimensionality of latent space
             decoder_net=[200,200]) 
    list_of_archs = [n_a0]
    list_of_archs_i = [0]


    # Test settings
    m_evaluation = 5
    k_evaluation = 100

    #Experiment log
    dt = datetime.datetime.now()
    date_ = str(dt.date())
    time_ = str(dt.time())
    time_2 = time_[0] + time_[1] + time_[3] + time_[4] + time_[6] + time_[7] 
    experiment_log = experiment_log_path+'experiment_' + date_ + '_' +time_2 +'.txt'
    print 'Saving experiment log to ' + experiment_log


    for k in list_of_k_samples:

        for m in list_of_models:

            for arch in list_of_archs_i:

                saved_parameter_file = m + '_struc' + str(arch) + '_k' + str(k) + '_1000.ckpt' 
                print 'Current:', saved_parameter_file
                with open(experiment_log, "a") as myfile:
                    myfile.write('\nCurrent:' + saved_parameter_file +'\n')

                #Train 

                #Initialize model
                if m == 'bvae':
                    model = VAE(list_of_archs[arch], batch_size=n_batch, n_particles=k)
                elif m == 'biwae':
                    model = IWAE(list_of_archs[arch], batch_size=n_batch, n_particles=k)

                start = time.time()

                model.train(train_x=train_x, valid_x=valid_x, display_step=9000, 
                            path_to_load_variables='', 
                            path_to_save_variables=home+'/data/'+saved_parameter_file, 
                            epochs=epochs)

                time_to_train = time.time() - start
                print 'Time to train', time_to_train
                with open(experiment_log, "a") as f:
                    f.write('Time to train '+  str(time_to_train) + '\n')


                #Evaluate

                #Initialize model
                if m == 'vae':
                    model = VAE(list_of_archs[arch], batch_size=1, n_particles=k_evaluation)
                elif m == 'iwae':
                    model = IWAE(list_of_archs[arch], batch_size=1, n_particles=k_evaluation)


                # time.sleep(10) # delays for 5 seconds
                

                #Load parameters
                model.load_parameters(path_to_load_variables=home+'/data/'+saved_parameter_file)

                start = time.time()

                #Log Likelihood Lower Bound
                IS_ELBO = model.IS_ELBO(data=test_x, n_z_samples=k_evaluation, n_W_samples=m_evaluation)

                print 'Model Log Likelihood is ' + str(IS_ELBO) + ' for ' + saved_parameter_file
                with open(experiment_log, "a") as myfile:
                    myfile.write('Model Log Likelihood is ' + str(IS_ELBO) + ' for ' + saved_parameter_file 
                        +'\n')

                time_to_eval = time.time() - start
                print 'time to evaluate', time_to_eval
                with open(experiment_log, "a") as myfile:
                    myfile.write('time to evaluate '+  str(time_to_eval) +'\n\n')
                print 



    #Examples: 
    # python experiments.py -m vae -k 1 -a train -s vae_1
    # python experiments.py -m vae -k 10 -a train -s vae_10_s6 -l vae_10 -ss 6 -es 6
    # # python experiments.py -m vae -k 10 -a evaluate -l vae_10
    # parser = argparse.ArgumentParser(description='Run experiments.')
    # parser.add_argument('--model', '-m', choices=['vae', 'iwae', 'mog_vae', 'mog_iwae'], 
    #     default='vae')
    # parser.add_argument('--k', '-k', type=int, default=1)
    # parser.add_argument('--action', '-a', choices=['train', 'evaluate', 'combined'], default='train')
    # parser.add_argument('--save_to', '-s', type=str, default='')
    # parser.add_argument('--load_from', '-l', type=str, default='')
    # parser.add_argument('--starting_stage', '-ss', type=int, default=0)
    # parser.add_argument('--ending_stage', '-es', type=int, default=5)
    # parser.add_argument('--n_clusters', '-c', type=int, default=2)
    # args = parser.parse_args()

    # path_to_save_variables, path_to_load_variables = user_defined_locations(args)

    # #Load data
    # train_x, valid_x, test_x = load_binarized_mnist(location=home+'/data/binarized_mnist.pkl')
    # print 'Train', train_x.shape
    # print 'Valid', valid_x.shape
    # print 'Test', test_x.shape



        # Check for the model variables, if not there train it then eval. 
        #Write results to a file in case something happens
        #Also have a timer going to say how long each took. 



    print 'All Done'














