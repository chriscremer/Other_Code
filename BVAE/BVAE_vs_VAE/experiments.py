


import numpy as np
import tensorflow as tf
import pickle, time, datetime
from os.path import expanduser
home = expanduser("~")
# import sys
# sys.path.insert(0, './BVAE_adding_eval_use_this')

# import argparse
# import os
# from scipy.stats import multivariate_normal as norm

from BVAE import BVAE
from VAE import VAE





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

    train_x = train_x[:50]
    print 'Train', train_x.shape


    # Training settings
    x_size = 784   #f_height=28f_width=28
    n_batch = 50
    epochs = 30000
    h1_size = 200
    S_training = 1  #number of weight samples

    #Experimental Variables
    list_of_models = ['vae', 'bvae']
    list_of_k_samples = [1]
    z_sizes = [10,100]#[10,50,100]

    # Test settings
    S_evaluation = 5 #2
    k_evaluation = 100 #500
    n_batch_eval = 1 #2

    #Experiment log
    dt = datetime.datetime.now()
    date_ = str(dt.date())
    time_ = str(dt.time())
    time_2 = time_[0] + time_[1] + time_[3] + time_[4] + time_[6] + time_[7] 
    experiment_log = experiment_log_path+'experiment_' + date_ + '_' +time_2 +'.txt'
    print 'Saving experiment log to ' + experiment_log


    for k_training in list_of_k_samples:

        for m in list_of_models:

            for z_size in z_sizes:

                saved_parameter_file = m + '_k' + str(k_training) + '_z'+str(z_size) + '_epochs'+str(epochs)+'_smalldata.ckpt' 
                print 'Current:', saved_parameter_file
                with open(experiment_log, "a") as myfile:
                    myfile.write('\nCurrent:' + saved_parameter_file +'\n')




                # #Train 
                # print 'Training'

                # hyperparams = {
                #     'learning_rate': .0001,
                #     'x_size': x_size,
                #     'z_size': z_size,
                #     'encoder_net': [x_size, h1_size, z_size*2],
                #     'decoder_net': [z_size, h1_size, x_size],
                #     'n_W_particles': S_training,
                #     'n_z_particles': k_training}

                # #Initialize model
                # if m == 'bvae':
                #     model = BVAE(hyperparams)
                # elif m == 'biwae':
                #     model = BIWAE(hyperparams)
                # elif m == 'vae':
                #     model = VAE(hyperparams)   

                # start = time.time()

                # model.train(train_x=train_x, valid_x=valid_x,
                #             epochs=epochs, batch_size=n_batch,
                #             display_step=[500,3000],
                #             path_to_load_variables='',
                #             path_to_save_variables=parameter_path+saved_parameter_file)

                # time_to_train = time.time() - start
                # print 'Time to train', time_to_train
                # with open(experiment_log, "a") as f:
                #     f.write('Time to train '+  str(time_to_train) + '\n')













                #Evaluate
                print 'Evaluating'

                hyperparams = {
                    'learning_rate': .0001,
                    'x_size': x_size,
                    'z_size': z_size,
                    'encoder_net': [x_size, h1_size, z_size*2],
                    'decoder_net': [z_size, h1_size, x_size],
                    'n_W_particles': S_evaluation,
                    'n_z_particles': k_evaluation}

                #Initialize model
                if m == 'bvae':
                    model = BVAE(hyperparams)
                elif m == 'biwae':
                    model = BIWAE(hyperparams)            
                elif m == 'vae':
                    model = VAE(hyperparams)   

                start = time.time()

                info = model.eval(data=test_x, batch_size=n_batch_eval, display_step=100,
                                        path_to_load_variables=parameter_path+saved_parameter_file)

                time_to_eval = time.time() - start
                print 'time to evaluate', time_to_eval
                print 'Model Log Likelihood is ' + str(info) + ' for ' + saved_parameter_file
                
                with open(experiment_log, "a") as myfile:
                    myfile.write('time to evaluate '+  str(time_to_eval) +'\n')
                    
                    myfile.write('iwae_elbo, log_px,log_pz,log_qz,log_pW,log_qW\n')

                    myfile.write('Info' + str(info) + ' for '+ saved_parameter_file +'\n')
                    
                print 





                
    with open(experiment_log, "a") as myfile:
        myfile.write('All Done.\n')
        
    print 'All Done'














