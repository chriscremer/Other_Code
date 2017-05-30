


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


from BNN2 import BNN
from NN2 import NN
from MNF import MNF





if __name__ == '__main__':

    print 'Loading data'
    with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
        mnist_data = pickle.load(f)

    train_x = mnist_data[0][0]
    train_y = mnist_data[0][1]
    valid_x = mnist_data[1][0]
    valid_y = mnist_data[1][1]
    test_x = mnist_data[2][0]
    test_y = mnist_data[2][1]
    print 'Train x', train_x.shape #[50000,784]
    print 'Valid x', valid_x.shape #[50000,784]
    print 'Test y', test_y.shape #[10000]




    parameter_path = home+'/Documents/tmp/'
    experiment_log_path = home+'/Documents/tmp/'




    train_x = train_x[:50]
    train_y = train_y[:50]
    print 'Train', train_x.shape


    # Training settings
    x_size = 784   #f_height=28f_width=28
    n_batch = 50
    y_size = 10
    epochs = 20000 #25000
    h1_size = 200
    S_training = 1  #number of weight samples

    #Experimental Variables
    list_of_models = ['mnf', 'bnn', 'nn'] #['nn', 'bnn']

    # Test settings
    S_evaluation = 5 #2
    n_batch_eval = 1 #2

    #Experiment log
    dt = datetime.datetime.now()
    date_ = str(dt.date())
    time_ = str(dt.time())
    time_2 = time_[0] + time_[1] + time_[3] + time_[4] + time_[6] + time_[7] 
    experiment_log = experiment_log_path+'experiment_' + date_ + '_' +time_2 +'.txt'
    print 'Saving experiment log to ' + experiment_log



    for m in list_of_models:


        saved_parameter_file = m + '_epochs'+str(epochs)+'.ckpt' 
        print 'Current:', saved_parameter_file
        with open(experiment_log, "a") as myfile:
            myfile.write('\nCurrent:' + saved_parameter_file +'\n')




        #Train 
        print 'Training'

        net = [x_size,h1_size,y_size] 

        #Initialize model
        if m == 'nn':
            model = NN(net)
        elif m == 'bnn':
            model = BNN(net)

        start = time.time()
        model.train(train_x=train_x, train_y=train_y, 
                    epochs=epochs, batch_size=n_batch, n_particles=S_training, display_step=[1000,5000],
                    path_to_load_variables='',
                    path_to_save_variables=parameter_path+saved_parameter_file)

        time_to_train = time.time() - start
        print 'Time to train', time_to_train
        with open(experiment_log, "a") as f:
            f.write('Time to train '+  str(time_to_train) + '\n')



        #Evaluate
        print 'Evaluating'

        #Initialize model
        if m == 'nn':
            model = NN(net)
        elif m == 'bnn':
            model = BNN(net)
        elif m == 'mnf':
            model = MNF(net)

        start = time.time()
        info = model.eval(test_x, test_y, path_to_load_variables=parameter_path+saved_parameter_file, 
                            batch_size=n_batch_eval, n_particles=S_evaluation)


        time_to_eval = time.time() - start
        print 'time to evaluate', time_to_eval
        print 'Model Log Likelihood is ' + str(info) + ' for ' + saved_parameter_file
        
        with open(experiment_log, "a") as myfile:
            myfile.write('time to evaluate '+  str(time_to_eval) +'\n')
            
            myfile.write('iwae_elbo\n')

            myfile.write('Info' + str(info) + ' for '+ saved_parameter_file +'\n')
            
        print 




            
    with open(experiment_log, "a") as myfile:
        myfile.write('All Done.\n')
        
    print 'All Done'














