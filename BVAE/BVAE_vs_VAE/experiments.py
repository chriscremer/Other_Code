


import numpy as np
import tensorflow as tf
import pickle, time, datetime
from os.path import expanduser
home = expanduser("~")
import matplotlib.pyplot as plt
# import sys
# sys.path.insert(0, './BVAE_adding_eval_use_this')

# import argparse
# import os
# from scipy.stats import multivariate_normal as norm

from BVAE import BVAE
from VAE import VAE
from VAE import VAE_no_reg






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

    save_log = 0
    train_ = 0
    eval_ = 0
    plot_histo = 0
    viz_sammples = 1

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

    # test_x = test_x[:100]


    # Training settings
    x_size = 784   #f_height=28f_width=28
    n_batch = 50
    epochs = 50000
    h1_size = 100  #hidden layer size
    S_training = 1  #number of weight samples

    #Experimental Variables
    list_of_models = ['vae', 'bvae', 'vae_no_reg']#['bvae']#['bvae'] #['vae', 'bvae', 'vae_no_reg']
    list_of_k_samples = [1]
    z_sizes = [30] #[10,100]#[10,50,100]   #latent layer size

    # Test settings
    S_evaluation = 5 #2  
    k_evaluation = 100 #500
    n_batch_eval = 1 #2

    #Experiment log
    if save_log:
        dt = datetime.datetime.now()
        date_ = str(dt.date())
        time_ = str(dt.time())
        time_2 = time_[0] + time_[1] + time_[3] + time_[4] + time_[6] + time_[7] 
        experiment_log = experiment_log_path+'experiment_' + date_ + '_' +time_2 +'.txt'
        print 'Saving experiment log to ' + experiment_log


    for k_training in list_of_k_samples:

        for m in list_of_models:

            for z_size in z_sizes:


                # to_load_parameter_file = m + '_k' + str(k_training) + '_z'+str(z_size) + '_epochs'+str(90000)+'_smalldata.ckpt' 

                saved_parameter_file = m + '_k' + str(k_training) + '_z'+str(z_size) + '_epochs'+str(epochs)+'_smalldata.ckpt' 
                print 'Current:', saved_parameter_file
                if save_log:
                    with open(experiment_log, "a") as myfile:
                        myfile.write('\n\n' + saved_parameter_file +'\n')



                if train_:
                    #Train 
                    print 'Training'

                    hyperparams = {
                        'learning_rate': .001,
                        'x_size': x_size,
                        'z_size': z_size,
                        'encoder_net': [x_size, h1_size, z_size*2],
                        'decoder_net': [z_size, h1_size, x_size],
                        'n_W_particles': S_training,
                        'n_z_particles': k_training}

                    #Initialize model
                    if m == 'bvae':
                        model = BVAE(hyperparams)
                    elif m == 'vae_no_reg':
                        model = VAE_no_reg(hyperparams)
                    elif m == 'vae':
                        model = VAE(hyperparams)   

                    start = time.time()

                    model.train(train_x=train_x, valid_x=valid_x,
                                epochs=epochs, batch_size=n_batch,
                                display_step=[500,3000],
                                path_to_load_variables='',
                                # path_to_load_variables=parameter_path+to_load_parameter_file,

                                path_to_save_variables=parameter_path+saved_parameter_file)

                    time_to_train = time.time() - start
                    print 'Time to train', time_to_train
                    with open(experiment_log, "a") as f:
                        f.write('Time to train '+  str(time_to_train) + '\n')




                


                if plot_histo:

                    print 'Plot histograms'

                    hyperparams = {
                        'learning_rate': .001,
                        'x_size': x_size,
                        'z_size': z_size,
                        'encoder_net': [x_size, h1_size, z_size*2],
                        'decoder_net': [z_size, h1_size, x_size],
                        'n_W_particles': 1,
                        'n_z_particles': 1}

                    #Initialize model
                    if m == 'bvae':
                        model = BVAE(hyperparams)
                    elif m == 'vae_no_reg':
                        model = VAE_no_reg(hyperparams)            
                    elif m == 'vae':
                        model = VAE(hyperparams) 


                    # PLOT HISTOGRAMS
                    means, logvars = model.get_means_logvars(path_to_load_variables=parameter_path+saved_parameter_file)
                    
                    for list_ in means:
                        print list_.shape
                        print np.mean(list_)

                        list_ = np.reshape(list_, [-1])
                        plt.hist(list_, 100, facecolor='green', alpha=0.75)
                        plt.show()

                    for list_ in logvars:
                        print list_.shape
                        print np.mean(list_)

                        list_ = np.reshape(list_, [-1])
                        plt.hist(list_, 100, facecolor='green', alpha=0.75)
                        plt.show()

                    print np.mean(means)
                    print np.mean(logvars)
                    # fsafsa




                if viz_sammples:

                    print 'Vizualize samples'

                    hyperparams = {
                        'learning_rate': .001,
                        'x_size': x_size,
                        'z_size': z_size,
                        'encoder_net': [x_size, h1_size, z_size*2],
                        'decoder_net': [z_size, h1_size, x_size],
                        'n_W_particles': 5,
                        'n_z_particles': 1}

                    #Initialize model
                    if m == 'bvae':
                        model = BVAE(hyperparams)
                    elif m == 'vae_no_reg':
                        model = VAE_no_reg(hyperparams)            
                    elif m == 'vae':
                        model = VAE(hyperparams)   

                    batch, recons, prior_x_ = model.get_recons_and_priors(train_x, path_to_load_variables=parameter_path+saved_parameter_file)

                    print recons.shape
                    print prior_x_.shape
                    print batch.shape


                    fsd

                    # print recons[0][0][0]
                    # print prior_x_[0][0][1]
                    # print batch[0]
                    
                    # plt.clf()

                    f,axarr=plt.subplots(2,3,figsize=(12,6))
                    # axarr[0].plot(data1[0],data1[1])
                    # axarr[0].set_title('Loss')

                    axarr[0,0].set_title(m+' Prior Samples')
                    axarr[0,0].set_ylabel('Train')

                    tmp = np.reshape(prior_x_,(-1,280,28)) # (10,280,28)
                    img = np.hstack([tmp[i] for i in range(10)])
                    axarr[0,0].imshow(img, cmap='gray')

                    axarr[0,1].set_title('Batch')
                    tmp = np.reshape(batch,(-1,280,28)) # (10,280,28)
                    img = np.hstack([tmp[i] for i in range(10)])
                    axarr[0,1].imshow(img, cmap='gray')

                    axarr[0,2].set_title('Recons')
                    tmp = np.reshape(recons,(-1,280,28)) # (10,280,28)
                    img = np.hstack([tmp[i] for i in range(10)])
                    axarr[0,2].imshow(img, cmap='gray')


                    batch, recons, prior_x_ = model.get_recons_and_priors(test_x, path_to_load_variables=parameter_path+saved_parameter_file)


                    axarr[1,0].set_ylabel('Test')
                    tmp = np.reshape(prior_x_,(-1,280,28)) # (10,280,28)
                    img = np.hstack([tmp[i] for i in range(10)])
                    axarr[1,0].imshow(img, cmap='gray')

                    tmp = np.reshape(batch,(-1,280,28)) # (10,280,28)
                    img = np.hstack([tmp[i] for i in range(10)])
                    axarr[1,1].imshow(img, cmap='gray')

                    tmp = np.reshape(recons,(-1,280,28)) # (10,280,28)
                    img = np.hstack([tmp[i] for i in range(10)])
                    axarr[1,2].imshow(img, cmap='gray')

                    plt.grid('off')
                    # plt.show()
                    plt.savefig(experiment_log_path+ m + '_k' + str(k_training) + '_z'+str(z_size) + '_epochs'+str(epochs)+'_smalldata.png')
                    print 'saved fig to' + experiment_log_path+ m + '_k' + str(k_training) + '_z'+str(z_size) + '_epochs'+str(epochs)+'_smalldata.png'

                    






                if eval_:


                    hyperparams = {
                        'learning_rate': .001,
                        'x_size': x_size,
                        'z_size': z_size,
                        'encoder_net': [x_size, h1_size, z_size*2],
                        'decoder_net': [z_size, h1_size, x_size],
                        'n_W_particles': S_evaluation,
                        'n_z_particles': k_evaluation}

                    #Initialize model
                    if m == 'bvae':
                        model = BVAE(hyperparams)
                    elif m == 'vae_no_reg':
                        model = VAE_no_reg(hyperparams)            
                    elif m == 'vae':
                        model = VAE(hyperparams)   


                    #Evaluate
                    print 'Evaluating'

                    start = time.time()
                    test_results, train_results, labels = model.eval(data=test_x, batch_size=n_batch_eval, display_step=100,
                                            path_to_load_variables=parameter_path+saved_parameter_file, data2=train_x)

                    time_to_eval = time.time() - start
                    print 'time to evaluate', time_to_eval
                    print 'Results: ' + str(test_results) + ' for ' + saved_parameter_file
                    
                    if save_log:
                        with open(experiment_log, "a") as myfile:

                            # myfile.write('time to evaluate '+  str(time_to_eval) +'\n')

                            myfile.write('Train set\n')
                            myfile.write('%10s' % 'elbo'+ '%10s' %'log_px' + '%10s' %'log_pz'+'%10s' %'log_qz'+ '%10s' %'log_pW'+'%10s' %'log_qW'+'\n')
                            results_str = ''
                            for val in train_results:
                                results_str += '%10.2f' % val
                            myfile.write(results_str +'\n')


                            myfile.write('Test set\n')
                            myfile.write('%10s' % 'iwae_elbo'+ '%10s' %'log_px' + '%10s' %'log_pz'+'%10s' %'log_qz'+ '%10s' %'log_pW'+'%10s' %'log_qW'+'\n')
                            results_str = ''
                            for val in test_results:
                                results_str += '%10.2f' % val
                            myfile.write(results_str +'\n')
                        


                print





    if save_log:             
        with open(experiment_log, "a") as myfile:
            myfile.write('\n\nAll Done.\n')
        
    print 'All Done'
































