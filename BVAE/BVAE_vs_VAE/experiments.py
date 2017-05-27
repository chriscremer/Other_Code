


import numpy as np
import tensorflow as tf
import pickle, time, datetime
from os.path import expanduser
home = expanduser("~")
import matplotlib.pyplot as plt
# from decimal import Decimal
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

    save_log = 1
    train_ = 1
    eval_ = 1
    plot_histo = 0
    viz_sammples = 0

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
    lr = .001
    h1_size = 100  #hidden layer size
    S_training = 1  #number of weight samples
    k_training = 1 #number of z samples
    z_size = 30

    #Experimental Variables
    list_of_models = ['bvae']  #['vae', 'bvae', 'vae_no_reg'] #['vae', 'bvae', 'vae_no_reg']
    # list_of_k_samples = [1]
    # z_sizes = [30] #[10,100]#[10,50,100]   #latent layer size
    qW_weights = [.0000001, 1.]
    lmbas = [0., 1.]

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


    for qW_weight in qW_weights:

        for m in list_of_models:

            for lmba in lmbas:

                hyperparams = {
                    'learning_rate': lr,
                    'x_size': x_size,
                    'z_size': z_size,
                    'encoder_net': [x_size, h1_size, z_size*2],
                    'decoder_net': [z_size, h1_size, x_size],
                    # 'n_W_particles': S_training,
                    # 'n_z_particles': k_training,
                    'qW_weight': qW_weight,
                    'lmba': lmba}


                # to_load_parameter_file = m + '_k' + str(k_training) + '_z'+str(z_size) + '_epochs'+str(90000)+'_smalldata.ckpt' 
                
                sci_not = '{:2e}'.format(qW_weight)
                sci_not = sci_not.replace(".", "")
                sci_not = sci_not.replace("0", "")
                # print sci_not

                saved_parameter_file = m + '_qW_' + str(sci_not) + '_lmba'+str(int(lmba)) + '_epochs'+str(epochs)+'_smalldata.ckpt' 
                print 'Current:', saved_parameter_file
                if save_log:
                    with open(experiment_log, "a") as myfile:
                        myfile.write('\n\n' + saved_parameter_file +'\n')



                if train_:
                    #Train 
                    print 'Training'

                    hyperparams['n_W_particles'] = S_training
                    hyperparams['n_z_particles'] = k_training

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
                    # with open(experiment_log, "a") as f:
                    #     f.write('Time to train '+  str(time_to_train) + '\n')




                


                if plot_histo:

                    print 'Plot histograms'

                    hyperparams['n_W_particles'] = 1
                    hyperparams['n_z_particles'] = 1


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


                    hyperparams['n_W_particles'] = 5
                    hyperparams['n_z_particles'] = 1


                    #Initialize model
                    if m == 'bvae':
                        model = BVAE(hyperparams)
                    elif m == 'vae_no_reg':
                        model = VAE_no_reg(hyperparams)            
                    elif m == 'vae':
                        model = VAE(hyperparams)   

                    batch_size=36
                    batch_size_sqrt = int(np.sqrt(batch_size))

                    f,axarr=plt.subplots(2,3,figsize=(12,6))



                    #Train set
                    batch, recons, prior_x_ = model.get_recons_and_priors(train_x, batch_size, path_to_load_variables=parameter_path+saved_parameter_file)

                    # print recons.shape  # [S,K,N,X]
                    # print prior_x_.shape  # [S,K,N,X]
                    # print batch.shape  # [N,X]



                    # Average over weights 
                    recons_means = np.mean(recons, axis=0)
                    recons_means = np.reshape(recons_means, [batch_size, x_size]) #[N,X]
                    prior_x_means = np.mean(prior_x_, axis=0)
                    prior_x_means = np.reshape(prior_x_means, [batch_size, x_size]) #[N,X]

                    #get variance of predictions
                    if m=='bvae':
                        recons_std = np.std(recons, axis=0)
                        recons_std = np.reshape(recons_std, [batch_size, x_size]) #[N,X]
                        recons_means[batch_size/2:] = recons_std[:batch_size/2]

                        # # print recons_std
                        # for jj in range(len(prior_x_[0][0][0])):
                        #         print prior_x_[0][0][0][jj], prior_x_[1][0][0][jj], prior_x_[2][0][0][jj], ' ', recons_std[0][jj]

                        prior_x_std = np.std(prior_x_, axis=0)
                        prior_x_std = np.reshape(prior_x_std, [batch_size, x_size]) #[N,X]
                        prior_x_means[batch_size/2:] = prior_x_std[:batch_size/2]

                        # print prior_x_std

                    recons = recons_means
                    prior_x_ = prior_x_means

                    axarr[0,0].set_title('Prior Samples')
                    axarr[0,0].set_ylabel('Train')
                    tmp = np.reshape(prior_x_,(-1,28*batch_size_sqrt,28)) 
                    img = np.hstack([tmp[i] for i in range(batch_size_sqrt)])
                    axarr[0,0].imshow(img, cmap='gray')
                    # axarr[0,0].axis('off')
                    axarr[0,0].set_yticklabels([])
                    axarr[0,0].set_xticklabels([])

                    axarr[0,1].set_title('Batch')
                    tmp = np.reshape(batch,(-1,28*batch_size_sqrt,28)) 
                    img = np.hstack([tmp[i] for i in range(batch_size_sqrt)])
                    axarr[0,1].imshow(img, cmap='gray')
                    axarr[0,1].axis('off')

                    axarr[0,2].set_title('Recons')
                    tmp = np.reshape(recons,(-1,28*batch_size_sqrt,28)) 
                    img = np.hstack([tmp[i] for i in range(batch_size_sqrt)])
                    axarr[0,2].imshow(img, cmap='gray')
                    axarr[0,2].axis('off')


                    #Test set
                    batch, recons, prior_x_ = model.get_recons_and_priors(test_x, batch_size, path_to_load_variables=parameter_path+saved_parameter_file)

                    # Average over weights 
                    recons_means = np.mean(recons, axis=0)
                    recons_means = np.reshape(recons_means, [batch_size, x_size]) #[N,X]
                    prior_x_means = np.mean(prior_x_, axis=0)
                    prior_x_means = np.reshape(prior_x_means, [batch_size, x_size]) #[N,X]

                    #get variance of predictions
                    if m=='bvae':
                        recons_std = np.std(recons, axis=0)
                        recons_std = np.reshape(recons_std, [batch_size, x_size]) #[N,X]
                        recons_means[batch_size/2:] = recons_std[:batch_size/2]

                        prior_x_std = np.std(prior_x_, axis=0)
                        prior_x_std = np.reshape(prior_x_std, [batch_size, x_size]) #[N,X]
                        prior_x_means[batch_size/2:] = prior_x_std[:batch_size/2]

                    recons = recons_means
                    prior_x_ = prior_x_means


                    axarr[1,0].set_ylabel('Test')
                    tmp = np.reshape(prior_x_,(-1,28*batch_size_sqrt,28)) 
                    img = np.hstack([tmp[i] for i in range(batch_size_sqrt)])
                    axarr[1,0].imshow(img, cmap='gray')
                    # axarr[1,0].axis('off')
                    axarr[1,0].set_yticklabels([])
                    axarr[1,0].set_xticklabels([])

                    tmp = np.reshape(batch,(-1,28*batch_size_sqrt,28)) 
                    img = np.hstack([tmp[i] for i in range(batch_size_sqrt)])
                    axarr[1,1].imshow(img, cmap='gray')
                    axarr[1,1].axis('off')

                    tmp = np.reshape(recons,(-1,28*batch_size_sqrt,28)) 
                    img = np.hstack([tmp[i] for i in range(batch_size_sqrt)])
                    axarr[1,2].imshow(img, cmap='gray')
                    axarr[1,2].axis('off')

                    plt.grid('off')
                    # plt.show()
                    f.text(.05,.5,m)

                    plt.savefig(experiment_log_path+ m + '_k' + str(k_training) + '_z'+str(z_size) + '_epochs'+str(epochs)+'_smalldata.png')
                    print 'saved fig to' + experiment_log_path+ m + '_k' + str(k_training) + '_z'+str(z_size) + '_epochs'+str(epochs)+'_smalldata.png'

                    






                if eval_:

                    hyperparams['n_W_particles'] = S_evaluation
                    hyperparams['n_z_particles'] = k_evaluation


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
                    test_results, train_results, test_labels, train_labels = model.eval(data=test_x, batch_size=n_batch_eval, display_step=100,
                                            path_to_load_variables=parameter_path+saved_parameter_file, data2=train_x)

                    time_to_eval = time.time() - start
                    print 'time to evaluate', time_to_eval
                    print 'Results: ' + str(test_results) + ' for ' + saved_parameter_file
                    
                    if save_log:
                        with open(experiment_log, "a") as myfile:

                            # myfile.write('time to evaluate '+  str(time_to_eval) +'\n')

                            myfile.write('Train set\n')
                            labels_str = ''
                            for val in train_labels:
                                labels_str += '%11s' % val
                            myfile.write(labels_str +'\n')
                            # myfile.write('%11s' % 'elbo'+ '%11s' %'log_px' + '%11s' %'log_pz'+'%11s' %'log_qz'+ '%11s' %'log_pW'+'%11s' %'log_qW'+'\n')
                            results_str = ''
                            for val in train_results:
                                results_str += '%11.2f' % val
                            myfile.write(results_str +'\n')


                            myfile.write('Test set\n')
                            labels_str = ''
                            for val in test_labels:
                                labels_str += '%11s' % val
                            myfile.write(labels_str +'\n')

                            # myfile.write('%11s' % 'iwae_elbo'+ '%11s' %'log_px' + '%11s' %'log_pz'+'%11s' %'log_qz'+ '%11s' %'log_pW'+'%11s' %'log_qW'+'\n')
                            
                            results_str = ''
                            for val in test_results:
                                results_str += '%11.2f' % val
                            myfile.write(results_str +'\n')
                        


                print





    if save_log:             
        with open(experiment_log, "a") as myfile:
            myfile.write('\n\nAll Done.\n')
        
    print 'All Done'
































