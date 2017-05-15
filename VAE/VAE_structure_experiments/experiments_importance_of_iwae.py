
import numpy as np
import tensorflow as tf
import argparse
import os
from os.path import expanduser
home = expanduser("~")
import pickle
from scipy.stats import multivariate_normal as norm
import time
import datetime

from IWAE import VAE
from IWAE import IWAE





def load_binarized_mnist(location):

    with open(location, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    train_x = train_set
    valid_x = valid_set
    test_x = test_set

    return train_x, valid_x, test_x


def user_defined_locations(args):

    if args.save_to != '':
        path_to_save_variables = home+'/data/'+args.save_to+'.ckpt'
    else:
        path_to_save_variables = ''

    if args.load_from != '':
        path_to_load_variables = home+'/data/'+args.load_from+'.ckpt'
    else:
        path_to_load_variables = ''

    return path_to_save_variables, path_to_load_variables



def evaluate(model, data, n_samples):

    iwae_elbos = []

    min_value= 10.**(-50)
    max_value= 1.-(10.**(-10)) # this is the largest it can be until it becomes 1

    # For each datapoint
    for i in range(len(data)):

        if i %100 ==0:
            print i, '/', len(data)

        # Encode data
        input_ = np.reshape(data[i], [1,784])
        # [1,Z]
        recog_means, recog_log_vars = model.encode(input_)
        recog_means = np.reshape(recog_means, [-1])
        recog_log_vars = np.reshape(recog_log_vars, [-1])
        recog_vars = np.exp(recog_log_vars)

        n_z = len(recog_means)
        
        # [P,Z]
        sample_from_N0I = np.random.randn(n_samples,n_z)
        # [P,Z]
        samples = sample_from_N0I * np.sqrt(recog_vars) + recog_means
        
        #Prob under prior
        p_z_dist = norm(mean=np.zeros([n_z]))
        # [P]
        p_z = p_z_dist.pdf(samples)

        q_i_dist = norm(mean=recog_means, cov=recog_vars)
        q_z_given_x = q_i_dist.pdf(samples)
        # q_z_given_x += q_i * weights[c]

        # Reconstruct sample
        samples = np.reshape(samples, [n_samples, 1, n_z])
        x_mean_no_sigmoig = model.decode(samples)
        x_mean_no_sigmoig = np.reshape(x_mean_no_sigmoig, [n_samples, 784])

        # [P] 
        log_p_x_given_z = -np.sum(np.maximum(x_mean_no_sigmoig, 0) 
                            - x_mean_no_sigmoig * data[i]
                            + np.log(1. + np.exp(-np.abs(x_mean_no_sigmoig))),
                             axis=1)

        if np.isnan(log_p_x_given_z).any():

            print np.sum(x_mean)
            print np.sum(log_p_x_given_z)
            print np.sum(data[i]*np.log(x_mean))
            print np.sum((1.-data[i])*np.log(1.-x_mean))
            fsafd

        log_p_z = np.log(p_z)
        log_q_z_given_x = np.log(q_z_given_x)
        log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
        # print np.mean(log_p_x_given_z), np.mean(log_p_z), np.mean(log_q_z_given_x)
        # fasdf


        max_ = np.max(log_w)
        iwae_elbo = np.log(np.mean(np.exp(log_w-max_))) + max_
        iwae_elbos.append(iwae_elbo)

    # Average IWAE ELBOs over all datapoints
    L = np.mean(iwae_elbos)

    return L



if __name__ == '__main__':
    #Examples: 
    # python experiments.py -m vae -k 1 -a train -s vae_1
    # python experiments.py -m vae -k 10 -a train -s vae_10_s6 -l vae_10 -ss 6 -es 6
    # python experiments.py -m vae -k 10 -a evaluate -l vae_10
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--model', '-m', choices=['vae', 'iwae', 'mog_vae', 'mog_iwae'], 
        default='vae')
    parser.add_argument('--k', '-k', type=int, default=1)
    parser.add_argument('--action', '-a', choices=['train', 'evaluate', 'combined'], default='train')
    parser.add_argument('--save_to', '-s', type=str, default='')
    parser.add_argument('--load_from', '-l', type=str, default='')
    parser.add_argument('--starting_stage', '-ss', type=int, default=0)
    parser.add_argument('--ending_stage', '-es', type=int, default=5)
    parser.add_argument('--n_clusters', '-c', type=int, default=2)
    args = parser.parse_args()

    path_to_save_variables, path_to_load_variables = user_defined_locations(args)

    #Load data
    train_x, valid_x, test_x = load_binarized_mnist(location=home+'/data/binarized_mnist.pkl')
    print 'Train', train_x.shape
    print 'Valid', valid_x.shape
    print 'Test', test_x.shape


    # Combine training and validation, maybe theyre supplementary has a typo
    # train_x = np.concatenate([train_x, valid_x], axis=0)
    
    # if args.action == 'train':

    #     #Initialize model
    #     if args.model == 'vae':
    #         model = VAE(network_architecture, learning_rate=0.001, batch_size=n_batch, n_particles=args.k)
    #     elif args.model == 'iwae':
    #         model = IWAE(network_architecture, learning_rate=0.001, batch_size=n_batch, n_particles=args.k)
    #     elif args.model == 'mog_vae':
    #         model = MoG_VAE(network_architecture, learning_rate=0.001, batch_size=n_batch, n_particles=args.k, 
    #             n_clusters=args.n_clusters)
    #     elif args.model == 'mog_iwae':
    #         model = MoG_IWAE(network_architecture, learning_rate=0.001, batch_size=n_batch, 
    #             n_particles=args.k, n_clusters=args.n_clusters)

    #     #Train
    #     model.train(train_x=train_x, valid_x=valid_x, display_step=9000, 
    #                 path_to_load_variables=path_to_load_variables, 
    #                 path_to_save_variables=path_to_save_variables, 
    #                 starting_stage=args.starting_stage, ending_stage=args.ending_stage, 
    #                 path_to_save_training_info='')



    # elif args.action == 'evaluate':

    #     #Initialize model
    #     if args.model == 'vae':
    #         model = VAE(network_architecture, batch_size=1, n_particles=args.k)
    #     elif args.model == 'iwae':
    #         model = IWAE(network_architecture, batch_size=1, n_particles=args.k)
    #     elif args.model == 'mog_vae':
    #         model = MoG_VAE(network_architecture, batch_size=1, n_particles=args.k, 
    #             n_clusters=args.n_clusters)
    #     elif args.model == 'mog_iwae':
    #         model = MoG_IWAE(network_architecture, batch_size=1, n_particles=args.k, 
    #             n_clusters=args.n_clusters)

    #     #Load parameters
    #     model.load_parameters(path_to_load_variables=path_to_load_variables)

    #     #Log Likelihood Lower Bound
    #     LL_LB = evaluate(model, data=test_x, n_samples=args.k)
    #     print 'Model Log Likelihood is ' + str(LL_LB)




    # if args.action == 'combined':
    if 1:


        # Check for the model variables, if not there train it then eval. 
        #Write results to a file in case something happens
        #Also have a timer going to say how long each took. 


        k_evaluation = 5000

        dt = datetime.datetime.now()
        date_ = str(dt.date())
        time_ = str(dt.time())
        time_2 = time_[0] + time_[1] + time_[3] + time_[4] + time_[6] + time_[7] 

        experiment_log = home+'/data/experiment_' + date_ + '_' +time_2 +'.txt'

        with open(experiment_log, "a") as myfile:
            myfile.write("Evaluation k=" +str(k_evaluation) +'\n')

        print 'saving experiment log to ' +experiment_log


        # list_of_models = ['iwae', 'vae']  #mog vae vs mog iwae
        list_of_models = ['iwae']#, 'vae']  #mog vae vs mog iwae

        # list_model_structures = range(4)
        list_of_k_samples = [50,10,1] #[1,5,50] #[1,3,12,60]
        # list_of_donts = [[]]  #[c,k]

        #Define model
        f_height=28
        f_width=28
        n_batch = 100
        epochs = 500

        n_a0 = \
            dict(n_input=f_height*f_width, # 784 image
                 encoder_net=[200,200], 
                 n_z=10,  # dimensionality of latent space
                 decoder_net=[200,200]) 

        n_a1 = \
            dict(n_input=f_height*f_width, # 784 image
                 encoder_net=[200,200], 
                 n_z=50,  # dimensionality of latent space
                 decoder_net=[200,200]) 

        n_a2 = \
            dict(n_input=f_height*f_width, # 784 image
                 encoder_net=[200,200], 
                 n_z=100,  # dimensionality of latent space
                 decoder_net=[200,200]) 

        # n_a3 = \
        #     dict(n_input=f_height*f_width, # 784 image
        #          encoder_net=[400,400,400], 
        #          n_z=50,  # dimensionality of latent space
        #          decoder_net=[400,400,400]) 

        list_of_archs = [n_a0, n_a1, n_a2]
        list_of_archs_i = [0,1,2]
                 
        for k in list_of_k_samples:

            for m in list_of_models:

                for arch in list_of_archs_i:

                    saved_parameter_file = m + '_struc' + str(arch) + '_k' + str(k) + '_1000.ckpt' 
                    # 60 means the train + validation set

                    print 'Current:', saved_parameter_file
                    with open(experiment_log, "a") as myfile:
                        myfile.write('\nCurrent:' + saved_parameter_file +'\n')

                    # model = None

                    #See if its already been trained
                    if saved_parameter_file in os.listdir(home+'/data'):
                        print 'Already trained'
                        with open(experiment_log, "a") as myfile:
                            myfile.write('Already trained\n')
                    else:
                        print 'Not trained yet'
                        with open(experiment_log, "a") as myfile:
                            myfile.write('Not trained yet\n')

                        #Train 
                        if m == 'vae':
                            model = VAE(list_of_archs[arch], batch_size=n_batch, n_particles=k)
                        elif m == 'iwae':
                            model = IWAE(list_of_archs[arch], batch_size=n_batch, n_particles=k)

                        start = time.time()

                        # model.train(train_x=train_x, valid_x=valid_x, display_step=9000, 
                        #             path_to_load_variables='', 
                        #             path_to_save_variables=home+'/data/'+saved_parameter_file, 
                        #             starting_stage=0, ending_stage=4, 
                        #             path_to_save_training_info='')

                        model.train(train_x=train_x, valid_x=valid_x, display_step=9000, 
                                    path_to_load_variables='', 
                                    path_to_save_variables=home+'/data/'+saved_parameter_file, 
                                    epochs=epochs)
                        print 'time to train', (time.time() - start)
                        with open(experiment_log, "a") as myfile:
                            myfile.write('time to train '+  str(time.time() - start) + '\n')


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
                    LL_LB = evaluate(model, data=test_x, n_samples=k_evaluation)
                    print 'Model Log Likelihood is ' + str(LL_LB) + ' for ' + saved_parameter_file
                    with open(experiment_log, "a") as myfile:
                        myfile.write('Model Log Likelihood is ' + str(LL_LB) + ' for ' + saved_parameter_file 
                            +'\n')

                    print 'time to evaluate', (time.time() - start)
                    with open(experiment_log, "a") as myfile:
                        myfile.write('time to evaluate '+  str(time.time() - start) +'\n\n')
                    print 

                    # with open(experiment_log, "a") as myfile:
                    #     myfile.write('\n')







    print 'All Done'














