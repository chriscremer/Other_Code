
import numpy as np
import argparse
import os
from os.path import expanduser
home = expanduser("~")
import pickle
from scipy.stats import multivariate_normal as norm
import time
import datetime

from autoencoders import VAE
# from IWAE import IWAE
# from autoencoders import IW_MoG_AE
from MoG_VAE import MoG_VAE
# from MoG_IWAE import MoG_IWAE

import tensorflow as tf


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

        if i %500 ==0:
            print i, '/', len(data)

        # Encode data
        input_ = np.reshape(data[i], [1,784])
        recog_means, recog_log_vars, weights = model.encode(input_)

        # if i %100 ==0:
        #     print weights

        #Recog_means [B,Z*C]
        #Recog_log_vars [B,Z*C]
        #weights [B,C]

        n_batch = len(weights)
        n_clusters = len(weights[0])
        n_z = len(recog_means[0]) / n_clusters

        # Get rid of the batch dimension
        recog_means = np.reshape(recog_means, [-1])
        recog_log_vars = np.reshape(recog_log_vars, [-1])
        weights = np.reshape(weights, [-1])

        recog_vars = np.exp(recog_log_vars)

        # [P]
        component = np.random.choice(n_clusters, size=[n_samples], p=weights)
        #Convert to one hot
        component_one_hot = np.zeros((n_samples, n_clusters))
        # [P,C]
        component_one_hot[np.arange(n_samples), component] = 1

        # [C,Z]
        recog_means = np.reshape(recog_means, [n_clusters, n_z])
        recog_vars = np.reshape(recog_vars, [n_clusters, n_z])
        # [P,Z]
        sample_means = np.dot(component_one_hot, recog_means)
        sample_vars = np.dot(component_one_hot, recog_vars)
        # [P,Z]
        sample_from_N0I = np.random.randn(n_samples,n_z)
        # [P,Z]
        samples = sample_from_N0I * np.sqrt(sample_vars) + sample_means
        
        #Prob under prior
        p_z_dist = norm(mean=np.zeros([n_z]))
        # [P]
        p_z = p_z_dist.pdf(samples)

        q_z_given_x = np.zeros([n_samples])
        for c in range(n_clusters):
            # dists.append(norm(mean=recog_means[c], cov=recog_vars[c]))
            q_i_dist = norm(mean=recog_means[c], cov=recog_vars[c])
            q_i = q_i_dist.pdf(samples)
            q_z_given_x += q_i * weights[c]

        # Reconstruct sample
        samples = np.reshape(samples, [n_samples, 1, n_z])
        x_mean_no_sigmoig = model.decode(samples)
        x_mean_no_sigmoig = np.reshape(x_mean_no_sigmoig, [n_samples, 784])



        # x_mean = np.array([.5, 10.**(-20), 1.-(10.**(-50)), 1.-(10.**(-10))])
        # print x_mean
        # x_mean = np.clip(x_mean, min_value, max_value)
        # print x_mean
        # print np.log(x_mean)

        # fsda


        # [P] 
        # log_p_x_given_z = np.sum(data[i]*np.log(x_mean) + ((1.-data[i])*np.log(1.-x_mean)), axis=1)

        log_p_x_given_z = -np.sum(np.maximum(x_mean_no_sigmoig, 0) 
                            - x_mean_no_sigmoig * data[i]
                            + np.log(1. + np.exp(-np.abs(x_mean_no_sigmoig))),
                             axis=1)



        if np.isnan(log_p_x_given_z).any():

            print np.sum(x_mean)
            print np.sum(log_p_x_given_z)
            print np.sum(data[i]*np.log(x_mean))
            print np.sum((1.-data[i])*np.log(1.-x_mean))

            # for val in 1.-x_mean:
            fsafd

        log_p_z = np.log(p_z)
        log_q_z_given_x = np.log(q_z_given_x)
        log_w = log_p_x_given_z + log_p_z - log_q_z_given_x



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


    #Define model
    f_height=28
    f_width=28
    n_batch = 10
    network_architecture = \
        dict(n_hidden_recog_1=200, # 1st layer encoder neurons
             n_hidden_recog_2=200, # 2nd layer encoder neurons
             n_hidden_gener_1=200, # 1st layer decoder neurons
             n_hidden_gener_2=200, # 2nd layer decoder neurons
             n_input=f_height*f_width, # 784 image
             n_z=50)  # dimensionality of latent space


    
    if args.action == 'train':

        #Initialize model
        if args.model == 'vae':
            model = VAE(network_architecture, learning_rate=0.001, batch_size=n_batch, n_particles=args.k)
        elif args.model == 'iwae':
            model = IWAE(network_architecture, learning_rate=0.001, batch_size=n_batch, n_particles=args.k)
        elif args.model == 'mog_vae':
            model = MoG_VAE(network_architecture, learning_rate=0.001, batch_size=n_batch, n_particles=args.k, 
                n_clusters=args.n_clusters)
        elif args.model == 'mog_iwae':
            model = MoG_IWAE(network_architecture, learning_rate=0.001, batch_size=n_batch, 
                n_particles=args.k, n_clusters=args.n_clusters)

        #Train
        model.train(train_x=train_x, valid_x=valid_x, display_step=9000, 
                    path_to_load_variables=path_to_load_variables, 
                    path_to_save_variables=path_to_save_variables, 
                    starting_stage=args.starting_stage, ending_stage=args.ending_stage, 
                    path_to_save_training_info='')



    elif args.action == 'evaluate':

        #Initialize model
        if args.model == 'vae':
            model = VAE(network_architecture, batch_size=1, n_particles=args.k)
        elif args.model == 'iwae':
            model = IWAE(network_architecture, batch_size=1, n_particles=args.k)
        elif args.model == 'mog_vae':
            model = MoG_VAE(network_architecture, batch_size=1, n_particles=args.k, 
                n_clusters=args.n_clusters)
        elif args.model == 'mog_iwae':
            model = MoG_IWAE(network_architecture, batch_size=1, n_particles=args.k, 
                n_clusters=args.n_clusters)

        #Load parameters
        model.load_parameters(path_to_load_variables=path_to_load_variables)

        #Log Likelihood Lower Bound
        LL_LB = evaluate(model, data=test_x, n_samples=args.k)
        print 'Model Log Likelihood is ' + str(LL_LB)




    elif args.action == 'combined':

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


        list_of_models = ['mog_vae'] #['mog_vae','mog_iwae'] #mog vae vs mog iwae
        list_of_n_clusters = [2,3,4,5,6]
        list_of_n_samples = [1,12,60] #[1,3,12,60]
        # list_of_donts = [[]]  #[c,k]


        for m in list_of_models:

            for c in list_of_n_clusters:

                for k in list_of_n_samples:

                    if k % c != 0:
                        print 'skipping c' +str(c) + '_k' + str(k)+ ' ' +m 
                        with open(experiment_log, "a") as myfile:
                            myfile.write('\nskipping c' +str(c) + '_k' + str(k)+ ' ' +m +'\n')

                        continue

                    saved_parameter_file = m + '_c' + str(c) + '_k' + str(k) + '_s5_3.ckpt' 
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
                        if m == 'mog_vae':
                            model = MoG_VAE(network_architecture, batch_size=n_batch, 
                                n_particles=k, n_clusters=c)
                        elif m == 'mog_iwae':
                            model = MoG_IWAE(network_architecture, batch_size=n_batch, 
                                n_particles=k, n_clusters=c)

                        start = time.time()

                        model.train(train_x=train_x, valid_x=valid_x, display_step=9000, 
                                    path_to_load_variables='', 
                                    path_to_save_variables=home+'/data/'+saved_parameter_file, 
                                    starting_stage=0, ending_stage=5, 
                                    path_to_save_training_info='')

                        print 'time to train', (time.time() - start)
                        with open(experiment_log, "a") as myfile:
                            myfile.write('time to train '+  str(time.time() - start) + '\n')


                    #Evaluate
                    k_for_this_model = k_evaluation
                    while k_for_this_model % c != 0:
                        k_for_this_model +=1
                    print 'k for this model ' + str(k_for_this_model)
                    with open(experiment_log, "a") as myfile:
                        myfile.write('k for this model ' + str(k_for_this_model) +'\n')

                    #Initialize model
                    if m == 'mog_vae':
                        model = MoG_VAE(network_architecture, batch_size=1, n_particles=k_for_this_model, 
                            n_clusters=c)
                    elif m == 'mog_iwae':
                        model = MoG_IWAE(network_architecture, batch_size=1, n_particles=k_for_this_model, 
                            n_clusters=c)


                    # time.sleep(10) # delays for 5 seconds
                    

                    #Load parameters
                    model.load_parameters(path_to_load_variables=home+'/data/'+saved_parameter_file)

                    start = time.time()

                    #Log Likelihood Lower Bound
                    LL_LB = evaluate(model, data=test_x, n_samples=k_for_this_model)
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














