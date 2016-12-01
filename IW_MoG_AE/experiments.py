

import numpy as np
import argparse
from os.path import expanduser
home = expanduser("~")
import pickle
from scipy.stats import multivariate_normal as norm
# from scipy.stats import bernoulli

# from autoencoders import VAE
# from autoencoders import IWAE
# from autoencoders import IW_MoG_AE
from MoG_VAE import MoG_VAE
from MoG_IWAE import MoG_IWAE




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


def prob_under_MoG(sample, means, log_vars, weights):

    n_clusters = len(weights)
    n_z = len(means) / n_clusters

    sum_ = 0
    for i in range(n_clusters):

        sum_ += weights[i] * norm.pdf(sample, mean=means[i*n_z:i*n_z+n_z:1], 
                    cov=np.diag(np.exp(log_vars[i*n_z:i*n_z+n_z:1])))

    return sum_


def evaluate(model, data, n_samples):

    iwae_elbos = []

    # For each datapoint
    for i in range(len(data)):

        if i %2 ==0:
            print i, '/', len(data)

        # Encode data
        input_ = np.reshape(data[i], [1,784])
        recog_means, recog_log_vars, weights = model.encode(input_)

        if i %50 ==0:
            print weights

        #Get rid of the batch dimension
        recog_means = np.reshape(recog_means, [-1])
        recog_log_vars = np.reshape(recog_log_vars, [-1])
        weights = np.reshape(weights, [-1])

        n_clusters = len(weights)
        n_z = len(recog_means) / n_clusters

        # log_weights = np.log(weights)
        
        ws = []
        # For number of samples
        for j in range(n_samples):

            # Sample the latent space: first select which component
            component = np.random.choice(n_clusters, size=1, p=weights)[0]

            # Sample that component, which is a diagonal Gaussian
            sample = (np.random.randn(n_z) * 
                np.sqrt(np.exp(recog_log_vars[component*n_z:component*n_z+n_z:1])) 
                + recog_means[component*n_z:component*n_z+n_z:1]) 

            # Get probability of sample under recognition model and prior
            p_z = norm.pdf(sample, mean=np.zeros([n_z]), cov=np.diag(np.ones([n_z])))
            q_z_given_x = prob_under_MoG(sample, recog_means, recog_log_vars, weights)

            # Reconstruct sample
            x_mean = model.decode([[sample],[sample]])[0]
            x_mean = np.reshape(x_mean, [784])

            # Get probability of reconstruction
            # p_x_given_z = bernoulli.pmf(x=data[i], p=x_mean)
            log_p_x_given_z = np.sum(data[i]*np.log(x_mean) + (1-data[i])*np.log(1-x_mean))

            log_p_z = np.log(p_z)
            log_q_z_given_x = np.log(q_z_given_x)

            # Compute w
            # w = p_x_given_z * p_z / q_z_given_x
            log_w = log_p_x_given_z + log_p_z - log_q_z_given_x

            if np.isnan(log_w):
                print log_p_x_given_z
                print log_p_z
                print log_q_z_given_x
                fsafd
            # w = np.exp(log_w)

            ws.append(log_w)

        # Calc IWAE ELBO over all samples

        # iwae_elbo = np.log(np.mean(ws))
        ws = np.array(ws)
        max_ = np.max(ws)
        iwae_elbo = np.log(np.mean(np.exp(ws-max_))) + max_
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
    parser.add_argument('--action', '-a', choices=['train', 'evaluate'], default='train')
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


    #Define model
    f_height=28
    f_width=28
    n_batch = 20
    k = args.k
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
            model = VAE(network_architecture, learning_rate=0.001, batch_size=n_batch, n_particles=k)
        elif args.model == 'iwae':
            model = IWAE(network_architecture, learning_rate=0.001, batch_size=n_batch, n_particles=k)
        elif args.model == 'mog_vae':
            model = MoG_VAE(network_architecture, learning_rate=0.001, batch_size=n_batch, n_particles=k, 
                n_clusters=args.n_clusters)
        elif args.model == 'mog_iwae':
            model = MoG_IWAE(network_architecture, learning_rate=0.001, batch_size=n_batch, n_particles=k, 
                n_clusters=args.n_clusters)


        #Train
        model.train(train_x=train_x, valid_x=valid_x, display_step=2000, 
                    path_to_load_variables=path_to_load_variables, 
                    path_to_save_variables=path_to_save_variables, 
                    starting_stage=args.starting_stage, ending_stage=args.ending_stage, 
                    path_to_save_training_info='')



    elif args.action == 'evaluate':

        #Initialize model
        if args.model == 'vae':
            model = VAE(network_architecture, batch_size=1, n_particles=2)
        elif args.model == 'iwae':
            model = IWAE(network_architecture, batch_size=1, n_particles=2)
        elif args.model == 'mog_vae':
            model = MoG_VAE(network_architecture, batch_size=1, n_particles=2, 
                n_clusters=args.n_clusters)
        elif args.model == 'mog_iwae':
            model = MoG_IWAE(network_architecture, batch_size=1, n_particles=2, 
                n_clusters=args.n_clusters)

        #Load parameters
        model.load_parameters(path_to_load_variables=path_to_load_variables)

        #Log Likelihood Lower Bound
        LL_LB = evaluate(model, data=test_x, n_samples=1000)
        print 'Model Log Likelihood is ' + str(LL_LB)




























