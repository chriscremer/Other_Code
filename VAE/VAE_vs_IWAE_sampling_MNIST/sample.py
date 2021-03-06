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


from VAE_IWAE import VAE
from VAE_IWAE import IWAE

# import matplotlib.pyplot as plt
import scipy 

import numpy.random as npr





def load_binarized_mnist(location):

    with open(location, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    train_x = train_set
    valid_x = valid_set
    test_x = test_set

    return train_x, valid_x, test_x




# path_to_save_variables = ''
# path_to_load_variables = ''
# path_to_save_variables = home+'/data/'+args.save_to+'.ckpt'
# path_to_load_variables = home+'/data/'+args.load_from+'.ckpt'


data_location = home + '/Documents/MNIST_data/'
parameter_location = home + '/Documents/tmp/'

#Load data
train_x, valid_x, test_x = load_binarized_mnist(location=data_location+'binarized_mnist.pkl')
print 'Train', train_x.shape
print 'Valid', valid_x.shape
print 'Test', test_x.shape



# list_of_models = ['vae','iwae']  #mog vae vs mog iwae
list_of_models = ['iwae']  #mog vae vs mog iwae

# list_model_structures = range(4)
list_of_k_samples = [50]#[1,10,50] #[1,5,50] #[1,3,12,60]
# list_of_donts = [[]]  #[c,k]
sampling_type =  'vae'
# sampling_type = 'vae'


#Define model
f_height=28
f_width=28
n_batch = 100
epochs = 1000
batch_size = 5

n_a0 = \
    dict(n_input=f_height*f_width, # 784 image
         encoder_net=[200,200], 
         n_z=2,  # dimensionality of latent space
         decoder_net=[200,200]) 


    
list_of_archs = [n_a0]
list_of_archs_i = [0]


rs = npr.RandomState(0)

#Ramdomly select a batch
batch = []
while len(batch) != batch_size:
    datapoint = test_x[rs.randint(0,len(test_x))]
    batch.append(datapoint)


for k in list_of_k_samples:

    for m in list_of_models:

        for arch in list_of_archs_i:

            # saved_parameter_file = m + '_struc' + str(arch) + '_k' + str(k) + '_1000.ckpt' 
            # 60 means the train + validation set
            saved_parameter_file = m + '_latent' + str(n_a0['n_z']) + '_k' + str(k) + '_epochs'+str(epochs)+'.ckpt' 


            print 'Current:', saved_parameter_file

            #Initialize model
            # if m == 'vae':
            #     model = VAE(list_of_archs[arch], batch_size=batch_size, n_particles=k)
            # elif m == 'iwae':
            model = IWAE(list_of_archs[arch], batch_size=batch_size, n_particles=k)

            model.load_parameters(path_to_load_variables=parameter_location+saved_parameter_file)


            reconstructions, batch = model.reconstruct(sampling=sampling_type, data=batch)
            # print reconstructions.shape
            # print batch.shape

            # fadfas

            while len(reconstructions) < 10:
                reconstructions1, batch = model.reconstruct(sampling=sampling_type, data=batch)
                reconstructions = np.concatenate((reconstructions, reconstructions1), axis=0)



            for b_ in range(batch_size):
                for k_ in range(len(reconstructions)):

                    #only get 10 samps
                    if k_ < 10:

                        img = np.reshape(reconstructions[k_][b_], [28,28])

                        if k_ == 0:
                            real_img = np.reshape(batch[b_], [28,28])
                            space = np.zeros([28,28])
                            concat = np.concatenate((real_img, space), axis=1)
                            concat = np.concatenate((concat, img), axis=1)

                        else:
                            concat = np.concatenate((concat, img), axis=1)

                if b_ == 0:
                    concat2 = concat
                if b_ != 0:
                    concat2 = np.concatenate((concat2, concat), axis=0)

            # print k
            # print m
            # if k==50 and m=='vae':
            #     # print 'yess'
            #     concat3 = concat2

            # else:
            #     concat3 = np.concatenate((concat3, concat2), axis=0)

            # print concat3.shape



            # scipy.misc.imsave(home+'/data/recon_'+str(m)+'_k'+str(k)+'.png', concat2)
            # print 'saved ' + home+'/data/recon_'+str(m)+'_k'+str(k)+'.png'

<<<<<<< HEAD


# scipy.misc.imsave(home+'/data/recon_sampling'+str(sampling_type)+'.png', concat3)
# print 'saved ' + home+'/data/recon_'+str(sampling_type)+'.png'
=======
scipy.misc.imsave(parameter_location+'recon_sampling'+str(sampling_type)+'.png', concat3)
print 'saved ' + parameter_location+ 'recon_'+str(sampling_type)+'.png'
>>>>>>> b9f2d289c242753459930336a0b2afa44307fd0f





#change sampling type

sampling_type = 'iwae'


for k in list_of_k_samples:

    for m in list_of_models:

        for arch in list_of_archs_i:

            # saved_parameter_file = m + '_struc' + str(arch) + '_k' + str(k) + '_1000.ckpt' 
            # 60 means the train + validation set
            saved_parameter_file = m + '_latent' + str(n_a0['n_z']) + '_k' + str(k) + '_epochs'+str(epochs)+'.ckpt' 

            print 'Current:', saved_parameter_file

            #Initialize model
            if m == 'vae':
                model = VAE(list_of_archs[arch], batch_size=batch_size, n_particles=k)
            elif m == 'iwae':
                model = IWAE(list_of_archs[arch], batch_size=batch_size, n_particles=k)

            model.load_parameters(path_to_load_variables=home+'/data/'+saved_parameter_file)


            reconstructions, batch = model.reconstruct(sampling=sampling_type, data=batch)
            # print reconstructions.shape
            while len(reconstructions) < 10:
                reconstructions1, batch = model.reconstruct(sampling=sampling_type, data=batch)
                reconstructions = np.concatenate((reconstructions, reconstructions1), axis=0)



            for b_ in range(batch_size):
                for k_ in range(len(reconstructions)):

                    #only get 10 samps
                    if k_ < 10:

                        img = np.reshape(reconstructions[k_][b_], [28,28])

                        if k_ == 0:
                            # real_img = np.reshape(batch[b_], [28,28])
                            # space = np.zeros([28,28])
                            # concat = np.concatenate((real_img, space), axis=1)
                            # concat = np.concatenate((concat, img), axis=1)
                            concat = img

                        else:
                            concat = np.concatenate((concat, img), axis=1)

                if b_ == 0:
                    concat2_2 = concat
                if b_ != 0:
                    concat2_2 = np.concatenate((concat2_2, concat), axis=0)

            # print k
            # print m
            # if k==50 and m=='vae':
            #     # print 'yess'
            #     concat3 = concat2

            # else:
            #     concat3 = np.concatenate((concat3, concat2), axis=0)

            # print concat3.shape



            # scipy.misc.imsave(home+'/data/recon_'+str(m)+'_k'+str(k)+'.png', concat2)
            # print 'saved ' + home+'/data/recon_'+str(m)+'_k'+str(k)+'.png'


zeros = np.zeros([batch_size*28, 28])
print zeros.shape
print concat2.shape
print concat2_2.shape
concat3 = np.concatenate((concat2, zeros), axis=1)
concat3 = np.concatenate((concat3, concat2_2), axis=1)


scipy.misc.imsave(home+'/data/recon_sampling'+str(sampling_type)+'.png', concat3)
print 'saved ' + home+'/data/recon_'+str(sampling_type)+'.png'



scipy.misc.imsave(home+'/data/recon_sampling'+str(sampling_type)+'.eps', concat3)
print 'saved ' + home+'/data/recon_'+str(sampling_type)+'.eps'




