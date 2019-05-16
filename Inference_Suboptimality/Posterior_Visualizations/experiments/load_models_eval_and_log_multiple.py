



import sys, os
sys.path.insert(0, '../models')
sys.path.insert(0, '../models/utils')

import time
import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")
import matplotlib.pyplot as plt

import gzip

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from ais3 import test_ais

from pytorch_vae_v5 import VAE

from approx_posteriors_v5 import standard
from approx_posteriors_v5 import flow1
from approx_posteriors_v5 import aux_nf
from approx_posteriors_v5 import hnf

from approx_posteriors_v5 import standard_layernorm

from approx_posteriors_v6 import FFG_LN
from approx_posteriors_v6 import ANF_LN
from pytorch_vae_v6 import VAE

import argparse


# directory = home+'/Documents/tmp/first_try'
# directory = home+'/Documents/tmp/small_N'
# directory = home+'/Documents/tmp/large_N'
# directory = home+'/Documents/tmp/large_N_time_2'

# directory = home+'/Documents/tmp/large_N_2'

# directory = home+'/Documents/tmp/new_training'

# directory = home+'/Documents/tmp/new_training_2'

# directory = home+'/Documents/tmp/fashion_2'

# directory = home+'/Documents/tmp/batch20_correct'


directory = home+'/Documents/tmp/fashion_1'





# directory = home+'/Documents/tmp/2D_models'





#since theres too moany checkpoints select which ones to eval:
# checkpoints = [1000,1900,2800]


# checkpoints = [100,400,700,1000]

checkpoints = [400,1300,2500]#,3100]


# checkpoints = [1,2,3,4,5]


# checkpoints = [1000,2000,3000]


#Steps:
# take as arg, which model you want to eval
# 

# models = ['standard', 'flow1', 'aux_nf']#, 'hnf']

# models = ['standard', 'standard_large_encoder']#, 'aux_nf', 'aux_large_encoder']#, 'hnf']
# models = ['standard_large_encoder']#, 'aux_nf', 'aux_large_encoder']#, 'hnf']


# models = ['hnf']

# models = ['standard', 'aux_nf']
models = ['FFG']

# models = ['standard']









def test_prior(model, data_x, batch_size, display, k):
    
    time_ = time.time()
    elbos = []
    data_index= 0
    for i in range(int(len(data_x)/ batch_size)):

        batch = data_x[data_index:data_index+batch_size]
        data_index += batch_size

        batch = Variable(torch.from_numpy(batch)).type(model.dtype)

        elbo = model.forward3_prior(batch, k=k)

        elbos.append(elbo.data[0])

        if i%display==0:
            print (i,len(data_x)/ batch_size, np.mean(elbos))

    mean_ = np.mean(elbos)
    print(mean_, 'T:', time.time()-time_)

    return mean_#, time.time()-time_






def test_vae(model, data_x, batch_size, display, k):
    
    time_ = time.time()
    elbos = []
    data_index= 0
    for i in range(int(len(data_x)/ batch_size)):

        batch = data_x[data_index:data_index+batch_size]
        data_index += batch_size

        batch = Variable(torch.from_numpy(batch)).type(model.dtype)

        elbo, logpxz, logqz = model.forward2(batch, k=k)

        elbos.append(elbo.data[0])

        if i%display==0:
            print (i,len(data_x)/ batch_size, np.mean(elbos))

    mean_ = np.mean(elbos)
    print(mean_, 'T:', time.time()-time_)

    return mean_#, time.time()-time_







def test(model, data_x, batch_size, display, k):
    
    time_ = time.time()
    elbos = []
    data_index= 0
    for i in range(int(len(data_x)/ batch_size)):

        batch = data_x[data_index:data_index+batch_size]
        data_index += batch_size

        batch = Variable(torch.from_numpy(batch)).type(model.dtype)

        elbo, logpxz, logqz = model(batch, k=k)

        elbos.append(elbo.data[0])

        if i%display==0:
            print (i,len(data_x)/ batch_size, np.mean(elbos))

    mean_ = np.mean(elbos)
    print(mean_, 'T:', time.time()-time_)

    return mean_#, time.time()-time_







#fashion


def load_mnist(path, kind='train'):

    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(-1, 784)

    return images#, labels


path = home+'/Documents/fashion_MNIST'

train_x = load_mnist(path=path)
test_x = load_mnist(path=path, kind='t10k')

train_x = train_x / 255.
test_x = test_x / 255.
train_x = train_x[:1000]
test_x = test_x[:1000]

print (train_x.shape)
print (test_x.shape)












# print ('Loading data')
# with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
#     mnist_data = pickle.load(f, encoding='latin1')
# train_x = mnist_data[0][0]
# valid_x = mnist_data[1][0]
# test_x = mnist_data[2][0]
# train_x = np.concatenate([train_x, valid_x], axis=0)
# print (train_x.shape)

# # #For debug purposes
# # train_x = train_x[:1000]
# # test_x = test_x[:500]

# print (train_x.shape)
# print (test_x.shape)

# train_x = train_x[:1000]
# test_x = test_x[:1000]



# print (train_x.shape)
# print (test_x.shape)




















x_size = 784
z_size = 50
# z_size = 2



eval_file = '/log_eval_only1000datapoints.txt'


# parser = argparse.ArgumentParser()
# parser.add_argument("-m",'--model', default="standard")
# args = parser.parse_args()


for model_ in models:




    if model_ == 'standard':

        this_dir = directory+'/standard'
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
            print ('Made directory:'+this_dir)


        experiment_log = this_dir+eval_file

        with open(experiment_log, "a") as myfile:
            myfile.write("Standard" +'\n')
        


        print('Init standard model')
        
        hyper_config = { 
                        'x_size': x_size,
                        'z_size': z_size,
                        'act_func': F.tanh,# F.relu,
                        'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
                        'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                        'q_dist': standard_layernorm # standard,#hnf,#aux_nf,#flow1,#,
                    }

        model = VAE(hyper_config)



    elif model_ == 'flow1':

        this_dir = directory+'/flow1'
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
            print ('Made directory:'+this_dir)

        experiment_log = this_dir+eval_file

        with open(experiment_log, "a") as myfile:
            myfile.write("Flow1" +'\n')

        print('Init flow model')
        hyper_config = { 
                        'x_size': x_size,
                        'z_size': z_size,
                        'act_func': F.tanh,# F.relu,
                        'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
                        'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                        'q_dist': flow1,#hnf,#aux_nf,#,#standard,#, #
                        'n_flows': 2,
                        'flow_hidden_size': 100
                    }

        model = VAE(hyper_config)



    elif model_ == 'aux_nf':

        this_dir = directory+'/aux_nf'
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
            print ('Made directory:'+this_dir)

        experiment_log = this_dir+eval_file

        with open(experiment_log, "a") as myfile:
            myfile.write("aux_nf" +'\n')

        print('Init aux nf model')
        hyper_config = { 
                        'x_size': x_size,
                        'z_size': z_size,
                        'act_func': F.tanh,# F.relu,
                        'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
                        'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                        'q_dist': aux_nf,#aux_nf,#flow1,#standard,#, #, #, #,#, #,# ,
                        'n_flows': 2,
                        'qv_arch': [[x_size,200],[200,200],[200,z_size*2]],
                        'qz_arch': [[x_size+z_size,200],[200,200],[200,z_size*2]],
                        'rv_arch': [[x_size+z_size,200],[200,200],[200,z_size*2]],
                        'flow_hidden_size': 100
                    }

        model = VAE(hyper_config)



    elif model_ == 'hnf':

        this_dir = directory+'/hnf'
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
            print ('Made directory:'+this_dir)

        experiment_log = this_dir+eval_file

        with open(experiment_log, "a") as myfile:
            myfile.write("hnf" +'\n')

        print('Init hnf model')
        hyper_config = { 
                        'x_size': x_size,
                        'z_size': z_size,
                        'act_func': F.tanh,# F.relu,
                        'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
                        'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                        'q_dist': hnf,#aux_nf,#flow1,#standard,#, #, #, #,#, #,# ,
                        'n_flows': 2,
                        'qv_arch': [[x_size,200],[200,200],[200,z_size*2]],
                        'qz_arch': [[x_size+z_size,200],[200,200],[200,z_size*2]],
                        'rv_arch': [[x_size+z_size,200],[200,200],[200,z_size*2]],
                        'flow_hidden_size': 100
                    }

        model = VAE(hyper_config)





    elif model_ == 'standard_large_encoder':

        this_dir = directory+'/standard_large_encoder'
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
            print ('Made directory:'+this_dir)


        experiment_log = this_dir+eval_file

        with open(experiment_log, "a") as myfile:
            myfile.write("Standard LE" +'\n')
        


        print('Init standard LE model')
            
        hyper_config = { 
                        'x_size': x_size,
                        'z_size': z_size,
                        'act_func': F.tanh,# F.relu,
                        'encoder_arch': [[x_size,500],[500,500],[500,200],[200,z_size*2]],
                        'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                        'q_dist': standard,#hnf,#aux_nf,#flow1,#,
                    }


        # model = VAE(hyper_config)



    elif model_ == 'aux_large_encoder':

        this_dir = directory+'/aux_large_encoder'
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
            print ('Made directory:'+this_dir)


        experiment_log = this_dir+eval_file

        with open(experiment_log, "a") as myfile:
            myfile.write("aux nf LE" +'\n')
        


        print('Aux LE')
        hyper_config = { 
                        'x_size': x_size,
                        'z_size': z_size,
                        'act_func': F.tanh,# F.relu,
                        'encoder_arch': [[x_size,500],[500,500],[500,200],[200,z_size*2]],
                        'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                        'q_dist': aux_nf,#aux_nf,#flow1,#standard,#, #, #, #,#, #,# ,
                        'n_flows': 2,
                        'qv_arch': [[x_size,200],[200,200],[200,z_size*2]],
                        'qz_arch': [[x_size+z_size,200],[200,200],[200,z_size*2]],
                        'rv_arch': [[x_size+z_size,200],[200,200],[200,z_size*2]],
                        'flow_hidden_size': 100
                    }


        # model = VAE(hyper_config)


    elif model_ == 'FFG':

        this_dir = directory+'/FFG'
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
            print ('Made directory:'+this_dir)


        experiment_log = this_dir+eval_file
        with open(experiment_log, "a") as myfile:
            myfile.write("ffg " +'\n')
        

        print('Init FFG model')
        
        # hyper_config = { 
        #                 'x_size': x_size,
        #                 'z_size': z_size,
        #                 'act_func': F.elu,#F.tanh,# F.relu,
        #                 'encoder_arch': [[x_size,200],[200,200],[200,200],[200,z_size*2]],
        #                 'decoder_arch': [[z_size,200],[200,200],[200,200],[200,x_size]],
        #                 'q_dist': FFG_LN#standard,#hnf,#aux_nf,#flow1,#,
        #             }



        hyper_config = { 
                        'x_size': x_size,
                        'z_size': z_size,
                        'act_func': F.tanh,#F.elu,#,# F.relu,
                        'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
                        'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                        'q_dist': FFG_LN#standard,#hnf,#aux_nf,#flow1,#,
                    }






    elif model_ == 'FFG_LE':

        this_dir = directory+'/FFG'
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
            print ('Made directory:'+this_dir)

        experiment_log = this_dir+eval_file
        with open(experiment_log, "a") as myfile:
            myfile.write("ffg LE" +'\n')
        

        print('Init FFG model')
        
        hyper_config = { 
                        'x_size': x_size,
                        'z_size': z_size,
                        'act_func': F.elu,#F.tanh,# F.relu,
                        'encoder_arch': [[x_size,500],[500,500],[500,500],[500,z_size*2]],
                        'decoder_arch': [[z_size,200],[200,200],[200,200],[200,x_size]],
                        'q_dist': FFG_LN#standard,#hnf,#aux_nf,#flow1,#,
                    }





    else:
        print ('What')
        print (model_)
        fadas






    # #Train params
    # learning_rate = .0001
    # batch_size = 100
    # k = 1
    # epochs = 3000

    # #save params and compute IW and AIS
    # start_at = 100
    # save_freq = 300
    # display_epoch = 10


    # Test params
    # k_IW = 5000
    k_IW = 50

    batch_size_IW = 20

    k_AIS = 100
    batch_size_AIS = 100
    n_intermediate_dists = 500


    # print('\nTraining')
    # print('k='+str(k), 'lr='+str(learning_rate), 'batch_size='+str(batch_size))
    # print('\nModel:', hyper_config,'\n')


    # with open(experiment_log, "a") as myfile:
    #     myfile.write(str(hyper_config)+'\n\n')
    # with open(experiment_log, "a") as myfile:
    #     myfile.write('k='+str(k)+' lr='+str(learning_rate)+' batch_size='+str(batch_size) +'\n\n')



    model = VAE(hyper_config)


    # path_to_load_variables=''
    # path_to_load_variables=home+'/Documents/tmp/pytorch_bvae.pt'
    # path_to_save_variables=home+'/Documents/tmp/pytorch_vae'+str(epochs)+'.pt'
    path_to_save_variables=this_dir+'/params_'+model_+'_'

    # path_to_save_variables=''


    if torch.cuda.is_available():
        model.cuda()














    prior_train_list = []
    prior_test_list = []
    vae_train_list = []
    vae_test_list = []
    iw_train_list = []
    iw_test_list = []
    ais_train_list = []
    ais_test_list = []

    for ckt in checkpoints:



        #find model
        this_ckt_file = path_to_save_variables + str(ckt) + '.pt'

        #load model
        model.load_params(path_to_load_variables=this_ckt_file)


        # log it
        with open(experiment_log, "a") as myfile:
            myfile.write('Checkpoint' +str(ckt)+'\n')

        start_time = time.time()

        k_prior = 50000
        batch_prior =1

        # compute LL 
        # print('\nTesting with prior, Train set, B'+str(batch_prior)+' k'+str(k_prior))
        # vae_train = test_prior(model=model, data_x=train_x, batch_size=batch_prior, display=10, k=k_prior)
        # print ('prior_train', vae_train)
        # with open(experiment_log, "a") as myfile:
        #     myfile.write('prior_train '+ str(vae_train) +'\n')
        #     myfile.write('time'+str(time.time()-start_time)+'\n\n')
        # prior_train_list.append(vae_train)
                    
        # print('\nTesting with prior, Test set, B'+str(batch_prior)+' k'+str(k_prior))
        # vae_test = test_prior(model=model, data_x=test_x, batch_size=batch_prior, display=10, k=k_prior)
        # print ('prior_test', vae_test)
        # with open(experiment_log, "a") as myfile:
        #     myfile.write('prior_test '+ str(vae_test) +'\n')
        #     myfile.write('time'+str(time.time()-start_time)+'\n\n')
        # prior_test_list.append(vae_test)



        print('\nTesting with VAE, Train set, B'+str(batch_size_IW)+' k'+str(k_IW))
        vae_train = test_vae(model=model, data_x=train_x, batch_size=batch_size_IW, display=10, k=k_IW)
        print ('vae_train', vae_train)
        with open(experiment_log, "a") as myfile:
            myfile.write('vae_train '+ str(vae_train) +'\n')
            myfile.write('time'+str(time.time()-start_time)+'\n\n')
        vae_train_list.append(vae_train)
                    
        print('\nTesting with VAE, Test set, B'+str(batch_size_IW)+' k'+str(k_IW))
        vae_test = test_vae(model=model, data_x=test_x, batch_size=batch_size_IW, display=10, k=k_IW)
        print ('vae_test', vae_test)
        with open(experiment_log, "a") as myfile:
            myfile.write('vae_test '+ str(vae_test) +'\n')
            myfile.write('time'+str(time.time()-start_time)+'\n\n')
        vae_test_list.append(vae_test)



        print('\nTesting with IW, Train set, B'+str(batch_size_IW)+' k'+str(k_IW))
        IW_train = test(model=model, data_x=train_x, batch_size=batch_size_IW, display=10, k=k_IW)
        print ('IW_train', IW_train)
        with open(experiment_log, "a") as myfile:
            myfile.write('IW_train '+ str(IW_train) +'\n')
            myfile.write('time'+str(time.time()-start_time)+'\n\n')
        iw_train_list.append(IW_train)
                    
        print('\nTesting with IW, Test set, B'+str(batch_size_IW)+' k'+str(k_IW))
        IW_test = test(model=model, data_x=test_x, batch_size=batch_size_IW, display=10, k=k_IW)
        print ('IW_test', IW_test)
        with open(experiment_log, "a") as myfile:
            myfile.write('IW_test '+ str(IW_test) +'\n')
            myfile.write('time'+str(time.time()-start_time)+'\n\n')
        iw_test_list.append(IW_test)


        #uncomment this next time

        # print('\nTesting with AIS, Train set, B'+str(batch_size_AIS)+' k'+str(k_AIS)+' intermediates'+str(n_intermediate_dists))
        # AIS_train = test_ais(model=model, data_x=train_x, batch_size=batch_size_AIS, display=2, k=k_AIS, n_intermediate_dists=n_intermediate_dists)
        # print ('AIS_train', AIS_train)
        # with open(experiment_log, "a") as myfile:
        #     myfile.write('AIS_train '+ str(AIS_train) +'\n')
        #     myfile.write('time'+str(time.time()-start_time)+'\n\n')
        # ais_train_list.append(AIS_train)


        # print('\nTesting with AIS, Test set, B'+str(batch_size_AIS)+' k'+str(k_AIS)+' intermediates'+str(n_intermediate_dists))
        # AIS_test = test_ais(model=model, data_x=test_x, batch_size=batch_size_AIS, display=2, k=k_AIS, n_intermediate_dists=n_intermediate_dists)
        # print ('AIS_test', AIS_test)
        # with open(experiment_log, "a") as myfile:
        #     myfile.write('AIS_test '+ str(AIS_test) +'\n\n')
        #     myfile.write('time'+str(time.time()-start_time)+'\n\n')
        # ais_test_list.append(AIS_test)



    # # log results
    # with open(experiment_log, "a") as myfile:
    #     myfile.write('\nIW_train '+str(IW_train)
    #                     +'\nIW_test '+str(IW_test)
    #                     +'\nAIS_train '+str(AIS_train)
    #                     +'\nAIS_test '+str(AIS_test))

    print(vae_train_list)
    print(vae_test_list)
    print(iw_train_list)
    print(iw_test_list)
    print(ais_train_list)
    print(ais_test_list)





    # log it




# train_with_log(model=model, train_x=train_x, test_x=test_x, k=k, batch_size=batch_size, learning_rate=learning_rate,
#                     epochs=epochs, start_at=start_at, save_freq=save_freq, display_epoch=display_epoch, 
#                     path_to_save_variables=path_to_save_variables, experiment_log=experiment_log)







with open(experiment_log, "a") as myfile:
    myfile.write('\n\nDone.\n')
print ('Done.')












