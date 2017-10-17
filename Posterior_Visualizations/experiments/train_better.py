


# same as before, except small N, larger LR. 
    # less epochs, actually nervermind, each epoch has less training sicne dataset is smaller

import sys, os
sys.path.insert(0, '../models')
sys.path.insert(0, '../models/utils')

import time
import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")
import matplotlib.pyplot as plt


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from ais2 import test_ais

from pytorch_vae_v5 import VAE

from approx_posteriors_v5 import standard
from approx_posteriors_v5 import flow1
from approx_posteriors_v5 import aux_nf
from approx_posteriors_v5 import hnf

import argparse


# directory = home+'/Documents/tmp/small_N'
# directory = home+'/Documents/tmp/large_encoder'
# directory = home+'/Documents/tmp/test_can_delete'

directory = home+'/Documents/tmp/new_training'





if not os.path.exists(directory):
    os.makedirs(directory)
    print ('Made directory:'+directory)









print ('Loading data')
with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
    mnist_data = pickle.load(f, encoding='latin1')
train_x = mnist_data[0][0]
valid_x = mnist_data[1][0]
test_x = mnist_data[2][0]
train_x = np.concatenate([train_x, valid_x], axis=0)
print (train_x.shape)

# #For debug purposes
# train_x = train_x[:1000]
# test_x = test_x[:500]

print (train_x.shape)
print (test_x.shape)


#Small N
# train_x = train_x[:1000]


print (train_x.shape)
print (test_x.shape)


x_size = 784
z_size = 50




parser = argparse.ArgumentParser()
parser.add_argument("-m",'--model', default="standard")
args = parser.parse_args()



if args.model == 'standard':

    this_dir = directory+'/standard'
    if not os.path.exists(this_dir):
        os.makedirs(this_dir)
        print ('Made directory:'+this_dir)


    # experiment_log = this_dir+'/log.txt'

    # with open(experiment_log, "a") as myfile:
    #     myfile.write("Standard" +'\n')
    


    print('Init standard model')
    
    hyper_config = { 
                    'x_size': x_size,
                    'z_size': z_size,
                    'act_func': F.tanh,# F.relu,
                    'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
                    'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                    'q_dist': standard,#hnf,#aux_nf,#flow1,#,
                }

    # model = VAE(hyper_config)



elif args.model == 'flow1':

    this_dir = directory+'/flow1'
    if not os.path.exists(this_dir):
        os.makedirs(this_dir)
        print ('Made directory:'+this_dir)

    experiment_log = this_dir+'/log.txt'

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

    # model = VAE(hyper_config)



elif args.model == 'aux_nf':

    this_dir = directory+'/aux_nf'
    if not os.path.exists(this_dir):
        os.makedirs(this_dir)
        print ('Made directory:'+this_dir)

    # experiment_log = this_dir+'/log.txt'

    # with open(experiment_log, "a") as myfile:
    #     myfile.write("aux_nf" +'\n')

    print('Init flow model')
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

    # model = VAE(hyper_config)



elif args.model == 'hnf':

    this_dir = directory+'/hnf'
    if not os.path.exists(this_dir):
        os.makedirs(this_dir)
        print ('Made directory:'+this_dir)

    experiment_log = this_dir+'/log.txt'

    with open(experiment_log, "a") as myfile:
        myfile.write("hnf" +'\n')

    print('Init flow model')
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

    # model = VAE(hyper_config)










elif args.model == 'standard_large_encoder':

    this_dir = directory+'/standard'
    if not os.path.exists(this_dir):
        os.makedirs(this_dir)
        print ('Made directory:'+this_dir)


    # experiment_log = this_dir+'/log.txt'

    # with open(experiment_log, "a") as myfile:
    #     myfile.write("Standard" +'\n')
    


    print('Init standard model')
    
    hyper_config = { 
                    'x_size': x_size,
                    'z_size': z_size,
                    'act_func': F.tanh,# F.relu,
                    'encoder_arch': [[x_size,500],[500,500],[500,200],[200,z_size*2]],
                    'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                    'q_dist': standard,#hnf,#aux_nf,#flow1,#,
                }

    # model = VAE(hyper_config)









elif args.model == 'aux_large_encoder':

    this_dir = directory+'/aux_nf'
    if not os.path.exists(this_dir):
        os.makedirs(this_dir)
        print ('Made directory:'+this_dir)

    # experiment_log = this_dir+'/log.txt'

    # with open(experiment_log, "a") as myfile:
    #     myfile.write("aux_nf" +'\n')

    print('Init flow model')
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



else:
    print ('What')
    fadas


model = VAE(hyper_config)
# model.load_params(home+'/Documents/tmp/first_try/'+args.model+'/params_'+args.model+'_2800.pt')



#Train params
# learning_rate = .001
batch_size = 50
k = 1
# epochs = 3000

#save params and compute IW and AIS
start_at = 100
save_freq = 300
display_epoch = 10


# # Test params
# k_IW = 2000
# batch_size_IW = 20
# k_AIS = 10
# batch_size_AIS = 100
# n_intermediate_dists = 100

print('\nTraining')
# print('k='+str(k), 'lr='+str(learning_rate), 'batch_size='+str(batch_size))
print('\nModel:', hyper_config,'\n')


# with open(experiment_log, "a") as myfile:
#     myfile.write(str(hyper_config)+'\n\n')
# with open(experiment_log, "a") as myfile:
#     myfile.write('k='+str(k)+' lr='+str(learning_rate)+' batch_size='+str(batch_size) +'\n\n')





# path_to_load_variables=''
# path_to_load_variables=home+'/Documents/tmp/pytorch_bvae.pt'
# path_to_save_variables=home+'/Documents/tmp/pytorch_vae'+str(epochs)+'.pt'
path_to_save_variables=this_dir+'/params_'+args.model+'_'

# path_to_save_variables=''


if torch.cuda.is_available():
    model.cuda()













# def test(model, data_x, batch_size, display, k):
    
#     time_ = time.time()
#     elbos = []
#     data_index= 0
#     for i in range(int(len(data_x)/ batch_size)):

#         batch = data_x[data_index:data_index+batch_size]
#         data_index += batch_size

#         batch = Variable(torch.from_numpy(batch)).type(model.dtype)

#         elbo, logpxz, logqz = model(batch, k=k)

#         elbos.append(elbo.data[0])

#         if i%display==0:
#             print (i,len(data_x)/ batch_size, np.mean(elbos))

#     mean_ = np.mean(elbos)
#     print(mean_, 'T:', time.time()-time_)

#     return mean_#, time.time()-time_




# def train_with_log(model, train_x, test_x, k, batch_size, learning_rate,
#                     epochs, start_at, save_freq, display_epoch, 
#                     path_to_save_variables, experiment_log):

#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     time_ = time.time()
#     start_time = time.time()
    
#     n_data = len(train_x)
#     arr = np.array(range(n_data))

#     for epoch in range(1, epochs + 1):

#         #shuffle
#         np.random.shuffle(arr)
#         train_x = train_x[arr]

#         data_index= 0
#         for i in range(int(n_data/batch_size)):
#             batch = train_x[data_index:data_index+batch_size]
#             data_index += batch_size

#             batch = Variable(torch.from_numpy(batch)).type(model.dtype)
#             optimizer.zero_grad()

#             elbo, logpxz, logqz = model.forward(batch, k=k)

#             loss = -(elbo)
#             loss.backward()
#             optimizer.step()


#         if epoch%display_epoch==0:
#             print ('Train Epoch: {}/{}'.format(epoch, epochs),
#                 'LL:{:.3f}'.format(-loss.data[0]),
#                 'logpxz:{:.3f}'.format(logpxz.data[0]),
#                 'logqz:{:.3f}'.format(logqz.data[0]),
#                 'T:{:.2f}'.format(time.time()-time_),
#                 )
#             time_ = time.time()


#         if epoch >= start_at and (epoch-start_at)%save_freq==0:

#             # save params
#             save_file = path_to_save_variables+str(epoch)+'.pt'
#             torch.save(model.state_dict(), save_file)
#             print ('saved variables ' + save_file)
#             with open(experiment_log, "a") as myfile:
#                 myfile.write('\nepoch'+str(epoch)+'\nsaved variables ' + save_file+'\n')
#                 myfile.write('time'+str(time.time()-start_time)+'\n\n')


#             # compute LL 
#             # print('\nTesting with IW, Train set, B'+str(batch_size_IW)+' k'+str(k_IW))
#             # IW_train = test(model=model, data_x=train_x, batch_size=batch_size_IW, display=1000, k=k_IW)
#             # print('\nTesting with IW, Test set, B'+str(batch_size_IW)+' k'+str(k_IW))
#             # IW_test = test(model=model, data_x=test_x, batch_size=batch_size_IW, display=1000, k=k_IW)
#             # print('\nTesting with AIS, Train set, B'+str(batch_size_AIS)+' k'+str(k_AIS)+' intermediates'+str(n_intermediate_dists))
#             # AIS_train = test_ais(model=model, data_x=train_x, batch_size=batch_size_AIS, display=1000, k=k_AIS, n_intermediate_dists=n_intermediate_dists)
#             # print('\nTesting with AIS, Test set, B'+str(batch_size_AIS)+' k'+str(k_AIS)+' intermediates'+str(n_intermediate_dists))
#             # AIS_test = test_ais(model=model, data_x=test_x, batch_size=batch_size_AIS, display=10000, k=k_AIS, n_intermediate_dists=n_intermediate_dists)


#             # # log results
#             # with open(experiment_log, "a") as myfile:
#             #     myfile.write('\nIW_train '+str(IW_train)
#             #                     +'\nIW_test '+str(IW_test)
#             #                     +'\nAIS_train '+str(AIS_train)
#             #                     +'\nAIS_test '+str(AIS_test))






def train_lr_schedule(model, train_x, test_x, k, batch_size,
                    start_at, save_freq, display_epoch, 
                    path_to_save_variables):

    #IWAE paper training strategy

    time_ = time.time()

    n_data = len(train_x)
    arr = np.array(range(n_data))

    total_epochs = 0

    i_max = 6

    warmup_over_epochs = 100.

    for i in range(0,i_max+1):
        lr = .001 * 10**(-i/float(i_max))
        print (i, 'LR:', lr)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        epochs = 3**(i)

        for epoch in range(1, epochs + 1):

            #shuffle
            np.random.shuffle(arr)
            train_x = train_x[arr]

            data_index= 0
            for i in range(int(n_data/batch_size)):
                batch = train_x[data_index:data_index+batch_size]
                data_index += batch_size

                batch = Variable(torch.from_numpy(batch)).type(model.dtype)
                optimizer.zero_grad()

                warmup = total_epochs/warmup_over_epochs
                if warmup > 1.:
                    warmup = 1.

                elbo, logpxz, logqz = model.forward(batch, k=k, warmup=warmup)

                loss = -(elbo)
                loss.backward()
                optimizer.step()

            total_epochs += 1


            if total_epochs%display_epoch==0:
                print ('Train Epoch: {}/{}'.format(epoch, epochs),
                    'total_epochs {}'.format(total_epochs),
                    'LL:{:.3f}'.format(-loss.data[0]),
                    'logpxz:{:.3f}'.format(logpxz.data[0]),
                    'logqz:{:.3f}'.format(logqz.data[0]),
                    'warmup:{:.3f}'.format(warmup),
                    'T:{:.2f}'.format(time.time()-time_),
                    )
                time_ = time.time()


            if total_epochs >= start_at and (total_epochs-start_at)%save_freq==0:

                # save params
                save_file = path_to_save_variables+str(total_epochs)+'.pt'
                torch.save(model.state_dict(), save_file)
                print ('saved variables ' + save_file)
                # with open(experiment_log, "a") as myfile:
                #     myfile.write('\nepoch'+str(epoch)+'\nsaved variables ' + save_file+'\n')
                #     myfile.write('time'+str(time.time()-start_time)+'\n\n')



    # save params
    save_file = path_to_save_variables+str(total_epochs)+'.pt'
    torch.save(model.state_dict(), save_file)
    print ('saved variables ' + save_file)
    print ('done training')






# train_with_log(model=model, train_x=train_x, test_x=test_x, k=k, batch_size=batch_size, learning_rate=learning_rate,
#                     epochs=epochs, start_at=start_at, save_freq=save_freq, display_epoch=display_epoch, 
#                     path_to_save_variables=path_to_save_variables, experiment_log=experiment_log)


train_lr_schedule(model=model, train_x=train_x, test_x=test_x, k=k, batch_size=batch_size,
                    start_at=start_at, save_freq=save_freq, display_epoch=display_epoch, 
                    path_to_save_variables=path_to_save_variables)

# with open(experiment_log, "a") as myfile:
#     myfile.write('\n\nDone.\n')
print ('Done.')


















