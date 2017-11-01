

import os
import gzip
import numpy as np


from os.path import expanduser
home = expanduser("~")




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

from pytorch_vae_v6 import VAE

from approx_posteriors_v6 import FFG_LN
from approx_posteriors_v6 import ANF_LN

import argparse



#FASHION
# def load_mnist(path, kind='train'):

#     images_path = os.path.join(path,
#                                '%s-images-idx3-ubyte.gz'
#                                % kind)

#     with gzip.open(images_path, 'rb') as imgpath:
#         images = np.frombuffer(imgpath.read(), dtype=np.uint8,
#                                offset=16).reshape(-1, 784)

#     return images#, labels


# path = home+'/Documents/fashion_MNIST'

# train_x = load_mnist(path=path)
# test_x = load_mnist(path=path, kind='t10k')

# train_x = train_x / 255.
# test_x = test_x / 255.

# print (train_x.shape)
# print (test_x.shape)

# print (np.max(train_x))
# print (test_x[3])
# fsda




os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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

# print (train_x.shape)
# print (test_x.shape)






# directory = home+'/Documents/tmp/small_N'
# directory = home+'/Documents/tmp/large_encoder'
# directory = home+'/Documents/tmp/test_can_delete'

# directory = home+'/Documents/tmp/no_warmup'

directory = home+'/Documents/tmp/batch20_correct'






if not os.path.exists(directory):
    os.makedirs(directory)
    print ('Made directory:'+directory)










x_size = 784
z_size = 50


parser = argparse.ArgumentParser()
parser.add_argument("-m",'--model')
args = parser.parse_args()

model_name = args.model

if model_name == 'FFG':

    this_dir = directory+'/FFG'
    if not os.path.exists(this_dir):
        os.makedirs(this_dir)
        print ('Made directory:'+this_dir)


    print('Init FFG model')
    
    hyper_config = { 
                    'x_size': x_size,
                    'z_size': z_size,
                    'act_func': F.tanh,# F.relu,
                    'encoder_arch': [[x_size,200],[200,200],[200,200],[200,z_size*2]],
                    'decoder_arch': [[z_size,200],[200,200],[200,200],[200,x_size]],
                    'q_dist': FFG_LN#standard,#hnf,#aux_nf,#flow1,#,
                }



elif model_name == 'Flex':

    this_dir = directory+'/Flex'
    if not os.path.exists(this_dir):
        os.makedirs(this_dir)
        print ('Made directory:'+this_dir)


    print('Init Flex model')
    hyper_config = { 
                    'x_size': x_size,
                    'z_size': z_size,
                    'act_func': F.tanh,# F.relu,
                    'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
                    'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                    'q_dist': ANF_LN,#aux_nf,#flow1,#standard,
                    'n_flows': 2,
                    'qv_arch': [[x_size,200],[200,200],[200,z_size*2]],
                    'qz_arch': [[x_size+z_size,200],[200,200],[200,z_size*2]],
                    'rv_arch': [[x_size+z_size,200],[200,200],[200,z_size*2]],
                    'flow_hidden_size': 100
                }

else:
    print ('What')
    fadas


model = VAE(hyper_config)
# model.load_params(home+'/Documents/tmp/first_try/'+args.model+'/params_'+args.model+'_2800.pt')



#Train params
# learning_rate = .001
batch_size = 20
k = 1
# epochs = 3000

#save params 
start_at = 100
save_freq = 300
display_epoch = 3


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
path_to_save_variables=this_dir+'/params_'+model_name+'_'

# path_to_save_variables=''


if torch.cuda.is_available():
    model.cuda()











# def train_lr_schedule(model, train_x, test_x, k, batch_size,
#                     start_at, save_freq, display_epoch, 
#                     path_to_save_variables):

#     #IWAE paper training strategy

#     time_ = time.time()

#     n_data = len(train_x)
#     arr = np.array(range(n_data))

#     total_epochs = 0

#     i_max = 7

#     warmup_over_epochs = 100.

#     for i in range(0,i_max+1):

#         lr = .001 * 10**(-i/float(i_max))
#         print (i, 'LR:', lr)

#         optimizer = optim.Adam(model.parameters(), lr=lr)

#         epochs = 3**(i)

#         for epoch in range(1, epochs + 1):

#             #shuffle
#             np.random.shuffle(arr)
#             train_x = train_x[arr]

#             data_index= 0
#             for i in range(int(n_data/batch_size)):
#                 batch = train_x[data_index:data_index+batch_size]
#                 data_index += batch_size

#                 batch = Variable(torch.from_numpy(batch)).type(model.dtype)
#                 optimizer.zero_grad()

#                 warmup = total_epochs/warmup_over_epochs
#                 if warmup > 1.:
#                     warmup = 1.

#                 elbo, logpxz, logqz = model.forward(batch, k=k, warmup=warmup)

#                 loss = -(elbo)
#                 loss.backward()
#                 optimizer.step()

#             total_epochs += 1


#             if total_epochs%display_epoch==0:
#                 print ('Train Epoch: {}/{}'.format(epoch, epochs),
#                     'total_epochs {}'.format(total_epochs),
#                     'LL:{:.3f}'.format(-loss.data[0]),
#                     'logpxz:{:.3f}'.format(logpxz.data[0]),
#                     'logqz:{:.3f}'.format(logqz.data[0]),
#                     'warmup:{:.3f}'.format(warmup),
#                     'T:{:.2f}'.format(time.time()-time_),
#                     )
#                 time_ = time.time()


#             if total_epochs >= start_at and (total_epochs-start_at)%save_freq==0:

#                 # save params
#                 save_file = path_to_save_variables+str(total_epochs)+'.pt'
#                 torch.save(model.state_dict(), save_file)
#                 print ('saved variables ' + save_file)




#     # save params
#     save_file = path_to_save_variables+str(total_epochs)+'.pt'
#     torch.save(model.state_dict(), save_file)
#     print ('saved variables ' + save_file)
#     print ('done training')







def train_lr_schedule(model, train_x, test_x, k, batch_size,
                    start_at, save_freq, display_epoch, 
                    path_to_save_variables):

    train_y = torch.from_numpy(np.zeros(len(train_x)))
    train_x = torch.from_numpy(train_x).float().type(model.dtype)

    train_ = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True)

    #IWAE paper training strategy
    time_ = time.time()
    # n_data = len(train_x)
    # arr = np.array(range(n_data))

    total_epochs = 0

    i_max = 7

    warmup_over_epochs = 100.

    for i in range(0,i_max+1):

        lr = .001 * 10**(-i/float(i_max))
        print (i, 'LR:', lr)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        epochs = 3**(i)

        for epoch in range(1, epochs + 1):

            # #shuffle
            # np.random.shuffle(arr)
            # train_x = train_x[arr]

            # data_index= 0
            # for i in range(int(n_data/batch_size)):
            #     batch = train_x[data_index:data_index+batch_size]
            #     data_index += batch_size

            #     batch = Variable(torch.from_numpy(batch)).type(model.dtype)

            for batch_idx, (data, target) in enumerate(train_loader):

                batch = Variable(data)#.type(model.dtype)

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




    # save params
    save_file = path_to_save_variables+str(total_epochs)+'.pt'
    torch.save(model.state_dict(), save_file)
    print ('saved variables ' + save_file)
    print ('done training')








train_lr_schedule(model=model, train_x=train_x, test_x=test_x, k=k, batch_size=batch_size,
                    start_at=start_at, save_freq=save_freq, display_epoch=display_epoch, 
                    path_to_save_variables=path_to_save_variables)


print ('Done.')





































