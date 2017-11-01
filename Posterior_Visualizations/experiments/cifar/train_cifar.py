

# adding warmup


import sys, os
sys.path.insert(0, '../../models')
sys.path.insert(0, '../../models/utils')


import time
import numpy as np
import pickle#, cPickle
import _pickle as cPickle
from os.path import expanduser
home = expanduser("~")

# import matplotlib.pyplot as plt


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import lognormal2 as lognormal
from utils import log_bernoulli

from scipy.misc import toimage



from ais2 import test_ais

from pytorch_vae_v5 import VAE

from approx_posteriors_v5 import standard
from approx_posteriors_v5 import flow1
from approx_posteriors_v5 import aux_nf
from approx_posteriors_v5 import hnf

import argparse

from vae_model_cifar import VAE_deconv1


# reload(sys)  
# sys.setdefaultencoding('utf8')












def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo,encoding='latin1')
    return dict




def train_with_log(model, train_x, test_x, k, batch_size, learning_rate,
                    epochs, start_at, save_freq, display_epoch, 
                    path_to_save_variables, experiment_log):

    print ('Train starting')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    time_ = time.time()
    start_time = time.time()
    
    n_data = len(train_x)
    arr = np.array(range(n_data))

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

            elbo, logpxz, logqz = model.forward(batch, k=k)

            loss = -(elbo)
            loss.backward()
            optimizer.step()


        if epoch%display_epoch==0:
            print ('Train Epoch: {}/{}'.format(epoch, epochs),
                'LL:{:.3f}'.format(-loss.data[0]),
                'logpxz:{:.3f}'.format(logpxz.data[0]),
                'logqz:{:.3f}'.format(logqz.data[0]),
                'T:{:.2f}'.format(time.time()-time_),
                )
            time_ = time.time()


        if epoch >= start_at and (epoch-start_at)%save_freq==0:

            # save params
            save_file = path_to_save_variables+str(epoch)+'.pt'
            torch.save(model.state_dict(), save_file)
            print ('saved variables ' + save_file)
            # with open(experiment_log, "a") as myfile:
            #     myfile.write('\nepoch'+str(epoch)+'\nsaved variables ' + save_file+'\n')
            #     myfile.write('time'+str(time.time()-start_time)+'\n\n')




# def train(model, train_x, train_y, valid_x=[], valid_y=[], 
#             path_to_load_variables='', path_to_save_variables='', 
#             epochs=10, batch_size=20, display_epoch=2, k=1):
    

#     if path_to_load_variables != '':
#         model.load_state_dict(torch.load(path_to_load_variables))
#         # print 'loaded variables ' + path_to_load_variables

#     train = torch.utils.data.TensorDataset(train_x, train_y)
#     train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
#     optimizer = optim.Adam(model.parameters(), lr=.0001)

#     for epoch in range(1, epochs + 1):

#         for batch_idx, (data, target) in enumerate(train_loader):

#             # if data.is_cuda:
#             if torch.cuda.is_available():
#                 data = Variable(data).type(torch.cuda.FloatTensor)# , Variable(target).type(torch.cuda.LongTensor) 
#             else:
#                 data = Variable(data)#, Variable(target)

#             optimizer.zero_grad()


#             #warm up
#             warmup = epoch / 100.
#             if warmup > 1.:
#                 warmup = 1.

#             elbo, logpx, logpz, logqz = model.forward(data, k=k, warmup=warmup)
#             loss = -(elbo)

#             loss.backward()
#             optimizer.step()

#             # if (epoch%display_epoch==0 or epoch==1) and batch_idx == 0:
#             #     print 'Train Epoch: {}/{}'.format(epoch, epochs), \
#             #         'Loss:{:.4f}'.format(loss.data[0]), \
#             #         'logpx:{:.4f}'.format(logpx.data[0]), \
#             #         'logpz:{:.4f}'.format(logpz.data[0]), \
#             #         'logqz:{:.4f}'.format(logqz.data[0]), \
#             #         'warmup:{:.2f}'.format(warmup)

#     if path_to_save_variables != '':
#         torch.save(model.state_dict(), path_to_save_variables)
#         # print 'Saved variables to ' + path_to_save_variables




# def test(model, data_x, path_to_load_variables='', batch_size=20, display_epoch=4, k=10):
    

#     if path_to_load_variables != '':
#         model.load_state_dict(torch.load(path_to_load_variables))
#         # print 'loaded variables ' + path_to_load_variables

#     elbos = []
#     data_index= 0
#     for i in range(len(data_x)/ batch_size):

#         batch = data_x[data_index:data_index+batch_size]
#         data_index += batch_size

#         # if data.is_cuda:
#         if torch.cuda.is_available():
#             data = Variable(batch).type(torch.cuda.FloatTensor)
#         else:
#             data = Variable(batch)#, Variable(target)

#         elbo, logpx, logpz, logqz = model.forward(data, k=k)
#         # print elbo, logpx, logpz, logqz
#         # fasdfa

#         # elbo, logpx, logpz, logqz = model(Variable(batch), k=k)
#         elbos.append(elbo.data[0])

#         # if i%display_epoch==0:
#         #     print i,len(data_x)/ batch_size, np.mean(elbos)

#     return np.mean(elbos)




# def load_params(model, path_to_load_variables=''):

#     if path_to_load_variables != '':
#         # model.load_state_dict(torch.load(path_to_load_variables))
#         model.load_state_dict(torch.load(path_to_load_variables, lambda storage, loc: storage)) 
#         # print 'loaded variables ' + path_to_load_variables






















if __name__ == "__main__":

    # directory = home+'/Documents/tmp/small_N'
    directory = home+'/Documents/tmp/cifar_exp'

    if not os.path.exists(directory):
        os.makedirs(directory)
        print ('Made directory:'+directory)





    print ('Loading data')
    file_ = home+'/Documents/cifar-10-batches-py/data_batch_'

    for i in range(1,6):
        file__ = file_ + str(i)
        b1 = unpickle(file__)
        if i ==1:
            train_x = b1['data']
            train_y = b1['labels']
        else:
            train_x = np.concatenate([train_x, b1['data']], axis=0)
            train_y = np.concatenate([train_y, b1['labels']], axis=0)

    file__ = home+'/Documents/cifar-10-batches-py/test_batch'
    b1 = unpickle(file__)
    test_x = b1['data']
    test_y = b1['labels']


    train_x = train_x / 255.
    test_x = test_x / 255.

    # train_x = torch.from_numpy(train_x).float()
    # test_x = torch.from_numpy(test_x)
    # train_y = torch.from_numpy(train_y)

    print (train_x.shape)
    print (test_x.shape)
    print (train_y.shape)



    batch_size = 50
    epochs = 200
    display_epoch = 10










    # standard model

    this_dir = directory+'/standard'
    if not os.path.exists(this_dir):
        os.makedirs(this_dir)
        print ('Made directory:'+this_dir)


    # experiment_log = this_dir+'/log.txt'

    # with open(experiment_log, "a") as myfile:
    #     myfile.write("Standard" +'\n')
    

    # print('Init standard model')
    
    # hyper_config = { 
    #                 'x_size': x_size,
    #                 'z_size': z_size,
    #                 'act_func': F.tanh,# F.relu,
    #                 'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
    #                 'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
    #                 'q_dist': standard,#hnf,#aux_nf,#flow1,#,
    #             }





    print ('Init model')
    # print 'conv encoder, deconv decoder'
    model = VAE_deconv1()
    print ('Init done')

    if torch.cuda.is_available():
        # print 'GPU available, loading cuda'#, torch.cuda.is_available()
        model.cuda()
        # train_x = train_x.cuda()

    path_to_load_variables=''
    # path_to_load_variables=home+'/Documents/tmp/pytorch_first.pt'
    # path_to_save_variables=home+'/Documents/tmp/pytorch_vae_cifar_deconv_2.pt'
    path_to_save_variables=this_dir+'/params_standard2_'

    # path_to_save_variables=''
    experiment_log = ''


    # train(model=model, train_x=train_x, train_y=train_y, valid_x=[], valid_y=[], 
    #             path_to_load_variables=path_to_load_variables, 
    #             path_to_save_variables=path_to_save_variables, 
    #             epochs=epochs, batch_size=batch_size, display_epoch=display_epoch, k=1)


    #Train params
    learning_rate = .0001
    batch_size = 50
    k = 1
    epochs = 3000

    #save params and compute IW and AIS
    start_at = 100
    save_freq = 300
    display_epoch = 5


    train_with_log(model=model, train_x=train_x, test_x=test_x, k=k, batch_size=batch_size, learning_rate=learning_rate,
                        epochs=epochs, start_at=start_at, save_freq=save_freq, display_epoch=display_epoch, 
                        path_to_save_variables=path_to_save_variables, experiment_log=experiment_log)


    # with open(experiment_log, "a") as myfile:
    #     myfile.write('\n\nDone.\n')
    print ('Done.')


    # print test(model=model, data_x=test_x, path_to_load_variables='', 
    #             batch_size=5, display_epoch=100, k=1000)

    # print 'Done.'
    fsadfa



































# TO VIZ


# print 'conv encoder, deconv decoder'
# model = VAE_deconv1()
# path_to_save_variables=home+'/Documents/tmp/pytorch_deconv.pt'

model2 = VAE_deconv1()
load_params(model2, home+'/Documents/tmp/pytorch_deconv.pt')

#Reconstruct images
n_imgs = 10
fig, axes1 = plt.subplots(n_imgs,4,figsize=(3,6))


for j in range(n_imgs):
    # for i in range(n_imgs):
    i = j
    # i = np.random.choice(range(50))
    axes1[j][0].set_axis_off()
    axes1[j][1].set_axis_off()
    axes1[j][2].set_axis_off()
    axes1[j][3].set_axis_off()


    real_img = train_x[i].numpy()* 255
    real_img = np.reshape(real_img, [3,32,32])
    real_img = real_img.transpose(1,2,0).astype("uint8")
    axes1[j][0].imshow(real_img)


    x = Variable(train_x[i].view(1,3072))


    model1.forward(x)
    recon = model1.x_hat_sigmoid
    recon = recon.data.numpy()* 255
    recon = np.reshape(recon, [3,32,32])
    recon = recon.transpose(1,2,0).astype("uint8")
    axes1[j][1].imshow(recon)


    model2.forward(x)
    recon = model2.x_hat_sigmoid
    recon = recon.data.numpy()* 255
    recon = np.reshape(recon, [3,32,32])
    recon = recon.transpose(1,2,0).astype("uint8")
    axes1[j][2].imshow(recon)


    model3.forward(x)
    recon = model3.x_hat_sigmoid
    recon = recon.data.numpy()* 255
    recon = np.reshape(recon, [3,32,32])
    recon = recon.transpose(1,2,0).astype("uint8")
    axes1[j][3].imshow(recon)

plt.annotate('Real', xy=(0, 0), xytext=(.18,.9), textcoords='figure fraction', size=6)
plt.annotate('FC', xy=(0, 0), xytext=(.4,.9), textcoords='figure fraction', size=6)
plt.annotate('Deconv', xy=(0, 0), xytext=(.56,.9), textcoords='figure fraction', size=6)
plt.annotate('Deconv+Conv', xy=(0, 0), xytext=(.74,.9), textcoords='figure fraction', size=6)

# ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5))

plt.show()
fasdfa















