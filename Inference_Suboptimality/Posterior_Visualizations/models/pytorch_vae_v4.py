

#Refactored


import numpy as np
import pickle
# import cPickle as pickle
from os.path import expanduser
home = expanduser("~")
import time
import sys
sys.path.insert(0, 'utils')

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import lognormal2 as lognormal
from utils import log_bernoulli
from ais import test_ais

from approx_posteriors_v4 import standard
from approx_posteriors_v4 import flow1
from approx_posteriors_v4 import aux_nf
from approx_posteriors_v4 import hnf








class VAE(nn.Module):
    def __init__(self, hyper_config, seed=1):
        super(VAE, self).__init__()

        torch.manual_seed(seed)


        self.z_size = hyper_config['z_size']
        self.x_size = hyper_config['x_size']
        self.act_func = hyper_config['act_func']
        self.flow_bool = hyper_config['flow_bool']

        self.q_dist = hyper_config['q_dist'](self, hyper_config=hyper_config)


        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
            self.q_dist.cuda()
        else:
            self.dtype = torch.FloatTensor
            

        #Decoder
        self.fc4 = nn.Linear(self.z_size, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, self.x_size)

        # See params
        # for aaa in self.parameters():
        #     print (aaa.size())
        # fsadfsa


    def decode(self, z):
        k = z.size()[0]
        B = z.size()[1]
        z = z.view(-1, self.z_size)

        out = self.act_func(self.fc4(z))
        out = self.act_func(self.fc5(out))
        out = self.fc6(out)

        x = out.view(k, B, self.x_size)
        return x


    def forward(self, x, k):

        self.B = x.size()[0] #batch size
        self.zeros = Variable(torch.zeros(self.B, self.z_size).type(self.dtype))

        logposterior = lambda aa: lognormal(aa, self.zeros, self.zeros) + log_bernoulli(self.decode(aa), x)

        z, logqz = self.q_dist.forward(k, x, logposterior)

        logpxz = logposterior(z)

        #Compute elbo
        elbo = logpxz - logqz #[P,B]
        if k>1:
            max_ = torch.max(elbo, 0)[0] #[B]
            elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
            
        elbo = torch.mean(elbo) #[1]
        logpxz = torch.mean(logpxz) #[1]
        logqz = torch.mean(logqz)

        return elbo, logpxz, logqz














    def train(self, train_x, k, epochs, batch_size, display_epoch, learning_rate):

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        time_ = time.time()
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

                batch = Variable(torch.from_numpy(batch)).type(self.dtype)
                optimizer.zero_grad()

                elbo, logpxz, logqz = model.forward(batch, k=k)

                loss = -(elbo)
                loss.backward()
                optimizer.step()


            if epoch%display_epoch==0:
                print ('Train Epoch: {}/{}'.format(epoch, epochs),
                    'LL:{:.3f}'.format(-loss.data[0]),
                    'logpxz:{:.3f}'.format(logpxz.data[0]),
                    # 'logpz:{:.3f}'.format(logpz.data[0]),
                    'logqz:{:.3f}'.format(logqz.data[0]),
                    'T:{:.2f}'.format(time.time()-time_),
                    )

                time_ = time.time()





    def test(self, data_x, batch_size, display, k):
        
        time_ = time.time()
        elbos = []
        data_index= 0
        for i in range(int(len(data_x)/ batch_size)):

            batch = data_x[data_index:data_index+batch_size]
            data_index += batch_size

            batch = Variable(torch.from_numpy(batch)).type(self.dtype)

            elbo, logpxz, logqz = model(batch, k=k)

            elbos.append(elbo.data[0])

            if i%display==0:
                print (i,len(data_x)/ batch_size, np.mean(elbos))

        mean_ = np.mean(elbos)
        print(mean_, 'T:', time.time()-time_)





    def load_params(self, path_to_load_variables=''):
        # model.load_state_dict(torch.load(path_to_load_variables))
        model.load_state_dict(torch.load(path_to_load_variables, map_location=lambda storage, loc: storage))
        print ('loaded variables ' + path_to_load_variables)


    def save_params(self, path_to_save_variables=''):
        torch.save(model.state_dict(), path_to_save_variables)
        print ('saved variables ' + path_to_save_variables)














if __name__ == "__main__":

    load_params = 0
    train_ = 1
    eval_IW = 1
    eval_AIS = 0

    print ('Loading data')
    with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
        mnist_data = pickle.load(f, encoding='latin1')

    train_x = mnist_data[0][0]
    valid_x = mnist_data[1][0]
    test_x = mnist_data[2][0]

    train_x = np.concatenate([train_x, valid_x], axis=0)

    print (train_x.shape)

    hyper_config = { 
                    'x_size': 784,
                    'z_size': 50,
                    'act_func': F.tanh,# F.relu,
                    'flow_bool': False,
                    'q_dist': hnf, #aux_nf, #flow1, #standard,#, #,# ,
                    'n_flows': 2
                }

    
    model = VAE(hyper_config)

    if torch.cuda.is_available():
        model.cuda()



    #Train params
    learning_rate = .0001
    batch_size = 100
    epochs = 3000
    display_epoch = 2
    k = 1

    path_to_load_variables=''
    # path_to_load_variables=home+'/Documents/tmp/pytorch_bvae.pt'
    path_to_save_variables=home+'/Documents/tmp/pytorch_vae'+str(epochs)+'.pt'
    # path_to_save_variables=''



    if load_params:
        print ('\nLoading parameters')
        model.load_params(path_to_save_variables)

    if train_:

        print('\nTraining')
        model.train(train_x=train_x, k=k, epochs=epochs, batch_size=batch_size, 
                    display_epoch=display_epoch, learning_rate=learning_rate)
        model.save_params(path_to_save_variables)


    if eval_IW:
        k_IW = 2000
        batch_size = 20
        print('\nTesting with IW, Train set[:10000], B'+str(batch_size)+' k'+str(k_IW))
        model.test(data_x=train_x[:10000], batch_size=batch_size, display=100, k=k_IW)

        print('\nTesting with IW, Test set, B'+str(batch_size)+' k'+str(k_IW))
        model.test(data_x=test_x, batch_size=batch_size, display=100, k=k_IW)

    if eval_AIS:
        k_AIS = 10
        batch_size = 100
        n_intermediate_dists = 100
        print('\nTesting with AIS, Train set[:10000], B'+str(batch_size)+' k'+str(k_AIS)+' intermediates'+str(n_intermediate_dists))
        test_ais(model, data_x=train_x[:10000], batch_size=batch_size, display=10, k=k_AIS, n_intermediate_dists=n_intermediate_dists)

        print('\nTesting with AIS, Test set, B'+str(batch_size)+' k'+str(k_AIS)+' intermediates'+str(n_intermediate_dists))
        test_ais(model, data_x=test_x, batch_size=batch_size, display=10, k=k_AIS, n_intermediate_dists=n_intermediate_dists)




















