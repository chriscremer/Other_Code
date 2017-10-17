


import time
import numpy as np
import pickle#, cPickle
import _pickle as cPickle
from os.path import expanduser
home = expanduser("~")



import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F




import sys
sys.path.insert(0, '../utils')

from utils import lognormal2 as lognormal
from utils import log_bernoulli
from ais import test_ais

# from approx_posteriors_v5 import standard
# from approx_posteriors_v5 import flow1
# from approx_posteriors_v5 import aux_nf
# from approx_posteriors_v5 import hnf

from approx_posteriors_cifar import standard



class VAE_deconv1(nn.Module):
    def __init__(self):
        super(VAE_deconv1, self).__init__()

        torch.manual_seed(1)

        self.x_size = 3072
        self.z_size = 200


        self.act_func = F.tanh

        # self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=2, padding=0, dilation=1, bias=True)

        # self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=2, padding=0, dilation=1, bias=True)

        # self.fc1 = nn.Linear(1960, 200)
        # self.fc2 = nn.Linear(200, self.z_size*2)




        self.q_dist = standard(self)



        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
            self.q_dist.cuda()
        else:
            self.dtype = torch.FloatTensor
            



        self.fc3 = nn.Linear(self.z_size, 200)
        # self.fc4 = nn.Linear(200, 1960)
        self.fc4 = nn.Linear(200, 250)


        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=5, stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)

        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size=5, stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)


    # def encode(self, x):

    #     x = x.view(-1, 3, 32, 32)
    #     x = F.relu(self.conv1(x))

    #     x = x.view(-1, 1960)

    #     h1 = F.relu(self.fc1(x))
    #     h2 = self.fc2(h1)
    #     mean = h2[:,:self.z_size]
    #     logvar = h2[:,self.z_size:]
    #     return mean, logvar



    # def sample(self, mu, logvar, k):
    #     if torch.cuda.is_available():
    #         eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_()).cuda() #[P,B,Z]
    #         z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
    #         logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size).cuda()), 
    #                             Variable(torch.zeros(self.B, self.z_size)).cuda())  #[P,B]
    #         logqz = lognormal(z, mu, logvar)
    #     else:
    #         eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_())#[P,B,Z]
    #         z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
    #         logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size)), 
    #                             Variable(torch.zeros(self.B, self.z_size)))  #[P,B]
    #         logqz = lognormal(z, mu, logvar) 
    #     return z, logpz, logqz


    def decode(self, z):

        # print (z.size())  #[P,B,Z]
        # fsdafa
        k = z.size()[0]
        B = z.size()[1]

        z = z.view(-1,self.z_size) #[PB,Z]

        # z = self.act_func(self.fc3(z)) 
        # z = self.act_func(self.fc4(z))  #[PB,1960]

        z = F.relu(self.fc3(z)) 
        z = F.relu(self.fc4(z))  #[PB,1960]

        # print (z.size())  #[PB,250]
        # fsdafa

        # z = z.view(-1, 10, 14, 14)
        z = z.view(-1, 10, 5, 5)

        z = self.deconv1(z)
        z = F.relu(z)
        # z = self.act_func(z)
        z = self.deconv2(z)
        # z = z.view(-1, self.x_size)


        x = z.view(k, B, self.x_size)

        # print (x.size())  #[P,B,3072]
        # fsdafa

        return x



    def forward(self, x, k=1, warmup=1.):
        
        self.B = x.size()[0] #batch size
        self.zeros = Variable(torch.zeros(self.B, self.z_size).type(self.dtype))  #[B,Z]

        self.logposterior = lambda aa: lognormal(aa, self.zeros, self.zeros) + log_bernoulli(self.decode(aa), x)


        z, logqz = self.q_dist.forward(k, x, self.logposterior)

        # [PB,Z]
        # z = z.view(-1,self.z_size)

        logpxz = self.logposterior(z)

        #Compute elbo
        elbo = logpxz - logqz #[P,B]
        if k>1:
            max_ = torch.max(elbo, 0)[0] #[B]
            elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
            
        elbo = torch.mean(elbo) #[1]
        logpxz = torch.mean(logpxz) #[1]
        logqz = torch.mean(logqz)


        # mu, logvar = self.encode(x)
        # z, logpz, logqz = self.sample(mu, logvar, k=k)  #[P,B,Z]
        # x_hat = self.decode(z)  #[PB,X]
        # x_hat = x_hat.view(k, self.B, -1)
        # logpx = log_bernoulli(x_hat, x)  #[P,B]

        # elbo = logpx +  warmup*(logpz - logqz)  #[P,B]

        # if k>1:
        #     max_ = torch.max(elbo, 0)[0] #[B]
        #     elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

        # elbo = torch.mean(elbo) #[1]

        # #for printing
        # logpx = torch.mean(logpx)
        # logpz = torch.mean(logpz)
        # logqz = torch.mean(logqz)
        # self.x_hat_sigmoid = F.sigmoid(x_hat)

        # return elbo, logpx, logpz, logqz


        return elbo, logpxz, logqz 
















    def sample_q(self, x, k):

        self.B = x.size()[0] #batch size
        self.zeros = Variable(torch.zeros(self.B, self.z_size).type(self.dtype))

        self.logposterior = lambda aa: lognormal(aa, self.zeros, self.zeros) + log_bernoulli(self.decode(aa), x)

        # print (x)
        # fsda
        z, logqz = self.q_dist.forward(k=k, x=x, logposterior=self.logposterior)

        return z


    def logposterior_func(self, x, z):
        self.B = x.size()[0] #batch size
        self.zeros = Variable(torch.zeros(self.B, self.z_size).type(self.dtype))

        # print (x)  #[B,X]
        # print(z)    #[P,Z]
        z = Variable(z).type(self.dtype)
        z = z.view(-1,self.B,self.z_size)
        return lognormal(z, self.zeros, self.zeros) + log_bernoulli(self.decode(z), x)



    def logposterior_func2(self, x, z):
        self.B = x.size()[0] #batch size
        self.zeros = Variable(torch.zeros(self.B, self.z_size).type(self.dtype))

        # print (x)  #[B,X]
        # print(z)    #[P,Z]
        # z = Variable(z).type(self.dtype)
        z = z.view(-1,self.B,self.z_size)

        # print (z)
        return lognormal(z, self.zeros, self.zeros) + log_bernoulli(self.decode(z), x)



    def forward2(self, x, k):

        self.B = x.size()[0] #batch size
        self.zeros = Variable(torch.zeros(self.B, self.z_size).type(self.dtype))

        self.logposterior = lambda aa: lognormal(aa, self.zeros, self.zeros) + log_bernoulli(self.decode(aa), x)

        z, logqz = self.q_dist.forward(k, x, self.logposterior)

        logpxz = self.logposterior(z)

        #Compute elbo
        elbo = logpxz - logqz #[P,B]
        # if k>1:
        #     max_ = torch.max(elbo, 0)[0] #[B]
        #     elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
            
        elbo = torch.mean(elbo) #[1]
        logpxz = torch.mean(logpxz) #[1]
        logqz = torch.mean(logqz)

        return elbo, logpxz, logqz
















    def train(self, train_x, k, epochs, batch_size, display_epoch, learning_rate):

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
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

                elbo, logpxz, logqz = self.forward(batch, k=k)

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

            elbo, logpxz, logqz = self(batch, k=k)

            elbos.append(elbo.data[0])

            if i%display==0:
                print (i,len(data_x)/ batch_size, np.mean(elbos))

        mean_ = np.mean(elbos)
        print(mean_, 'T:', time.time()-time_)





    def load_params(self, path_to_load_variables=''):
        # model.load_state_dict(torch.load(path_to_load_variables))
        self.load_state_dict(torch.load(path_to_load_variables, map_location=lambda storage, loc: storage))
        print ('loaded variables ' + path_to_load_variables)


    def save_params(self, path_to_save_variables=''):
        torch.save(self.state_dict(), path_to_save_variables)
        print ('saved variables ' + path_to_save_variables)
































