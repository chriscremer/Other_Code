







import time






import numpy as np
import pickle
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

from bnn_pytorch import BNN





def train(model, train_x, train_y, valid_x=[], valid_y=[], 
            path_to_load_variables='', path_to_save_variables='', 
            epochs=10, batch_size=20, display_epoch=2, k=1):
    

    if path_to_load_variables != '':
        model.load_state_dict(torch.load(path_to_load_variables))
        print 'loaded variables ' + path_to_load_variables


    # # own_state = 
    # print len(model.state_dict())
    # fds
    # for name, param in model.state_dict():
    #     print name, param

    #     # if name not in own_state:
    #     #      continue
    #     # if isinstance(param, Parameter):
    #     #     # backwards compatibility for serialized parameters
    #     #     param = param.data
    #     # own_state[name].copy_(param)
    # fasdfasd

    train = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)



    # params = list(model.parameters()) + list(model.decoder.params)

    # print 'Params'
    # for parameter_i in range(len(params)):
    #     print parameter_i, params[parameter_i].size()
    #     # print params[parameter_i]
    #     # print(parameter.size())

    # print 'Params222'
    # for parameter in model.parameters():
    #     print(parameter.size())

    # optimizer = optim.Adam(params, lr=.0001)
    optimizer = optim.Adam(model.parameters(), lr=.001)


    # print 'passed'

    if torch.cuda.is_available():
        print 'GPU available, loading cuda'#, torch.cuda.is_available()
        model.cuda()


    for epoch in range(1, epochs + 1):

        for batch_idx, (data, target) in enumerate(train_loader):

            if torch.cuda.is_available():
                data = Variable(data.cuda())#, Variable(target)#.type(torch.cuda.LongTensor)
            else:
                data = Variable(data)#, Variable(target)

            optimizer.zero_grad()

            # start = time.time()

            # print 'forward'
            elbo, logpx, logpz, logqz, logpW, logqW = model.forward(data, k=k)
            loss = -(elbo)
            # print 'backward', epoch, display_epoch, batch_idx

            # print(time.time() - start), 'forward'


            # start = time.time()
            loss.backward()
            # print(time.time() - start), 'backward'

            # start = time.time()
            optimizer.step()
            # print(time.time() - start), 'step'

            if epoch%display_epoch==0 and batch_idx == 0:
                print 'Train Epoch: {}/{} [{}/{} ({:.0f}%)]'.format(epoch, epochs, 
                        batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader)), \
                    'Loss:{:.3f}'.format(loss.data[0]), \
                    'logpx:{:.3f}'.format(logpx.data[0]), \
                    'logpz:{:.3f}'.format(logpz.data[0]), \
                    'logqz:{:.3f}'.format(logqz.data[0]), \
                    'logpW:{:.3f}'.format(logpW.data[0]), \
                    'logqW:{:.3f}'.format(logqW.data[0])

    if path_to_save_variables != '':
        torch.save(model.state_dict(), path_to_save_variables)
        print 'Saved variables to ' + path_to_save_variables




# def test(model, data_x, path_to_load_variables='', batch_size=20, display_epoch=4, k=10):
    

#     if path_to_load_variables != '':
#         model.load_state_dict(torch.load(path_to_load_variables))
#         print 'loaded variables ' + path_to_load_variables

#     elbos = []
#     data_index= 0
#     for i in range(len(data_x)/ batch_size):

#         batch = data_x[data_index:data_index+batch_size]
#         data_index += batch_size

#         elbo, logpx, logpz, logqz = model(Variable(batch), k=k)
#         elbos.append(elbo.data[0])

#         if i%display_epoch==0:
#             print i,len(data_x)/ batch_size, elbo.data[0]

#     return np.mean(elbos)


# self.module_list = nn.ModuleList()
# for i in range(5):
#     self.module_list += make_sequence()



class BVAE(nn.Module):
    def __init__(self):
        super(BVAE, self).__init__()

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            

        self.z_size = 20
        self.input_size = 784

        #Encoder
        self.fc1 = nn.Linear(self.input_size, 200)
        self.fc2 = nn.Linear(200, self.z_size*2)
        #Decoder
        # self.fc3 = nn.Linear(self.z_size, 200)
        # self.fc4 = nn.Linear(200, 784)
        # self.decoder = BNN([self.z_size, 200, 784], [torch.nn.Softplus, torch.nn.Softplus])
        self.decoder = BNN([self.z_size, 200, 784], [F.relu, F.relu])

        # self.add_module('BNN', self.decoder)
        # for idx, m in enumerate(self.modules()):
        #     print(idx, '->', m)
        # fsdf

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = self.fc2(h1)
        mean = h2[:,:self.z_size]
        logvar = h2[:,self.z_size:]
        return mean, logvar

    def sample(self, mu, logvar, k):
        
        # if torch.cuda.is_available():
        #     eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_()).cuda() #[P,B,Z]
        # else:
        eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]

        z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]

        # if torch.cuda.is_available():
        #     logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size).cuda()), 
        #                     Variable(torch.zeros(self.B, self.z_size)).cuda())  #[P,B]
        # else:
        logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size).type(self.dtype)), 
                            Variable(torch.zeros(self.B, self.z_size)).type(self.dtype))  #[P,B]


        logqz = lognormal(z, mu, logvar)
        return z, logpz, logqz

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        # return self.fc4(h3)

        z = z.view(-1, self.z_size)

        Ws, log_p_W_sum, log_q_W_sum = self.decoder.sample_weights()

        x = self.decoder.forward(Ws, z)

        x = x.view(self.k, self.B, self.input_size)

        return x, log_p_W_sum, log_q_W_sum


    def forward(self, x, k=1):
        self.k = k
        self.B = x.size()[0]
        mu, logvar = self.encode(x)
        z, logpz, logqz = self.sample(mu, logvar, k=k)
        x_hat, logpW, logqW = self.decode(z)

        logpx = log_bernoulli(x_hat, x)  #[P,B]


        elbo = logpx + logpz - logqz + (logpW - logqW)*.00000001  #[P,B]

        if k>1:
            max_ = torch.max(elbo, 0)[0] #[B]
            elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

        elbo = torch.mean(elbo) #[1]

        #for printing
        logpx = torch.mean(logpx)
        logpz = torch.mean(logpz)
        logqz = torch.mean(logqz)
        self.x_hat_sigmoid = F.sigmoid(x_hat)

        return elbo, logpx, logpz, logqz, logpW, logqW






print 'Loading data'
with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
    mnist_data = pickle.load(f)

train_x = mnist_data[0][0]
train_y = mnist_data[0][1]
valid_x = mnist_data[1][0]
valid_y = mnist_data[1][1]
test_x = mnist_data[2][0]
test_y = mnist_data[2][1]

train_x = torch.from_numpy(train_x)
test_x = torch.from_numpy(test_x)
train_y = torch.from_numpy(train_y)

print train_x.shape
print test_x.shape
print train_y.shape


model = BVAE()

# if torch.cuda.is_available():
#     print 'GPU available, loading cuda'#, torch.cuda.is_available()
#     model.cuda()
#     train_x = train_x.cuda()


# path_to_load_variables=''
path_to_load_variables=home+'/Documents/tmp/pytorch_bvae.pt'
path_to_save_variables=home+'/Documents/tmp/pytorch_bvae.pt'
# path_to_save_variables=''



train(model=model, train_x=train_x, train_y=train_y, valid_x=[], valid_y=[], 
            path_to_load_variables=path_to_load_variables, 
            path_to_save_variables=path_to_save_variables, 
            epochs=10, batch_size=100, display_epoch=1, k=1)



# print test(model=model, data_x=test_x, path_to_load_variables='', 
#             batch_size=20, display_epoch=100, k=1000)

print 'Done.'


















