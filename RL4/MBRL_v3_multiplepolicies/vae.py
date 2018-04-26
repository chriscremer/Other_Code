


import numpy as np
import pickle#, cPickle
from os.path import expanduser
home = expanduser("~")

# import matplotlib.pyplot as plt


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



from vae_utils import lognormal2 as lognormal
from vae_utils import log_bernoulli

# from scipy.misc import toimage





class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # self.x_size = 3072
        self.x_size = 84
        self.z_size = 100

        image_channels = 2

        # # self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=2, padding=0, dilation=1, bias=True)
        # # self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=3, kernel_size=3, stride=1, padding=0, dilation=1, bias=True)

        # self.conv1 = nn.Conv2d(image_channels, 32, 8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        # self.intermediate_size = 32*7*7

        # self.fc1 = nn.Linear(self.intermediate_size, 200)
        # self.fc2 = nn.Linear(200, self.z_size*2)
        # self.fc3 = nn.Linear(self.z_size, 200)
        # self.fc4 = nn.Linear(200, self.intermediate_size)

        # # self.deconv1 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=1, kernel_size=5, stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)
        # self.deconv1 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        # self.deconv2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2)
        # self.deconv3 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=image_channels, kernel_size=8, stride=4)


        #v2

        self.conv1 = nn.Conv2d(image_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.intermediate_size = 32*7*7   #1: (84-(8/2) -> 80, 2: 80-4/2 -> 78, 3: 

        self.fc1 = nn.Linear(self.intermediate_size, 200)
        self.fc2 = nn.Linear(200, self.z_size*2)
        self.fc3 = nn.Linear(self.z_size, 200)
        self.fc4 = nn.Linear(200, self.intermediate_size)

        # self.deconv1 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=1, kernel_size=5, stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.deconv3 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=image_channels, kernel_size=8, stride=4)










        # self.optimizer = optim.Adam(self.parameters(), lr=.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=.0001, weight_decay=.00001)




    def encode(self, x):

        # x = x.view(-1, 1, self.x_size, self.x_size)
        # print (x.shape)

        x = F.relu(self.conv1(x))

        # print (x.shape)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # print (x.size())

        x = x.view(-1, self.intermediate_size)

        h1 = F.relu(self.fc1(x))
        h2 = self.fc2(h1)
        mean = h2[:,:self.z_size]
        logvar = h2[:,self.z_size:]
        return mean, logvar


    def sample(self, mu, logvar, k):

        # print (mu)
        # print (logvar)


        if torch.cuda.is_available():
            eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_()).cuda() #[P,B,Z]

            # print (mu.size())
            # print (logvar.size())
            # print (eps.size())

            z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
            logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size).cuda()), 
                                Variable(torch.zeros(self.B, self.z_size)).cuda())  #[P,B]
            logqz = lognormal(z, mu, logvar)
        else:
            eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_())#[P,B,Z]
            z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
            logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size)), 
                                Variable(torch.zeros(self.B, self.z_size)))  #[P,B]
            logqz = lognormal(z, mu, logvar) 
        return z, logpz, logqz



    def decode(self, z):

        z = F.relu(self.fc3(z)) 
        z = F.relu(self.fc4(z))  #[B,1960]


        z = z.view(-1, 32, 7, 7)

        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        x = self.deconv3(z)

        # print (x.size())
        # fdsa


        # z = z.view(-1, 10, 14, 14)
        # z = self.deconv1(z)
        # z = z.view(-1, self.x_size)

        # x = x.view(-1, 84*84)
        
        return x




    def forward(self, x, k=1):
        # x: [B,2,84,84]
        
        self.B = x.size()[0]

        mu, logvar = self.encode(x)
        # print (mu.size())
        # print (logvar.size())
        # mu = mu.unsqueeze(0)
        # logvar = logvar.unsqueeze(0)
        z, logpz, logqz = self.sample(mu, logvar, k=k)  #[P,B,Z]
        x_hat = self.decode(z)  #[PB,X]
        x_hat = x_hat.view(k, self.B, -1)
        # print x_hat.size()

        x = x.view(self.B, -1)

        # print (x_hat.size())
        # print (x.size())
        # fasdfd

        logpx = log_bernoulli(x_hat, x)  #[P,B]

        elbo = logpx + logpz - logqz  #[P,B]

        if k>1:
            max_ = torch.max(elbo, 0)[0] #[B]
            elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

        elbo = torch.mean(elbo) #[1]

        #for printing
        logpx = torch.mean(logpx)
        logpz = torch.mean(logpz)
        logqz = torch.mean(logqz)
        self.x_hat_sigmoid = F.sigmoid(x_hat)

        return elbo, logpx, logpz, logqz




    def reconstruction(self, x):
        # x: [B,2,84,84]

        x = Variable(torch.from_numpy(np.array(x)).float()).cuda() 

        self.B = x.size()[0]
        k =1

        mu, logvar = self.encode(x)
        # print (mu.size())
        # print (logvar.size())
        # mu = mu.unsqueeze(0)
        # logvar = logvar.unsqueeze(0)
        z, logpz, logqz = self.sample(mu, logvar, k=k)  #[P,B,Z]
        x_hat = self.decode(z)  #[PB,X]
        # x_hat = x_hat.view(k, self.B, -1)
        # print x_hat.size()

        # x = x.view(self.B, -1)


        return F.sigmoid(x_hat).data.cpu().numpy() #[B,X]





    # def forward2(self, x, k=1):


    #     x = x  / 255.0
        
    #     self.B = x.size()[0]
    #     mu, logvar = self.encode(x)
    #     z, logpz, logqz = self.sample(mu, logvar, k=k)  #[P,B,Z]
    #     x_hat = self.decode(z)  #[PB,X]
    #     x_hat = x_hat.view(k, self.B, -1)
    #     # print x_hat.size()

    #     x = x.view(self.B, -1)

    #     # print (x_hat.size())
    #     # print (x.size())
    #     # fasdfd

    #     logpx = log_bernoulli(x_hat, x)  #[P,B]

    #     elbo = logpx #+ logpz - logqz  #[P,B]

    #     if k>1:
    #         max_ = torch.max(elbo, 0)[0] #[B]
    #         elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

    #     # elbo = torch.mean(elbo) #[1]

    #     # #for printing
    #     # logpx = torch.mean(logpx)
    #     # logpz = torch.mean(logpz)
    #     # logqz = torch.mean(logqz)
    #     # self.x_hat_sigmoid = F.sigmoid(x_hat)

    #     return elbo





    # def forward3(self, x, k=1):
        
    #     self.B = x.size()[0]
    #     mu, logvar = self.encode(x)
    #     z, logpz, logqz = self.sample(mu, logvar, k=k)  #[P,B,Z]
    #     x_hat = self.decode(z)  #[PB,X]
    #     x_hat = x_hat.view(k, self.B, -1)
    #     # print x_hat.size()

    #     x = x.view(self.B, -1)

    #     # print (x_hat.size())
    #     # print (x.size())
    #     # fasdfd

    #     logpx = log_bernoulli(x_hat, x)  #[P,B]

    #     elbo = logpx #+ logpz - logqz  #[P,B]

    #     if k>1:
    #         max_ = torch.max(elbo, 0)[0] #[B]
    #         elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

    #     elbo = torch.mean(elbo) #[1]

    #     #for printing
    #     logpx = torch.mean(logpx)
    #     logpz = torch.mean(logpz)
    #     logqz = torch.mean(logqz)
    #     self.x_hat_sigmoid = F.sigmoid(x_hat)

    #     return elbo, logpx, logpz, logqz, self.x_hat_sigmoid






    # def update(self, data):

    #     k=1
    #     data = data  / 255.0

    #     # if data.is_cuda:
    #     if torch.cuda.is_available():
    #         data = Variable(data).type(torch.cuda.FloatTensor)# , Variable(target).type(torch.cuda.LongTensor) 
    #     else:
    #         data = Variable(data)#, Variable(target)

    #     self.optimizer.zero_grad()

    #     elbo, logpx, logpz, logqz = self.forward(data, k=k)
    #     loss = -(elbo)

    #     loss.backward()
    #     self.optimizer.step()


    #     return elbo

    #     # print ('vae step')
    #     # print 
    #     # if epoch%display_epoch==0 and batch_idx == 0:
    #     #     print 'Train Epoch: {}/{} [{}/{} ({:.0f}%)]'.format(epoch, epochs, 
    #     #             batch_idx * len(data), len(train_loader.dataset),
    #     #             100. * batch_idx / len(train_loader)), \
    #     #         'Loss:{:.4f}'.format(loss.data[0]), \
    #     #         'logpx:{:.4f}'.format(logpx.data[0]), \
    #     #         'logpz:{:.4f}'.format(logpz.data[0]), \
    #     #         'logqz:{:.4f}'.format(logqz.data[0]) 



    def train(self, train_x, epochs):

        batch_size = 20
        k=1
        display_step = 100 
        # epochs = 20

        train_y = torch.from_numpy(np.zeros(len(train_x)))
        train_x = torch.from_numpy(np.array(train_x)).float().cuda() #.type(model.dtype)
        train_ = torch.utils.data.TensorDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True)

        total_steps = 0
        for epoch in range(epochs):

            for batch_idx, (data, target) in enumerate(train_loader):

                batch = Variable(data) /255.0

                # # if data.is_cuda:
                # if torch.cuda.is_available():
                #     data = Variable(data).type(torch.cuda.FloatTensor)# , Variable(target).type(torch.cuda.LongTensor) 
                # else:
                #     data = Variable(data)#, Variable(target)

                self.optimizer.zero_grad()

                elbo, logpx, logpz, logqz = self.forward(batch, k=k)
                loss = -(elbo)

                loss.backward()
                self.optimizer.step()

                total_steps+=1

                if total_steps%display_step==0: # and batch_idx == 0:
                    print ('Train Epoch: {}/{}'.format(epoch+1, epochs),
                        # 'total_epochs {}'.format(total_epochs),
                        'LL:{:.3f}'.format(-loss.data[0]),
                        'logpx:{:.3f}'.format(logpx.data[0]),
                        'logpz:{:.3f}'.format(logpz.data[0]),
                        'logqz:{:.3f}'.format(logqz.data[0]),
                        )





    def save_params(self, path_to_save_variables):
        torch.save(self.state_dict(), path_to_save_variables)
        print ('Saved variables to ' + path_to_save_variables)


    def load_params(self, path_to_load_variables):
        self.load_state_dict(torch.load(path_to_load_variables))
        print ('loaded variables ' + path_to_load_variables)












# def train(model, train_x, train_y, valid_x=[], valid_y=[], 
#             path_to_load_variables='', path_to_save_variables='', 
#             epochs=10, batch_size=20, display_epoch=2, k=1):
    

#     # if path_to_load_variables != '':
#     #     model.load_state_dict(torch.load(path_to_load_variables))
#     #     print 'loaded variables ' + path_to_load_variables

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

#             elbo, logpx, logpz, logqz = model.forward(data, k=k)
#             loss = -(elbo)

#             loss.backward()
#             optimizer.step()

#             if epoch%display_epoch==0 and batch_idx == 0:
#                 print 'Train Epoch: {}/{} [{}/{} ({:.0f}%)]'.format(epoch, epochs, 
#                         batch_idx * len(data), len(train_loader.dataset),
#                         100. * batch_idx / len(train_loader)), \
#                     'Loss:{:.4f}'.format(loss.data[0]), \
#                     'logpx:{:.4f}'.format(logpx.data[0]), \
#                     'logpz:{:.4f}'.format(logpz.data[0]), \
#                     'logqz:{:.4f}'.format(logqz.data[0]) 







