








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




        self.act_func = F.leaky_relu# F.relu # # F.tanh ##  F.elu F.softplus



        # self.optimizer = optim.Adam(self.parameters(), lr=.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=.0005, weight_decay=.00001)




    def encode(self, x):

        # x = x.view(-1, 1, self.x_size, self.x_size)
        # print (x.shape)

        x = self.act_func(self.conv1(x))

        # print (x.shape)
        x = self.act_func(self.conv2(x))
        x = self.act_func(self.conv3(x))

        # print (x.size())

        x = x.view(-1, self.intermediate_size)

        h1 = self.act_func(self.fc1(x))
        h2 = self.fc2(h1)
        mean = h2[:,:self.z_size]
        logvar = h2[:,self.z_size:]

        #this solves the nan grad problem.
        logvar = torch.clamp(logvar, min=-20.)


        self.mean = mean
        self.logvar = logvar


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



            # logqz = lognormal(z, mu, logvar)

            logqz = lognormal(z, Variable(mu.data), Variable(logvar.data))



        else:
            eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_())#[P,B,Z]
            z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
            logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size)), 
                                Variable(torch.zeros(self.B, self.z_size)))  #[P,B]
            logqz = lognormal(z, mu, logvar) 
        return z, logpz, logqz



    def decode(self, z):

        z = self.act_func(self.fc3(z)) 
        z = self.act_func(self.fc4(z))  #[B,1960]


        z = z.view(-1, 32, 7, 7)

        z = self.act_func(self.deconv1(z))
        z = self.act_func(self.deconv2(z))
        x = self.deconv3(z)

        # print (x.size())
        # fdsa


        # z = z.view(-1, 10, 14, 14)
        # z = self.deconv1(z)
        # z = z.view(-1, self.x_size)

        # x = x.view(-1, 84*84)
        
        return x




    def forward(self, x, policy, k=1):
        # x: [B,2,84,84]
        
        self.B = x.size()[0]

        mu, logvar = self.encode(x)
        # print (mu.size())
        # print (logvar.size())
        # mu = mu.unsqueeze(0)
        # logvar = logvar.unsqueeze(0)
        z, logpz, logqz = self.sample(mu, logvar, k=k)  #[P,B,Z]
        x_hat = self.decode(z)  #[PB,X]

        
        # dist_recon = policy.action_dist(F.sigmoid(x_hat)*255.)
        log_dist_recon = policy.action_logdist(F.sigmoid(x_hat)*255.)



        # print (torch.sum(torch.autograd.grad(torch.sum(torch.log(dist_recon)), self.deconv3.weight)[0]))
        # print (torch.sum(torch.autograd.grad(torch.sum(torch.log(dist_recon)), self.deconv3.weight)[0]))
        # print (torch.sum(torch.autograd.grad(torch.sum(x_hat*10), self.deconv3.weight)[0]))
        # fsadf

        # dist_true = policy.action_dist(x*255.)
        log_dist_true = policy.action_logdist(x*255.)



        # print (dist_true)
        # print (dist_recon)

        # fasdf
        




        # # print (x_hat.size())
        # # print (dist_recon)
        # # print (dist_true)

        # # fads

        flat_x_hat = x_hat.view(k, self.B, -1)
        # # print x_hat.size()

        flat_x = x.view(self.B, -1)

        # # print (x_hat.size())
        # # print (x.size())
        # # fasdfd

        logpx = log_bernoulli(flat_x_hat, flat_x)  #[P,B]
        # print (logpx.size())
        
        # elbo = logpx + logpz - logqz  #[P,B]


        # action_dif = torch.mean((dist_recon-dist_true)**2)

        # action_dif = torch.sum((torch.log(dist_true) - torch.log(dist_recon))*dist_true)
        action_dist_kl = torch.sum((log_dist_true - log_dist_recon)*torch.exp(log_dist_true), dim=1) #[B]

        # print (action_dif.size())
        # fasdf



        # neg_action_dif = - action_dif 



        # print (torch.sum(torch.autograd.grad(neg_action_dif, policy.conv1.weight)[0]))      # ZERO 
        # fdsa

        # print (torch.sum(torch.autograd.grad(neg_action_dif, self.deconv3.weight)[0]))
        # print (torch.sum(torch.autograd.grad(neg_action_dif, self.deconv1.weight)[0]))
        # print (torch.sum(torch.autograd.grad(neg_action_dif, self.conv1.weight)[0]))



        # print (torch.sum(torch.autograd.grad(torch.sum(torch.log(dist_recon)), self.deconv3.weight)[0]))
        # print (torch.sum(torch.autograd.grad(torch.sum(torch.log(dist_true)), self.deconv3.weight)[0]))

        # fadf

        # elbo = logpx + logpz - logqz 


        # if k>1:
        #     max_ = torch.max(elbo, 0)[0] #[B]
        #     elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

        # elbo = torch.mean(elbo) #[1]


        # neg_action_dif = neg_action_dif
        # logpz = logpz*.01
        # logqz = logqz*.01
        # logpx = logpx*.01

        # elbo = torch.mean(logpx) + torch.mean(logpz) - torch.mean(logqz) - torch.mean(action_dist_kl)  #[1]

        #for printing
        action_dist_kl = torch.mean(action_dist_kl)
        weight = .0000001 # .00001
        logpx = torch.mean(logpx) * weight
        logpz = torch.mean(logpz) * weight  *.1
        logqz = torch.mean(logqz) * weight  *.1

        elbo = torch.mean(logpx) + torch.mean(logpz) - torch.mean(logqz) - torch.mean(action_dist_kl)  #[1]

        # self.x_hat_sigmoid = F.sigmoid(x_hat)

        # return elbo, logpx, logpz, logqz

        # print (logpx != logpx)

        # if (logpx != logpx).data.cpu().numpy():

        #     print( 'NAN')
        #     fasd


        return elbo, logpx, logpz, logqz, action_dist_kl





    def reconstruction(self, x):
        # x: [B,2,84,84]

        x = Variable(torch.from_numpy(np.array(x)).float()).cuda() /255.0

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




    def get_action_dist(self, x, policy):

        x = Variable(torch.from_numpy(np.array(x)).float()).cuda()

        # dist_recon = policy.action_dist(F.sigmoid(x_hat)*255.)

        dist = policy.action_dist(x)

        return dist.data.cpu().numpy()





    def get_kl_error(self, x, policy):

        k=1

        x = Variable(torch.from_numpy(np.array(x)).float()).cuda()

        self.B = x.size()[0]

        mu, logvar = self.encode(x)
        z, logpz, logqz = self.sample(mu, logvar, k=k)  #[P,B,Z]
        x_hat = self.decode(z)  #[PB,X]

        # print (torch.max(x))
        
        log_dist_recon = policy.action_logdist(F.sigmoid(x_hat)*255.)
        log_dist_true = policy.action_logdist(x)

        dist_recon = policy.action_dist(F.sigmoid(x_hat)*255.)
        dist_true = policy.action_dist(x)

        print ()
        print (dist_true)
        print (log_dist_true)
        print (torch.log(dist_true))
        print ()

        print ()
        print (dist_recon)
        print (log_dist_recon)
        print (torch.log(dist_recon))
        print ()
        # fasdf


        action_dist_kl = torch.sum((log_dist_true - log_dist_recon)*torch.exp(log_dist_true), dim=1) #[B]

        action_dist_kl_v2 = torch.sum((torch.log(dist_true) - torch.log(dist_recon))*dist_true, dim=1) #[B]

        return action_dist_kl, action_dist_kl_v2













    def train(self, train_x, epochs, policy):

        batch_size = 40
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

                # if batch_idx < 5: 
                #     continue
                # print (torch.max(data))
                # fad
                batch = Variable(data) /255.0

                # # if data.is_cuda:
                # if torch.cuda.is_available():
                #     data = Variable(data).type(torch.cuda.FloatTensor)# , Variable(target).type(torch.cuda.LongTensor) 
                # else:
                #     data = Variable(data)#, Variable(target)

                self.optimizer.zero_grad()

                elbo, logpx, logpz, logqz, action_dist_kl = self.forward(batch, policy, k=k)
                loss = -(elbo)

                loss.backward()

                # print(self.fc3.weight.grad) #().grad)
                # print(torch.sum(self.fc3.weight.grad)) #().grad)
                # print(torch.sum(self.fc1.weight.grad)) #().grad)
                # print(torch.sum(self.deconv3.weight.grad)) #().grad)
                # fadsf

                nn.utils.clip_grad_norm(self.parameters(), .5)

                # print (self.parameters())
                # fsda


                # #WHEN LOOKING FOR NANS
                # # print (policy)
                # for param in self.parameters():
                #     # print (param.size())

                #     # print (param.grad)
                #     aa = np.any((param.grad != param.grad).data.cpu().numpy())
                #     bb = np.any((param.grad > 99999.).data.cpu().numpy())
                #     cc = np.any((param.grad < -99999.).data.cpu().numpy())
                #     dd = np.any((param != param).data.cpu().numpy())


                #     if aa:
                #         print ("NAN grad")


                #         print ('Train Epoch: {}/{}'.format(epoch+1, epochs),
                #             # 'total_epochs {}'.format(total_epochs),
                #             'LL:{:.3f}'.format(-loss.data[0]),
                #             'logpx:{:.3f}'.format(logpx.data[0]),
                #             'logpz:{:.5f}'.format(logpz.data[0]),
                #             'logqz:{:.5f}'.format(logqz.data[0]),
                #             'action_kl:{:.3f}'.format(action_dist_kl.data[0])
                #             )

                #         print (self.mean)
                #         print ()
                #         print (self.logvar)
                #         print ()

                #         print (torch.max(self.logvar))
                #         print ()

                #         print (torch.min(self.logvar))



                #         print (aa,bb,cc,dd)

                #         for name, param in self.named_parameters():
                #             if np.any((param.grad != param.grad).data.cpu().numpy()):
                #                 print (name, param.size())
                #         print (self)
                #         fdsfa

                #     if bb:
                #         print ('INF')
                #         print (aa,bb,cc,dd)
                #         fdsfa

                #     if cc:
                #         print ('-INF')
                #         print (aa,bb,cc,dd)
                #         fdsfa

                #     if dd:
                #         print ("NAN param")
                #         print (aa,bb,cc,dd)
                #         print (param.size())

                #         fdsfa
                #     # bb = np.any((param.grad == float('inf')).data.cpu().numpy())



                        
                # fad


                self.optimizer.step()

                # total_steps+=1

                if total_steps%display_step==0: # and batch_idx == 0:
                    print ('Train Epoch: {}/{}'.format(epoch+1, epochs),
                        # 'total_epochs {}'.format(total_epochs),
                        'LL:{:.3f}'.format(-loss.data[0]),
                        'logpx:{:.4f}'.format(logpx.data[0]),
                        'logpz:{:.5f}'.format(logpz.data[0]),
                        'logqz:{:.5f}'.format(logqz.data[0]),
                        'action_kl:{:.4f}'.format(action_dist_kl.data[0])
                        )

                if (logpx != logpx).data.cpu().numpy():
                    print( 'NAN')

                    print ('Train Epoch: {}/{}'.format(epoch+1, epochs),
                        # 'total_epochs {}'.format(total_epochs),
                        'LL:{:.3f}'.format(-loss.data[0]),
                        'logpx:{:.3f}'.format(logpx.data[0]),
                        'logpz:{:.3f}'.format(logpz.data[0]),
                        'logqz:{:.3f}'.format(logqz.data[0]),
                        'action_kl:{:.3f}'.format(action_dist_kl.data[0])
                        )

                    for param in self.parameters():
                        aa = np.any((param != param).data.cpu().numpy())
                        if aa:

                            print (param.size())
                            # print (param)


                    print (policy)

                    fasd


                total_steps+=1





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







