






import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import lognormal2 as lognormal
from utils import lognormal333

from utils import log_bernoulli

import time

import pickle

quick = 0


from flow_model import flow1





def optimize_local_gaussian_mean_logvar(logposterior, model, x):

    # B = x.shape[0]
    B = x.size()[0] #batch size
    # input to log posterior is z, [P,B,Z]
    # I think B will be 1 for now



        # self.zeros = Variable(torch.zeros(self.B, self.z_size).type(self.dtype))

    mean = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)
    logvar = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)

    optimizer = optim.Adam([mean, logvar], lr=.001)
    # time_ = time.time()
    # n_data = len(train_x)
    # arr = np.array(range(n_data))

    P = 50


    last_100 = []
    best_last_100_avg = -1
    consecutive_worse = 0
    for epoch in range(1, 99999): # 999999):

        if quick:
        # if 1:

            break

        #Sample
        eps = Variable(torch.FloatTensor(P, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz = lognormal(z, mean, logvar) #[P,B]

        logpx = logposterior(z)

        # print (logpx)
        # print (logqz)

        # fsda

        # data_index= 0
        # for i in range(int(n_data/batch_size)):
            # batch = train_x[data_index:data_index+batch_size]
            # data_index += batch_size

            # batch = Variable(torch.from_numpy(batch)).type(self.dtype)
        optimizer.zero_grad()

        # elbo, logpxz, logqz = self.forward(batch, k=k)

        loss = -(torch.mean( logpx-logqz))
        # print (torch.mean( logpx))
        # print (torch.mean( logqz))
        # fadaf


        loss_np = loss.data.cpu().numpy()
        # print (epoch, loss_np)
        # fasfaf

        loss.backward()
        optimizer.step()

        last_100.append(loss_np)
        if epoch % 100 ==0:

            last_100_avg = np.mean(last_100)
            if last_100_avg< best_last_100_avg or best_last_100_avg == -1:
                consecutive_worse=0
                best_last_100_avg = last_100_avg
            else:
                consecutive_worse +=1 
                # print(consecutive_worse)
                if consecutive_worse> 10:
                    # print ('done')
                    break

            print (epoch, last_100_avg, consecutive_worse,mean)
            # print (torch.mean(logpx))

            last_100 = []

        if epoch %1000 ==0:
            # print (logpx)
            # print (logqz)
            print (torch.mean( logpx))
            print (torch.mean( logqz))
            print (torch.std( logpx))
            print (torch.std( logqz))

        # if epoch%display_epoch==0:
        #     print ('Train Epoch: {}/{}'.format(epoch, epochs),
        #         'LL:{:.3f}'.format(-loss.data[0]),
        #         'logpxz:{:.3f}'.format(logpxz.data[0]),
        #         # 'logpz:{:.3f}'.format(logpz.data[0]),
        #         'logqz:{:.3f}'.format(logqz.data[0]),
        #         'T:{:.2f}'.format(time.time()-time_),
        #         )

        #     time_ = time.time()


    # Compute VAE and IWAE bounds



    #Sample
    # eps = Variable(torch.FloatTensor(1000, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
    # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
    # logqz = lognormal(z, mean, logvar) #[P,B]

    # # print (logqz)
    # # fad
    # logpx = logposterior(z)

    # elbo = logpx-logqz #[P,B]
    # vae = torch.mean(elbo)

    # max_ = torch.max(elbo, 0)[0] #[B]
    # elbo_ = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
    # iwae = torch.mean(elbo_)

    return mean, logvar, z





























def optimize_local_gaussian_mean_logvar2(logposterior, model, x):

    # B = x.shape[0]
    B = x.size()[0] #batch size
    # input to log posterior is z, [P,B,Z]
    # I think B will be 1 for now

    mean = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)
    logvar = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)

    optimizer = optim.Adam([mean, logvar], lr=.001)
    # time_ = time.time()
    # n_data = len(train_x)
    # arr = np.array(range(n_data))

    P = 50


    last_100 = []
    best_last_100_avg = -1
    consecutive_worse = 0
    for epoch in range(1, 99999): # 999999):

        if quick:
        # if 1:

            break

        #Sample
        eps = Variable(torch.FloatTensor(P, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz = lognormal(z, mean, logvar) #[P,B]
        logpx = logposterior(z)

        loss = -(torch.mean( 1.5 * logpx-logqz))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_np = loss.data.cpu().numpy()
        last_100.append(loss_np)
        if epoch % 100 ==0:

            last_100_avg = np.mean(last_100)
            if last_100_avg< best_last_100_avg or best_last_100_avg == -1:
                consecutive_worse=0
                best_last_100_avg = last_100_avg
            else:
                consecutive_worse +=1 
                # print(consecutive_worse)
                if consecutive_worse> 10:
                    # print ('done')
                    break

            print (epoch, last_100_avg, consecutive_worse,mean)
            # print (torch.mean(logpx))

            last_100 = []

        if epoch %1000 ==0:
            # print (logpx)
            # print (logqz)
            print (torch.mean( logpx))
            print (torch.mean( logqz))
            print (torch.std( logpx))
            print (torch.std( logqz))





    #Round 2

    last_100 = []
    best_last_100_avg = -1
    consecutive_worse = 0
    for epoch in range(1, 99999): # 999999):

        if quick:
        # if 1:
            break

        #Sample
        eps = Variable(torch.FloatTensor(P, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz = lognormal(z, mean, logvar) #[P,B]
        logpx = logposterior(z)

        loss = -(torch.mean(logpx-logqz))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_np = loss.data.cpu().numpy()
        last_100.append(loss_np)
        if epoch % 100 ==0:

            last_100_avg = np.mean(last_100)
            if last_100_avg< best_last_100_avg or best_last_100_avg == -1:
                consecutive_worse=0
                best_last_100_avg = last_100_avg
            else:
                consecutive_worse +=1 
                # print(consecutive_worse)
                if consecutive_worse> 10:
                    # print ('done')
                    break

            print (epoch, last_100_avg, consecutive_worse,mean, '2')
            # print (torch.mean(logpx))

            last_100 = []

        if epoch %1000 ==0:
            # print (logpx)
            # print (logqz)
            print (torch.mean( logpx))
            print (torch.mean( logqz))
            print (torch.std( logpx))
            print (torch.std( logqz))


















    return mean, logvar, z












































def optimize_local_flow1(logposterior, model, x):




    # x_size = 784
    # z_size = 2

    # hyper_config = { 
    #                 'x_size': x_size,
    #                 'z_size': z_size,
    #                 'act_func': F.tanh,# F.relu,
    #                 'n_flows': 2,
    #                 # 'flow_hidden_size': 30
    #                 'flow_hidden_size': 120
    #             }



    # # B = x.shape[0]
    # B = x.size()[0] #batch size

    # n_flows = 2 # 8 #2 #hyper_config['n_flows']

    # z_size = model.z_size
    # x_size = model.x_size
    # act_func = model.act_func

    # all_params = []


    # mean = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)
    # logvar = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)

    # all_params.append(mean)
    # all_params.append(logvar)


    # h_s = hyper_config['flow_hidden_size']

    # params = []
    # for i in range(n_flows):
    #     params.append([nn.Linear(z_size, h_s).cuda(), nn.Linear(h_s, z_size).cuda(), nn.Linear(h_s, z_size).cuda()])

    #     all_params.append(params[i][0].weight)
    #     all_params.append(params[i][1].weight)
    #     all_params.append(params[i][2].weight)




    # optimizer = optim.Adam(all_params, lr=.001)

    n_flows = 6
    flow = flow1(n_flows=n_flows).cuda()


    P = 100
    # k = P


    last_100 = []
    best_last_100_avg = -1
    consecutive_worse = 0
    for epoch in range(1, 99999):

        z, logq = flow.sample(P)
        logpx = logposterior(z)

        loss = -(torch.mean( logpx-logq))
        
        flow.optimizer.zero_grad()
        loss.backward()
        flow.optimizer.step()

        loss_np = loss.data.cpu().numpy()
        last_100.append(loss_np)
        if epoch % 100 ==0:

            last_100_avg = np.mean(last_100)
            if last_100_avg< best_last_100_avg or best_last_100_avg == -1:
                consecutive_worse=0
                best_last_100_avg = last_100_avg
            else:
                consecutive_worse +=1 
                # print(consecutive_worse)
                if consecutive_worse> 10:
                    # print ('done')
                    break

            print (epoch, last_100_avg, consecutive_worse)
            # print(z[0])
            # print (torch.mean(logpx).data.cpu().numpy())
            # print (torch.mean(logqz0).data.cpu().numpy(),torch.mean(logqv0).data.cpu().numpy(),torch.mean(logdetsum).data.cpu().numpy(),torch.mean(logrvT).data.cpu().numpy())

            last_100 = []




    # #Sample
    # k=1000

    # z, logq = sample(k)



    return flow














