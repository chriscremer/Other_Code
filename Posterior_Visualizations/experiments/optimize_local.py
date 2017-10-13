
import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import lognormal2 as lognormal
from utils import log_bernoulli

import time




def optimize_local_gaussian(logposterior, model, x):

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
    for epoch in range(1, 999999):

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

        loss = -(torch.mean(logpx-logqz))

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
                print(consecutive_worse)
                if consecutive_worse> 10:
                    # print ('done')
                    break

            # print (epoch, last_100_avg, consecutive_worse)

            last_100 = []


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
    eps = Variable(torch.FloatTensor(1000, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
    z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
    logqz = lognormal(z, mean, logvar) #[P,B]
    logpx = logposterior(z)

    elbo = logpx-logqz #[P,B]
    vae = torch.mean(elbo)

    max_ = torch.max(elbo, 0)[0] #[B]
    elbo_ = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
    iwae = torch.mean(elbo_)

    return vae, iwae


























