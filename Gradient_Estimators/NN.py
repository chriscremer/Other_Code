

import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


import os



def H(x):
    if x > .5:
        return torch.tensor([1])
    else:
        return torch.tensor([0])




class NN(nn.Module):
    def __init__(self, seed=1):
        super(NN, self).__init__()

        torch.manual_seed(seed)

        self.input_size = 1
        self.output_size = 1
        h_size = 50

        # self.net = nn.Sequential(
        #   nn.Linear(self.input_size,h_size),
        #   nn.ReLU(),
        #   nn.Linear(h_size,self.output_size)
        # )
        self.net = nn.Sequential(
          nn.Linear(self.input_size,h_size),
          # nn.Tanh(),
          # nn.Linear(h_size,h_size),
          nn.Tanh(),
          nn.Linear(h_size,self.output_size),
          # nn.Tanh(),
          # nn.Linear(h_size,self.output_size)
        )

        # self.optimizer = optim.Adam(self.parameters(), lr=.01)
        self.optimizer = optim.Adam(self.parameters(), lr=.0004)




    def train(self, func, dist, save_dir):

        done = 0
        step_count = 0
        last_100 = []
        best_100 = 0
        not_better_counter = 0
        first = 1
        while not done:

            #sample batch
            # samps = sampler(20) #[B,X]
            # samps = dist.sample(1)

            bs = []
            zs = []
            for i in range(20):
                samps = dist.sample()

                zs.append(samps)
                bs.append(H(samps))


            # zs = torch.from_numpy(np.array(zs)).unsqueeze(1)
            # bs = torch.from_numpy(np.array(bs)).unsqueeze(1)

            zs = torch.FloatTensor(zs).unsqueeze(1)
            bs = torch.FloatTensor(bs).unsqueeze(1)

            # samps = H(samps)

            # # print (samps.size())
            # obj_value = func(H(samps)).unsqueeze(1) #[B,1]
            # pred = self.net(samps) #[B,1]

            # print (zs.shape)
            # print (bs.shape)
            # fas

            obj_value = func(bs) #[B,1]
            pred = self.net(zs) #[B,1]

            # print (obj_value.shape)
            # print (pred.shape)
            # fas

            # loss = torch.mean(((obj_value - pred)/.1)**2 ) #[1]
            loss = torch.mean((obj_value - pred)**2 ) #[1]

            # print (samps)
            # print (pred)
            # print (loss)

            # fdsa

            # print(step_count, loss.data.numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            

            if len(last_100) <  200:
                last_100.append(loss)
            else:
                last_100_avg = torch.mean(torch.stack(last_100))
                # print(step_count, loss.data.numpy())

                # if step_count % 500==0:
                #     print(step_count, last_100_avg.data.numpy(), not_better_counter)
                
                # print(best_100, last_100_avg)
                # print (last_100_avg< best_100)
                if first or (last_100_avg< best_100).data.numpy():
                    first = 0
                    best_100 = last_100_avg
                    not_better_counter = 0
                else:

                    if step_count % 5==0:
                        print(step_count, last_100_avg.data.numpy(), not_better_counter)
                    
                    not_better_counter +=1
                    # if not_better_counter == 200:
                    if not_better_counter == 100:
                        break
                last_100 = []

            step_count+=1

        self.save_params_v3(save_dir=save_dir, step=step_count, name='')




    def load_params_v3(self, save_dir, step, name=''):
        save_to=os.path.join(save_dir, "net_params_" +name + str(step)+".pt")
        state_dict = torch.load(save_to)
        # # # print (state_dict)
        # for key, val in state_dict.items():
        #     print (key)
        # fddsf
        self.load_state_dict(state_dict)
        print ('loaded params', save_to)


    def save_params_v3(self, save_dir, step, name=''):
        save_to=os.path.join(save_dir, "net_params_"+name + str(step)+".pt")
        torch.save(self.state_dict(), save_to)
        print ('saved params', save_to)
        





