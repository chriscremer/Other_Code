


import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F






class NN(nn.Module):
    def __init__(self, seed=1):
        super(NN, self).__init__()

        torch.manual_seed(seed)

        self.input_size = 5
        self.output_size = 1
        h_size = 50

        self.net = nn.Sequential(
          nn.Linear(self.input_size,h_size),
          nn.ReLU(),
          nn.Linear(h_size,self.output_size)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=.001)


    def forward(self, input_x):

        input_x = Variable(torch.FloatTensor(input_x))
        return self.net(input_x)





    def train(self, data_x, data_y):
        #data_x: [B,X]
        #data_y: [B,Y]

        data_x = Variable(torch.FloatTensor(data_x))
        data_y = Variable(torch.FloatTensor(data_y))

        done = 0
        step_count = 0
        last_100 = []
        best_100 = 0
        not_better_counter = 0
        first = 1
        # while not done:

        #sample batch
        # samps = sampler(20) #[B,X]
        # print (samps.size())
        # obj_value = objective(samps).unsqueeze(1) #[B,1]
        pred = self.net(data_x) #[B,Y]

        loss = torch.mean((data_y - pred)**2) #[1]

        # print(step_count, loss.data.numpy())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # break

            

            # if len(last_100) < 100:
            #     last_100.append(loss)
            # else:
            #     last_100_avg = torch.mean(torch.stack(last_100))
            #     # print(step_count, loss.data.numpy())
            #     print(step_count, last_100_avg.data.numpy())
            #     # print(best_100, last_100_avg)
            #     # print (last_100_avg< best_100)
            #     if first or (last_100_avg< best_100).data.numpy():
            #         first = 0
            #         best_100 = last_100_avg
            #         not_better_counter = 0
            #     else:
            #         not_better_counter +=1
            #         if not_better_counter == 3:
            #             break
            #     last_100 = []

            # step_count+=1



    def train_on_dataset(self, dataset):

        for step in range(100):

            #Sample datapoint
            datapoint = dataset[np.random.randint(len(dataset))]
            s_t, a_t, r_t, s_tp1, done = datapoint

            next_state_and_rand_action = np.array([np.concatenate([s_tp1,np.array([np.random.randint(2)])])])
            Q_next_state = self.forward(next_state_and_rand_action).data.numpy()
            if not done:
                R_and_Q = r_t + Q_next_state 
            else:
                R_and_Q = np.reshape(np.array([r_t]), [1,1])
            
            # print (data_x.shape)  #[B,X+A]
            data_x = np.array([np.concatenate([s_t,np.array([a_t])])])
            self.train(data_x, R_and_Q)








