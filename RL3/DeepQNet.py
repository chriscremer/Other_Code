

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


import numpy as np


class Qnet(nn.Module):
    def __init__(self, specs):
        super(Qnet, self).__init__()


        self.input_size = specs['n_input']
        self.z_size = specs['n_z']
        self.action_size = specs['n_actions']

        self.encoder_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_size+self.z_size+self.action_size, specs['encoder_net'][0]),
            torch.nn.ReLU(),
            torch.nn.Linear(specs['encoder_net'][0], specs['encoder_net'][1]),
            torch.nn.ReLU(),
            torch.nn.Linear(specs['encoder_net'][1], self.z_size*2),
        )


        self.optimizer = optim.Adam(model.parameters(), lr=.001)



    def forward(self, x, a):
        '''
        x: [B,X]
        a: [B,A]
        output: elbo scalar
        '''
        
        self.B = x.size()[0]
        self.T = x.size()[1]
        self.k = k

        return elbo


    def load_params(self,path_to_load_variables):
        model.load_state_dict(torch.load(path_to_load_variables))
        print('loaded variables ' + path_to_load_variables)


    def save_params(self,path_to_save_variables):
        torch.save(model.state_dict(), path_to_save_variables)
        print('Saved variables to ' + path_to_save_variables)


    def step(self):

        observations, actions = Variable(torch.from_numpy(np.array(batch))), Variable(torch.from_numpy(np.array(batch_actions)))

        optimizer.zero_grad()
        elbo, logpx, logpz, logqz = model.forward(observations, actions, k=k)
        loss = -(elbo)
        loss.backward()
        optimizer.step()


