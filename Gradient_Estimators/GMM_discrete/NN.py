

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
    def __init__(self, input_size, output_size, seed=1):
        super(NN, self).__init__()

        torch.manual_seed(seed)

        self.input_size = input_size
        self.output_size = output_size
        h_size = 50

        # self.net = nn.Sequential(
        #   nn.Linear(self.input_size,h_size),
        #   nn.ReLU(),
        #   nn.Linear(h_size,self.output_size)
        # )

        # self.net = nn.Sequential(
        #   nn.Linear(self.input_size,h_size),
        #   # nn.Tanh(),
        #   # nn.Linear(h_size,h_size),
        #   nn.Tanh(),
        #   nn.Linear(h_size,self.output_size),
        #   # nn.Tanh(),
        #   # nn.Linear(h_size,self.output_size)
        # )

        self.net = nn.Sequential(
          nn.Linear(self.input_size,h_size),
          # nn.Tanh(),
          # nn.Linear(h_size,h_size),
          nn.LeakyReLU(),
          nn.Linear(h_size,h_size),
          nn.LeakyReLU(),
          nn.Linear(h_size,self.output_size)
        )

        # self.optimizer = optim.Adam(self.parameters(), lr=.01)
        # self.optimizer = optim.Adam(self.parameters(), lr=.0004)







class NN2(nn.Module):
    def __init__(self, input_size, output_size, seed=1):
        super(NN2, self).__init__()

        torch.manual_seed(seed)

        self.input_size = input_size
        self.output_size = output_size
        h_size = 50

        self.net = nn.Sequential(
          nn.Linear(self.input_size,h_size),
          # nn.Tanh(),
          # nn.Linear(h_size,h_size),
          nn.LeakyReLU(),
          nn.Linear(h_size,h_size),
          # nn.Tanh(),
          nn.LeakyReLU(),
          nn.Linear(h_size,h_size),
          # nn.Tanh(),
          nn.LeakyReLU(),
          nn.Linear(h_size,self.output_size),
        )


















class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  
                        # nn.LeakyReLU(inplace=True),
                        nn.LeakyReLU(),
                        nn.Linear(in_features, in_features),
                        nn.LayerNorm(in_features),
                        # nn.LeakyReLU(inplace=True),
                        nn.LeakyReLU(),
                        nn.Linear(in_features, in_features),
                        # nn.LayerNorm(in_features),
                        ]
                        # nn.BatchNorm1d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class NN3(nn.Module):
    def __init__(self, input_size, output_size, seed=1, n_residual_blocks=3, width=50):
        super(NN3, self).__init__()

        torch.manual_seed(seed)

        self.input_size = input_size
        self.output_size = output_size
        h_size = width

        self.first_layer = nn.Linear(self.input_size,h_size)
        self.last_layer = nn.Linear(h_size,self.output_size)

        # n_residual_blocks = 5
        model = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(h_size)]
        self.part3 = nn.Sequential(*model)

    def net(self, x):

        # out = F.leaky_relu(self.first_layer(x))
        out = self.first_layer(x)
        out = self.part3(out)
        out = self.last_layer(out)

        return out


    def load_params_v3(self, save_dir, name, step):
        save_to=os.path.join(save_dir, name + str(step)+".pt")
        state_dict = torch.load(save_to)
        # # # print (state_dict)
        # for key, val in state_dict.items():
        #     print (key)
        # fddsf
        self.load_state_dict(state_dict)
        print ('loaded params', save_to)


    def save_params_v3(self, save_dir, name, step):
        save_to=os.path.join(save_dir, name + str(step)+".pt")
        torch.save(self.state_dict(), save_to)
        print ('saved params', save_to)
        












class NN4(nn.Module):
    def __init__(self, input_size, output_size, seed=1, n_residual_blocks=3):
        super(NN4, self).__init__()

        torch.manual_seed(seed)

        self.input_size = input_size
        self.output_size = output_size
        h_size = 50

        # self.net = nn.Sequential(
        #   nn.Linear(self.input_size,h_size),
        #   nn.BatchNorm1d(h_size),
        #   # nn.Tanh(),
        #   # nn.Linear(h_size,h_size),
        #   nn.LeakyReLU(),
        #   nn.Linear(h_size,h_size),
        #   nn.BatchNorm1d(h_size),
        #   # nn.Tanh(),
        #   nn.LeakyReLU(),
        #   nn.Linear(h_size,h_size),
        #   nn.BatchNorm1d(h_size),
        #   # nn.Tanh(),
        #   nn.LeakyReLU(),
        #   nn.Linear(h_size,self.output_size),
        # )

        self.first_layer = nn.Linear(self.input_size,h_size)
        
        # n_residual_blocks = 5
        model = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(h_size)]
        self.part3 = nn.Sequential(*model)

        self.last_layer = nn.Linear(h_size,self.output_size)


        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(h_size)]
        self.part4 = nn.Sequential(*model2)

        self.last_layer2 = nn.Linear(h_size,self.output_size)

    def net(self, x):

        # out = F.leaky_relu(self.first_layer(x))
        out = self.first_layer(x)

        out1 = self.part3(out)
        out2 = self.part4(out)

        out1 = self.last_layer(out1)
        out2 = torch.sigmoid(self.last_layer2(out2))

        return out1, out2












# class NN5(nn.Module):
#     def __init__(self, input_size, output_size, seed=1, n_residual_blocks=3):
#         super(NN4, self).__init__()

#         torch.manual_seed(seed)

#         self.input_size = input_size
#         self.output_size = output_size
#         h_size = 50

#         # self.net = nn.Sequential(
#         #   nn.Linear(self.input_size,h_size),
#         #   nn.BatchNorm1d(h_size),
#         #   # nn.Tanh(),
#         #   # nn.Linear(h_size,h_size),
#         #   nn.LeakyReLU(),
#         #   nn.Linear(h_size,h_size),
#         #   nn.BatchNorm1d(h_size),
#         #   # nn.Tanh(),
#         #   nn.LeakyReLU(),
#         #   nn.Linear(h_size,h_size),
#         #   nn.BatchNorm1d(h_size),
#         #   # nn.Tanh(),
#         #   nn.LeakyReLU(),
#         #   nn.Linear(h_size,self.output_size),
#         # )

#         self.first_layer = nn.Linear(self.input_size,h_size)
        
#         # n_residual_blocks = 5
#         model = []
#         # Residual blocks
#         for _ in range(n_residual_blocks):
#             model += [ResidualBlock(h_size)]
#         self.part3 = nn.Sequential(*model)

#         self.last_layer = nn.Linear(h_size,self.output_size)


#         # model2 = []
#         # # Residual blocks
#         # for _ in range(n_residual_blocks):
#         #     model2 += [ResidualBlock(h_size)]
#         # self.part4 = nn.Sequential(*model2)

#         # self.last_layer2 = nn.Linear(h_size,self.output_size)

#     def net(self, x):

#         # out = F.leaky_relu(self.first_layer(x))
#         out = self.first_layer(x)

#         out1 = self.part3(out)
#         # out2 = self.part4(out)

#         out1 = torch.sigmoid(self.last_layer(out1))
#         # out2 = torch.sigmoid(self.last_layer2(out2))

#         return out1















































