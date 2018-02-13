


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F







class bottleneck_NN(nn.Module):

    def __init__(self, hyper_config):
        super(bottleneck_NN, self).__init__()

        self.hyper_config = hyper_config
        self.act_func = hyper_config['act_func']

        if hyper_config['use_gpu']:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        arch_1 = hyper_config['first_half']
        arch_2 = hyper_config['second_half']


        #First Half
        self.first_half_weights = []
        count =1
        for i in range(len(arch_1)):
            self.first_half_weights.append(nn.Linear(arch_1[i][0], arch_1[i][1]))

            self.add_module('firsthalf_'+str(count), self.first_half_weights[-1])
            count+=1


        #Second Half
        self.second_half_weights = []
        count =1
        for i in range(len(arch_2)):
            self.second_half_weights.append(nn.Linear(arch_2[i][0], arch_2[i][1]))

            self.add_module('secondhalf_'+str(count), self.second_half_weights[-1])
            count+=1

        self.loss = torch.nn.CrossEntropyLoss()



    def feedforward(self, x):

        for i in range(len(self.first_half_weights)-1):

            x = self.act_func(self.first_half_weights[i](x))
            # print (x.size())

        x = self.first_half_weights[-1](x)
        # print (x.size())

        for i in range(len(self.second_half_weights)-1):
            x = self.act_func(self.second_half_weights[i](x))
            # print (x.size())

        y_hat = self.second_half_weights[-1](x)

        return y_hat



    def forward(self, x, y):

        y_hat = self.feedforward(x,y) #[B,O]

        loss = self.loss(y_hat, y) #[1]

        return loss





    def accuracy(self, x, y):

        y_hat = self.feedforward(x,y) #[B,O]
        y_hat = F.softmax(y_hat)

        values, indices = torch.max(y_hat, 1)
        mean = torch.mean((indices == y).type(torch.FloatTensor))

        return mean


    def encode(self, x):

        for i in range(len(self.first_half_weights)-1):

            x = self.act_func(self.first_half_weights[i](x))
            # print (x.size())

        x = self.first_half_weights[-1](x)

        return x


    def predict_from_encoding(self, encodings):

        x = encodings

        # print (x.size())

        for i in range(len(self.second_half_weights)-1):
            x = self.act_func(self.second_half_weights[i](x))
            # print (x.size())

        y_hat = self.second_half_weights[-1](x)
        y_hat = F.softmax(y_hat)

        return y_hat



