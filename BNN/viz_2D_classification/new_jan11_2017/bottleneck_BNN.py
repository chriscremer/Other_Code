






import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F







class bottleneck_BNN(nn.Module):

    def __init__(self, hyper_config):
        super(bottleneck_BNN, self).__init__()

        self.hyper_config = hyper_config
        self.act_func = hyper_config['act_func']

        if hyper_config['use_gpu']:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        arch_1 = hyper_config['first_half']
        self.arch_2 = hyper_config['second_half']


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
        for i in range(len(self.arch_2)):
            self.second_half_weights.append(nn.Linear(self.arch_2[i][0], self.arch_2[i][1]))

            self.add_module('secondhalf_'+str(count), self.second_half_weights[-1])
            count+=1

        self.loss = torch.nn.CrossEntropyLoss()


    def encode(self, x):

        for i in range(len(self.first_half_weights)-1):
            x = self.act_func(self.first_half_weights[i](x))
        x = self.first_half_weights[-1](x)
        return x


    def decode(self, x):

        for i in range(len(self.second_half_weights)-1):
            pre_act = self.second_half_weights[i](x) #[B,D]
            # pre_act_with_noise = Variable(torch.randn(1, self.arch_2[i][1]).type(self.dtype)) * pre_act
            probs = torch.ones(1, self.arch_2[i][1]) * .5
            pre_act_with_noise = Variable(torch.bernoulli(probs).type(self.dtype)) * pre_act
            x = self.act_func(pre_act_with_noise)
        y_hat = self.second_half_weights[-1](x)
        return y_hat



    def feedforward(self, x):

        x = self.encode(x)
        y_hat = self.decode(x)
        return y_hat



    def forward(self, x, y):

        y_hat = self.feedforward(x) #[B,O]
        loss = self.loss(y_hat, y) #[1]

        return loss



    def accuracy(self, x, y):

        #Integrate out noise
        y_hats = []
        for i in range(10):
            y_hat = self.feedforward(x) #[B,O]
            y_hat = F.softmax(y_hat)
            y_hats.append(y_hat)

        y_hat = torch.stack(y_hats)  #[P,B,O]
        y_hat = torch.mean(y_hat, 0) #[B,O]

        values, indices = torch.max(y_hat, 1)
        mean = torch.mean((indices == y).type(torch.FloatTensor))

        return mean



    def predict_from_encoding(self, x):

        #Integrate out noise
        y_hats = []
        for i in range(10):
            y_hat = self.decode(x) #[B,O]
            y_hat = F.softmax(y_hat)
            y_hats.append(y_hat)

        y_hat = torch.stack(y_hats)  #[P,B,O]
        y_hat = torch.mean(y_hat, 0) #[B,O]

        return y_hat



