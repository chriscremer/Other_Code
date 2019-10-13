import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn

import numpy as np
import pdb




'''
Convolution Layer with zero initialisation
'''
class Conv2dZeroInit(nn.Conv2d):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=0, logscale=3.):
        super().__init__(channels_in, channels_out, filter_size, stride=stride, padding=padding)
        self.register_parameter("logs", nn.Parameter(torch.zeros(channels_out, 1, 1)))
        self.logscale_factor = logscale

    def reset_parameters(self):        
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        out = super().forward(input)
        return out  * torch.exp(self.logs * self.logscale_factor)

'''
Convolution Interlaced with Actnorm
'''
class Conv2dActNorm(nn.Module):
    def __init__(self, channels_in, channels_out, filter_size, args, stride=1, padding=None):
        from invertible_layers import ActNorm
        super(Conv2dActNorm, self).__init__()
        padding = (filter_size - 1) // 2 or padding
        self.conv = nn.Conv2d(channels_in, channels_out, filter_size, padding=padding, bias=False)
        self.actnorm = ActNorm(channels_out, args=args)

    def forward(self, x):
        x = self.conv(x)
        x = self.actnorm.forward_(x, -1)[0]
        return x

'''
Linear layer zero initialization
'''
class LinearZeroInit(nn.Linear):
    def reset_parameters(self):
        self.weight.data.fill_(0.)
        self.bias.data.fill_(0.)


# '''
# Shallow NN used for skip connection. Labelled `f` in the original repo.
# '''
# # def NN(in_channels, hidden_channels=512, channels_out=None):
# def NN(in_channels, args, hidden_channels=128, channels_out=None, filter_size=3):
# # def NN(in_channels, hidden_channels=256, channels_out=None):
#     channels_out = channels_out or in_channels
#     padding = (filter_size - 1) // 2 #or padding
#     return nn.Sequential(
#         Conv2dActNorm(in_channels, hidden_channels, filter_size, stride=1, padding=padding, args=args),
#         nn.ReLU(inplace=True),
#         Conv2dActNorm(hidden_channels, hidden_channels, 1, stride=1, padding=0, args=args),
#         nn.ReLU(inplace=True),
#         Conv2dZeroInit(hidden_channels, channels_out, filter_size, stride=1, padding=padding))
#         # Conv2dActNorm(hidden_channels, channels_out, 3, stride=1, padding=1))



class NN(nn.Module):
    def __init__(self, in_channels, args, hidden_channels=128, channels_out=None, filter_size=3):
        super(NN, self).__init__()

        filter_size = 3 #7
        padding = (filter_size - 1) // 2 or padding

        self.conv1 = Conv2dActNorm(in_channels, hidden_channels, filter_size, stride=1, padding=padding, args=args)
        self.conv2 = Conv2dActNorm(hidden_channels, hidden_channels, 1, stride=1, padding=0, args=args)
        self.conv3 = Conv2dZeroInit(hidden_channels, channels_out, filter_size, stride=1, padding=padding)


        # self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=4, padding=padding)
        # self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, filter_size, stride=1, padding=padding)
        # self.conv3 = nn.ConvTranspose2d(hidden_channels, channels_out, filter_size, stride=4, padding=padding, output_padding=3)


    def forward(self, x):

        # print ('input', x.shape)

        x = self.conv1(x)
        # print ('conv1', x.shape)
        x = F.relu(x)
        x = self.conv2(x)
        # print ('conv2', x.shape)

        x = F.relu(x)
        x = self.conv3(x)
        # print ('conv3', x.shape)
        # fsad

        return x








class NN_FC(nn.Module):
    def __init__(self, in_shape):
        super(NN_FC, self).__init__()

        self.in_shape=in_shape

        # self.net1 = nn.Linear(in_shape[0]*in_shape[1]*in_shape[2] // 2, 10)
        # self.net2 = nn.Linear(10, in_shape[0]*in_shape[1]*in_shape[2])

        intermediate = 3

        self.net1 = nn.Linear(in_shape[2], intermediate)
        self.net2 = nn.Linear(intermediate, in_shape[0]*in_shape[1]*in_shape[2] // 2)

        # filter_size = 3 #7
        # padding = (filter_size - 1) // 2 or padding

        # self.conv1 = Conv2dActNorm(in_channels, hidden_channels, filter_size, stride=1, padding=padding, args=args)
        # self.conv2 = Conv2dActNorm(hidden_channels, hidden_channels, 1, stride=1, padding=0, args=args)
        # self.conv3 = Conv2dZeroInit(hidden_channels, channels_out, filter_size, stride=1, padding=padding)


        # self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=4, padding=padding)
        # self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, filter_size, stride=1, padding=padding)
        # self.conv3 = nn.ConvTranspose2d(hidden_channels, channels_out, filter_size, stride=4, padding=padding, output_padding=3)


    def forward(self, x):

        # print ('input', x.shape)

        B = x.shape[0]
        # print (x.shape)

        x = x.view(B,-1)

        # print (x.shape)

        x = self.net1(x)
        # print ('conv1', x.shape)
        x = F.relu(x)
        x = self.net2(x)
        # print ('conv2', x.shape)

        # x = F.relu(x)
        # x = self.conv3(x)
        # # print ('conv3', x.shape)
        # fsad
        x = x.view(B, self.in_shape[0] // 2, self.in_shape[1], self.in_shape[2] )

        return x




