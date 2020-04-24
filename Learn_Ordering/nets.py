


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# class Res(nn.Module):
#     def __init__(self, in_channels, hidden_channels=128): #, channels_out=None): #, filter_size=3):
#         super(Res, self).__init__()

#         filter_size = 5 #7
#         padding = (filter_size - 1) // 2 or padding
#         # self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=2, padding=padding)

#         self.pad = nn.ReflectionPad2d(2)

#         self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=1, padding=0)

#         self.conv2 = nn.Conv2d(hidden_channels, in_channels, filter_size, stride=1, padding=0)

#         # self.conv3 = nn.ConvTranspose2d(hidden_channels, in_channels, filter_size, stride=2, padding=padding, output_padding=1)
#         # self.conv3 = nn.ConvTranspose2d(hidden_channels, in_channels, filter_size, stride=1, padding=padding, output_padding=0)

#     def forward(self, x):

#         x_init = x
#         # print (x.shape)
#         x = self.pad(x)
#         x = self.conv1(x)
#         # print (x.shape)
#         x = F.leaky_relu(x)
#         x = self.pad(x)
#         x = self.conv2(x)
#         # print (x.shape)

#         # x = F.leaky_relu(x)
#         # x = self.conv3(x)

#         # print (x_init[0][0])
#         # print (x[0][0])     
#         # print (x_init.shape)
#         # print (x.shape)

#         x = x_init + x


#         #TODO confirm init didnt change 
#         # fadsf

#         return x


# class NN(nn.Module):
#     def __init__(self, in_channels, hidden_channels=128, channels_out=None): #, filter_size=3):
#         super(NN, self).__init__()

#         filter_size = 5 #7
#         padding = (filter_size - 1) // 2 #or padding

#         # self.conv1 = Conv2dActNorm(in_channels, hidden_channels, filter_size, stride=1, padding=padding, args=args)
#         # self.conv2 = Conv2dActNorm(hidden_channels, hidden_channels, 1, stride=1, padding=0, args=args)
#         # self.conv3 = Conv2dZeroInit(hidden_channels, channels_out, filter_size, stride=1, padding=padding)

#         # self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=4, padding=padding)
#         # self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, filter_size, stride=1, padding=padding)
#         # self.conv3 = nn.ConvTranspose2d(hidden_channels, channels_out, filter_size, stride=4, padding=padding, output_padding=3)

#         stride = 1

#         self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=stride, padding=padding)

#         self.resnet1 = Res(hidden_channels, hidden_channels) #, hidden_channels)
#         self.resnet2 = Res(hidden_channels, hidden_channels) #, hidden_channels)
#         self.resnet3 = Res(hidden_channels, hidden_channels) #, channels_out)

#         # self.deconv1 = nn.ConvTranspose2d(hidden_channels, channels_out, 1, 1, 0)
#         # self.deconv1 = nn.ConvTranspose2d(hidden_channels, channels_out, filter_size, stride=stride, padding=padding, output_padding=0)

#         self.pad = nn.ReflectionPad2d(3)
#         self.conv2 = nn.Conv2d(hidden_channels, channels_out, 7) #, stride=1) #, padding=3)


#         self.resnet_softmax = Res(hidden_channels, hidden_channels) #, channels_out)
#         # self.deconv_softmax = nn.ConvTranspose2d(hidden_channels, hidden_channels, filter_size, stride=stride, padding=padding, output_padding=0)
#         self.pad2 = nn.ReflectionPad2d(2)
#         self.conv_softmax = nn.Conv2d(hidden_channels, 1, filter_size, stride=1, padding=0)
#         # self.conv_softmax2 = nn.Conv2d(hidden_channels, 1, 1, stride=1, padding=0)

#     def forward(self, x):

#         B = x.shape[0]
#         HW = x.shape[2]

#         # print ('input', x.shape)

#         x = self.conv1(x)

#         x = self.resnet1(x)
#         # print ('conv1', x.shape)
#         x = F.leaky_relu(x)

#         x = self.resnet2(x)

#         # print ('conv2', x.shape)

#         x_fork = x

#         x = F.leaky_relu(x)
#         x = self.resnet3(x)
        
#         # print ('conv3', x.shape)

#         # x = F.leaky_relu(x)
#         # x = self.deconv1(x)
#         # print (x.shape)
#         # fasdf

#         # print ('conv3', x.shape)
#         x = self.pad(x)
#         x = self.conv2(x)
#         # print ('output', x.shape)



#         x_softmax = F.leaky_relu(x_fork)
#         x_softmax = self.resnet_softmax(x_softmax)
#         # x_softmax = F.leaky_relu(x_softmax)
#         # x_softmax = self.deconv_softmax(x_softmax)
#         x_softmax = self.pad2(x_softmax)
#         x_softmax = self.conv_softmax(x_softmax)
#         # x_softmax = self.conv_softmax2(x_softmax)

#         # x_softmax = x_softmax.view(B,-1)
#         # x_softmax = torch.softmax(x_softmax, dim=1)
#         # # x_softmax = x_softmax.view(B,1,HW,HW)
#         x_softmax = x_softmax.view(B,HW,HW)

#         return x, x_softmax























# class Res2(nn.Module):
#     def __init__(self, in_channels, hidden_channels=128): #, channels_out=None): #, filter_size=3):
#         super(Res2, self).__init__()

#         filter_size = 5 #7
#         padding = (filter_size - 1) // 2 or padding
#         # self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=2, padding=padding)

#         self.pad = nn.ReflectionPad2d(2)

#         self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=1, padding=0)

#         self.conv2 = nn.Conv2d(hidden_channels, in_channels, filter_size, stride=1, padding=0)

#         # self.conv3 = nn.ConvTranspose2d(hidden_channels, in_channels, filter_size, stride=2, padding=padding, output_padding=1)
#         # self.conv3 = nn.ConvTranspose2d(hidden_channels, in_channels, filter_size, stride=1, padding=padding, output_padding=0)

#     def forward(self, x):

#         x_init = x
#         # print (x.shape)
#         x = self.pad(x)
#         x = self.conv1(x)
#         # print (x.shape)
#         x = F.leaky_relu(x)
#         x = self.pad(x)
#         x = self.conv2(x)
#         # print (x.shape)

#         # x = F.leaky_relu(x)
#         # x = self.conv3(x)

#         # print (x_init[0][0])
#         # print (x[0][0])     
#         # print (x_init.shape)
#         # print (x.shape)

#         x = x_init + x


#         #TODO confirm init didnt change 
#         # fadsf

#         return x

# class NN2(nn.Module):
#     def __init__(self, in_channels, hidden_channels=128, channels_out=None): #, filter_size=3):
#         super(NN2, self).__init__()

#         filter_size = 5 #7
#         padding = (filter_size - 1) // 2 #or padding
#         stride = 2
        
#         self.pad1 = nn.ReflectionPad2d(1)
#         self.pad2 = nn.ReflectionPad2d(2)
#         self.pad3 = nn.ReflectionPad2d(3)

#         # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.upsample = nn.UpsamplingNearest2d(scale_factor=2)



#         self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=stride, padding=0)
#         self.resnet1 = Res2(hidden_channels, hidden_channels) #, hidden_channels)
#         self.resnet2 = Res2(hidden_channels, hidden_channels) #, hidden_channels)
#         self.resnet22 = Res2(hidden_channels, hidden_channels) #, hidden_channels)


#         # IMAGE PRED
#         self.resnet3 = Res2(hidden_channels, hidden_channels) #, channels_out)
#         # self.deconv1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, filter_size, stride=stride, padding=2, output_padding=1)
#         # self.deconv1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, filter_size, stride=stride, padding=0, output_padding=0)
#         self.conv_mean = nn.Conv2d(hidden_channels, channels_out//2, 7) #, stride=1) #, padding=3)
#         # self.conv_mean2 = nn.Conv2d(hidden_channels, channels_out//2, 7) #, stride=1) #, padding=3)
#         self.conv_logsd = nn.Conv2d(hidden_channels, channels_out//2, 7) #, stride=1) #, padding=3)


#         # ERROR PRED
#         self.resnet_softmax = Res2(hidden_channels, hidden_channels) #, channels_out)
#         # self.deconv_softmax = nn.ConvTranspose2d(hidden_channels, hidden_channels, filter_size, stride=stride, padding=0, output_padding=0)
#         self.conv_softmax = nn.Conv2d(hidden_channels, 1, 7) #, stride=1) #, padding=3)




#     def forward(self, x):

#         B = x.shape[0]
#         HW = x.shape[2]

#         # print ('input', x.shape)

#         # SHARED WEIGHTS
#         x = self.pad2(x)
#         x = self.conv1(x)        
#         x = F.leaky_relu(self.resnet1(x))
#         x = F.leaky_relu(self.resnet2(x))
#         x = self.resnet22(x)

#         x_fork = x




#         # IMAGE PRED
#         x = F.leaky_relu(x)
#         x = self.resnet3(x)
#         # print (x.shape)
        
        
#         # x = self.pad2(x)
#         # print (x.shape)
#         x = self.upsample(x)
#         # x = self.deconv1(x)
#         # print (x.shape)
#         # fddafs

#         x = self.pad3(x)
#         x_mean = self.conv_mean(x)
#         # x_mean = self.pad3(x_mean)
#         # x_mean = self.conv_mean2(x_mean)

#         x_logsd = self.conv_logsd(x)

#         # print (x_mean.shape)
#         # fassadf




#         # ERROR PRED
#         x_softmax = F.leaky_relu(x_fork)
#         x_softmax = self.resnet_softmax(x_softmax)
#         # x_softmax = self.pad1(x_softmax)
#         # x_softmax = self.deconv_softmax(x_softmax)

#         x_softmax = self.upsample(x_softmax)

#         x_softmax = self.pad3(x_softmax)
#         x_softmax = self.conv_softmax(x_softmax)
#         x_softmax = x_softmax.view(B,HW,HW)

#         return x_mean, x_logsd, x_softmax




































# # class Res3(nn.Module):
# #     def __init__(self, in_channels, hidden_channels=128): #, channels_out=None): #, filter_size=3):
# #         super(Res3, self).__init__()

# #         filter_size = 5 #7
# #         padding = (filter_size - 1) // 2 or padding

# #         self.pad = nn.ReflectionPad2d(2)
# #         self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=1, padding=0)
# #         self.conv2 = nn.Conv2d(hidden_channels, in_channels, filter_size, stride=1, padding=0)


# #     def forward(self, x):
# #         x_init = x
# #         x = self.pad(x)
# #         x = self.conv1(x)
# #         x = F.leaky_relu(x)
# #         x = self.pad(x)
# #         x = self.conv2(x)

# #         x = x_init + x
# #         return x




# class Res3(nn.Module):
#     def __init__(self, in_channels, hidden_channels=128): #, channels_out=None): #, filter_size=3):
#         super(Res3, self).__init__()

#         filter_size = 5 #7
#         padding = (filter_size - 1) // 2 or padding

#         stride = 4

#         self.pad = nn.ReflectionPad2d(2)
#         self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=stride, padding=0)

#         self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, filter_size, stride=1, padding=0)

#         self.upsample = nn.UpsamplingNearest2d(scale_factor=stride)
#         self.conv3 = nn.Conv2d(hidden_channels, in_channels, filter_size, stride=1, padding=0)


#         self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, 7, stride=1, padding=0)
#         self.conv5 = nn.ConvTranspose2d(hidden_channels, hidden_channels, 7, stride=1, padding=0)



#     def forward(self, x):
#         x_init = x
#         x = self.pad(x)
#         x = F.leaky_relu(self.conv1(x))
        
#         # x = self.pad(x)
#         # x = F.leaky_relu(self.conv2(x))

#         # print (x.shape)
#         x = F.leaky_relu(self.conv4(x))
#         # print (x.shape)
#         x = F.leaky_relu(self.conv5(x))
#         # print (x.shape)
#         # fad

#         x = self.upsample(x)

#         x = self.pad(x)
#         x = F.leaky_relu(self.conv3(x))

#         x = x_init + x
#         return x



# class NN3(nn.Module):
#     def __init__(self, in_channels, hidden_channels=128, channels_out=None): #, filter_size=3):
#         super(NN3, self).__init__()

#         filter_size = 5 #7
#         padding = (filter_size - 1) // 2 #or padding
#         stride = 4
        
#         self.pad1 = nn.ReflectionPad2d(1)
#         self.pad2 = nn.ReflectionPad2d(2)
#         self.pad3 = nn.ReflectionPad2d(3)

#         self.upsample = nn.UpsamplingNearest2d(scale_factor=stride)

#         self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=stride, padding=0)
#         self.resnet1 = Res3(hidden_channels, hidden_channels) #, hidden_channels)
#         self.resnet2 = Res3(hidden_channels, hidden_channels) #, hidden_channels)
#         self.resnet22 = Res3(hidden_channels, hidden_channels) #, hidden_channels)

#         # IMAGE PRED
#         self.resnet3 = Res3(hidden_channels, hidden_channels) 
#         self.resnet4 = Res3(hidden_channels, hidden_channels) 
#         self.conv_postup = nn.Conv2d(hidden_channels, hidden_channels*2, 5) 

#         # self.conv_mean1 = nn.Conv2d(hidden_channels*2, hidden_channels*2, 5) #, stride=1) #, padding=3)
#         # self.conv_logsd1 = nn.Conv2d(hidden_channels*2, hidden_channels*2, 5) #, stride=1) #, padding=3)     
#         self.conv_mean = nn.Conv2d(hidden_channels*2, channels_out//2, 5) #, stride=1) #, padding=3)
#         self.conv_logsd = nn.Conv2d(hidden_channels*2, channels_out//2, 5) #, stride=1) #, padding=3)

#         # ERROR PRED
#         self.resnet_softmax = Res3(hidden_channels, hidden_channels) 
#         self.resnet_softmax2 = Res3(hidden_channels, hidden_channels) 
#         self.conv_softmax = nn.Conv2d(hidden_channels, 1, 7) 




#     def forward(self, x):

#         B = x.shape[0]
#         HW = x.shape[2]

#         # SHARED WEIGHTS
#         x = self.pad2(x)
#         x = self.conv1(x)        
#         x = F.leaky_relu(self.resnet1(x))
#         x = F.leaky_relu(self.resnet2(x))
#         x = self.resnet22(x)

#         x_fork = x

#         # IMAGE PRED
#         x = self.resnet3(F.leaky_relu(x))
#         x = self.resnet4(F.leaky_relu(x))
#         x = self.upsample(x)
#         x = self.pad2(x)
#         x = F.leaky_relu(self.conv_postup(x))
#         x = self.pad2(x)


#         # x_mean = self.conv_mean1(x)
#         # x_mean = self.pad2(x_mean)
#         x_mean = self.conv_mean(x)
#         # x_mean = torch.sigmoid(x_mean)

#         # x_logsd = self.conv_logsd1(x)
#         # x_logsd = self.pad2(x_logsd)
#         x_logsd = self.conv_logsd(x)


#         # ERROR PRED
#         x_softmax = self.resnet_softmax(F.leaky_relu(x_fork))
#         x_softmax = self.resnet_softmax2(F.leaky_relu(x_softmax))
#         x_softmax = self.upsample(x_softmax)
#         x_softmax = self.conv_softmax(self.pad3(x_softmax)).view(B,HW,HW)

#         return x_mean, x_logsd, x_softmax




















# class Res3(nn.Module):
#     def __init__(self, in_channels, hidden_channels=128): #, channels_out=None): #, filter_size=3):
#         super(Res3, self).__init__()

#         filter_size = 5 #7
#         padding = (filter_size - 1) // 2 or padding

#         stride = 4

#         self.pad = nn.ReflectionPad2d(2)
#         self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=stride, padding=0)

#         self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, filter_size, stride=1, padding=0)

#         self.upsample = nn.UpsamplingNearest2d(scale_factor=stride)
#         self.conv3 = nn.Conv2d(hidden_channels, in_channels, filter_size, stride=1, padding=0)


#         self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, 7, stride=1, padding=0)
#         self.conv5 = nn.ConvTranspose2d(hidden_channels, hidden_channels, 7, stride=1, padding=0)



#     def forward(self, x):
#         x_init = x
#         x = self.pad(x)
#         x = F.leaky_relu(self.conv1(x))
        
#         # x = self.pad(x)
#         # x = F.leaky_relu(self.conv2(x))

#         # print (x.shape)
#         x = F.leaky_relu(self.conv4(x))
#         # print (x.shape)
#         x = F.leaky_relu(self.conv5(x))
#         # print (x.shape)
#         # fad

#         x = self.upsample(x)

#         x = self.pad(x)
#         x = F.leaky_relu(self.conv3(x))

#         x = x_init + x
#         return x



# class NN4(nn.Module):
#     def __init__(self, in_channels, hidden_channels=128, channels_out=None): #, filter_size=3):
#         super(NN4, self).__init__()

#         filter_size = 5 #7
#         padding = (filter_size - 1) // 2 #or padding
#         stride = 4
        
#         self.pad1 = nn.ReflectionPad2d(1)
#         self.pad2 = nn.ReflectionPad2d(2)
#         self.pad3 = nn.ReflectionPad2d(3)

#         self.upsample = nn.UpsamplingNearest2d(scale_factor=stride)

#         self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=stride, padding=0)
#         self.resnet1 = Res3(hidden_channels, hidden_channels) #, hidden_channels)
#         self.resnet2 = Res3(hidden_channels, hidden_channels) #, hidden_channels)
#         self.resnet22 = Res3(hidden_channels, hidden_channels) #, hidden_channels)

#         # IMAGE PRED
#         self.resnet3 = Res3(hidden_channels, hidden_channels) 
#         self.resnet4 = Res3(hidden_channels, hidden_channels) 
#         self.conv_postup = nn.Conv2d(hidden_channels, hidden_channels*2, 5) 

#         # self.conv_mean1 = nn.Conv2d(hidden_channels*2, hidden_channels*2, 5) #, stride=1) #, padding=3)
#         # self.conv_logsd1 = nn.Conv2d(hidden_channels*2, hidden_channels*2, 5) #, stride=1) #, padding=3)     
#         self.conv_mean = nn.Conv2d(hidden_channels*2, channels_out//2, 5) #, stride=1) #, padding=3)
#         self.conv_logsd = nn.Conv2d(hidden_channels*2, channels_out//2, 5) #, stride=1) #, padding=3)

#         # # ERROR PRED
#         # self.resnet_softmax = Res3(hidden_channels, hidden_channels) 
#         # self.resnet_softmax2 = Res3(hidden_channels, hidden_channels) 
#         # self.conv_softmax = nn.Conv2d(hidden_channels, 1, 7) 




#     def forward(self, x):

#         B = x.shape[0]
#         HW = x.shape[2]

#         # SHARED WEIGHTS
#         x = self.pad2(x)
#         x = self.conv1(x)        
#         x = F.leaky_relu(self.resnet1(x))
#         x = F.leaky_relu(self.resnet2(x))
#         x = self.resnet22(x)

#         # x_fork = x

#         # IMAGE PRED
#         x = self.resnet3(F.leaky_relu(x))
#         x = self.resnet4(F.leaky_relu(x))
#         x = self.upsample(x)
#         x = self.pad2(x)
#         x = F.leaky_relu(self.conv_postup(x))
#         x = self.pad2(x)


#         # x_mean = self.conv_mean1(x)
#         # x_mean = self.pad2(x_mean)
#         x_mean = self.conv_mean(x)
#         # x_mean = torch.sigmoid(x_mean)

#         # x_logsd = self.conv_logsd1(x)
#         # x_logsd = self.pad2(x_logsd)
#         x_logsd = self.conv_logsd(x)


#         # # ERROR PRED
#         # x_softmax = self.resnet_softmax(F.leaky_relu(x_fork))
#         # x_softmax = self.resnet_softmax2(F.leaky_relu(x_softmax))
#         # x_softmax = self.upsample(x_softmax)
#         # x_softmax = self.conv_softmax(self.pad3(x_softmax)).view(B,HW,HW)

#         return x_mean, x_logsd #, x_softmax


































class Res_Global(nn.Module):
    def __init__(self, in_channels, global_filter_size, hidden_channels=128): #, channels_out=None): #, filter_size=3):
        super(Res_Global, self).__init__()

        filter_size = 5 #7
        padding = (filter_size - 1) // 2 or padding
        stride = 2

        self.pad = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=stride, padding=0)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, global_filter_size, stride=1, padding=0)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=stride)
        self.conv3 = nn.ConvTranspose2d(hidden_channels, hidden_channels, global_filter_size, stride=1, padding=0)
        self.conv4 = nn.Conv2d(hidden_channels, in_channels, filter_size, stride=1, padding=0)

    def forward(self, x):
        x_init = x
        x = self.pad(x)
        x = F.leaky_relu(self.conv1(x))  #downsample with stride
        x = F.leaky_relu(self.conv2(x)) # downsample with glocal filter
        x = F.leaky_relu(self.conv3(x)) #upsample with global conv transpose
        x = self.upsample(x)
        x = self.pad(x)
        x = self.conv4(x)
        x = x_init + x
        return x




class Res_Local(nn.Module):
    def __init__(self, in_channels, hidden_channels=128): #, channels_out=None): #, filter_size=3):
        super(Res_Local, self).__init__()

        filter_size = 5 #7
        padding = (filter_size - 1) // 2 or padding

        stride = 2

        self.pad = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=stride, padding=0)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, filter_size, stride=1, padding=0)

        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, filter_size, stride=1, padding=0)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=stride)
        self.conv4 = nn.Conv2d(hidden_channels, in_channels, filter_size, stride=1, padding=0)

        # self.conv5 = nn.ConvTranspose2d(hidden_channels, hidden_channels, 28, stride=1, padding=0)



    def forward(self, x):

        # print (x.shape)
        x_init = x
        x = self.pad(x)
        x = F.leaky_relu(self.conv1(x))  #downsample with stride
        
        x = self.pad(x)
        x = F.leaky_relu(self.conv2(x))

        x = self.pad(x)
        x = F.leaky_relu(self.conv3(x))

        x = self.upsample(x)

        x = self.pad(x)
        x = self.conv4(x)

        x = x_init + x
        return x



class NN4(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, channels_out=None, k=4, input_size=32): 
        super(NN4, self).__init__()

        filter_size = 5 #7
        padding = (filter_size - 1) // 2 #or padding
        stride = 2

        k = k

        if input_size == 32:
        	global_filter_size = 8
        if input_size == 112:
        	global_filter_size = 28

        
        self.pad1 = nn.ReflectionPad2d(1)
        self.pad2 = nn.ReflectionPad2d(2)
        self.pad3 = nn.ReflectionPad2d(3)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=stride)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, filter_size, stride=stride, padding=0)
        self.resnet1 = Res_Global(hidden_channels, global_filter_size, hidden_channels) #, hidden_channels)
        self.resnet2 = Res_Local(hidden_channels, hidden_channels) #, hidden_channels)
        self.resnet22 = Res_Global(hidden_channels, global_filter_size, hidden_channels) #, hidden_channels)
        self.resnet3 = Res_Local(hidden_channels, hidden_channels) 
        # self.resnet4 = Res_Global(hidden_channels, hidden_channels) 
        self.conv_postup = nn.Conv2d(hidden_channels, hidden_channels*2, 5) 

        # self.conv_mean1 = nn.Conv2d(hidden_channels*2, hidden_channels*2, 5) #, stride=1) #, padding=3)
        # self.conv_logsd1 = nn.Conv2d(hidden_channels*2, hidden_channels*2, 5) #, stride=1) #, padding=3)     
        self.conv_mean = nn.Conv2d(hidden_channels*2, channels_out * k, 5) #, stride=1) #, padding=3)
        self.conv_logsd = nn.Conv2d(hidden_channels*2, channels_out * k, 5) #, stride=1) #, padding=3)

        self.conv_mixture = nn.Conv2d(hidden_channels*2, channels_out*k, 5) 





    def forward(self, x):

        B = x.shape[0]
        HW = x.shape[2]

        # SHARED WEIGHTS
        x = self.pad2(x)
        x = self.conv1(x)   # downsample with strides     
        x = F.leaky_relu(self.resnet1(x))
        x = F.leaky_relu(self.resnet2(x))
        x = self.resnet22(x)
        x = self.resnet3(F.leaky_relu(x))
        # x = self.resnet4(F.leaky_relu(x))
        x = self.upsample(x)
        x = self.pad2(x)
        x = F.leaky_relu(self.conv_postup(x))
        x = self.pad2(x)

        # x_mean = self.conv_mean1(x)
        # x_mean = self.pad2(x_mean)
        x_mean = self.conv_mean(x)
        # x_mean = torch.sigmoid(x_mean)

        # x_logsd = self.conv_logsd1(x)
        # x_logsd = self.pad2(x_logsd)
        x_logsd = self.conv_logsd(x)

        x_mix = self.conv_mixture(x)


        # # ERROR PRED
        # x_softmax = self.resnet_softmax(F.leaky_relu(x_fork))
        # x_softmax = self.resnet_softmax2(F.leaky_relu(x_softmax))
        # x_softmax = self.upsample(x_softmax)
        # x_softmax = self.conv_softmax(self.pad3(x_softmax)).view(B,HW,HW)

        return x_mean, x_logsd, x_mix #, x_softmax


























