

import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
Module = nn.Module
import collections
from itertools import repeat






from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import imageio



# In[2]:


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


# In[3]:


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


# In[4]:


class Conv2dLocal(Module):
 
    def __init__(self, in_height, in_width, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2dLocal, self).__init__()
 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
 
        self.in_height = in_height
        self.in_width = in_width
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        self.weight = Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_channels, self.out_height, self.out_width))
        else:
            self.register_parameter('bias', None)
 
        self.reset_parameters()
 
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
 
    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
 
    def forward(self, input):
        return conv2d_local(
            input, self.weight, self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation)


# In[5]:


unfold = F.unfold


# In[6]:


def conv2d_local(input, weight, bias=None, padding=0, stride=1, dilation=1):
    if input.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
    if weight.dim() != 6:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))
 
    outH, outW, outC, inC, kH, kW = weight.size()
    kernel_size = (kH, kW)
 
    # N x [inC * kH * kW] x [outH * outW]
    cols = F.unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
    cols = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)
 
    out = torch.matmul(cols, weight.view(outH * outW, outC, inC * kH * kW).permute(0, 2, 1))
    out = out.view(cols.size(0), outH, outW, outC).permute(0, 3, 1, 2)
 
    if bias is not None:
        out = out + bias.expand_as(out)
    return out










# lc = Conv2dLocal(in_height=3, in_width=3, in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=0, bias=True, dilation=1)
# print (lc(torch.autograd.Variable(torch.randn((1,64,3,3)))).size())



print ('testing Conv2dLocal')
lc = Conv2dLocal(in_height=64, in_width=64, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, bias=True, dilation=1)

print (lc(torch.autograd.Variable(torch.randn((1,3,64,64)))).size())



# outH = int(math.floor(in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
# outH =  ((H + (2*P) - (D*(K-1)-1) /S) +1
# outH =  ((H + (2*0) - (1*(K-1)-1) /1) +1
# outH =  H - K+1 -1 +1
# outH =  H - K +1

# weight: outH x outW x outC x inC x kH x kW
iH = 64
iW = 64
inC = 3
kH = 3
kW = 3
outH = iH - kH +1
outW = iW - kW +1
outC = 4
B = 1 


aa = conv2d_local(input=torch.randn((B,inC,iH,iW)), weight=torch.randn((outH, outW, outC, inC, kH, kW)))
#output: [B,outC,outH,outW]  so same as a regular conv
# why does the weight need outH,outW dimensions?
    # in regular conv, weight is [inC,k,k,outC]
    #ok so now since weights arent shared, I need make filters, which if stride=1, then its iH-k+1
# I dont get exactly what the code does, but I get the higher level idea of what its doing. 

#ok given [outH, outW] precisions, can I generate the (outH, outW, outC, inC, kH, kW) filters? 
    # for inC , ill just copy the weights for each channel
    # for kH, kW, good
    # for outH, outW, ??
    # for outC, ???


print (aa.size())



print ()
print ()
print ('testing Conv2dLocal matmul dimensions')
input1 = torch.randn(5,2,1,3) 
weight = torch.randn(2,3,4) 

print (input1.size())
print (weight.size())
print (torch.matmul(input1, weight).size())











# print ()
# import torch.legacy.nn as legacy
# print (legacy.SpatialConvolutionLocal)
# print ()




# print ()
# x = torch.arange(1, 9)
# print (x.size())
# x = x.unfold(0, 2, 1)
# print (x.size())
# print ()

# fsadf

# x = torch.Tensor(1, 1, 10, 10)
# # print (x)
# print (x.size())
# x = x.unfold(dimension=2, size=2, step=2)
# print (x.size())
# x = x.unfold(dimension=3, size=2, step=2)
# print (x.size())
# x = x.contiguous().view(1,1,5,5,4)
# print (x.size())
# print ()


import numpy as np

print ()
print ()
print ()
print ()


#TO GENERWTE GAUSSIAN KERNELS


# First a 1-D  Gaussian
print ('1D gaussian kernel')
var = .5
t = np.linspace(-1, 1, 5)
print(t)
print(t.shape)
bump = np.exp(-var*(t**2))
# bump /= np.trapz(bump) # normalize the integral to 1

# make a 2-D kernel out of it
# kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
kernel = np.reshape(bump, [-1,1]) * np.reshape(bump, [1,-1])

print (kernel.shape)
print (kernel)
print ()
print ()
print ()
print ()


#now try it with a matrix of variances
print ('2D gaussian kernel')
print ()
C=1
k=5
M=2
N=2
print ('C=',C)
print ('k=',k)
print ('M=',M)
print ('N=',N)
print ()


prec = np.array([[.1,.2],[.3,.4]])  #[m,n]
k= 5
t = np.linspace(-1, 1, k) #[k]
print ('precision matrix')
prec = np.reshape(prec, [2,2,1])
print (prec)
print (prec.shape)
print ('[M,N,1]')
print ()


t = np.reshape(t, [1,1,k])

filters = -prec*(t**2)
# print (filters)
# print (filters.shape)  #[m,n,k]
# print ()

filters = np.exp(filters)
# print (filters)
# print (filters.shape)  #[m,n,k]
# print ()



print ('gaussian filters')
filters = np.reshape(filters, [M,N,k,1]) * np.reshape(filters, [M,N,1,k])
print (filters)
print (filters.shape)  #[m,n,k,k]
print ('[M,N,K,K]')  #[m,n,k,k]
print ()


print ('ok do above but with outC and inC dimension, ')
print ('which are both 1 since Ill do each dim individually but with the same weights')
filters = np.reshape(filters, [M,N,1,1,k,k])
print (filters.shape)  #[m,n,k,k]
print ('[M,N,outC,inC,K,K]')  #[m,n,k,k]
print ('[M,N,1,1,K,K]')  #[m,n,k,k]

print ()
print ('Done. Ready to do it on frames')





print ()
print ()
print ()
print ()

print ('Load image')
frame = imageio.imread(home+'/Documents/tmp/Doom/frmame.png')
frame = frame[:,:,:3]
print (frame.shape) #[H,W,3]


print (np.max(frame))
print (np.min(frame))
frame = frame / 255.
# fas



# #TO SEE REAL FRAME
# plt.imshow(frame)
# # save_dir = home+'/Documents/tmp/Doom/'
# plt_path = home+'/Documents/tmp/Doom/temp_frmae.png' 
# plt.tight_layout()
# plt.savefig(plt_path)
# print ('saved viz',plt_path)
# plt.close()
# dsfa




frame = np.rollaxis(frame, 2, 1)
frame = np.rollaxis(frame, 1, 0)

print (frame.shape) #[3,H,W]
print ()


print ('Make 2D Gaussian Filters')
H = frame.shape[1]
W = frame.shape[2]
stride = 1
dilation = 1
# padding = 2
# K = 5
# padding = 3
# K = 7
# padding = 4
# K = 9
# padding = 5
# K = 11
# padding = 6
# K = 13
# padding = 7
# K = 15

# padding = 10
# K = 21

padding = 15
K = 31

outH = int(math.floor((H + 2 * padding - dilation * (K - 1) - 1) / stride + 1))
outW = int(math.floor((W + 2 * padding - dilation * (K - 1) - 1) / stride + 1))
# outH = int(np.floor((H + (2*padding) - (dilation*(K-1)-1) /stride) +1))
print ('stride',stride)
print ('padding',padding)
print ('dilation',dilation)
print ('K',K)
# print ()
print ('outH',outH)
print ('outW',outW)

#ok problem, I need the output to have same dimension as input, but I dont want H*W filters. 
# stride reudces output dim
# dilation increaes receptive field
# padding only applied to output dims I think
# I guess I want some parameter sharing.. its hard to have something between none and all parameter sharing
#  no sharing is fine , but it requires many params and might be slow.
# ya its over 7M params






prec = np.ones([outH,outW,1]) * .000001
x = np.linspace(-1, 1, K) #[k]
x = np.reshape(x, [1,1,K])
filters = -prec*(x**2)
filters = np.exp(filters)
filters = np.reshape(filters, [outH,outW,K,1]) * np.reshape(filters, [outH,outW,1,K])
filters = np.reshape(filters, [outH,outW,1,1,K,K])  #

print (filters.shape)
# print (filters[0][0][0][0])
# fsda
print ('[M,N,1,1,K,K]')  
print ('[M,N,outC,inC,K,K]') 
print ()
sum_ = np.sum(filters, axis=5, keepdims=1)
sum_ = np.sum(sum_, axis=4, keepdims=1)
filters = filters / sum_
# filters = filters / np.mean(filters, axis=4, keepdims=1)
# print (filters.shape)
# fas
filters = torch.from_numpy(filters).float()



print ('First channel')
frame_c0 = frame[0]
frame_c0 = np.expand_dims(frame_c0, axis=0)
frame_c0 = np.expand_dims(frame_c0, axis=0)
print (frame_c0.shape) #[1,1,H,W] = [B,C,H,W]
print ('Input: [B,C,H,W]')

print ('Local Conv')
frame_c0 = torch.from_numpy(frame_c0).float()

frame_c0 = conv2d_local(input=frame_c0, weight=filters, bias=None, padding=padding, stride=1, dilation=1)

print (frame_c0.size())
print ('Output: [B,outC,outH,outW]')
print ()




print ('Second channel')
frame_c1 = frame[1]
frame_c1 = np.expand_dims(frame_c1, axis=0)
frame_c1 = np.expand_dims(frame_c1, axis=0)
print (frame_c1.shape) #[1,1,H,W] = [B,C,H,W]
print ('Input: [B,C,H,W]')

print ('Local Conv')
frame_c1 = torch.from_numpy(frame_c1).float()
frame_c1 = conv2d_local(input=frame_c1, weight=filters, bias=None, padding=padding, stride=1, dilation=1)

print (frame_c1.size())
print ('Output: [B,outC,outH,outW]')
print ()




print ('Third channel')
frame_c2 = frame[2]
frame_c2 = np.expand_dims(frame_c2, axis=0)
frame_c2 = np.expand_dims(frame_c2, axis=0)
print (frame_c2.shape) #[1,1,H,W] = [B,C,H,W]
print ('Input: [B,C,H,W]')

print ('Local Conv')
frame_c2 = torch.from_numpy(frame_c2).float()

# print (torch.max(frame_c2)[0])
frame_c2 = conv2d_local(input=frame_c2, weight=filters, bias=None, padding=padding, stride=1, dilation=1)

# print (torch.max(frame_c2)[0])
# print (torch.min(frame_c2)[0])
# fdsa


print (frame_c2.size())
print ('Output: [B,outC,outH,outW]')
print ()
print ()
print ()

blurred_image = [frame_c0, frame_c1, frame_c2]
blurred_image = torch.stack(blurred_image)
print (blurred_image.shape)
# fdsa
blurred_image = blurred_image.view(3,600,800)
print (blurred_image.size())

frame = blurred_image.numpy()

frame = np.rollaxis(frame, 1, 0)
frame = np.rollaxis(frame, 2, 1)

# print (frame.shape)
# fsd

plt.imshow(frame)
# save_dir = home+'/Documents/tmp/Doom/'
plt_path = home+'/Documents/tmp/Doom/temp_frmae_blurred_k'+str(K)+'.png' 
plt.tight_layout()
plt.savefig(plt_path)
print ('saved viz',plt_path)
plt.close()




fdsaf




print ()




















# # weights dont have a batch dimension, nvm they do. since each datpoint has its own weigths
# print ('ok do above but with batch dimension')

# B=3
# k=5
# M=2
# N=2
# print ('B=',B)
# print ('k=',k)
# print ('M=',M)
# print ('N=',N)
# print ()

# print ('precision matrix')
# prec = np.array([[.1,.2],[.3,.4]])  #[m,n]
# prec = np.reshape(prec, [1,M,N,1])
# prec = np.repeat(prec, B, axis=0)
# # print (prec)
# print (prec.shape)
# print ('[B,M,N,1]')
# print ()

# t = np.linspace(-1, 1, k) #[k]
# t = np.reshape(t, [1,1,1,k])
# filters = -prec*(t**2)
# filters = np.exp(filters)
# print ('gaussian filter 1D')
# print (filters.shape) 
# print ('[B,M,N,K]')
# print ()

# print ('gaussian filters 2D')
# filters = np.reshape(filters, [B,M,N,k,1]) * np.reshape(filters, [B,M,N,1,k])
# # print (filters)
# print (filters.shape)  #[m,n,k,k]
# print ('[B,M,N,K,K]')  #[m,n,k,k]
# print ()
# print ()








# print ('ok do above but with outC and inC dimension, ')
# print ('which are both 1 since Ill do each dim individually but with the same weights')

# C=1
# k=5
# M=2
# N=2
# print ('C=',C)
# print ('k=',k)
# print ('M=',M)
# print ('N=',N)
# print ()

# print ('precision matrix')
# prec = np.array([[.1,.2],[.3,.4]])  #[m,n]
# prec = np.reshape(prec, [1,M,N,1])
# prec = np.repeat(prec, B, axis=0)
# # print (prec)
# print (prec.shape)
# print ('[B,M,N,1]')
# print ()

# t = np.linspace(-1, 1, k) #[k]
# t = np.reshape(t, [1,1,1,k])
# filters = -prec*(t**2)
# filters = np.exp(filters)
# print ('gaussian filter 1D')
# print (filters.shape) 
# print ('[B,M,N,K]')
# print ()

# print ('gaussian filters 2D')
# filters = np.reshape(filters, [B,M,N,k,1]) * np.reshape(filters, [B,M,N,1,k])
# # print (filters)
# print (filters.shape)  #[m,n,k,k]
# print ('[B,M,N,K,K]')  #[m,n,k,k]
# print ()
# print ()

fsadfa










#then repeat for 3 channels [m,n,3,k,k]

#image: [B,3,h,w] -> [m,n,B,3,S*S]   using unfold with size and step S, m=h/S n=w/S

#for each m
#   for each n
#       fold = conv2d(fold_mn,kernel_mn)

# Con: folds arent independent, it breaks things into blocks, but id rather it be smooth 










#ok now how to use in local connected layer
# I could just unfold the image into areas of the same size as the filters

# do regular conv on the folds of the image usign dif gaussian kernels 




# print (filters)
# print (filters.shape)  #[m,n,k,k]

# # print (list(np.arange(-1,1,.1)))
# # fsad


# x = torch.arange(-.6, .61, .3)
# print(x)
# x = torch.ger(x, x)
# print (x.size())
# print (x)
# x= -(x**2)
# print (x)

# # x = torch.exp(x)
# print (x.size())
# print(x)
# print ()


# #ok once I have -x**2 filter matrix, then divide by variacne and exp it all -> [m,n,k,k]
# #problem is that cross product comes after exp, but I need matrix of -x**2 first ..
# #could maybe log divide exp.. 

# m=3
# n=3
# k=5
# filters = x.view(1,1,k,k)
# filters = filters.repeat(m,n,1,1)
# print (filters.size()) #[m,n,k,k]

# var = torch.exp(torch.Tensor(m,n,1,1).uniform_(-1, 1)) #[m,n,1,1]
# print (var.size())

# filters = torch.log(filters)
# filters = filters / var
# filters = torch.exp(filters)
# print (filters.size())
# print (var[0][0])
# print (filters[0,0])
# print ()
# print ()







# fsdf



# x = torch.arange(-.6, .61, .3)
# x = torch.ger(x, x)
# print (x)


# fsda







# print (np.mgrid[-1:1.1:.5,-1:1.1:.5])
# # print (np.meshgrid(-1:1:.5,-1:1:.5))



# fasd
# # ind = -np.floor(N/2) : np.floor(N/2);
# # [X Y] = np.meshgrid(ind, ind);

# # # %// Create Gaussian Mask
# # h = exp(-(X.^2 + Y.^2) / (2*sigma*sigma));

# # fdsfa






# fads
# print (np.mgrid[0:5,0:5].shape)

# fdsa

# xy = np.mgrid[-5:5.1:0.5, -5:5.1:0.5].reshape(2, -1).T

# print (xy.shape)

# fads


# nx, ny = (3, 2)
# x = np.linspace(0, 1, nx)
# y = np.linspace(0, 1, ny)
# xv, yv = np.meshgrid(x, y)

# import numpy as np


# X,Y = np.mgrid[-5:5.1:0.5, -5:5.1:0.5]
# X, Y = np.mgrid[-5:5:21j, -5:5:21j]
# xy = np.vstack((X.flatten(), Y.flatten())).T






