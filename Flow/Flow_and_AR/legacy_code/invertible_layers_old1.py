import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch.distributions.bernoulli import Bernoulli

import numpy as np
import pdb

from layers import * 
from utils import * 


from os.path import expanduser
home = expanduser("~")



# import sys, os
# sys.path.insert(0, os.path.abspath(home+'/nsf-master/'))
# sys.path.insert(0, os.path.abspath(home+'/nsf-master/nde/'))
# sys.path.insert(0, os.path.abspath(home+'/nsf-master/nde/transforms/'))
# sys.path.insert(0, os.path.abspath(home+'/nsf-master/nde/transforms/splines/'))
# import rational_quadratic


# sys.path.insert(0, './spline_code/')
# sys.path.insert(0, './spline_code/nde/')
# sys.path.insert(0, './spline_code/nde/transforms/')
# sys.path.insert(0, './spline_code/nde/transforms/splines/')
# from nde import *
# # import utils2 #as utils
# # from conv import OneByOneConvolution

# from quadratic import *
# from linear import *

# import utils2

# print (utils)
# sfda

# from nsf-master import utils

# fsad

import sys, os
sys.path.insert(0, os.path.abspath('./PixelCNN/'))

from utils import * 
from model import * 
from PIL import Image


# ------------------------------------------------------------------------------
# Abstract Classes to define common interface for invertible functions
# ------------------------------------------------------------------------------

# Abstract Class for bijective functions
class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    def forward_(self, x, objective):
        raise NotImplementedError

    def reverse_(self, y, objective):
        raise NotImplementedError

# Wrapper for stacking multiple layers 
class LayerList(Layer):
    def __init__(self, list_of_layers=None):
        super(LayerList, self).__init__()
        self.layers = nn.ModuleList(list_of_layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward_(self, x, objective):
        for layer in self.layers:

            x, objective = layer.forward_(x, objective)


            # aa = x.clone()

            # xx, objective = layer.forward_(x, objective)

            # bb = x.clone()
            # print ()
            # print (str(layer)[:6])
            # print(torch.mean((aa - bb)**2))

            # if 'Add' in str(layer)[:6]:

            #     fadads

            # x = xx

            # print ()


            # if ((x!=x).any() or torch.max(x) > 99999 or torch.min(x) < -99999 
            #         or (objective!=objective).any() or torch.max(objective) > 999999 or torch.min(objective) < -999999  ):

            if ((x!=x).any() or (objective!=objective).any()  ):

                print (str(layer)[:6])

                # h = layer.conv_zero(x_pre)
                # mean, logs = h[:, 0::2], h[:, 1::2]

                # print (torch.min(x_pre), torch.max(x_pre))
                print ('x', torch.min(x), torch.max(x))
                print ('obj', torch.min(objective), torch.max(objective))
                # print ((x_pre!=x_pre).any(), (mean!=mean).any(), (logs!=logs).any())
                # print ( layer.conv_zero.logs)
                # fadfas
                print ('bad stuff in forward')
                fasdf





        return x, objective






    # def forward_2(self, x, objective):
    #     for layer in self.layers:

    #         # x, objective = layer.forward_(x, objective)


    #         aa = x.clone()


    #         if 'Add' in str(layer)[:6]:

    #             xx, objective = layer.forward_2(x, objective)
    #             # bb = x.clone()

    #             # print ()
    #             print (str(layer)[:6], torch.mean((aa - x)**2))
    #             print ('xx vs x', str(layer)[:6], torch.mean((xx - x)**2))
    #             # print()
    #             # fadads



    #         else:

    #             xx, objective = layer.forward_(x, objective)


    #         x = xx


    #     return x, objective





    def reverse_(self, x, objective, args=None):
        count=0
        for layer in reversed(self.layers): 

            # if count ==13:
            #     x_pre = x.clone()

            # if 'Split' in str(layer):
            #     x, objective = layer.reverse_(x, objective, use_stored_sample=True)


            # else:
            

            x, objective = layer.reverse_(x, objective, args=args)


            # print (count, layer)


            # if count ==13:
            #     if (x==x_pre).all():
            #         print ('yes')
            #     else:
            #         print ('no')
            #     fsadf

            # if count == 50:
            #     print (layer.conv_zero.logs)

            #     fasfa

            if (x!=x).any() or torch.max(x) > 999999:
                print (count, layer)

                # h = layer.conv_zero(x_pre)
                # mean, logs = h[:, 0::2], h[:, 1::2]

                print (torch.min(x_pre), torch.max(x_pre))
                print (torch.min(x), torch.max(x))
                # print ((x_pre!=x_pre).any(), (mean!=mean).any(), (logs!=logs).any())
                # print ( layer.conv_zero.logs)
                # fadfas
                print ('bad stuff in reverse')
                fasdf

            x_pre = x.clone()    
            count+=1

        # fsadfa

        return x, objective






    # def forward_andgetlayers_(self, x, objective):
    #     layers = []
    #     layers.append(x)
    #     names = []
    #     names.append('x')
    #     for layer in self.layers:
    #         # print (layer)
    #         # print (str(layer)[:10])
    #         names.append(str(layer)[:6])
    #         # fas

    #         # aa = x.clone()

    #         # xx, objective = layer.forward_(x, objective)


    #         if 'Split' in str(layer):
    #             xx, objective = layer.reverse_(x, objective, use_stored_sample=True)

    #         else:
    #             xx, objective = layer.reverse_(x, objective)


    #         # bb = x.clone()
    #         # print ()
    #         # print (str(layer)[:6])
    #         # print(torch.mean((aa - bb)**2))


    #         x = xx


    #         layers.append(x.clone())

    #     # print(torch.mean((layers[3] - layers[33])**2))
    #     # fasd


    #     return layers, names



    # def reverse_andgetlayers_(self, x, objective):

        
        

    #     layers = []
    #     layers.append(x)
    #     for layer in reversed(self.layers): 

    #         # aa = x.clone()

    #         xx, objective = layer.reverse_(x, objective)


    #         # bb = x.clone()
    #         # print(torch.mean((aa - bb)**2))
            

    #         x = xx

    #         layers.append(x.clone())


    #     # fdsaa


    #     return layers










# ------------------------------------------------------------------------------
# Permutation Layers 
# ------------------------------------------------------------------------------



# Shuffling on the channel axis
class Shuffle(Layer):
    def __init__(self, num_channels):
        super(Shuffle, self).__init__()
        indices = np.arange(num_channels)
        np.random.shuffle(indices)
        rev_indices = np.zeros_like(indices)
        for i in range(num_channels): 
            rev_indices[indices[i]] = i

        indices = torch.from_numpy(indices).long()
        rev_indices = torch.from_numpy(rev_indices).long()
        self.register_buffer('indices', indices)
        self.register_buffer('rev_indices', rev_indices)
        # self.indices, self.rev_indices = indices.cuda(), rev_indices.cuda()

    def forward_(self, x, objective):
        return x[:, self.indices], objective

    def reverse_(self, x, objective, args=None):
        return x[:, self.rev_indices], objective
        





# Reversing on the channel axis
class Reverse(Shuffle):
    def __init__(self, num_channels):
        super(Reverse, self).__init__(num_channels)
        indices = np.copy(np.arange(num_channels)[::-1])
        indices = torch.from_numpy(indices).long()
        self.indices.copy_(indices)
        self.rev_indices.copy_(indices)

# Invertible 1x1 convolution
class Invertible1x1Conv(Layer, nn.Conv2d):
    def __init__(self, num_channels):
        self.num_channels = num_channels
        nn.Conv2d.__init__(self, num_channels, num_channels, 1, bias=False)

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init = np.linalg.qr(np.random.randn(self.num_channels, self.num_channels))[0]
        w_init = torch.from_numpy(w_init.astype('float32'))
        w_init = w_init.unsqueeze(-1).unsqueeze(-1)
        self.weight.data.copy_(w_init)

    def forward_(self, x, objective):
        dlogdet = torch.det(self.weight.squeeze()).abs().log() * x.size(-2) * x.size(-1) 
        objective += dlogdet
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, \
            self.dilation, self.groups)
 
        return output, objective

    def reverse_(self, x, objective):
        dlogdet = torch.det(self.weight.squeeze()).abs().log() * x.size(-2) * x.size(-1) 
        objective -= dlogdet
        weight_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
        output = F.conv2d(x, weight_inv, self.bias, self.stride, self.padding, \
                    self.dilation, self.groups)
        return output, objective








# ------------------------------------------------------------------------------
# Layers involving squeeze operations defined in RealNVP / Glow. 
# ------------------------------------------------------------------------------

# Trades space for depth and vice versa
class Squeeze(Layer):
    def __init__(self, input_shape, factor=2):
        super(Squeeze, self).__init__()
        assert factor > 1 and isinstance(factor, int), 'no point of using this if factor <= 1'
        self.factor = factor
        self.input_shape = input_shape

    def squeeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert h % self.factor == 0 and w % self.factor == 0, pdb.set_trace()
        
        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c, h // self.factor, self.factor, w // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(bs, c * self.factor * self.factor, h // self.factor, w // self.factor)

        return x
 
    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c // self.factor ** 2, self.factor, self.factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(bs, c // self.factor ** 2, h * self.factor, w * self.factor)
        return x
    
    def forward_(self, x, objective):
        if len(x.size()) != 4: 
            raise NotImplementedError # Maybe ValueError would be more appropriate

        return self.squeeze_bchw(x), objective
        
    def reverse_(self, x, objective, args=None):
        if len(x.size()) != 4: 
            raise NotImplementedError

        return self.unsqueeze_bchw(x), objective






# class Squeeze_Reverse(Layer):
#     def __init__(self, input_shape, factor=2):
#         super(Squeeze, self).__init__()
#         assert factor > 1 and isinstance(factor, int), 'no point of using this if factor <= 1'
#         self.factor = factor
#         self.input_shape = input_shape

#     def squeeze_bchw(self, x):
#         bs, c, h, w = x.size()
#         assert h % self.factor == 0 and w % self.factor == 0, pdb.set_trace()
        
#         # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
#         x = x.view(bs, c, h // self.factor, self.factor, w // self.factor, self.factor)
#         x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
#         x = x.view(bs, c * self.factor * self.factor, h // self.factor, w // self.factor)

#         return x
 
#     def unsqueeze_bchw(self, x):
#         bs, c, h, w = x.size()
#         assert c >= 4 and c % 4 == 0

#         # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
#         x = x.view(bs, c // self.factor ** 2, self.factor, self.factor, h, w)
#         x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
#         x = x.view(bs, c // self.factor ** 2, h * self.factor, w * self.factor)
#         return x
    
#     def forward_(self, x, objective):
#         if len(x.size()) != 4: 
#             raise NotImplementedError # Maybe ValueError would be more appropriate

#         return self.squeeze_bchw(x), objective
        
#     def reverse_(self, x, objective, args=None):
#         if len(x.size()) != 4: 
#             raise NotImplementedError

#         return self.unsqueeze_bchw(x), objective








































# ------------------------------------------------------------------------------
# Layers involving prior
# ------------------------------------------------------------------------------



class AR_Prior(Layer):
    def __init__(self, input_shape, args):
        super(AR_Prior, self).__init__()
        self.input_shape = input_shape
        # print(args.learntop)
        # if args.learntop: 
        #     self.conv = Conv2dZeroInit(2 * input_shape[1], 2 * input_shape[1], 3, padding=(3 - 1) // 2)
        # else: 
        #     self.conv = None

        # self.conv = Conv2dZeroInit(2 * input_shape[1], 2 * input_shape[1], 3, padding=(3 - 1) // 2)

        self.model = PixelCNN(nr_resnet=args.AR_resnets, nr_filters=args.AR_channels, 
                    input_channels=input_shape[1], nr_logistic_mix=2)
        # self.model = PixelCNN(nr_resnet=3, nr_filters=128, 
        #             input_channels=input_shape[1], nr_logistic_mix=2)
        # self.model = PixelCNN(nr_resnet=2, nr_filters=32, 
        #             input_channels=input_shape[1], nr_logistic_mix=2)
        # model = model.cuda()

        self.loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)

        # logsd_limitvalue
        self.lm = 10.

    def gauss_log_prob(self, x, mean, logsd):
        Log2PI = float(np.log(2 * np.pi))
        # return  -0.5 * (Log2PI + 2. * logsd + ((x - mean) ** 2) / torch.exp(2. * logsd))

        aaa = Log2PI + 2. * logsd 
        var = torch.exp(2. * logsd)
        bbb = ((x - mean) ** 2) / var
        return  -0.5 * (aaa + bbb)
 
    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c // 2 ** 2, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(bs, c // 2 ** 2, h * 2, w * 2)
        return x



    def forward_(self, x, objective):
        B = x.shape[0]
        # mean_and_logsd = torch.cat([torch.zeros_like(x) for _ in range(2)], dim=1)
        # # if self.conv:

        # #this could just be some parameters we learn, 
        # # doesnt need to output of conv 
        # mean_and_logsd = self.conv(mean_and_logsd)

        # mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)

        # logsd = torch.clamp(logsd, min=-6., max=2.)

        # # print(self.conv)
        # # # print (mean)
        # # fdsfa

        # pz = gaussian_diag(mean, logsd)
        # objective += pz.logp(x) 

        # print (x.shape)
        # x = self.unsqueeze_bchw(x)

        # print (x.shape)
        mean_and_logsd = self.model(x)
        # print (mean_and_logsd.shape)

        # outputs = torch.sum(d)
        # torch.autograd.grad(outputs, inputs)

        # fsada

        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)

        # print (torch.min(mean), torch.max(mean))
        # print (torch.min(logsd), torch.max(logsd))

        # print ()


        # logsd = torch.clamp(logsd, min=-6., max=2.)

        mean = torch.tanh(mean / 100.) * 100.

        # logsd = (torch.tanh(logsd /4.) * 4.) -2.

        logsd = (torch.tanh(logsd /self.lm ) * self.lm ) - (self.lm /2.)



        
        LL = self.gauss_log_prob(x, mean=mean, logsd=logsd)

        # print (torch.min(LL), torch.max(LL))

        # fafds

        # output = self.loss_op(x,output)
        # print (LL.shape)

        LL = LL.view(B,-1).sum(-1)

        # fasdfa
        objective += LL

        # this way, you can encode and decode back the same image. 
        return x, objective



    # def sample(model):
    #     model.train(False)
    #     with torch.no_grad():

    #         data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    #         data = data.cuda()
    #         for i in range(obs[1]):
    #             for j in range(obs[2]):
    #                 # data_v = Variable(data, volatile=True)
    #                 out   = model(data, sample=True)
    #                 out_sample = sample_op(out)
    #                 data[:, :, i, j] = out_sample.data[:, :, i, j]

    #     return data






    def reverse_(self, x, objective, args=None):
        bs, c, h, w = self.input_shape
        samp = torch.zeros(bs, c, h, w).cuda()
        for i in range(h):
            for j in range(w):
                mean_and_logsd   = self.model(samp)
                mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)

                mean = torch.tanh(mean / 100.) * 100.
                # logsd = (torch.tanh(logsd /4.) * 4.) -2.
                logsd = (torch.tanh(logsd /self.lm ) * self.lm ) - (self.lm /2.)

                eps = torch.zeros_like(mean).normal_().cuda()
                x_ = mean + torch.exp(logsd) * eps 

                samp[:, :, i, j] = x_[:, :, i, j]

        # I THINK the model cahgnes smap.. need to copy it first ..
        # B = x.shape[0]
        samp1 = samp.clone()
        B = bs
        mean_and_logsd = self.model(samp)
        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
        mean = torch.tanh(mean / 100.) * 100.
        # logsd = (torch.tanh(logsd /4.) * 4.) -2.
        logsd = (torch.tanh(logsd /self.lm ) * self.lm ) - (self.lm /2.)
        LL = self.gauss_log_prob(samp, mean=mean, logsd=logsd)
        LL = LL.view(B,-1).sum(-1)
        objective -= LL



        # if args is not None and 'temp' in args:
        #     temp = args['temp']
        # else:
        #     temp =1.

        # if args is not None and 'batch_size' in args:
        #     bs = args['batch_size']


        # mean_and_logsd = torch.cuda.FloatTensor(bs, 2 * c, h, w).fill_(0.)
        
        # # if self.conv: 
        # mean_and_logsd = self.conv(mean_and_logsd)

        # mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)

        # logsd = torch.clamp(logsd, min=-6., max=2.)




        # pz = gaussian_diag(mean, logsd, temp=temp)
        # z = pz.sample() if x is None else x
        # objective -= pz.logp(z)

        # this way, you can encode and decode back the same image. 
        return samp1, objective
         

















# Split Layer for multi-scale architecture. Factor of 2 hardcoded.
class Split(Layer):
    def __init__(self, input_shape):
        super(Split, self).__init__()
        bs, c, h, w = input_shape
        self.conv_zero = Conv2dZeroInit(c // 2, c, 3, padding=(3 - 1) // 2)

    def split2d_prior(self, x, args=None):
        h = self.conv_zero(x)
        mean, logs = h[:, 0::2], h[:, 1::2]

        logs = torch.clamp(logs, min=-6., max=2.)

        # if (logs < -10).any():
        #     print (logs)
        #     print ('small logs!!')
        #     fasdfa

        if args is not None:
            temp = args['temp']
        else:
            temp = 1.

        return gaussian_diag(mean, logs, temp=temp)

    def forward_(self, x, objective):
        bs, c, h, w = x.size()
        z1, z2 = torch.chunk(x, 2, dim=1)

        pz = self.split2d_prior(z1)
        # mean = torch.zeros_like(z1)
        # logs = torch.zeros_like(z1)
        # pz = gaussian_diag(mean, logs)
        objective += pz.logp(z2) 
        self.sample = z2

        return z1, objective

    def reverse_(self, x, objective, args=None):

        pz = self.split2d_prior(x, args=args)
        # mean = torch.zeros_like(x)
        # logs = torch.zeros_like(x)
        # pz = gaussian_diag(mean, logs)

        if args is not None and 'use_stored_sample' in args:
            use_stored_sample = args['use_stored_sample']
        else:
            use_stored_sample = 0


        z2 = self.sample if use_stored_sample else pz.sample() 
        # print (x.shape)
        # print (z2.shape)
        # fsdfa
        z = torch.cat([x, z2], dim=1)
        objective -= pz.logp(z2) 
        return z, objective





# Gaussian Prior that's compatible with the Layer framework
class GaussianPrior(Layer):
    def __init__(self, input_shape, args):
        super(GaussianPrior, self).__init__()
        self.input_shape = input_shape
        # print(args.learntop)
        # if args.learntop: 
        #     self.conv = Conv2dZeroInit(2 * input_shape[1], 2 * input_shape[1], 3, padding=(3 - 1) // 2)
        # else: 
        #     self.conv = None

        self.conv = Conv2dZeroInit(2 * input_shape[1], 2 * input_shape[1], 3, padding=(3 - 1) // 2)

    def forward_(self, x, objective):
        mean_and_logsd = torch.cat([torch.zeros_like(x) for _ in range(2)], dim=1)
        # if self.conv:

        #this could just be some parameters we learn, 
        # doesnt need to output of conv 
        mean_and_logsd = self.conv(mean_and_logsd)

        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)

        logsd = torch.clamp(logsd, min=-6., max=2.)

        # print(self.conv)
        # # print (mean)
        # fdsfa

        pz = gaussian_diag(mean, logsd)
        objective += pz.logp(x) 

        # this way, you can encode and decode back the same image. 
        return x, objective

    def reverse_(self, x, objective, args=None):
        bs, c, h, w = self.input_shape

        if args is not None and 'temp' in args:
            temp = args['temp']
        else:
            temp =1.

        if args is not None and 'batch_size' in args:
            bs = args['batch_size']


        mean_and_logsd = torch.cuda.FloatTensor(bs, 2 * c, h, w).fill_(0.)
        
        # if self.conv: 
        mean_and_logsd = self.conv(mean_and_logsd)

        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)

        logsd = torch.clamp(logsd, min=-6., max=2.)




        pz = gaussian_diag(mean, logsd, temp=temp)
        z = pz.sample() if x is None else x
        objective -= pz.logp(z)

        # this way, you can encode and decode back the same image. 
        return z, objective
         







class MoGPrior(Layer):
    def __init__(self, input_shape, args):
        super(MoGPrior, self).__init__()

        bs, c, h, w = input_shape
        self.input_shape = [16, c, h, w]


        self.mean1 = 0. #2.
        self.mean2 = 0. # -2.
        self.logsd = 0. #np.log(.1)

    def gauss_log_prob(self, x, mean, logsd):
        Log2PI = float(np.log(2 * np.pi))
        # return  -0.5 * (Log2PI + 2. * logsd + ((x - mean) ** 2) / torch.exp(2. * logsd))

        aaa = Log2PI + 2. * logsd 
        var = np.exp(2. * logsd)
        bbb = ((x - mean) ** 2) / var
        return  -0.5 * (aaa + bbb)

    def MoG_log_prob(self, x):

        B = x.shape[0]

        logpz_1 = self.gauss_log_prob(x, self.mean1, self.logsd) #.view(B,-1)
        logpz_2 = self.gauss_log_prob(x, self.mean2, self.logsd) #.view(B,-1)

        # log_sum_exp
        # concat = torch.cat([logpz_1,logpz_2], 1)
        max_ = torch.max(logpz_1, logpz_2)

        logpz = torch.log(.5 * (torch.exp(logpz_1 - max_) + torch.exp(logpz_2 - max_))) + max_

        logpz = logpz.view(B,-1).sum(-1)

        return logpz   

    def MoG_sample(self):

        prob = torch.ones(self.input_shape) * .5
        bern = Bernoulli(prob)
        b = bern.sample().cuda()

        eps = torch.zeros_like(b).normal_().cuda()
        z1 = self.mean1 + torch.exp(self.logsd) * eps 
        z2 = self.mean2 + torch.exp(self.logsd) * eps
        z = b * z1 + (1.-b) * z2
        return z

    def forward_(self, x, objective):

        

        logpz = self.MoG_log_prob(x)
        objective += logpz

        return x, objective

    def reverse_(self, x, objective, args=None):
        # bs, c, h, w = self.input_shape

        # if args is not None and 'temp' in args:
        #     temp = args['temp']
        # else:
        #     temp =1.

        # if args is not None and 'batch_size' in args:
        #     bs = args['batch_size']


        z = self.MoG_sample()
        logpz = self.MoG_log_prob(z)
         
        objective -= logpz
        return z, objective
         




class Split_MoG(Layer):
    def __init__(self, input_shape):
        super(Split_MoG, self).__init__()
        bs, c, h, w = input_shape
        self.input_shape = [16, c//2, h, w]
        # print (self.input_shape)

        self.mean1 = 0. #2.
        self.mean2 = 0. #-2.
        self.logsd = 0. #np.log(.1)


    def gauss_log_prob(self, x, mean, logsd):
        Log2PI = float(np.log(2 * np.pi))
        # return  -0.5 * (Log2PI + 2. * logsd + ((x - mean) ** 2) / torch.exp(2. * logsd))

        aaa = Log2PI + 2. * logsd 
        var = np.exp(2. * logsd)
        bbb = ((x - mean) ** 2) / var
        return  -0.5 * (aaa + bbb)

    def MoG_log_prob(self, x):

        B = x.shape[0]

        logpz_1 = self.gauss_log_prob(x, self.mean1, self.logsd) #.view(B,-1)
        logpz_2 = self.gauss_log_prob(x, self.mean2, self.logsd) #.view(B,-1)

        # log_sum_exp
        # concat = torch.cat([logpz_1,logpz_2], 1)
        max_ = torch.max(logpz_1, logpz_2)

        logpz = torch.log(.5 * (torch.exp(logpz_1 - max_) + torch.exp(logpz_2 - max_))) + max_

        logpz = logpz.view(B,-1).sum(-1)

        return logpz   

    def MoG_sample(self):

        prob = torch.ones(self.input_shape) * .5
        bern = Bernoulli(prob)
        b = bern.sample().cuda()

        eps = torch.zeros_like(b).normal_().cuda()
        z1 = self.mean1 + self.logsd * eps 
        z2 = self.mean2 + self.logsd * eps
        z = b * z1 + (1.-b) * z2
        return z


    def forward_(self, x, objective):
        bs, c, h, w = x.size()
        z1, z2 = torch.chunk(x, 2, dim=1)

        objective += self.MoG_log_prob(z2)
        # self.sample = z2

        return z1, objective


    def reverse_(self, x, objective, args=None):

        # pz = self.split2d_prior(x, args=args)
        # mean = torch.zeros_like(x)
        # logs = torch.zeros_like(x)
        # pz = gaussian_diag(mean, logs)

        # if args is not None and 'use_stored_sample' in args:
        #     use_stored_sample = args['use_stored_sample']
        # else:
        #     use_stored_sample = 0


        z2 = self.MoG_sample()
        logpz = self.MoG_log_prob(z2)

        # z2 = self.sample if use_stored_sample else pz.sample() 
        # print (x.shape)
        # print (z2.shape)
        # fsdfa
        # print (z2.shape)
        # print (x.shape)
        z = torch.cat([x, z2], dim=1)
        objective -= logpz
        return z, objective




























# ------------------------------------------------------------------------------
# Coupling Layers
# ------------------------------------------------------------------------------

# Additive Coupling Layer
class AdditiveCoupling(Layer):
    def __init__(self, num_features, hidden_channels):
        super(AdditiveCoupling, self).__init__()
        assert num_features % 2 == 0
        self.NN = NN(num_features // 2, hidden_channels=hidden_channels, 
                        filter_size=3)


    def forward_(self, x, objective):

        z1, z2 = torch.chunk(x, 2, dim=1)
        z2 += self.NN(z1)
        output = torch.cat([z1, z2], dim=1)
        return output, objective

    # #due to my findings:
    #     # this doesnt work becasue x loses its grad somehow 
    #     # so have to stick with original
    # def forward_(self, x, objective):

    #     z1, z2 = torch.chunk(x, 2, dim=1)
    #     z2 += self.NN(z1)
    #     # output = torch.cat([z1, z2], dim=1)
    #     return x, objective


    # def forward_2(self, x, objective):

    #     aa = x.clone()

    #     z1, z2 = torch.chunk(x, 2, dim=1)

    #     # z1_pre = z1.clone()


    #     z2 += self.NN(z1)


    #     print('in additive', torch.mean((aa - x)**2))
    #     # print('in additive z1', torch.mean((z1_pre - z1)**2)) not changing

    #     output = torch.cat([z1, z2], dim=1)

    #     print('in additive x vs output', torch.mean((output - x)**2))

    #     # bb = x.clone()
    #     # print ()
    #     # print (str(layer)[:6])
        
    #     # fafds
        
    #     return output, objective


    def reverse_(self, x, objective, args=None):
        z1, z2 = torch.chunk(x, 2, dim=1)
        z2 -= self.NN(z1)
        return torch.cat([z1, z2], dim=1), objective







# Additive Coupling Layer
class AffineCoupling(Layer):
    def __init__(self, num_features, args, hidden_channels=128):
        super(AffineCoupling, self).__init__()
        # assert num_features % 2 == 0
        self.NN = NN(num_features // 2, channels_out=num_features, hidden_channels=hidden_channels, args=args)

    def forward_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = h[:, 1::2]
        # scale = torch.tanh(scale/4.) /4. + 1.
        scale = torch.tanh(scale) /4. + 1.


        # scale = torch.tanh(scale) + 1.2

        # scale = torch.clamp(scale, min=-4., max=.5)
        # scale = torch.exp(scale)

        # scale = torch.sigmoid(h[:, 1::2] + 2.)

        # shift = torch.clamp(shift, min=-5., max=5.)
        # scale = torch.sigmoid(h[:, 1::2] )

        # shift = torch.clamp(shift, min=-5., max=5.)
        # scale = torch.exp(h[:, 1::2] )

        z2 += shift
        z2 *= scale
        objective += flatten_sum(torch.log(scale))

        # print (objective.shape)
        # print (flatten_sum(torch.log(scale)).shape)
        # fasd

        return torch.cat([z1, z2], dim=1), objective

    def reverse_(self, x, objective, args=None):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = h[:, 1::2]
        # scale = torch.tanh(scale/4.) /4. + 1.
        scale = torch.tanh(scale) /4. + 1.


        # scale = torch.tanh(scale) + 1.2

        # scale = torch.clamp(scale, min=-4., max=.5)
        # scale = torch.exp(scale)

        # shift = torch.clamp(shift, min=-5., max=5.)

        # # scale = torch.sigmoid(h[:, 1::2] + 2.)
        # scale = torch.sigmoid(h[:, 1::2] )

        z2 /= scale
        z2 -= shift
        objective -= flatten_sum(torch.log(scale))

        return torch.cat([z1, z2], dim=1), objective








class SplineCoupling(Layer):
    def __init__(self, num_features, hidden_channels=128):
        super(SplineCoupling, self).__init__()
        # assert num_features % 2 == 0
        self.num_bins = 5 
        dim_multiplier = self.num_bins * 2 - 1

        self.transform_net = NN(num_features // 2, channels_out=num_features //2 * dim_multiplier, hidden_channels=hidden_channels)


    def forward_(self, x, objective):

        identity_split, transform_split = torch.chunk(x, 2, dim=1)
        # identity_split = inputs[:, self.identity_features, ...]
        # transform_split = inputs[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_split)
        # print (transform_params.shape)
        # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
        b, c, h, w = transform_split.shape
        transform_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        # print (transform_params.shape)
        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins:]

        transformed, logabsdet = unconstrained_quadratic_spline(transform_split, 
                                            unnormalized_widths=unnormalized_widths, 
                                            unnormalized_heights=unnormalized_heights)

        output = torch.cat([identity_split, transformed], dim=1)

        objective += utils2.sum_except_batch(logabsdet)

        return output, objective




    def reverse_(self, x, objective, args=None):

        identity_split, transform_split = torch.chunk(x, 2, dim=1)
        # identity_split = inputs[:, self.identity_features, ...]
        # transform_split = inputs[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_split)
        # print (transform_params.shape)
        # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
        b, c, h, w = transform_split.shape
        transform_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        # print (transform_params.shape)
        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins:]

        transformed, logabsdet = unconstrained_quadratic_spline(transform_split, 
                                            unnormalized_widths=unnormalized_widths, 
                                            unnormalized_heights=unnormalized_heights,
                                            inverse=True)

        output = torch.cat([identity_split, transformed], dim=1)


        objective += utils2.sum_except_batch(logabsdet)

        return output, objective











class LinearSplineCoupling(Layer):
    def __init__(self, num_features, hidden_channels=128):
        super(LinearSplineCoupling, self).__init__()
        # assert num_features % 2 == 0
        self.num_bins = 3
        dim_multiplier = self.num_bins 

        self.transform_net = NN(num_features // 2, channels_out=num_features //2 * dim_multiplier, hidden_channels=hidden_channels)


    def forward_(self, x, objective):

        identity_split, transform_split = torch.chunk(x, 2, dim=1)
        # identity_split = inputs[:, self.identity_features, ...]
        # transform_split = inputs[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_split)
        # print (transform_params.shape)
        # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
        b, c, h, w = transform_split.shape
        transform_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        # print (transform_params.shape)
        # unnormalized_widths = transform_params[..., :self.num_bins]
        # unnormalized_heights = transform_params[..., self.num_bins:]

        transformed, logabsdet = unconstrained_linear_spline(transform_split, 
                                            unnormalized_pdf=transform_params)

        output = torch.cat([identity_split, transformed], dim=1)

        objective += utils2.sum_except_batch(logabsdet)

        return output, objective




    def reverse_(self, x, objective, args=None):

        identity_split, transform_split = torch.chunk(x, 2, dim=1)
        # identity_split = inputs[:, self.identity_features, ...]
        # transform_split = inputs[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_split)
        # print (transform_params.shape)
        # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
        b, c, h, w = transform_split.shape
        transform_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        # print (transform_params.shape)
        # unnormalized_widths = transform_params[..., :self.num_bins]
        # unnormalized_heights = transform_params[..., self.num_bins:]

        transformed, logabsdet = unconstrained_linear_spline(transform_split, 
                                            unnormalized_pdf=transform_params,
                                            inverse=True)

        output = torch.cat([identity_split, transformed], dim=1)


        objective += utils2.sum_except_batch(logabsdet)

        return output, objective





















# ------------------------------------------------------------------------------
# Normalizing Layers
# ------------------------------------------------------------------------------

# ActNorm Layer with data-dependant init
class ActNorm(Layer):
    def __init__(self, num_features, args, logscale_factor=1., scale=1.):
        super(Layer, self).__init__()
        if args.load_step > 0:
            self.initialized = True
        else:
            self.initialized = False
        self.logscale_factor = logscale_factor
        self.scale = scale
        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1)))
        self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1)))

    def forward_(self, input, objective):
        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)

        if not self.initialized: 
            self.initialized = True
            unsqueeze = lambda x: x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = input.size(0) * input.size(-1)
            b = -torch.sum(input, dim=(0, -1)) / sum_size
            vars = unsqueeze(torch.sum((input + unsqueeze(b)) ** 2, dim=(0, -1))/sum_size)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6)) / self.logscale_factor
          
            self.b.data.copy_(unsqueeze(b).data)
            self.logs.data.copy_(logs.data)

        logs = self.logs * self.logscale_factor
        b = self.b
        
        output = (input + b) * torch.exp(logs)
        dlogdet = torch.sum(logs) * input.size(-1) # c x h  

        return output.view(input_shape), objective + dlogdet

    def reverse_(self, input, objective, args=None):
        assert self.initialized
        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        output = input * torch.exp(-logs) - b
        dlogdet = torch.sum(logs) * input.size(-1) # c x h  

        return output.view(input_shape), objective - dlogdet

# (Note: a BatchNorm layer can be found in previous commits)







# Inverse Sigmoid
class InverseSigmoid(Layer):
    def __init__(self):
        super(InverseSigmoid, self).__init__()
        # assert num_features % 2 == 0
        # self.NN = NN(num_features // 2)

    def forward_(self, x, objective):


        # SHOULD MODIFY OBJECTIVE
        # however wouldnt affect the gradient since x cant be cahnged, its the data
        jac = (1/x) + (1/(1-x))
        objective += flatten_sum(torch.log(jac))

        x = -torch.log( (1/x) - 1)
        # print (objective.shape)
        # fasfd

        return x, objective

    def reverse_(self, x, objective, args=None):

        x = 1/ (torch.exp(-x) + 1)

        # SHOULD MODIFY OBJECTIVE
        jac = 1/x + (1/(1-x))
        objective += flatten_sum(torch.log(jac))

        return x, objective






# Inverse Sigmoid
class LeakyRelu(Layer):
    def __init__(self):
        super(LeakyRelu, self).__init__()
        # assert num_features % 2 == 0
        # self.NN = NN(num_features // 2)

        self.slope = .01

    def forward_(self, x, objective):

        above_zero = (x > 0).float()
        below_zero = 1. - above_zero

        x_transformed = x * below_zero * self.slope

        output = (x * above_zero) +  x_transformed


        # objective += flatten_sum(np.log(self.slope)*below_zero)


        # # SHOULD MODIFY OBJECTIVE
        # # however wouldnt affect the gradient since x cant be cahnged, its the data
        # jac = (1/x) + (1/(1-x))
        # objective += flatten_sum(torch.log(jac))

        # x = -torch.log( (1/x) - 1)
        # # print (objective.shape)
        # # fasfd

        return output, objective

    def reverse_(self, x, objective, args=None):

        above_zero = (x > 0).float()
        below_zero = 1. -above_zero

        x_transformed = x * below_zero / self.slope

        output = (x * above_zero) +  x_transformed


        # objective -= flatten_sum(np.log(self.slope)*below_zero)

        # x = 1/ (torch.exp(-x) + 1)

        # # SHOULD MODIFY OBJECTIVE
        # jac = 1/x + (1/(1-x))
        # objective += flatten_sum(torch.log(jac))

        return x, objective















# ------------------------------------------------------------------------------
# Stacked Layers
# ------------------------------------------------------------------------------



# # 1 step of the flow (see Figure 2 a) in the original paper)
# class RevNetStep(LayerList):
#     def __init__(self, num_channels, args):
#         super(RevNetStep, self).__init__()
#         self.args = args
#         layers = []

#         if args.norm == 'actnorm': 
#             layers += [ActNorm(num_channels)]
#         else: 
#             assert not args.norm               
 
#         if args.permutation == 'reverse':
#             layers += [Reverse(num_channels)]
#         elif args.permutation == 'shuffle': 
#             layers += [Shuffle(num_channels)]
#         elif args.permutation == 'conv':
#             # layers += [Invertible1x1Conv(num_channels)]
#             layers += [OneByOneConvolution(num_channels)]


#         else: 
#             raise ValueError

#         if args.coupling == 'additive': 
#             layers += [AdditiveCoupling(num_channels)]
#         elif args.coupling == 'affine':
#             layers += [AffineCoupling(num_channels, hidden_channels=args.hidden_channels)]
#         elif args.coupling == 'spline':
#             layers += [SplineCoupling(num_channels)]
#         else: 
#             raise ValueError

#         self.layers = nn.ModuleList(layers)





# trying new setup
class RevNetStep(LayerList):
    def __init__(self, num_channels, args):
        super(RevNetStep, self).__init__()
        self.args = args
        layers = []


        

        if args.norm == 'actnorm': 
            layers += [ActNorm(num_channels, args=args)]
        else: 
            assert not args.norm     




        # layers += [LeakyRelu()]


 
        if args.permutation == 'reverse':
            layers += [Reverse(num_channels)]
        elif args.permutation == 'shuffle': 
            layers += [Shuffle(num_channels)]
        elif args.permutation == 'conv':
            # layers += [Invertible1x1Conv(num_channels)]
            layers += [OneByOneConvolution(num_channels)]


        else: 
            raise ValueError

        if args.coupling == 'additive': 
            layers += [AdditiveCoupling(num_channels, hidden_channels=args.hidden_channels)]
        elif args.coupling == 'affine':
            layers += [AffineCoupling(num_channels, hidden_channels=args.hidden_channels, args=args)]
        elif args.coupling == 'spline':
            layers += [SplineCoupling(num_channels, hidden_channels=args.hidden_channels)]
        elif args.coupling == 'linear_spline':
            


            
            layers += [AdditiveCoupling(num_channels, hidden_channels=args.hidden_channels)]
            layers += [LinearSplineCoupling(num_channels, hidden_channels=args.hidden_channels // 4 )]
            
            
        else: 
            raise ValueError




        # if args.norm == 'actnorm': 
        #     layers += [ActNorm(num_channels)]
        # else: 
        #     assert not args.norm       


        # layers += [Reverse(num_channels)]


        # if args.coupling == 'additive': 
        #     layers += [AdditiveCoupling(num_channels, hidden_channels=args.hidden_channels)]
        # elif args.coupling == 'affine':
        #     layers += [AffineCoupling(num_channels, hidden_channels=args.hidden_channels)]
        # elif args.coupling == 'spline':
        #     layers += [SplineCoupling(num_channels, hidden_channels=args.hidden_channels)]
        # else: 
        #     raise ValueError



        self.layers = nn.ModuleList(layers)


# # Full model
# class Glow_(LayerList, nn.Module):
#     def __init__(self, input_shape, args):
#         super(Glow_, self).__init__()
#         layers = []
#         output_shapes = []
#         B, C, H, W = input_shape


#         layers += [InverseSigmoid()]
        
#         for i in range(args.n_levels):

#             if W > 10:
#                 # Squeeze Layer 
#                 layers += [Squeeze(input_shape)]
#                 C, H, W = C * 4, H // 2, W // 2
#                 output_shapes += [(-1, C, H, W)]
                
#             # RevNet Block
#             layers += [RevNetStep(C, args) for _ in range(args.depth)]
#             output_shapes += [(-1, C, H, W) for _ in range(args.depth)]

#             if i < args.n_levels - 1: 
#                 # Split Layer
#                 layers += [Split(output_shapes[-1])]
#                 C = C // 2
#                 output_shapes += [(-1, C, H, W)]

#         layers += [GaussianPrior((B, C, H, W), args)]
#         output_shapes += [output_shapes[-1]]

#         # # print (output_shapes)
#         # # fadsf

        
#         for i in range(len(output_shapes)):
#             print (i, output_shapes[i])
#         # fdsa

#         self.layers = nn.ModuleList(layers)
#         self.output_shapes = output_shapes
#         self.args = args
#         self.flatten()


# Full model - trying different architecutres
class Glow_(LayerList, nn.Module):
    def __init__(self, input_shape, args):
        super(Glow_, self).__init__()
        layers = []
        output_shapes = []
        B, C, H, W = input_shape

        # depths = [4,4,4,4,4,30]

        layers += [InverseSigmoid()]
        
        for i in range(args.n_levels):

            if W > 10:
                # Squeeze Layer 
                layers += [Squeeze(input_shape)]
                C, H, W = C * 4, H // 2, W // 2
                output_shapes += [(-1, C, H, W)]
                # print (output_shapes)
                # fads
                
            # # RevNet Block
            layers += [RevNetStep(C, args) for _ in range(args.depth)]
            output_shapes += [(-1, C, H, W) for _ in range(args.depth)]
            # layers += [RevNetStep(C, args) for _ in range(depths[i])]
            # output_shapes += [(-1, C, H, W) for _ in range(depths[i])]

            if i < args.n_levels - 1 and args.base_dist in ['Mog', 'Gauss']: 
                # Split Layer
                if args.base_dist == 'MoG':
                    layers += [Split_MoG(output_shapes[-1])]
                elif args.base_dist == 'Gauss':
                    layers += [Split(output_shapes[-1])]                    

                C = C // 2
                output_shapes += [(-1, C, H, W)]

        while W > 30:
            # Squeeze Layer 
            layers += [Squeeze(input_shape)]
            C, H, W = C * 4, H // 2, W // 2
            output_shapes += [(-1, C, H, W)]
            # print (output_shapes)
            # fads

        if args.base_dist == 'MoG':
            layers += [MoGPrior((B, C, H, W), args)]
        elif args.base_dist == 'AR':
            layers += [AR_Prior((B, C, H, W), args)]
        else:
            layers += [GaussianPrior((B, C, H, W), args)]


        output_shapes += [output_shapes[-1]]

        # # print (output_shapes)
        # # fadsf

        
        # for i in range(len(output_shapes)):
        #     print (i, output_shapes[i])
        # # fdsa

        self.layers = nn.ModuleList(layers)
        self.output_shapes = output_shapes
        self.args = args
        self.flatten()





    def forward(self, *inputs):
        return self.forward_(*inputs)

    def sample(self, args=None):
        with torch.no_grad():
            samples = self.reverse_(None, 0., args=args)[0]
            return samples





    # def forward_andgetlayers(self, *inputs):
    #     return self.forward_andgetlayers_(*inputs)

    # def reverse_andgetlayers(self, x, objective):
    #     with torch.no_grad():
    #         return self.reverse_andgetlayers_(x, 0.)






    def flatten(self):
        # flattens the list of layers to avoid recursive call every time. 
        processed_layers = []
        to_be_processed = [self]
        while len(to_be_processed) > 0:
            current = to_be_processed.pop(0)
            if isinstance(current, LayerList):
                to_be_processed = [x for x in current.layers] + to_be_processed
            elif isinstance(current, Layer):
                processed_layers += [current]
        
        self.layers = nn.ModuleList(processed_layers)










    def load_params_v3(self, load_dir, step, name):
        save_to=os.path.join(load_dir, name + str(step)+".pt")
        state_dict = torch.load(save_to)
        # # # print (state_dict)
        # for key, val in state_dict.items():
        #     print (key)
        # fddsf
        self.load_state_dict(state_dict)
        print ('loaded params', save_to)


    def save_params_v3(self, save_dir, step, name):
        save_to=os.path.join(save_dir, name + str(step)+".pt")
        torch.save(self.state_dict(), save_to)
        print ('saved params', save_to)
        
























    
