

#inputs to layers is a list of the zs
#so that AR has access to all zs instead of just the final z


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

from Prior_Att import Att_Prior

import torch.distributions as dist

import sys, os
sys.path.insert(0, os.path.abspath('./PixelCNN/'))

from utils import * 
from model import * 
from PIL import Image

#https://forums.fast.ai/t/show-gpu-utilization-metrics-inside-training-loop-without-subprocess-call/26594
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
def get_gpu_mem(concat_string=''):
    res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(concat_string, f'mem: {100 * (res.used / res.total):.3f}%') # percentage 





def squeeze_bchw(x):
    bs, c, h, w = x.size()
    assert h % 2 == 0 and w % 2 == 0 #, pdb.set_trace()
    # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
    x = x.view(bs, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(bs, c * 2 * 2, h // 2, w // 2)
    return x


def unsqueeze_bchw(x):
    bs, c, h, w = x.size()
    assert c >= 4 and c % 4 == 0
    # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
    x = x.view(bs, c // 2 ** 2, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(bs, c // 2 ** 2, h * 2, w * 2)
    return x








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

        return [x], objective

    def reverse_(self, x, objective, args=None):

        assert len(x) == 1

        x = x[0]

        x = 1/ (torch.exp(-x) + 1)

        # SHOULD MODIFY OBJECTIVE
        jac = 1/x + (1/(1-x))
        objective += flatten_sum(torch.log(jac))

        return x, objective


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

    def forward_(self, x_list, objective):
        x = x_list.pop(-1)
        x = x[:, self.indices]
        x_list.append(x)
        return x_list, objective

    def reverse_(self, x_list, objective, args=None):
        x = x_list.pop(-1)
        x = x[:, self.rev_indices]
        x_list.append(x)
        return x_list, objective
        



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
    
    def forward_(self, x_list, objective):
        # if len(x.size()) != 4: 
        #     raise NotImplementedError # Maybe ValueError would be more appropriate

        x = x_list.pop(-1)
        x = self.squeeze_bchw(x)
        x_list.append(x)

        return x_list, objective
        
    def reverse_(self, x_list, objective, args=None):
        # if len(x.size()) != 4: 
        #     raise NotImplementedError

        x = x_list.pop(-1)
        x = self.unsqueeze_bchw(x)
        x_list.append(x)

        return x_list, objective





# Split Layer for multi-scale architecture. Factor of 2 hardcoded.
class Split(Layer):
    def __init__(self, input_shape):
        super(Split, self).__init__()
        # bs, c, h, w = input_shape
        # self.conv_zero = Conv2dZeroInit(c // 2, c, 3, padding=(3 - 1) // 2)

    # def split2d_prior(self, x, args=None):
    #     h = self.conv_zero(x)
    #     mean, logs = h[:, 0::2], h[:, 1::2]

    #     logs = torch.clamp(logs, min=-6., max=2.)

    #     # if (logs < -10).any():
    #     #     print (logs)
    #     #     print ('small logs!!')
    #     #     fasdfa

    #     # if args is not None:
    #     #     temp = args['temp']
    #     # else:
    #     #     temp = 1.

    #     return gaussian_diag(mean, logs, temp=temp)

    def forward_(self, x_list, objective):
        # bs, c, h, w = x.size()

        x = x_list.pop(-1)
        z1, z2 = torch.chunk(x, 2, dim=1)
        x_list.append(z1)
        x_list.append(z2)

        # pz = self.split2d_prior(z1)
        # mean = torch.zeros_like(z1)
        # logs = torch.zeros_like(z1)
        # pz = gaussian_diag(mean, logs)
        # objective += pz.logp(z2) 
        # self.sample = z2

        return x_list, objective

    def reverse_(self, x_list, objective, args=None):

        z2 = x_list.pop(-1)
        z1 = x_list.pop(-1)
        z = torch.cat([z1, z2], dim=1)
        x_list.append(z)

        # pz = self.split2d_prior(x, args=args)
        # mean = torch.zeros_like(x)
        # logs = torch.zeros_like(x)
        # pz = gaussian_diag(mean, logs)

        # if args is not None and 'use_stored_sample' in args:
        #     use_stored_sample = args['use_stored_sample']
        # else:
        #     use_stored_sample = 0


        # z2 = self.sample if use_stored_sample else pz.sample() 
        # print (x.shape)
        # print (z2.shape)
        # fsdfa
        
        # objective -= pz.logp(z2) 
        return x_list, objective











# Additive Coupling Layer
class AffineCoupling(Layer):
    def __init__(self, num_features, args, hidden_channels=128):
        super(AffineCoupling, self).__init__()
        # assert num_features % 2 == 0
        self.NN = NN(num_features // 2, channels_out=num_features, hidden_channels=hidden_channels, filter_size=3, args=args)

    def forward_(self, x_list, objective):


        x = x_list.pop(-1)


        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = h[:, 1::2]
        # scale = torch.tanh(scale/4.) /4. + 1.
        scale = torch.tanh(scale) /4. + 1.

        z2 += shift
        z2 *= scale
        objective += flatten_sum(torch.log(scale))


        x = torch.cat([z1, z2], dim=1)
        x_list.append(x)

        return x_list, objective

    def reverse_(self, x_list, objective, args=None):

        x = x_list.pop(-1)

        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = h[:, 1::2]
        # scale = torch.tanh(scale/4.) /4. + 1.
        scale = torch.tanh(scale) /4. + 1.

        z2 /= scale
        z2 -= shift
        objective -= flatten_sum(torch.log(scale))

        x = torch.cat([z1, z2], dim=1)
        x_list.append(x)

        return x_list, objective









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




# ------------------------------------------------------------------------------
# Normalizing Layers
# ------------------------------------------------------------------------------

# ActNorm Layer with data-dependant init
class ActNorm_Layer(Layer):
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

    def forward_(self, x_list, objective):

        input = x_list.pop(-1)

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

        x_list.append(output.view(input_shape))

        return x_list, objective + dlogdet

    def reverse_(self, x_list, objective, args=None):
        assert self.initialized

        input = x_list.pop(-1)

        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        output = input * torch.exp(-logs) - b
        dlogdet = torch.sum(logs) * input.size(-1) # c x h 

        x_list.append(output.view(input_shape))

        return x_list, objective - dlogdet

# (Note: a BatchNorm layer can be found in previous commits)












#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################


# ------------------------------------------------------------------------------
# Layers involving prior
# ------------------------------------------------------------------------------

class Combine(Layer):
    def __init__(self, split_sizes, args):
        super(Combine, self).__init__()    

        self.prior_max_width = args.prior_max_width

        # self.final_width = split_sizes[-1][1]
        self.final_width = args.final_width

        self.split_sizes = split_sizes
        print ('split sizes', self.split_sizes)
        print ('final width', self.final_width)

        self.split_sizes_samewidth = []
        for i in range(len(split_sizes)):
            shape = split_sizes[i]
            # while shape[1] > self.prior_max_width:
            while shape[1] != self.final_width:
                shape = [shape[0]*4, shape[1]//2, shape[2]//2,]
            self.split_sizes_samewidth.append(shape)

        print ('split sizes same width', self.split_sizes_samewidth)
        # fsdaf
        


    def forward_(self, x_list, objective):

        # make all zs have have width/height
        new_z_list = []
        for i in range(len(x_list)):
            z = x_list[i]
            # while z.shape[2] > self.prior_max_width:
            while z.shape[2] != self.final_width:
                z = squeeze_bchw(z)

            new_z_list.append(z)

        # for i in range(len(new_z_list)):
        #     print (new_z_list[i].shape)
        # fadfa
        
        # # Put zs together 
        z = new_z_list.pop(0)
        for i in range(len(new_z_list)):
            z = torch.cat([z,new_z_list[i]], dim=1)

        return z, objective



    def reverse_(self, z, objective, args=None):

        # print (self.split_sizes)

        #Seperate channels
        z_list = []
        for i in range(len(self.split_sizes_samewidth)-1):
            # C = self.split_sizes_samewidth[i][0]
            z1, z2 = torch.chunk(z, 2, dim=1)
            z_list.append(z1)
            z = z2
        z_list.append(z)

        # for i in range(len(z_list)):
        #     print (z_list[i].shape)

        #Unsqueeze to their original sizes
        new_z_list = []
        for i in range(len(self.split_sizes)):
            z =  z_list[i]
            # print (z.shape[1], self.split_sizes[i][0])

            while z.shape[1] != self.split_sizes[i][0]:
                z = unsqueeze_bchw(z)
            new_z_list.append(z)

        # for i in range(len(new_z_list)):
        #     print (new_z_list[i].shape)
        # fadfa

        # TODO

        return new_z_list, objective







class AR_Prior(Layer):
    def __init__(self, input_shape, args):
        super(AR_Prior, self).__init__()
        self.input_shape = input_shape

        # print ('p(z)', self.input_shape)
        # fdsaf


        self.model = PixelCNN(nr_resnet=args.AR_resnets, nr_filters=args.AR_channels, 
                    input_channels=input_shape[1], nr_logistic_mix=2)

        self.loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)

        self.lm = 10.

    def gauss_log_prob(self, x, mean, logsd):
        Log2PI = float(np.log(2 * np.pi))
        # return  -0.5 * (Log2PI + 2. * logsd + ((x - mean) ** 2) / torch.exp(2. * logsd))

        aaa = Log2PI + 2. * logsd 
        var = torch.exp(2. * logsd)
        bbb = ((x - mean) ** 2) / var
        return  -0.5 * (aaa + bbb)
 
    def get_logsd(self, x):

        # #v1
        # logsd = (torch.tanh(x /self.lm ) * self.lm ) - (self.lm /2.)

        # #v2 - softplus of scaled tanh(x)
        logsd = torch.log(torch.log(1+torch.exp((torch.tanh(x /self.lm ) * self.lm ) - (self.lm /2.))))

        #v3 - softplus
        # logsd = torch.log(torch.log(1+torch.exp(x)))

        return logsd




    def forward_(self, z, objective):

        # # make all zs have have width/height
        # new_z_list = []
        # for i in range(len(x_list)):
        #     z = x_list[i]
        #     while z.shape[2] > self.prior_max_width:
        #         z = squeeze_bchw(z)

        #     new_z_list.append(z)
        
        # # Put zs together 
        # z = new_z_list.pop(0)
        # for i in range(len(new_z_list)):
        #     z = torch.cat([z,new_z_list[i]], dim=1)


        B = z.shape[0]
        mean_and_logsd = self.model(z)
        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
        mean = torch.tanh(mean / 100.) * 100.
        logsd = self.get_logsd(logsd)

        LL = self.gauss_log_prob(z, mean=mean, logsd=logsd)
        LL = LL.view(B,-1).sum(-1)

        objective += LL

        return z, objective







    def reverse_(self, x, objective, args=None):
        bs, c, h, w = self.input_shape
        bs = args.sample_size

        samp = torch.zeros(bs, c, h, w).cuda()



        for i in range(h):

            if i%5==0:
                print ('h', i, '/'+str(h))
                # get_gpu_mem()


            if args:
                if args.special_sample:
                    if i%5==0:
                        print ('h', i, '/'+str(h))



            for j in range(w):
                mean_and_logsd   = self.model(samp)
                mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)

                # print (logsd.shape)
                # print (logsd[:,:,0,0])
                # fsdaf

                mean = torch.tanh(mean / 100.) * 100.
                # logsd = (torch.tanh(logsd /4.) * 4.) -2.
                # logsd = (torch.tanh(logsd /self.lm ) * self.lm ) - (self.lm /2.)
                logsd = self.get_logsd(logsd)

                # #MIN and MAX Standard deviations
                # print (np.exp((-1 * self.lm ) - (self.lm /2.)))
                # print (np.exp((1 * self.lm ) - (self.lm /2.)))
                # fasdf

                x_ = torch.zeros(bs, c, h, w).cuda()

                if args:
                    if args.special_sample:
                        for ii in range (bs):

                            if ii < 32:
                                if ii < i:
                                    eps = torch.zeros_like(mean).normal_().cuda()
                                    x_[ii] = mean[ii] + torch.exp(logsd)[ii] * eps[ii] 
                                else:
                                    x_[ii] = mean[ii]

                                # x_[i*j:] = mean[i*j:]
                                # x_[:i*j] = mean[i*j:] + torch.exp(logsd)[i*j:] * torch.zeros_like(mean).normal_().cuda()[i*j:]
                            else:
                                if ii-32 > i:
                                    eps = torch.zeros_like(mean).normal_().cuda()
                                    x_[ii] = mean[ii] + torch.exp(logsd)[ii] * eps[ii] 
                                else:
                                    x_[ii] = mean[ii]                            


                        # if i < args.special_h:
                        #     x_ = mean 

                    else:
                        eps = torch.zeros_like(mean).normal_().cuda()
                        x_ = mean + torch.exp(logsd) * eps  * args.temp


                    # print (i,j, torch.mean(torch.exp(logsd[:, :, i, j])).data.cpu().numpy(), 
                                    # torch.std(torch.exp(logsd[:, :, i, j])).data.cpu().numpy() )



                else:
                    eps = torch.zeros_like(mean).normal_().cuda()
                    x_ = mean + torch.exp(logsd) * eps * args.temp

                samp[:, :, i, j] = x_[:, :, i, j]

        # I THINK the model cahgnes smap.. need to copy it first ..
        # B = x.shape[0]


        

        # wont compute likelhood of samples for the moment
        # samp1 = samp.clone()
        # B = bs
        # mean_and_logsd = self.model(samp)
        # mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
        # mean = torch.tanh(mean / 100.) * 100.
        # # logsd = (torch.tanh(logsd /4.) * 4.) -2.
        # # logsd = (torch.tanh(logsd /self.lm ) * self.lm ) - (self.lm /2.)
        # logsd = self.get_logsd(logsd)
        # LL = self.gauss_log_prob(samp, mean=mean, logsd=logsd)
        # LL = LL.view(B,-1).sum(-1)
        # objective -= LL

        return samp, objective
         


















#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################






# Wrapper for stacking multiple layers 
class LayerList(Layer):
    def __init__(self, list_of_layers=None):
        super(LayerList, self).__init__()
        self.layers = nn.ModuleList(list_of_layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward_(self, x, objective):
        for layer in self.layers:

            # if str(layer)[:5] == 'Combi':

            #     idx = 2

            #     x_pre = x[idx].clone()  

            x, objective = layer.forward_(x, objective)

            # if str(layer)[:5] == 'Combi':
            #     # print (len(x))
            #     x, objective = layer.reverse_(x, objective)
            #     # print (len(x))

            #     if (x_pre == x[idx]).all():
            #         print (str(layer)[:9], 'pass')
            #     else:
            #         print (str(layer)[:9], 'fail')


            #     fdsfa


            # if ((x!=x).any() or (objective!=objective).any()  ):

            #     print (str(layer)[:6])

            #     # h = layer.conv_zero(x_pre)
            #     # mean, logs = h[:, 0::2], h[:, 1::2]

            #     # print (torch.min(x_pre), torch.max(x_pre))
            #     print ('x', torch.min(x), torch.max(x))
            #     print ('obj', torch.min(objective), torch.max(objective))
            #     # print ((x_pre!=x_pre).any(), (mean!=mean).any(), (logs!=logs).any())
            #     # print ( layer.conv_zero.logs)
            #     # fadfas
            #     print ('bad stuff in forward')
            #     fasdf

        return x, objective

    def reverse_(self, x, objective, args=None):
        # count=0
        for layer in reversed(self.layers): 


            x, objective = layer.reverse_(x, objective, args=args)


            # if (x!=x).any() or torch.max(x) > 999999:
            #     print (count, layer)

            #     # h = layer.conv_zero(x_pre)
            #     # mean, logs = h[:, 0::2], h[:, 1::2]

            #     print (torch.min(x_pre), torch.max(x_pre))
            #     print (torch.min(x), torch.max(x))
            #     # print ((x_pre!=x_pre).any(), (mean!=mean).any(), (logs!=logs).any())
            #     # print ( layer.conv_zero.logs)
            #     # fadfas
            #     print ('bad stuff in reverse')
            #     fasdf

            # x_pre = x.clone()    
            # count+=1

        # fsadfa

        return x, objective












# trying new setup
class RevNetStep(LayerList):
    def __init__(self, num_channels, args):
        super(RevNetStep, self).__init__()
        self.args = args
        layers = []


        

        if args.norm == 'actnorm': 
            layers += [ActNorm_Layer(num_channels, args=args)]
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
        total_dims = C*H*W
        prior_max_width = 17 #34 #17
        args.prior_max_width = prior_max_width

        min_channels = 13

        final_width = W
        while final_width > prior_max_width:
            final_width = final_width // 2

        # print (final_width)

        if int(H / 2**args.n_levels) > min_channels:
            final_width = np.minimum(final_width, int(H / 2**args.n_levels))


        # print (final_width)
        args.final_width = final_width
        # fsdaf

        final_channels = total_dims // final_width // final_width

        # depths = [4,4,4,4,4,30]
        # split_count = 0
        split_spaces = '' 
        split_sizes = []

        layers += [InverseSigmoid()]
        output_shapes += [('InvSigmoid', C, H, W)]

        for i in range(5):
            # Squeeze Layer 
            if W > prior_max_width:
                layers += [Squeeze(input_shape)]
                C, H, W = C * 4, H // 2, W // 2
                output_shapes += [(split_spaces + 'Squeeze', C, H, W)]
                # print (output_shapes)
                # fads      
        
        for i in range(args.n_levels):

            # if W > 20:
            #     # Squeeze Layer 
            #     layers += [Squeeze(input_shape)]
            #     C, H, W = C * 4, H // 2, W // 2
            #     output_shapes += [(split_spaces + 'Squeeze', C, H, W)]
            #     # print (output_shapes)
            #     # fads
                
            # # RevNet Block
            layers += [RevNetStep(C, args) for _ in range(args.depth)]
            output_shapes += [(split_spaces + 'RevNet', C, H, W) for _ in range(args.depth)]
            # layers += [RevNetStep(C, args) for _ in range(depths[i])]
            # output_shapes += [(-1, C, H, W) for _ in range(depths[i])]

            if i < args.n_levels - 1 : #and args.base_dist in ['Mog', 'Gauss']: 

                if C > min_channels:
                    # Split Layer
                    layers += [Split(output_shapes[-1])] 
                    # split_count +=1
                    split_spaces += '  '


                    # if args.base_dist == 'MoG':
                    #     layers += [Split_MoG(output_shapes[-1])]
                    # elif args.base_dist == 'Gauss':
                    #     layers += [Split(output_shapes[-1])]                    

                    C = C // 2
                    output_shapes += [(split_spaces + 'Split', C, H, W)]

                    split_sizes.append([C,H,W])

        split_sizes.append([C,H,W])

        # if args.base_dist in ['AR', 'AttPrior']:
        #     # while W > 30:
        #     while W > 17:
        #     # while W > 10:
        #         # Squeeze Layer 
        #         layers += [Squeeze(input_shape)]
        #         C, H, W = C * 4, H // 2, W // 2
        #         output_shapes += [('Squeeze', C, H, W)]
        #         # print (output_shapes)
        #         # fads

        # while W > prior_max_width:
        #     # Squeeze Layer 
        #     layers += [Squeeze(input_shape)]
        #     C, H, W = C * 4, H // 2, W // 2
        #     output_shapes += [('Squeeze', C, H, W)]
        #     # print (output_shapes)
        #     # fads

        # print (len(layers))


        # if args.base_dist == 'MoG':
        #     layers += [MoGPrior((B, C, H, W), args)]
        # elif args.base_dist == 'AR':
        #     layers += [AR_Prior((B, C, H, W), args)]
        # elif args.base_dist == 'Gauss':
        #     layers += [GaussianPrior((B, C, H, W), args)]
        # elif args.base_dist == 'AttPrior':
        #     layers += [Att_Prior((B, C, H, W), args)]

        # # print (len(layers))


        # TODO#, the shape of C should be larger 

        
        layers += [Combine(split_sizes, args)]
        output_shapes += [('Combine', final_channels, final_width, final_width)]

        layers += [AR_Prior((B, final_channels, final_width, final_width), args)]
        output_shapes += [('Prior', final_channels, final_width, final_width)]


        # output_shapes += [output_shapes[-1]]

        # # print (output_shapes)
        # # fadsf

        
        # print (input_shape)
        for i in range(len(output_shapes)):
            print (i, output_shapes[i])
        print()

        self.layers = nn.ModuleList(layers)
        # print(len(self.layers))
        # fadsf




        self.output_shapes = output_shapes
        self.args = args


        self.flatten()


        # count=0
        # for layer in self.layers:
        #     print (count, str(layer)[:6])
        #     count+=1

        # fdad


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




    def test_reversability(self, x, objective):

        print (x.shape)


        # layer = InverseSigmoid()

        # x_original = x.clone().data #.cpu()
        # print (torch.min(x[0]), torch.max(x[0]))

        # x, objective = layer.forward_(x, objective)
        # # x_list_forward = x
        # print (torch.min(x[0]), torch.max(x[0]))
        # x, objective = layer.reverse_(x, objective)
        # x_reverse = x

        # print (torch.min(x[0]), torch.max(x[0]))


        # print (x_reverse.shape)

        # print (x_original[0][0][0][:10])
        # print (x_reverse[0][0][0][:10])

        # if (x_reverse == x_original).all():

        #     print (str(layer)[:9], 'pass')

        # else:
        #     print (str(layer)[:9], 'fail')

        #     print ( torch.sum( ~(x_reverse == x_original)) )
        #     # print ( torch.sum( ~(x_reverse[0] == x_original[0])) )
        #     print (torch.max(x_reverse - x_original))



        x = [x]
        # layer = Shuffle(3)
        # layer = Squeeze(x[0].shape)
        layer = Split(x[0].shape)
        x_original = x[0].clone().data #.cpu()
        x, objective = layer.forward_(x, objective)
        x, objective = layer.reverse_(x, objective)
        x_reverse = x[0]
        if (x_reverse == x_original).all():
            print (str(layer)[:9], 'pass')
        else:
            print (str(layer)[:9], 'fail')



        # for layer in self.layers:

        #     x_original = x.clone().data.cpu()

        #     x, objective = layer.forward_(x, objective)

        #     x_forward = x.clone().data.cpu()

        #     x, objective = layer.reverse_(x, objective)

        #     x_reverse = x.clone().data.cpu()

        #     if (x_reverse == x_original).all():

        #         print (str(layer)[:6], 'pass')

        #     else:
        #         print (str(layer)[:6], 'fail')


        fdsafas











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
        







#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################




































