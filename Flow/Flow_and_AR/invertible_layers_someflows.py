


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

# from Prior_Att import Att_Prior

import torch.distributions as dist

import sys, os
sys.path.insert(0, os.path.abspath('./PixelCNN/'))

from utils import * 
from model import * 
from PIL import Image

# #https://forums.fast.ai/t/show-gpu-utilization-metrics-inside-training-loop-without-subprocess-call/26594
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
def get_gpu_mem(concat_string=''):
    res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(concat_string, f'mem: {100 * (res.used / res.total):.3f}%') # percentage 











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
        # return x, objective

    def reverse_(self, x, objective, args=None):

        assert len(x) == 1

        x = x[0]

        x = 1/ (torch.exp(-x) + 1)

        # SHOULD MODIFY OBJECTIVE
        jac = 1/x + (1/(1-x))
        objective += flatten_sum(torch.log(jac))

        return x, objective


















# ------------------------------------------------------------------------------
# Layers involving prior
# ------------------------------------------------------------------------------





class GaussPrior(Layer):
    def __init__(self, input_shape, args):
        super(GaussPrior, self).__init__()
        self.input_shape = input_shape

        dims = input_shape[1] * input_shape[2] * input_shape[3]
        # self.model = nn.Linear(1, dims*2)

        # self.means = torch.randn([input_shape[1] ,input_shape[2] ,input_shape[3]], requires_grad=True).cuda()
        # self.means = torch.randn([input_shape[1] ,input_shape[2] ,input_shape[3]], requires_grad=True).cuda()

        # self.means = nn.Parameter(torch.randn([1, input_shape[1] ,input_shape[2] ,input_shape[3]], device='cuda'))#.cuda())
        self.logsd = nn.Parameter(torch.randn([1, input_shape[1] ,input_shape[2] ,input_shape[3]], device='cuda'))#.cuda())
        # device=cuda is critical for the optimizer to get its gradients

        # self.register_parameter(self.means)
        # self.register_parameter(self.logsd)

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

    def get_mean_logsd(self, z):

        B = z.shape[0]
        mean_and_logsd = self.model(z)
        mean_and_logsd = mean_and_logsd.view(B, self.input_shape[1]*2, self.input_shape[2], self.input_shape[3])
        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
        mean = torch.tanh(mean / 100.) * 100.
        logsd = self.get_logsd(logsd)
        return mean, logsd


    def forward_(self, z, objective):

        z = z[0]

        B = z.shape[0]
        # tmp = torch.ones(B).cuda().view(B,1)
        # mean, logsd = self.get_mean_logsd(tmp)

        # mean = self.means# .unsqueeze(0) #.repeat(B,1,1,1)
        logsd = self.logsd#.unsqueeze(0) #.repeat(B,1,1,1)
        mean = torch.zeros_like(logsd).cuda()

        # print (mean.shape)
        # fsda


        LL = self.gauss_log_prob(z, mean=mean, logsd=logsd)
        LL = LL.view(B,-1).sum(-1)
        objective += LL
        return z, objective



    def reverse_(self, x, objective, args=None):
        bs, c, h, w = self.input_shape
        B = args.sample_size

        # tmp = torch.ones(B).cuda().view(B,1)
        # mean, logsd = self.get_mean_logsd(tmp)

        # mean = self.means.repeat(B,1,1,1) #.unsqueeze(0)
        logsd = self.logsd.repeat(B,1,1,1) #.unsqueeze(0)
        mean = torch.zeros_like(logsd).cuda()

        eps = torch.zeros_like(mean).normal_().cuda()
        samp = mean + torch.exp(logsd) * eps

        samp = [samp]
 
        return samp, objective
         








#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################



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




# Additive Coupling Layer
class AffineCoupling(Layer):
    def __init__(self, num_features, args, hidden_channels=128):
        super(AffineCoupling, self).__init__()


        # # assert num_features % 2 == 0
        # self.NN = NN(num_features // 2, channels_out=num_features, hidden_channels=hidden_channels, filter_size=3, args=args)
        # self.NN2 = NN(num_features // 2, channels_out=num_features, hidden_channels=hidden_channels, filter_size=3, args=args)


        self.NN = NN_FC(in_shape=args.shape_temp)
        self.NN2 = NN_FC(in_shape=args.shape_temp)


    def forward_(self, x_list, objective):

        x = x_list.pop(-1)

        # print (x.shape)
        z1, z2 = torch.chunk(x, 2, dim=1)

        shift1 = self.NN(z1[:,0,0])
        # h = self.NN(z1)
        # print (h.shape)

        # shift1, scale1 = torch.chunk(h, 2, dim=1)
        # scale1 = torch.tanh(scale1) /4. + 1.

        # shift = h[:, 0::2]
        # scale = h[:, 1::2]
        
        # z2 += shift1
        # z2 *= scale1

        z2 = z2 + shift1
        # z2 = z2 * scale1

        # objective += flatten_sum(torch.log(scale1))

        shift2 = self.NN2(z2[:,0,0])
        # h = self.NN2(z2)
        # shift = h[:, 0::2]
        # scale = h[:, 1::2]

        # shift2, scale2 = torch.chunk(h, 2, dim=1)
        # scale2 = torch.tanh(scale2) /4. + 1.

        # z1 += shift2
        # z1 *= scale2

        z1 = z1 + shift2
        # z1 = z1 * scale2

        # objective += flatten_sum(torch.log(scale2))

        x = torch.cat([z1, z2], dim=1)
        x_list.append(x)
        return x_list, objective


    def reverse_(self, x_list, objective, args=None):

        x = x_list.pop(-1)

        z1, z2 = torch.chunk(x, 2, dim=1)

        # h = self.NN(z1.contiguous())

        h = self.NN2(z2[:,0,0])
        # # h = self.NN2(z2)
        # shift = h[:, 0::2]
        # scale = h[:, 1::2]
        # scale = torch.tanh(scale) /4. + 1.
        # z1 /= scale
        z1 -= h #shift
        # objective -= flatten_sum(torch.log(scale))


        h = self.NN(z1[:,0,0])
        # # h = self.NN(z1)
        # shift = h[:, 0::2]
        # scale = h[:, 1::2]
        # scale = torch.tanh(scale) /4. + 1.
        # z2 /= scale
        z2 -= h #shift
        # objective -= flatten_sum(torch.log(scale))



        x = torch.cat([z1, z2], dim=1)
        x_list.append(x)

        return x_list, objective











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
            # print(str(layer))
            x, objective = layer.forward_(x, objective)
            # get_gpu_mem()
        return x, objective

    def reverse_(self, x, objective, args=None):
        # count=0
        for layer in reversed(self.layers): 
            
            
            x, objective = layer.reverse_(x, objective, args=args)
            

        return x, objective












# # trying new setup
# class RevNetStep(LayerList):
#     def __init__(self, num_channels, args):
#         super(RevNetStep, self).__init__()
#         self.args = args
#         layers = []

#         if args.norm == 'actnorm': 
#             layers += [ActNorm_Layer(num_channels, args=args)]
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
#             layers += [AdditiveCoupling(num_channels, hidden_channels=args.hidden_channels)]
#         elif args.coupling == 'affine':
#             layers += [AffineCoupling(num_channels, hidden_channels=args.hidden_channels, args=args)]
#         elif args.coupling == 'spline':
#             layers += [SplineCoupling(num_channels, hidden_channels=args.hidden_channels)]
#         elif args.coupling == 'linear_spline':
            


            
#             layers += [AdditiveCoupling(num_channels, hidden_channels=args.hidden_channels)]
#             layers += [LinearSplineCoupling(num_channels, hidden_channels=args.hidden_channels // 4 )]
            
            
#         else: 
#             raise ValueError

#         self.layers = nn.ModuleList(layers)













# Full model - trying different architecutres
class Glow_(LayerList, nn.Module):
    def __init__(self, input_shape, args):
        super(Glow_, self).__init__()
        layers = []
        output_shapes = []
        B, C, H, W = input_shape




        layers += [InverseSigmoid()]
        output_shapes += [('InvSigmoid', C, H, W)]

        layers += [Squeeze(input_shape)]
        C, H, W = C * 4, H // 2, W // 2
        output_shapes += [('Squeeze', C, H, W)]

        args.shape_temp = [C,H,W]
        layers += [AffineCoupling(C, hidden_channels=args.hidden_channels, args=args)]
        output_shapes += [('AffineCoupling', C, H, W)]

        layers += [GaussPrior((B, C, H, W), args)]
        output_shapes += [('Prior', C, H, W)]

        # print (layers[-1])
        # fsd


        
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



    def forward(self, *inputs):
        return self.forward_(*inputs)

    def sample(self, args=None):
        with torch.no_grad():
            samples = self.reverse_(None, 0., args=args)[0]
            return samples


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
        







#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################




































