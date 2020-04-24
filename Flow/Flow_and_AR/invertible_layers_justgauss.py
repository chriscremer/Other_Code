

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

# #https://forums.fast.ai/t/show-gpu-utilization-metrics-inside-training-loop-without-subprocess-call/26594
# import nvidia_smi

# nvidia_smi.nvmlInit()
# handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# def get_gpu_mem(concat_string=''):
#     res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#     print(concat_string, f'mem: {100 * (res.used / res.total):.3f}%') # percentage 











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

        # return [x], objective
        return x, objective

    def reverse_(self, x, objective, args=None):

        # assert len(x) == 1

        # x = x[0]

        x = 1/ (torch.exp(-x) + 1)

        # SHOULD MODIFY OBJECTIVE
        jac = 1/x + (1/(1-x))
        objective += flatten_sum(torch.log(jac))

        return x, objective


















# ------------------------------------------------------------------------------
# Layers involving prior
# ------------------------------------------------------------------------------





class Prior(Layer):
    def __init__(self, input_shape, args):
        super(Prior, self).__init__()
        self.input_shape = input_shape

        # print ('p(z)', self.input_shape)
        # fdsaf


        # self.model = PixelCNN(nr_resnet=args.AR_resnets, nr_filters=args.AR_channels, 
        #             input_channels=input_shape[1], nr_logistic_mix=2)

        # print (input_shape)
        # fasdfas

        #
        dims = input_shape[1] * input_shape[2] * input_shape[3]

        # self.mean_and_logsd = torch.randn()
        self.model = nn.Linear(1, dims*2)


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
        # mean_and_logsd = mean_and_logsd.view(B, 6, 32, 32)
        mean_and_logsd = mean_and_logsd.view(B, self.input_shape[1]*2, self.input_shape[2], self.input_shape[3])
        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
        mean = torch.tanh(mean / 100.) * 100.
        logsd = self.get_logsd(logsd)
        return mean, logsd





    def forward_(self, z, objective):

        # z = z[0]

        # print ()
        # print (torch.min(z))
        # print (torch.mean(z))
        # print (torch.median(z))
        # print (torch.max(z))
        # fsdaf



        B = z.shape[0]

        # print (z.shape)
        # mean_and_logsd = self.model(z)


        # mean_and_logsd = self.model(torch.rand(B).cuda().view(B,1))
        # mean_and_logsd = mean_and_logsd.view(B, 6, 32, 32)
        # mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
        # mean = torch.tanh(mean / 100.) * 100.
        # logsd = self.get_logsd(logsd)
        tmp = torch.ones(B).cuda().view(B,1)
        mean, logsd = self.get_mean_logsd(tmp)

        LL = self.gauss_log_prob(z, mean=mean, logsd=logsd)
        LL = LL.view(B,-1).sum(-1)

        # print (objective.shape)
        # print (LL.shape)
        objective += LL

        return z, objective







    def reverse_(self, x, objective, args=None):
        bs, c, h, w = self.input_shape
        B = args.sample_size

        tmp = torch.ones(B).cuda().view(B,1)
        mean, logsd = self.get_mean_logsd(tmp)

        eps = torch.zeros_like(mean).normal_().cuda()
        samp = mean + torch.exp(logsd) * eps
 
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
            x, objective = layer.forward_(x, objective)
        return x, objective

    def reverse_(self, x, objective, args=None):
        # count=0
        for layer in reversed(self.layers): 
            x, objective = layer.reverse_(x, objective, args=args)
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

        self.layers = nn.ModuleList(layers)













# Full model - trying different architecutres
class Glow_(LayerList, nn.Module):
    def __init__(self, input_shape, args):
        super(Glow_, self).__init__()
        layers = []
        output_shapes = []
        B, C, H, W = input_shape
        # total_dims = C*H*W
        # prior_max_width = 17 #34 #17
        # args.prior_max_width = prior_max_width

        # min_channels = 13

        # final_width = W
        # while final_width > prior_max_width:
        #     final_width = final_width // 2

        # # print (final_width)

        # if int(H / 2**args.n_levels) > min_channels:
        #     final_width = np.minimum(final_width, int(H / 2**args.n_levels))


        # # print (final_width)
        # args.final_width = final_width
        # # fsdaf

        # final_channels = total_dims // final_width // final_width

        # # depths = [4,4,4,4,4,30]
        # # split_count = 0
        # split_spaces = '' 
        # split_sizes = []



        layers += [InverseSigmoid()]
        output_shapes += [('InvSigmoid', C, H, W)]

        layers += [Prior((B, C, H, W), args)]
        output_shapes += [('Prior', C, H, W)]



        # for i in range(5):
        #     # Squeeze Layer 
        #     if W > prior_max_width:
        #         layers += [Squeeze(input_shape)]
        #         C, H, W = C * 4, H // 2, W // 2
        #         output_shapes += [(split_spaces + 'Squeeze', C, H, W)]
        #         # print (output_shapes)
        #         # fads      
        
        # for i in range(args.n_levels):

        #     # if W > 20:
        #     #     # Squeeze Layer 
        #     #     layers += [Squeeze(input_shape)]
        #     #     C, H, W = C * 4, H // 2, W // 2
        #     #     output_shapes += [(split_spaces + 'Squeeze', C, H, W)]
        #     #     # print (output_shapes)
        #     #     # fads
                
        #     # # RevNet Block
        #     layers += [RevNetStep(C, args) for _ in range(args.depth)]
        #     output_shapes += [(split_spaces + 'RevNet', C, H, W) for _ in range(args.depth)]
        #     # layers += [RevNetStep(C, args) for _ in range(depths[i])]
        #     # output_shapes += [(-1, C, H, W) for _ in range(depths[i])]

        #     if i < args.n_levels - 1 : #and args.base_dist in ['Mog', 'Gauss']: 

        #         if C > min_channels:
        #             # Split Layer
        #             layers += [Split(output_shapes[-1])] 
        #             # split_count +=1
        #             split_spaces += '  '


        #             # if args.base_dist == 'MoG':
        #             #     layers += [Split_MoG(output_shapes[-1])]
        #             # elif args.base_dist == 'Gauss':
        #             #     layers += [Split(output_shapes[-1])]                    

        #             C = C // 2
        #             output_shapes += [(split_spaces + 'Split', C, H, W)]

        #             split_sizes.append([C,H,W])

        # split_sizes.append([C,H,W])

        # # if args.base_dist in ['AR', 'AttPrior']:
        # #     # while W > 30:
        # #     while W > 17:
        # #     # while W > 10:
        # #         # Squeeze Layer 
        # #         layers += [Squeeze(input_shape)]
        # #         C, H, W = C * 4, H // 2, W // 2
        # #         output_shapes += [('Squeeze', C, H, W)]
        # #         # print (output_shapes)
        # #         # fads

        # # while W > prior_max_width:
        # #     # Squeeze Layer 
        # #     layers += [Squeeze(input_shape)]
        # #     C, H, W = C * 4, H // 2, W // 2
        # #     output_shapes += [('Squeeze', C, H, W)]
        # #     # print (output_shapes)
        # #     # fads

        # # print (len(layers))


        # # if args.base_dist == 'MoG':
        # #     layers += [MoGPrior((B, C, H, W), args)]
        # # elif args.base_dist == 'AR':
        # #     layers += [AR_Prior((B, C, H, W), args)]
        # # elif args.base_dist == 'Gauss':
        # #     layers += [GaussianPrior((B, C, H, W), args)]
        # # elif args.base_dist == 'AttPrior':
        # #     layers += [Att_Prior((B, C, H, W), args)]

        # # # print (len(layers))


        # # TODO#, the shape of C should be larger 

        
        # layers += [Combine(split_sizes, args)]
        # output_shapes += [('Combine', final_channels, final_width, final_width)]

        # layers += [AR_Prior((B, final_channels, final_width, final_width), args)]
        # output_shapes += [('Prior', final_channels, final_width, final_width)]


        # # output_shapes += [output_shapes[-1]]

        # # # print (output_shapes)
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




































