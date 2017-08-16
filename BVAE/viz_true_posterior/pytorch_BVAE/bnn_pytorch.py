


import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

# import matplotlib.pyplot as plt


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import lognormal3 as lognormal
from utils import log_bernoulli

from torch.nn.parameter import Parameter



# class just_var(nn.Module):

#     def __init__(self, in_size, out_size):
#         super(just_var, self).__init__()


#         self.param = Parameter(torch.Tensor(in_size, out_size))








# class BNN2(nn.Module):
#     r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         bias: If set to False, the layer will not learn an additive bias.
#             Default: True

#     Shape:
#         - Input: :math:`(N, *, in\_features)` where `*` means any number of
#           additional dimensions
#         - Output: :math:`(N, *, out\_features)` where all but the last dimension
#           are the same shape as the input.

#     Attributes:
#         weight: the learnable weights of the module of shape
#             (out_features x in_features)
#         bias:   the learnable bias of the module of shape (out_features)

#     Examples::

#         >>> m = nn.Linear(20, 30)
#         >>> input = autograd.Variable(torch.randn(128, 20))
#         >>> output = m(input)
#         >>> print(output.size())
#     """

#     def __init__(self, in_features, out_features):
#         super(BNN, self).__init__()



#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(4, 5))
#         # if bias:
#         #     self.bias = Parameter(torch.Tensor(out_features))
#         # else:
#         #     self.register_parameter('bias', None)
#         # self.reset_parameters()
#         self.params = [self.weight]

#     # def reset_parameters(self):
#     #     stdv = 1. / math.sqrt(self.weight.size(1))
#     #     self.weight.data.uniform_(-stdv, stdv)
#     #     if self.bias is not None:
#     #         self.bias.data.uniform_(-stdv, stdv)

#     # def forward(self, input):
#     #     return F.linear(input, self.weight, self.bias)

#     # def __repr__(self):
#     #     return self.__class__.__name__ + ' (' \
#     #         + str(self.in_features) + ' -> ' \
#     #         + str(self.out_features) + ')'



class BNN(nn.Module):
# class BNN2():

    def __init__(self, network_architecture, act_functions):
        super(BNN, self).__init__()
        


        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor




        self.net = network_architecture
        self.act_functions = act_functions

        # self.aaa = Parameter(torch.Tensor(4, 5))

        self.W_means = []
        self.W_logvars = []

        # for layer_i in range(len(self.net)-1):
        #     input_size_i = self.net[layer_i]+1 #plus 1 for bias
        #     output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1
        #     self.a = Parameter(torch.Tensor(input_size_i, output_size_i))
        #     self.W_means.append(self.a)
        #     self.b = Parameter(torch.Tensor(input_size_i, output_size_i))-5.
        #     self.W_logvars.append(self.b)

        for layer_i in range(len(self.net)-1):
            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1
            # self.a = just_var
            # self.W_means.append(Parameter(torch.Tensor(input_size_i, output_size_i)))
            # self.W_logvars.append(Parameter(torch.Tensor(input_size_i, output_size_i)-5.))

            # if torch.cuda.is_available():
            #     self.W_means.append(Parameter(torch.randn(input_size_i, output_size_i)/100.).cuda())
            #     self.W_logvars.append(Parameter(torch.randn(input_size_i, output_size_i)-8.).cuda())
            # else:
            # aaa = Parameter(torch.randn(input_size_i, output_size_i)/100.).type(self.dtype)
            # # aaa.is_leaf = True
            # bbb = Parameter(torch.randn(input_size_i, output_size_i)-8.).type(self.dtype)
            # # bbb.is_leaf = True

            aaa = Parameter(torch.randn(input_size_i, output_size_i).type(self.dtype)/100.)
            # aaa.is_leaf = True
            bbb = Parameter(torch.randn(input_size_i, output_size_i).type(self.dtype)-8.)
            # bbb.is_leaf = True

            self.W_means.append(aaa)
            self.W_logvars.append(bbb)

            # self.b = Parameter(torch.Tensor(input_size_i, output_size_i))-5.
            # self.W_logvars.append(just_var(input_size_i, output_size_i))


        # self.W_means, self.W_logvars = self.init_weights()
        self.params = self.W_means + self.W_logvars

        self.param_list = nn.ParameterList(self.params)




    # def init_weights(self):

    #     # def xavier_init(fan_in, fan_out, constant=1): 
    #     #     """ Xavier initialization of network weights"""
    #     #     # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    #     #     low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    #     #     high = constant*np.sqrt(6.0/(fan_in + fan_out))
    #     #     return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    #     W_means = []
    #     W_logvars = []

    #     for layer_i in range(len(self.net)-1):

    #         input_size_i = self.net[layer_i]+1 #plus 1 for bias
    #         output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

    #         #Define variables [IS,OS]
    #         # W_means.append(Variable(xavier_init(input_size_i, output_size_i)))
    #         # W_means.append(tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1)))
    #         # W_means.append(Parameter(torch.randn(input_size_i, output_size_i)))
    #         # W_means.append(Variable(torch.randn(input_size_i, output_size_i).type(torch.FloatTensor), requires_grad=True))
    #         W_means.append(Parameter(torch.Tensor(input_size_i, output_size_i)))


    #         # W_logvars.append(tf.Variable(xavier_init(input_size_i, output_size_i) - 10.))
    #         # W_logvars.append(tf.Variable(tf.random_normal([input_size_i, output_size_i], stddev=0.1))-5.)
    #         # W_logvars.append(Variable(torch.randn(input_size_i, output_size_i).type(torch.FloatTensor), requires_grad=True)-5.)
    #         # W_logvars.append(Parameter(torch.randn(input_size_i, output_size_i))-5.)
    #         W_logvars.append(Parameter(torch.Tensor(input_size_i, output_size_i))-5.)




    #     return W_means, W_logvars






    def sample_weights(self):

        Ws = []

        log_p_W_sum = 0
        log_q_W_sum = 0

        for layer_i in range(len(self.net)-1):

            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Get vars [I,O]
            W_means = self.W_means[layer_i]
            W_logvars = self.W_logvars[layer_i]

            #Sample weights [IS,OS]*[IS,OS]=[IS,OS]
            eps = Variable(torch.randn(input_size_i, output_size_i).type(self.dtype))
            # print eps
            # print torch.sqrt(torch.exp(W_logvars))
            # W = torch.add(W_means, torch.sqrt(torch.exp(W_logvars)) * eps)
            W =  (torch.sqrt(torch.exp(W_logvars)) * eps) + W_means 


            # W = W_means

            #Compute probs of samples  [1]
            flat_w = W.view(input_size_i*output_size_i) #[IS*OS]
            flat_W_means = W_means.view(input_size_i*output_size_i) #[IS*OS]
            flat_W_logvars = W_logvars.view(input_size_i*output_size_i) #[IS*OS]
            log_p_W_sum += lognormal(flat_w, Variable(torch.zeros([input_size_i*output_size_i]).type(self.dtype)), Variable(torch.log(torch.ones([input_size_i*output_size_i]).type(self.dtype))))
            # log_p_W_sum += log_normal3(flat_w, tf.zeros([input_size_i*output_size_i]), tf.log(tf.ones([input_size_i*output_size_i])*100.))

            log_q_W_sum += lognormal(flat_w, flat_W_means, flat_W_logvars)

            Ws.append(W)

        return Ws, log_p_W_sum, log_q_W_sum



    def forward(self, W_list, x):
        '''
        W: list of layers weights
        x: [B,X]
        y: [B,Y]
        '''
        B = x.size()[0]

        #[B,X]
        cur_val = x
        # #[B,X]->[B,1,X]
        # cur_val = tf.reshape(cur_val, [self.batch_size, 1, self.input_size])

        for layer_i in range(len(self.net)-1):

            #[X,X']
            W = W_list[layer_i]
            input_size_i = self.net[layer_i]+1 #plus 1 for bias
            output_size_i = self.net[layer_i+1] #plus 1 because we want layer i+1

            #Concat 1 to input for biases  [B,X]->[B,X+1]
            # print cur_val
            # print torch.ones([B, 1])
            # print cur_val


            # if torch.cuda.is_available():
            #     cur_val = torch.cat((cur_val,Variable(torch.ones([B, 1])).cuda()), 1)
            # else:
            cur_val = torch.cat((cur_val,Variable(torch.ones([B, 1]).type(self.dtype))), 1)


            # #[X,X']->[B,X,X']
            # W = tf.reshape(W, [1, input_size_i, output_size_i])
            # W = tf.tile(W, [self.batch_size, 1,1])

            cur_val = torch.mm(cur_val, W)

            #Forward Propagate  [B,X]*[X,X']->[B,X']
            if layer_i != len(self.net)-2:
                cur_val = self.act_functions[layer_i](cur_val)
            # else:




        # #[B,P,1,X']->[B,P,X']
        # cur_val = tf.reshape(cur_val, [self.batch_size,P,output_size_i])
        #[B,Y]
        # y = cur_val

        return cur_val

















