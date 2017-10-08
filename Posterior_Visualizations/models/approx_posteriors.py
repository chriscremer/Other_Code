


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F






class flow1(nn.Module):

    def __init__(self, model, hyper_config):
        super(flow1, self).__init__()

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.n_flows = hyper_config['n_flows']

        self.z_size = model.z_size

        self.params = []
        for i in range(self.n_flows):

            # self.params.append([nn.Linear(int(self.z_size/2), 100), nn.Linear(100, int(self.z_size/2)), nn.Linear(100, int(self.z_size/2))])
            self.params.append([nn.Linear(self.z_size, 100), nn.Linear(100, self.z_size), nn.Linear(100, self.z_size)])

        # self.param_list = nn.ParameterList(self.params)

        count =1
        for i in range(self.n_flows):

            self.add_module(str(count), self.params[i][0])
            count+=1
            self.add_module(str(count), self.params[i][1])
            count+=1
            self.add_module(str(count), self.params[i][2])
            count+=1

    

    def forward(self, z):
        '''
        z: [P,B,Z]
        '''
        self.P = z.size()[0]
        self.B = z.size()[1]


        logdetsum = 0.
        for i in range(self.n_flows):

            z, logdet = self.norm_flow(self.params[i],z)
            logdetsum += logdet

        return z, logdetsum


    def norm_flow(self, params, z):

        # [Z]
        mask = Variable(torch.zeros(self.z_size)).type(self.dtype)
        mask[:int(self.z_size/2)] = 1.
        mask = mask.view(1,1,-1)

        # [P,B,Z]
        z1 = z*mask
        # [PB,Z]
        z1 = z1.view(-1, self.z_size)

        h = F.tanh(params[0](z1))
        mew_ = params[1](h)
        sig_ = F.sigmoid(params[2](h)+5.) #[PB,Z]

        z = z.view(-1, self.z_size)
        mask = mask.view(1, -1)

        z2 = (z*sig_ +mew_)*(1-mask)
        z = z1 + z2
        # [PB]
        logdet = torch.sum((1-mask)*torch.log(sig_), 1)
        # [P,B]
        logdet = logdet.view(self.P,self.B)
        #[P,B,Z]
        z = z.view(self.P,self.B,self.z_size)


        #Other half

        # [Z]
        mask2 = Variable(torch.zeros(self.z_size)).type(self.dtype)
        mask2[int(self.z_size/2):] = 1.
        mask = mask2.view(1,1,-1)

        # [P,B,Z]
        z1 = z*mask
        # [PB,Z]
        z1 = z1.view(-1, self.z_size)

        h = F.tanh(params[0](z1))
        mew_ = params[1](h)
        sig_ = F.sigmoid(params[2](h)+5.) #[PB,Z]

        z = z.view(-1, self.z_size)
        mask = mask.view(1, -1)

        z2 = (z*sig_ +mew_)*(1-mask)
        z = z1 + z2
        # [PB]
        logdet2 = torch.sum((1-mask)*torch.log(sig_), 1)
        # [P,B]
        logdet2 = logdet2.view(self.P,self.B)
        #[P,B,Z]
        z = z.view(self.P,self.B,self.z_size)

        logdet = logdet + logdet2
        

        return z, logdet

























# class hnf1(nn.Module):

#     def __init__(self, model, hyper_config):
#         super(hnf1, self).__init__()

#         if torch.cuda.is_available():
#             self.dtype = torch.cuda.FloatTensor
#         else:
#             self.dtype = torch.FloatTensor

#         self.n_flows = hyper_config['n_flows']

#         self.z_size = model.z_size

#         self.params = []
#         for i in range(self.n_flows):

#             # self.params.append([nn.Linear(int(self.z_size/2), 100), nn.Linear(100, int(self.z_size/2)), nn.Linear(100, int(self.z_size/2))])
#             self.params.append([nn.Linear(self.z_size, 100), nn.Linear(100, self.z_size), nn.Linear(100, self.z_size)])

#         # self.param_list = nn.ParameterList(self.params)

#         count =1
#         for i in range(self.n_flows):

#             self.add_module(str(count), self.params[i][0])
#             count+=1
#             self.add_module(str(count), self.params[i][1])
#             count+=1
#             self.add_module(str(count), self.params[i][2])
#             count+=1

    

#     def forward(self, z):
#         '''
#         z: [P,B,Z]
#         '''
#         self.P = z.size()[0]
#         self.B = z.size()[1]


#         #Sample aux var 
#         eps = Variable(torch.FloatTensor(k, B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]


#         logdetsum = 0.
#         for i in range(self.n_flows):

#             z, logdet = self.norm_flow(self.params[i],z)
#             logdetsum += logdet

#         return z, logdetsum


#     def norm_flow(self, params, z):

#         # [Z]
#         mask = Variable(torch.zeros(self.z_size)).type(self.dtype)
#         mask[:int(self.z_size/2)] = 1.
#         mask = mask.view(1,1,-1)

#         # [P,B,Z]
#         z1 = z*mask
#         # [PB,Z]
#         z1 = z1.view(-1, self.z_size)

#         h = F.tanh(params[0](z1))
#         mew_ = params[1](h)
#         sig_ = F.sigmoid(params[2](h)+5.) #[PB,Z]

#         z = z.view(-1, self.z_size)
#         mask = mask.view(1, -1)

#         z2 = (z*sig_ +mew_)*(1-mask)
#         z = z1 + z2
#         # [PB]
#         logdet = torch.sum((1-mask)*torch.log(sig_), 1)
#         # [P,B]
#         logdet = logdet.view(self.P,self.B)
#         #[P,B,Z]
#         z = z.view(self.P,self.B,self.z_size)


#         #Other half

#         # [Z]
#         mask2 = Variable(torch.zeros(self.z_size)).type(self.dtype)
#         mask2[int(self.z_size/2):] = 1.
#         mask = mask2.view(1,1,-1)

#         # [P,B,Z]
#         z1 = z*mask
#         # [PB,Z]
#         z1 = z1.view(-1, self.z_size)

#         h = F.tanh(params[0](z1))
#         mew_ = params[1](h)
#         sig_ = F.sigmoid(params[2](h)+5.) #[PB,Z]

#         z = z.view(-1, self.z_size)
#         mask = mask.view(1, -1)

#         z2 = (z*sig_ +mew_)*(1-mask)
#         z = z1 + z2
#         # [PB]
#         logdet2 = torch.sum((1-mask)*torch.log(sig_), 1)
#         # [P,B]
#         logdet2 = logdet2.view(self.P,self.B)
#         #[P,B,Z]
#         z = z.view(self.P,self.B,self.z_size)

#         logdet = logdet + logdet2
        

#         return z, logdet















