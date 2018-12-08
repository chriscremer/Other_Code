

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler



# from encoder_decoder import Encoder, Decoder
from enc_dec_32_full import Encoder, Decoder
from distributions import lognormal, Flow1 #Normal,  #, IAF_flow
from torch.distributions import Beta




class Inference_Net(nn.Module):
    def __init__(self, kwargs):
        super(Inference_Net, self).__init__()


        kwargs['act_func'] = F.leaky_relu

        self.__dict__.update(kwargs)

        self.beta_scale = 100.

        # self.linear_hidden_size = 500 
        # self.linear_hidden_size2 = 200

        lr = .0004
        

        # q(z|x)
        self.image_encoder2 = Encoder(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=3, input_size=self.input_size)
        # self.qzx_fc1 = nn.Linear(self.x_enc_size, self.x_enc_size)
        # self.qzx_bn1 = nn.BatchNorm1d(self.x_enc_size)
        # self.qzx_fc2 = nn.Linear(self.x_enc_size, self.x_enc_size)
        # self.qzx_fc3 = nn.Linear(self.x_enc_size,self.z_size*2)

        # x_inf_params = [list(self.image_encoder2.parameters()) + list(self.qzx_fc1.parameters()) 
        #                 + list(self.qzx_bn1.parameters()) + list(self.qzx_fc2.parameters())
        #                 + list(self.qzx_fc3.parameters()) ]

        self.optimizer_x = optim.Adam(self.image_encoder2.parameters(), lr=lr, weight_decay=.0000001)
        self.scheduler_x = lr_scheduler.StepLR(self.optimizer_x, step_size=1, gamma=0.999995)



    # def inference_net(self, x):
    #     # b = x_enc.shape[0]

    #     mean = self.image_encoder2(x)
    #     logvar = None
    #     # input_ = x_enc
    #     # # padded_input = torch.cat([x_enc, torch.zeros(b, self.linear_hidden_size-self.x_enc_size).cuda()], 1)

    #     # out = self.act_func(self.qzx_bn1(self.qzx_fc1(input_)))
    #     # out = self.qzx_fc2(out)
    #     # out = out + input_ 
    #     # out = self.qzx_fc3(out)
    #     # mean = out[:,:self.z_size]
    #     # logvar = out[:,self.z_size:]
    #     # logvar = torch.clamp(logvar, min=-15., max=10.)
    #     return mean, logvar



    def inference_net(self, x):
        # b = x_enc.shape[0]

        mean_logvar = self.image_encoder2(x)

        mean = mean_logvar[:,:6]
        logvar = mean_logvar[:,6:]
        # input_ = x_enc
        # # padded_input = torch.cat([x_enc, torch.zeros(b, self.linear_hidden_size-self.x_enc_size).cuda()], 1)

        # out = self.act_func(self.qzx_bn1(self.qzx_fc1(input_)))
        # out = self.qzx_fc2(out)
        # out = out + input_ 
        # out = self.qzx_fc3(out)
        # mean = out[:,:self.z_size]
        # logvar = out[:,self.z_size:]
        # logvar = torch.clamp(logvar, min=-15., max=10.)
        return mean, logvar



    def load_params_v3(self, save_dir, step, name=''):
        save_to=os.path.join(save_dir, "infnet_params_" +name + str(step)+".pt")
        state_dict = torch.load(save_to)
        # # # print (state_dict)
        # for key, val in state_dict.items():
        #     print (key)
        # fddsf
        self.load_state_dict(state_dict)
        print ('loaded params', save_to)


    def save_params_v3(self, save_dir, step, name=''):
        save_to=os.path.join(save_dir, "infnet_params_"+name + str(step)+".pt")
        torch.save(self.state_dict(), save_to)
        print ('saved params', save_to)
        














