

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

        lr = .0004
        

        # q(z|x)
        self.image_encoder2 = Encoder(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=self.enc_res_blocks, input_size=self.input_size)

        self.optimizer_x = optim.Adam(self.image_encoder2.parameters(), lr=lr, weight_decay=.0000001)
        self.scheduler_x = lr_scheduler.StepLR(self.optimizer_x, step_size=1, gamma=0.999995)




    def inference_net(self, x):
        mean_logvar = self.image_encoder2(x)
        mean = mean_logvar[:,:6]
        logvar = mean_logvar[:,6:]
        logvar = torch.clamp(logvar, min=-15., max=10.)
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
        













class Inference_Q(nn.Module):
    def __init__(self, kwargs):
        super(Inference_Q, self).__init__()


        kwargs['act_func'] = F.leaky_relu

        self.__dict__.update(kwargs)

        self.beta_scale = 100.

        lr = .0004
        

        # q(z|x)
        self.image_encoder2 = Encoder(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=self.enc_res_blocks, input_size=self.input_size)

        self.optimizer_x = optim.Adam(self.image_encoder2.parameters(), lr=lr, weight_decay=.0000001)
        self.scheduler_x = lr_scheduler.StepLR(self.optimizer_x, step_size=1, gamma=0.999995)




    def inference_net(self, x):
        mean_logvar = self.image_encoder2(x)
        mean = mean_logvar[:,:6]
        logvar = mean_logvar[:,6:]
        logvar = torch.clamp(logvar, min=-15., max=10.)
        return mean, logvar


    def sample(self, x):

        mu, logvar = self.inference_net(x)

        B = mu.shape[0]
        eps = torch.FloatTensor(B,6,8,8).normal_().cuda() #[B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mu  #[B,Z]
        flat_z = z.view(B, -1)
        logqz = lognormal(flat_z, mu.view(B, -1).detach(), logvar.view(B, -1).detach())

        return z, logqz




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
        

















