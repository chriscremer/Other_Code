


import torch
import torch.nn as nn

from distributions import log_bernoulli

import sys, os

class Generator(nn.Module):

    def __init__(self, hyper_config):
        super(Generator, self).__init__()

        # if hyper_config['cuda']:
        #     self.dtype = torch.cuda.FloatTensor
        # else:
        #     self.dtype = torch.FloatTensor

        # self.z_size = hyper_config['z_size']
        # self.x_size = hyper_config['x_size']
        # self.act_func = hyper_config['act_func']

        self.__dict__.update(hyper_config)

        #Decoder
        self.decoder_weights = []
        self.layer_norms = []
        for i in range(len(hyper_config['decoder_arch'])):
            self.decoder_weights.append(nn.Linear(hyper_config['decoder_arch'][i][0], hyper_config['decoder_arch'][i][1]))

        count =1
        for i in range(len(self.decoder_weights)):
            self.add_module(str(count), self.decoder_weights[i])
            count+=1
   

    def decode(self, x, z):
        # k = z.size()[0]
        # B = z.size()[1]
        # z = z.view(-1, self.z_size)

        out = z
        for i in range(len(self.decoder_weights)-1):
            out = self.act_func(self.decoder_weights[i](out))
            # out = self.act_func(self.layer_norms[i].forward(self.decoder_weights[i](out)))
        out = self.decoder_weights[-1](out)

        logpx = log_bernoulli(pred_no_sig=out, target=x)

        # x = out.view(k, B, self.x_size)
        return torch.sigmoid(out), logpx








    def load_params_v3(self, save_dir, step, name):
        save_to=os.path.join(save_dir, name + str(step)+".pt")
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
        















