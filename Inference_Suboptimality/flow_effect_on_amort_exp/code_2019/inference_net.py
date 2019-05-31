







import torch
import torch.nn as nn

import sys, os

# import sys
# sys.path.insert(0, 'utils')
# from utils import lognormal2 as lognormal
# from utils import lognormal333



# from distributions import Gaussian
# from distributions import Flow



from distributions import Gauss
from distributions import Flow1
from distributions import Flow_Cond



class Inf_Net(nn.Module):

    def __init__(self, kwargs):
        super(Inf_Net, self).__init__()

        self.__dict__.update(kwargs)

        #Encoder
        self.encoder_weights = []
        for i in range(len(self.encoder_arch)):
            self.encoder_weights.append(nn.Linear(self.encoder_arch[i][0], self.encoder_arch[i][1]))

        count =1
        for i in range(len(self.encoder_weights)):
            self.add_module(str(count), self.encoder_weights[i])
            count+=1

        if self.q_dist == 'Gauss':
            self.q = Gauss(self.z_size) 

        elif self.q_dist == 'Flow':
            self.q = Flow1(kwargs) 

        elif self.q_dist == 'Flow_Cond':
            self.q = Flow_Cond(kwargs) 

    def encode(self, x):
        out = x
        for i in range(len(self.encoder_weights)-1):
            out = self.act_func(self.encoder_weights[i](out))
            # out = self.act_func(self.layer_norms[i].forward(self.encoder_weights[i](out)))
        out = self.encoder_weights[-1](out)
        mean = out[:,:self.z_size]  #[B,Z]
        logvar = out[:,self.z_size:]

        return mean, logvar


    def sample(self, x):

        mean, logvar = self.encode(x)

        z, logqz = self.q.sample(mean, logvar)

        return z, logqz






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
        












































