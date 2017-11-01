
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



import sys
sys.path.insert(0, 'utils')
from utils import lognormal2 as lognormal
from utils import lognormal333






class standard(nn.Module):

    def __init__(self, model, hyper_config=''):
        super(standard, self).__init__()

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor

            # change it to  torch.FloatTensor.cuda(gpu_number)
            # this might not work
            # might be easier to change env variable
        else:
            self.dtype = torch.FloatTensor

        self.z_size = model.z_size
        self.x_size = model.x_size
        self.act_func = model.act_func





        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=2, padding=0, dilation=1, bias=True)

        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=2, padding=0, dilation=1, bias=True)

        # self.fc1 = nn.Linear(1960, 200)
        self.fc1 = nn.Linear(250, 200)

        self.fc2 = nn.Linear(200, self.z_size*2)




        # #Encoder
        # self.encoder_weights = []
        # for i in range(len(hyper_config['encoder_arch'])):
        #     self.encoder_weights.append(nn.Linear(hyper_config['encoder_arch'][i][0], hyper_config['encoder_arch'][i][1]))

        # count =1
        # for i in range(len(self.encoder_weights)):
        #     self.add_module(str(count), self.encoder_weights[i])
        #     count+=1


    def forward(self, k, x, logposterior):
        '''
        k: number of samples
        x: [B,X]
        logposterior(z) -> [P,B]
        '''

        self.B = x.size()[0]

        # #Encode
        # out = x
        # for i in range(len(self.encoder_weights)-1):
        #     out = self.act_func(self.encoder_weights[i](out))
        # out = self.encoder_weights[-1](out)
        # mean = out[:,:self.z_size]
        # logvar = out[:,self.z_size:]


        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # print (x)
        # x = x.view(-1, 1960)
        x = x.view(-1, 250)

        h1 = F.relu(self.fc1(x))
        h2 = self.fc2(h1)
        mean = h2[:,:self.z_size]
        logvar = h2[:,self.z_size:]



        #Sample
        eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz = lognormal(z, mean, logvar) #[P,B]

        return z, logqz










    # def encode(self, x):

    #     x = x.view(-1, 3, 32, 32)
    #     x = F.relu(self.conv1(x))

    #     x = x.view(-1, 1960)

    #     h1 = F.relu(self.fc1(x))
    #     h2 = self.fc2(h1)
    #     mean = h2[:,:self.z_size]
    #     logvar = h2[:,self.z_size:]
    #     return mean, logvar

