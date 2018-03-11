







import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# from distributions import Categorical, DiagGaussian

from distributions import Categorical2 as Categorical

import numpy as np




# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1 or classname.find('Linear') != -1:
#         nn.init.orthogonal(m.weight.data)
#         if m.bias is not None:
#             m.bias.data.fill_(0)





class CNNPolicy(nn.Module):
    def __init__(self, num_inputs, action_space, n_contexts):
        super(CNNPolicy, self).__init__()

        # if action_space.__class__.__name__ == "Discrete":
        #     num_outputs = action_space.n
        #     self.dist = Categorical(512, num_outputs)
        # elif action_space.__class__.__name__ == "Box":
        #     num_outputs = action_space.shape[0]
        #     self.dist = DiagGaussian(512, num_outputs)
        # else:
        #     raise NotImplementedError



        num_outputs = action_space.n
        # print (num_outputs)
        # fda
        # self.dist = Categorical(num_outputs)
        self.dist = Categorical()


        self.num_inputs = num_inputs  #num of stacked frames
        self.num_outputs = num_outputs #action size

            
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        l_size = 10 #512
        self.linear1 = nn.Linear(32 * 7 * 7, l_size)

        n_contexts = 2

        self.action_linear = nn.Linear(l_size+n_contexts, 4)
        self.action_linear2 = nn.Linear(4, num_outputs)


        self.critic_linear = nn.Linear(l_size+n_contexts, 1)



    def encode_frames(self, inputs):

        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)

        x = F.elu(self.linear1(x))

        # self.z= x

        return x


    def predict_for_action(self, inputs):
        # print (inputs)
        inputs = F.elu(self.action_linear(inputs))
        for_action = self.action_linear2(inputs)
        return for_action

    def predict_for_value(self, inputs):
        for_value= self.critic_linear(inputs)
        return for_value

    def forward(self, x):
        # x = self.encode(inputs)
        for_action = self.predict_for_action(x)
        for_value = self.predict_for_value(x)
        return for_value, for_action


    # def action_dist(self, inputs):
    #     x = self.encode(inputs)
    #     for_action = self.predict_for_action(x)
    #     return self.dist.action_probs(for_action)



    def act(self, frames, context, deterministic=False):

        frames = self.encode_frames(frames) #[B,F]
        # print (frames, context)
        context = context.float()
        context = Variable(context).cuda()
        inputs = torch.cat((frames, context), dim=1)  #[B,F+Z]
        # inputs = torch.cat((frames, context), dim=1)  #[B,F+Z]


        value, x_action = self.forward(inputs)
        action, action_log_probs, dist_entropy = self.dist.sample(x_action, deterministic=deterministic)
        return value, action, action_log_probs, dist_entropy


    def graddd(self, frames, context):

        frames = self.encode_frames(frames) #[B,F]
        context = context.float()
        context = Variable(context, requires_grad=True).cuda()
        inputs = torch.cat((frames, context), dim=1)  #[B,F+Z]
        value, x_action = self.forward(inputs)


        for i in range(self.num_outputs):
        # for i in range(8):
            gradients = torch.autograd.grad(outputs=torch.mean(x_action[:,i]), inputs=(context), retain_graph=True, create_graph=True)[0]
            if i ==0:
                grad_sum = torch.mean(gradients**2)
            else:
                grad_sum += torch.mean(gradients**2)

        # print (torch.mean(gradients))
        # print (grad_sum)
        # fds
        return grad_sum

    # def predict_next_state(self, state, action):

    #     frame_size = 84
    #     z = self.encode(state)  #[P,Z]

    #     #convert aciton to one hot 
    #     action_onehot = torch.zeros(action.size()[0], self.num_outputs)
    #     # action_onehot[action.data.cpu()] = 1.

    #     action_onehot.scatter_(1, action.data.cpu(), 1)   #[P,A]
    #     action_onehot = Variable(action_onehot.float().cuda())

    #     z = torch.cat((z, action_onehot), dim=1)  #[P,Z+A] -> P,512+4

    #     z = self.state_pred_linear_1(z)
    #     z = z.view(-1, 32, 7, 7)

    #     z = self.deconv1(z)
    #     z = F.relu(z)
    #     z = self.deconv2(z)
    #     z = F.relu(z)
    #     z = self.deconv3(z)
    #     z = z*255.

        
    #     return z




    # def predict_next_state2(self, state, action):

    #     frame_size = 84

    #     # z = self.encode(state)  #[P,Z]
    #     z = self.z

    #     #convert aciton to one hot 
    #     action_onehot = torch.zeros(action.size()[0], self.num_outputs)
    #     # action_onehot[action.data.cpu()] = 1.

    #     action_onehot.scatter_(1, action.data.cpu(), 1)   #[P,A]
    #     action_onehot = Variable(action_onehot.float().cuda())

    #     #concat action and state, predict next one

    #     z = torch.cat((z, action_onehot), dim=1)  #[P,Z+A] -> P,512+4

    #     #deconv

    #     # print (z.size())

    #     z = self.state_pred_linear_1(z)
    #     z = z.view(-1, 32, 7, 7)

    #     z = self.deconv1(z)
    #     z = F.relu(z)
    #     z = self.deconv2(z)
    #     z = F.relu(z)
    #     z = self.deconv3(z)
    #     z = z*255.
        
    #     return z



















