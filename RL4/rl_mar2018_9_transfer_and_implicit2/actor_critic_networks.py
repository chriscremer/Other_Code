
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from distributions import Categorical, DiagGaussian

import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)





class CNNPolicy(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(512, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(512, num_outputs)
        else:
            raise NotImplementedError
        self.num_inputs = num_inputs  #num of stacked frames
        self.num_outputs = num_outputs #action size

            
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        self.critic_linear = nn.Linear(512, 1)


        # self.state_pred_linear_1 = nn.Linear(512+num_outputs, 32 * 7 * 7)
        # self.deconv1 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        # self.deconv2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2)
        # # self.deconv3 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=num_inputs, kernel_size=8, stride=4)
        # self.deconv3 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=8, stride=4)

        # # self.deconv1 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=3, stride=1)
        # # self.deconv2 = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2)
        # # # self.deconv3 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=num_inputs, kernel_size=8, stride=4)
        # # self.deconv3 = torch.nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=8, stride=4)



        # # self.state_pred_linear_2 = nn.Linear(32 * 7 * 7, 512)





        # self.train()
        # self.reset_parameters()

    # def reset_parameters(self):
    #     self.apply(weights_init)

    #     relu_gain = nn.init.calculate_gain('relu')
    #     self.conv1.weight.data.mul_(relu_gain)
    #     self.conv2.weight.data.mul_(relu_gain)
    #     self.conv3.weight.data.mul_(relu_gain)
    #     self.linear1.weight.data.mul_(relu_gain)

    #     if self.dist.__class__.__name__ == "DiagGaussian":
    #         self.dist.fc_mean.weight.data.mul_(0.01)


    def encode(self, inputs):

        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)

        x = self.linear1(x)

        self.z= x

        return x


    def predict_for_action(self, inputs):

        for_action = F.relu(inputs)

        return for_action

    def predict_for_value(self, inputs):

        x = F.relu(inputs)
        for_value= self.critic_linear(x)

        return for_value

    def forward(self, inputs):

        x = self.encode(inputs)
        for_action = self.predict_for_action(x)
        for_value = self.predict_for_value(x)

        return for_value, for_action


    def action_dist(self, inputs):
        x = self.encode(inputs)
        for_action = self.predict_for_action(x)

        return self.dist.action_probs(for_action)



    def act(self, inputs, deterministic=False):
        value, x_action = self.forward(inputs)
        # action = self.dist.sample(x_action, deterministic=deterministic)
        # action_log_probs, dist_entropy = self.dist.evaluate_actions(x_action, actions)

        # x_action.mean().backward()
        # fsadf

        action, action_log_probs, dist_entropy = self.dist.sample2(x_action, deterministic=deterministic)

        # action_log_probs.mean().backward()
        # fsadf

        return value, action, action_log_probs, dist_entropy

    # def evaluate_actions(self, inputs, actions):
    #     value, x = self(inputs)
    #     action_log_probs, dist_entropy = self.dist.evaluate_actions(x, actions)
    #     return value, action_log_probs, dist_entropy


    def predict_next_state(self, state, action):

        frame_size = 84

        # print (state.size())
        # print (action.size())
        # print (self.num_outputs) # aciton size
        # print (self.num_inputs)  # num stacks
        # # print (state)
        # print (action)
        

        z = self.encode(state)  #[P,Z]

        #convert aciton to one hot 
        action_onehot = torch.zeros(action.size()[0], self.num_outputs)
        # action_onehot[action.data.cpu()] = 1.

        action_onehot.scatter_(1, action.data.cpu(), 1)   #[P,A]
        action_onehot = Variable(action_onehot.float().cuda())

        # print (action_onehot)
        # fasda

        #concat action and state, predict next one
        # print (z)
        # print (action_onehot)
        z = torch.cat((z, action_onehot), dim=1)  #[P,Z+A] -> P,512+4

        # print (z.size())
        # fdsa

        #deconv

        z = self.state_pred_linear_1(z)
        z = z.view(-1, 32, 7, 7)

        z = self.deconv1(z)
        z = F.relu(z)
        z = self.deconv2(z)
        z = F.relu(z)
        z = self.deconv3(z)
        z = z*255.

        # print (z.size())
        # fdsfa

        
        return z




    def predict_next_state2(self, state, action):

        frame_size = 84

        # z = self.encode(state)  #[P,Z]
        z = self.z

        #convert aciton to one hot 
        action_onehot = torch.zeros(action.size()[0], self.num_outputs)
        # action_onehot[action.data.cpu()] = 1.

        action_onehot.scatter_(1, action.data.cpu(), 1)   #[P,A]
        action_onehot = Variable(action_onehot.float().cuda())

        #concat action and state, predict next one

        z = torch.cat((z, action_onehot), dim=1)  #[P,Z+A] -> P,512+4

        #deconv

        # print (z.size())

        z = self.state_pred_linear_1(z)
        z = z.view(-1, 32, 7, 7)

        z = self.deconv1(z)
        z = F.relu(z)
        z = self.deconv2(z)
        z = F.relu(z)
        z = self.deconv3(z)
        z = z*255.
        
        return z









# class CNNPolicy_dropout(CNNPolicy):

#     def forward(self, inputs):

#         x = self.encode(inputs)

#         x = self.linear1(x)
#         for_action = x
        
#         x = F.dropout(x, p=.5, training=True)  #training false has no stochasticity 
#         x = F.relu(x)
#         for_value= self.critic_linear(x)

#         return for_value, for_action










class CNNPolicy_trajectory_action_mask(CNNPolicy):

    def __init__(self, num_inputs, action_space):
        super(CNNPolicy_trajectory_action_mask, self).__init__(num_inputs, action_space)

        self.first = 1

        self.dropout_rate = .5



    def predict_for_action(self, inputs):

        if self.first:
            # probs = torch.ones_like(inputs)*.5  # need newer verion
            probs = torch.ones(*inputs.size())*self.dropout_rate 
            self.mask = Variable(torch.bernoulli(probs)).cuda() # [P,512]
            self.first =0

        for_action = F.relu(inputs*self.mask)

        return for_action


    def reset_mask(self, done):
        # done: [P] numpy array
        done = Variable(torch.FloatTensor([[1.0] if done_ else [0.0] for done_ in done])).cuda() #[P,1]

        # masks = Variable(masks).cuda()

        probs = torch.ones(*self.mask.size())*self.dropout_rate
        new_mask = Variable(torch.bernoulli(probs)).cuda()
        # self.mask =  (self.mask *(1.-done)) + (new_mask*done)
        self.mask =  (self.mask *(1.-done)) + (new_mask*done)


    def act(self, inputs, deterministic=True):

        value, x_action = self(inputs)
        action, action_log_probs, dist_entropy = self.dist.sample2(x_action, deterministic=False)
        return value, action, action_log_probs, dist_entropy
















