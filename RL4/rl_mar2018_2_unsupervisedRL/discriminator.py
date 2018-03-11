







import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# from distributions import Categorical, DiagGaussian

# from distributions import Categorical2 as Categorical

import numpy as np




# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1 or classname.find('Linear') != -1:
#         nn.init.orthogonal(m.weight.data)
#         if m.bias is not None:
#             m.bias.data.fill_(0)





class CNN_Discriminator(nn.Module):
    def __init__(self, n_stacked_frames, n_contexts, hparams):
        super(CNN_Discriminator, self).__init__()

        self.num_inputs = n_stacked_frames  #num of stacked frames
        self.num_outputs = n_contexts 

            
        self.conv1 = nn.Conv2d(n_stacked_frames, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        self.linear2 = nn.Linear(512, 200)

        self.linear3 = nn.Linear(200, n_contexts)


        self.optimizer = optim.Adam(params=self.parameters(), lr=hparams['lr'], eps=hparams['eps'])

        self.compute_loss = nn.CrossEntropyLoss(reduce=False)




    def predict(self, x):

        x = self.conv1(x / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)

        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = self.linear3(x)

        return x


    def update(self, list_frames, context):


        #Concat the frames, dropout frames, discriminator predict context
        nstep_frames = torch.stack(list_frames)  #[N, P, 1,84,84]  
        nstep_frames = torch.transpose(nstep_frames, 0,1)
        nstep_frames = torch.squeeze(nstep_frames) #its [P,N,84,84] so its like a batch of N dimensional images
        nstep_frames = Variable(nstep_frames).cuda()
        context_prediction = self.predict(nstep_frames)  #[P,C]

        # print(context_prediction, context)
        loss = self.compute_loss(context_prediction, Variable(torch.squeeze(context)).cuda()) #[B]

        # print (loss)
        # fds

        # fdsaads

        cost = torch.mean(loss)

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return loss





    # def predict_for_action(self, inputs):
    #     inputs = F.elu(self.action_linear(inputs))
    #     for_action = self.action_linear2(inputs)
    #     return for_action

    # def predict_for_value(self, inputs):
    #     for_value= self.critic_linear(inputs)
    #     return for_value

    # def forward(self, x):
    #     # x = self.encode(inputs)
    #     for_action = self.predict_for_action(x)
    #     for_value = self.predict_for_value(x)
    #     return for_value, for_action


    # # def action_dist(self, inputs):
    # #     x = self.encode(inputs)
    # #     for_action = self.predict_for_action(x)
    # #     return self.dist.action_probs(for_action)



    # def act(self, frames, context, deterministic=False):

    #     frames = self.encode_frames(frames) #[B,F]
    #     inputs = torch.concat((frames, context), dim=1)  #[B,F+Z]

    #     value, x_action = self.forward(inputs)
    #     action, action_log_probs, dist_entropy = self.dist.sample2(x_action, deterministic=deterministic)
    #     return value, action, action_log_probs, dist_entropy


















