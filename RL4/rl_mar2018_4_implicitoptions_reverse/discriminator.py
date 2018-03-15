







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
    def __init__(self, hparams):
        super(CNN_Discriminator, self).__init__()

        self.num_inputs = 2  #num of stacked frames
        self.num_outputs = hparams['action_size'] 

        # print (self.num_outputs)

        # width, height after conv = (x - floor(filter/2))*2 / stride 
        #since no padding, dims get lost based on how much of kernel can fit on channel.
        # its based on distance from center of filter , times 2
        # filter/kernel of 3 loses 1*2 dims, 4 loses 2*2 dims, 8 loses 4*2 dims

        self.conv1 = nn.Conv2d(self.num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        self.linear2 = nn.Linear(512, 200)

        self.linear3 = nn.Linear(200, self.num_outputs)


        # self.optimizer = optim.Adam(params=self.parameters(), lr=hparams['lr'], eps=hparams['eps'])
        self.optimizer = optim.Adam(params=self.parameters(), lr=.0001, eps=hparams['eps'])

        self.compute_loss = nn.CrossEntropyLoss(reduce=False)




    def predict(self, x):

        x = self.conv1(x / 255.0) #[32,32,20,20]
        x = F.relu(x)

        # print (x.size()) 

        x = self.conv2(x)  #[32,64,9,9]
        x = F.relu(x)

        # print (x.size())


        x = self.conv3(x)   #[32,32,7,7]
        x = F.relu(x)


        # print (x.size())
        # fdsafd


        # print (x.size())

        x = x.view(-1, 32 * 7 * 7)

        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = self.linear3(x)

        return x



    def forward(self, prev_frames, current_frames, actions):
        #frames:[P,1,84,84]
        #actions: [P,1]

        #Concat frames
        frames = Variable(torch.cat((prev_frames,current_frames), dim=1)).cuda() #[P,2,84,84]

        action_pred = self.predict(frames) #[P,A]

        neg_log_prob = self.compute_loss(action_pred, torch.squeeze(actions))  #[P]

        return neg_log_prob


    # def update(self, states, actions):


    #     # print (states, actions)  # [S+1,P,n_stacked,84,84], [S,P,1]

    #     #Make batch [B,X], [B,1]
    #         #last frme is newest so get those ones. [P*S,1,84,84]
    #     states = states[:,:,-1] #[S+1,P,84,84]
    #     # print (states.size())
    #     batch = []
    #     batch_y = []
    #     for s in range(len(states)-1):
    #         # print (len(states))
    #         two_states = torch.transpose(states[s:s+2],0,1).contiguous() #[P,2,84,84]
    #         # two_states = two_states.view(-1,2*84,84)
    #         batch.append(two_states)
    #         # batch.append(.view(-1,84,84)) #[2,P,84,84]
    #         batch_y.append(actions[s]) #[P,1]


    #     # print (len(batch))
    #     # print (batch[0].size())
    #     # print (batch_y[0].size())

    #     batch = torch.stack(batch)   #[P*S,2,84,84]
    #     # print (batch.size())
    #     # batch = batch.view(-1,1,2*84,84)  #[P*S,1,2*84,84]
    #     batch = batch.view(-1,2,84,84)  #[P*S,2,84,84]

    #     batch_y = torch.stack(batch_y)
    #     batch_y = batch_y.view(-1, 1) #[P*S,1]


    #     # print (batch.size())
    #     # print (batch_y.size())



    #     # fasfsd


    #     # #Concat the frames, dropout frames, discriminator predict context
    #     # nstep_frames = torch.stack(list_frames)  #[N, P, 1,84,84]  
    #     # nstep_frames = torch.transpose(nstep_frames, 0,1)
    #     # nstep_frames = torch.squeeze(nstep_frames) #its [P,N,84,84] so its like a batch of N dimensional images
    #     # nstep_frames = Variable(nstep_frames).cuda()
    #     # context_prediction = self.predict(nstep_frames)  #[P,C]

    #     batch = Variable(batch).cuda()
    #     batch_y = Variable(batch_y).cuda()

    #     action_pred = self.predict(batch)

    #     # print(context_prediction, context)
    #     # loss = self.compute_loss(action_pred, Variable(torch.squeeze(context)).cuda()) #[B]

    #     # print (action_pred.size())
    #     # print (batch_y.size())
    #     loss = self.compute_loss(action_pred, torch.squeeze(batch_y)) #[B]

    #     # print (loss)
    #     # fds

    #     # fdsaads

    #     cost = torch.mean(loss)

    #     self.optimizer.zero_grad()
    #     cost.backward()
    #     self.optimizer.step()

    #     return loss




    def optimize(self, errors):

        cost = torch.mean(errors)
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()



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


















