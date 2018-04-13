

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from collections import deque






class CNN(nn.Module):
    def __init__(self, n_outputs):
        super(CNN, self).__init__()

        n_channels = 4

        self.conv1 = nn.Conv2d(n_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        self.out_linear = nn.Linear(512, n_outputs)


    def forward(self, inputs):

        x = self.conv1(inputs  / 255.)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)

        x = self.linear1(x)
        x = F.relu(x)

        x = self.out_linear(x)

        return x





class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

        self.batch_size =  32
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 1)
        next_state = np.expand_dims(next_state, 1)
            
        # self.buffer.append((state, action, reward, next_state, done))

        # print (state[0].shape)
        # fasdf

        for i in range(len(done)):
            self.buffer.append([state[i], action[i], reward[i], next_state[i], done[i]])
    
    def sample(self):
        # state, action, reward, next_state, done = zip(*np.random.choice(self.buffer, self.batch_size))


        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for i in range(self.batch_size):

            ind = np.random.randint(len(self.buffer))
            datapoint = self.buffer[ind]
            states.append(datapoint[0])
            actions.append(datapoint[1])
            rewards.append(datapoint[2])
            next_states.append(datapoint[3])
            dones.append(datapoint[4])       



        # ind = np.random.randint(len(self.buffer))
        # datapoint = self.buffer[ind]
        # states = datapoint[0]
        # actions = datapoint[1]
        # rewards = datapoint[2]
        # next_states = datapoint[3]
        # dones = datapoint[4] 




        # print (actions)
        # return states, actions, rewards, next_states, dones
        return np.concatenate(states), np.array(actions), np.array(rewards), np.concatenate(next_states), np.array(dones)
        # return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
        # return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(next_states), torch.stack(dones)


    
    def __len__(self):
        return len(self.buffer)




class DQN(object):

    def __init__(self, envs, hparams):


        self.replay_buffer = ReplayBuffer(capacity=8000)

        self.q_net = CNN(n_outputs=hparams['action_size'])

        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=hparams['lr']) #, momentum=.9)


        if hparams['cuda']:
            self.q_net.cuda()

        self.hparams = hparams
        self.loss = Variable(torch.FloatTensor([2.]))



    def act(self, state, epsilon):
        if np.random.random() > epsilon:
            # state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.q_net.forward(state)  #[B,A]
            action =  q_value.max(1)[1].data.cpu().numpy()  #[B] 
        else:
            action = np.random.choice(self.hparams['action_size'], state.size()[0])

        return action



    def update(self):

        gamma = .99

        #Sample buffer
        state, action, reward, next_state, done = self.replay_buffer.sample()
        state      = Variable(torch.FloatTensor(np.float32(state))).cuda()
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True).cuda()
        action     = Variable(torch.LongTensor(action)).cuda()
        reward     = Variable(torch.FloatTensor(reward)).cuda()
        done       = Variable(torch.FloatTensor(done)).cuda()

        # print (state.size())
        # print (done.size())
        # print (reward.size())

        #Compute loss
        q_values      = self.q_net(state)
        next_q_values = self.q_net(next_state)

        # print (q_values.size())

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # print (q_value.size())

        next_q_value     = next_q_values.max(1)[0]
        # print (next_q_value.size())

        expected_q_value = reward + gamma * next_q_value * (1 - done)

        # print (expected_q_value.size())

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()


        # print (loss)
        # fdaf

        self.loss = loss

        #Update q net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

















