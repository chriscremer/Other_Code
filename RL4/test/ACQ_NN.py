


import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F






class NN(nn.Module):
    def __init__(self, seed=1):
        super(NN, self).__init__()

        torch.manual_seed(seed)

        self.action_size = 2
        self.state_size = 4
        self.value_size = 1

        
        h_size = 50

        self.actor = nn.Sequential(
          nn.Linear(self.state_size,h_size),
          nn.ReLU(),
          nn.Linear(h_size,self.action_size),
          # nn.log_softmax(dim=1)
        )

        self.critic = nn.Sequential(
          nn.Linear(self.state_size,h_size),
          nn.ReLU(),
          nn.Linear(h_size,self.value_size)
        )

        self.Q_func = nn.Sequential(
          nn.Linear(self.state_size + self.action_size,h_size),
          nn.ReLU(),
          nn.Linear(h_size,self.value_size)
        )

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=.0001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=.0001)
        self.optimizer_qfunc = optim.Adam(self.Q_func.parameters(), lr=.0001)


    def forward(self, input_x):

        input_x = Variable(torch.FloatTensor(input_x)).cuda()
        return self.Q_func(input_x)


    def forward2(self, state):

        input_x = Variable(torch.FloatTensor([state])).cuda()
        return F.softmax(self.actor(input_x), dim=1).data.cpu().numpy()


    def forward3(self, state):

        input_x = Variable(torch.FloatTensor([state])).cuda()
        return self.critic(input_x).data.cpu().numpy()





    # def train(self, data_x, data_y):
    #     #data_x: [B,X]
    #     #data_y: [B,Y]

    #     data_x = Variable(torch.FloatTensor(data_x)).cuda()
    #     data_y = Variable(torch.FloatTensor(data_y)).cuda()

    #     done = 0
    #     step_count = 0
    #     last_100 = []
    #     best_100 = 0
    #     not_better_counter = 0
    #     first = 1
    #     # while not done:

    #     #sample batch
    #     # samps = sampler(20) #[B,X]
    #     # print (samps.size())
    #     # obj_value = objective(samps).unsqueeze(1) #[B,1]
    #     pred = self.net(data_x) #[B,Y]

    #     loss = torch.mean((data_y - pred)**2) #[1]

    #     # print(step_count, loss.data.numpy())

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     # break

            

    #         # if len(last_100) < 100:
    #         #     last_100.append(loss)
    #         # else:
    #         #     last_100_avg = torch.mean(torch.stack(last_100))
    #         #     # print(step_count, loss.data.numpy())
    #         #     print(step_count, last_100_avg.data.numpy())
    #         #     # print(best_100, last_100_avg)
    #         #     # print (last_100_avg< best_100)
    #         #     if first or (last_100_avg< best_100).data.numpy():
    #         #         first = 0
    #         #         best_100 = last_100_avg
    #         #         not_better_counter = 0
    #         #     else:
    #         #         not_better_counter +=1
    #         #         if not_better_counter == 3:
    #         #             break
    #         #     last_100 = []

    #         # step_count+=1





    # def train_on_dataset(self, dataset):

    #     for step in range(100):

    #         #Sample datapoint
    #         datapoint = dataset[np.random.randint(len(dataset))]
    #         s_t, a_t, r_t, s_tp1, done = datapoint

    #         state = torch.unsqueeze(Variable(torch.FloatTensor(s_t)), 0).cuda()
    #         next_state = torch.unsqueeze(Variable(torch.FloatTensor(s_tp1)), 0).cuda()
    #         action = torch.unsqueeze(Variable(torch.FloatTensor([a_t])), 0).cuda()

    #         #Predict value
    #         v_value = self.critic(state)

    #         #Predict Q
    #         # q_value = self.Q_func(torch.cat((state, Variable((action.data).type(torch.FloatTensor))), 1))
    #         q_value = self.Q_func(torch.cat((state, action), 1))


    #         #Predict action distribution and sample
    #         action_x = self.actor(state)
    #         # action_dist = self.actor(state)
    #         # print (action_x)
    #         action_dist = F.softmax(action_x, dim=1)
    #         next_state_action = action_dist.multinomial().detach()
    #         log_probs = F.log_softmax(action_x, dim=1)
    #         dist_entropy = -(log_probs * action_dist).sum(-1)  #[P]
    #         action_log_prob = log_probs.gather(1, next_state_action)


    #         #Predict next state Q
    #         # next_state_action = Variable(torch.FloatTensor(np.array([np.random.randint(2)]))).cuda()
    #         # next_state_action = F.softmax(self.actor(next_state), dim=1).multinomial().detach()

    #         # print (next_state)
    #         # print (Variable(torch.unsqueeze((next_state_action.data).type(torch.FloatTensor),0)))
    #         # next_state_action = Variable(torch.unsqueeze((next_state_action.data).type(torch.FloatTensor),0)).cuda()
    #         next_state_action = Variable((next_state_action.data).type(torch.FloatTensor)).cuda()
    #         # print (next_state)
    #         # print (next_state_action)
    #         q_value_next_state = self.Q_func(torch.cat((next_state, next_state_action), 1)).detach()

    #         if not done:
    #             R_and_Q = r_t + q_value_next_state 
    #         else:
    #             R_and_Q = Variable(torch.FloatTensor(np.reshape(np.array([r_t]), [1,1]))).cuda()


    #         #Loses
    #         critic_loss = torch.mean((q_value - v_value)**2)
    #         q_func_loss = torch.mean((R_and_Q - q_value)**2)
    #         actor_loss = -torch.mean(((q_value - v_value)).detach()*action_log_prob)

    #         # if step==20:
    #         #     print (v_value.data.numpy(), q_value.data.numpy(), critic_loss.data.numpy()[0], q_func_loss.data.numpy()[0], actor_loss.data.numpy()[0])


    #         self.optimizer_qfunc.zero_grad()
    #         q_func_loss.backward(retain_graph=True)
    #         # q_func_loss.backward()
    #         self.optimizer_qfunc.step() 

    #         self.optimizer_actor.zero_grad()
    #         actor_loss.backward(retain_graph=True)
    #         self.optimizer_actor.step()

    #         self.optimizer_critic.zero_grad()
    #         critic_loss.backward()
    #         self.optimizer_critic.step()



    #         # fdafdsfa


    #         # # next_state_and_rand_action = np.array([np.concatenate([s_tp1,np.array([np.random.randint(2)])])])
    #         # Q_next_state = self.forward(next_state_and_rand_action).data.numpy()

            
    #         # # print (data_x.shape)  #[B,X+A]
    #         # data_x = np.array([np.concatenate([s_t,np.array([a_t])])])
    #         # self.train(data_x, R_and_Q)








    #batch version
    def train_on_dataset(self, dataset):

        batch_size = 1

        for step in range(100):

            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            for i in range(batch_size):
                #Sample datapoint
                datapoint = dataset[np.random.randint(len(dataset))]
                s_t, a_t, r_t, s_tp1, done = datapoint
                state_batch.append(s_t)
                action_batch.append(a_t)
                reward_batch.append(r_t)
                next_state_batch.append(s_tp1)            

            state = Variable(torch.FloatTensor(state_batch)).cuda()
            action = Variable(torch.FloatTensor(action_batch)).cuda()
            reward = Variable(torch.unsqueeze(torch.FloatTensor(reward_batch),1)).cuda()
            next_state = Variable(torch.FloatTensor(next_state_batch)).cuda()
            
            # #Predict value
            v_value = self.critic(state)

            #Predict Q
            q_value = self.Q_func(torch.cat((state, action), 1))


            #Predict action distribution and sample
            action_x = self.actor(next_state)  #[B,A]
            action_dist = F.softmax(action_x, dim=1) #[B,A]
            log_probs = F.log_softmax(action_x, dim=1) #[B,A]
            dist_entropy = -(log_probs * action_dist).sum(-1)  #[P]

            next_state_action = action_dist.multinomial().detach().data.cpu() #[B,1]
            next_state_action_onehot = Variable(torch.FloatTensor(batch_size, self.action_size).zero_().scatter_(1, next_state_action, 1)).cuda() #[B,A]
            next_state_action_onehot_L = Variable(torch.LongTensor(batch_size, self.action_size).zero_().scatter_(1, next_state_action, 1)).cuda() #[B,A]
            action_log_prob = log_probs.gather(1, next_state_action_onehot_L)

            # #Random action
            # a_t_onehot = np.zeros([batch_size, 2])
            # indexes = np.random.randint(2, size=(batch_size))
            # a_t_onehot[np.arange(batch_size),indexes] = 1.
            # next_state_action_onehot = Variable(torch.FloatTensor(a_t_onehot)).cuda()

            #Predict Q next state
            q_value_next_state = self.Q_func(torch.cat((next_state, next_state_action_onehot), 1)).detach()



            if not done:
                R_and_Q = reward + q_value_next_state 
            else:
                # R_and_Q = Variable(torch.FloatTensor(np.reshape(np.array([r_t]), [1,1]))).cuda()
                R_and_Q = reward

            #Loses
            critic_loss = torch.mean((q_value - v_value)**2)
            q_func_loss = torch.mean((R_and_Q - q_value)**2)
            actor_loss = -torch.mean(((q_value - v_value)).detach()*action_log_prob) - dist_entropy.mean()*.05

            # # print (step)
            # if step==10:
            #     # print (v_value.data.cpu().numpy(), q_value.data.cpu().numpy())
            #     print (critic_loss.data.cpu().numpy()[0], q_func_loss.data.cpu().numpy()[0], actor_loss.data.cpu().numpy()[0])
            #     # print (action_log_prob)

            self.optimizer_qfunc.zero_grad()
            q_func_loss.backward(retain_graph=True)
            # nn.utils.clip_grad_norm(self.Q_func.parameters(), .5)
            # q_func_loss.backward()
            self.optimizer_qfunc.step() 

            self.optimizer_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm(self.actor.parameters(), .5)
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            # nn.utils.clip_grad_norm(self.critic.parameters(), .5)
            critic_loss.backward()
            self.optimizer_critic.step()


            # next_state_action_onehot_L = Variable(torch.cuda.LongTensor(next_state_action_onehot))
            # next_state_action_onehot_F = Variable(torch.cuda.FloatTensor(next_state_action_onehot))


            # print (action_dist.data.cpu().numpy(), dist_entropy.data.cpu().numpy())



            # action_dist = F.sigmoid(action_x)

            # next_state_action = next_state_action.data.cpu().numpy()
            # next_state_action_onehot = np.zeros((batch_size,2))
            
            # next_state_action_onehot[:,next_state_action] = 1.
            # print (next_state_action_onehot)

            # next_state_action = action_dist.bernoulli().detach()
            # next_state_action_longtensor = Variable(torch.cuda.LongTensor(next_state_action.data.cpu().numpy()))
            # print (next_state_action)
            
            # log_probs = torch.nn.LogSigmoid(action_x, dim=1)
            # log_probs = F.logsigmoid(action_x)

            # print (log_probs)
            # fdfas
            

            # print (log_probs)
            # print (next_state_action_onehot_L)
            
            # # action_log_prob = log_probs

            # print (action_dist.data.cpu().numpy(), log_probs.data.cpu().numpy(), next_state_action.data.cpu().numpy())

            # fdsaa


            #Predict next state Q
            # next_state_action = Variable(torch.FloatTensor(np.array([np.random.randint(2)]))).cuda()
            # next_state_action = F.softmax(self.actor(next_state), dim=1).multinomial().detach()

            # print (next_state)
            # print (Variable(torch.unsqueeze((next_state_action.data).type(torch.FloatTensor),0)))
            # next_state_action = Variable(torch.unsqueeze((next_state_action.data).type(torch.FloatTensor),0)).cuda()

            # next_state_action = Variable((next_state_action.data).type(torch.FloatTensor)).cuda()
            



