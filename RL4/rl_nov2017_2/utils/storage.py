import torch
import numpy as np

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.states = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

    def cuda(self):
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, current_state, action, value_pred, reward, mask):
        self.states[step + 1].copy_(current_state)
        self.actions[step].copy_(action)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step+1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step+1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step+1] + self.rewards[step]  




# #convert to lists, so that I compute returns at any point.
# # I guess I dont need lists, just need to speify where to compute return? 
# # Im going to do it anyway, also need to pay attention to whats on gpu or cpu

# #the issue is that there are multiplt proceesses
# # ideally i would update once it reaches the end of teh episode or reaches a max number of steps
# # but cant do that sicne there are multiple processes
# but maybe could use for just the gif making

class RolloutStorage_list(object):
    def __init__(self):

        self.states = []
        self.actions = []
        self.rewards = []
        self.value_preds = []
        self.masks = []


    def insert(self, step, current_state, action, value_pred, reward, mask):

        self.states.append(current_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.value_preds.append(value_pred)
        self.masks.append(mask)



    def reset(self):

        self.states = []
        self.actions = []
        self.rewards = []
        self.value_preds = []
        self.masks = []
        self.returns = []


    def compute_returns(self, next_value, gamma):

        num_processes = next_value.size()[0]
        self.masks.append(torch.ones(num_processes, 1).cuda())
        self.returns = []
        next_value = 0.
        self.returns = [next_value] + self.returns  


        # print (next_value)
        # print (self.returns[0])
        # print (self.masks[0])
        # print (self.rewards[0] )

        #rewards: [S,P,1]
        
        # print (self.rewards)
        # print (self.returns)

        for step in reversed(range(len(self.rewards))):
            # self.returns = [self.rewards[step].cuda() + gamma*self.returns[0]*self.masks[step]] + self.returns
            # self.returns = [self.rewards[0][step].cuda() + gamma*self.returns[0]] + self.returns

            # self.returns = [self.rewards[step][0].numpy() + gamma*self.returns[0]] + self.returns
            self.returns = [self.rewards[step] + gamma*self.returns[0]] + self.returns




        # print (self.returns)
        # dsfadafd
            











