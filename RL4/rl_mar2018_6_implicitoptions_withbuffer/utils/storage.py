import torch
import numpy as np

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.states = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)

        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)


        # self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_preds = []


        # self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        # self.dist_entropy = torch.zeros(num_steps, num_processes, 1)
        self.action_log_probs = []
        self.dist_entropy = []

        self.state_preds = []
        self.real_states = []


        self.num_steps = num_steps
        self.num_processes = num_processes


    def cuda(self):
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

        # self.value_preds = self.value_preds.cuda()
        # self.action_log_probs = self.actions.cuda()
        # self.dist_entropy = self.masks.cuda()


    def insert(self, step, current_state, action, value_pred, reward, mask, action_log_prob, dist_entropy):
        self.states[step + 1].copy_(current_state)
        self.actions[step].copy_(action)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)

        # self.value_preds[step].copy_(value_pred)
        self.value_preds.append(value_pred)

        # self.action_log_probs[step].copy_(action_log_prob)
        # self.dist_entropy[step].copy_(dist_entropy)
        self.action_log_probs.append(action_log_prob)
        self.dist_entropy.append(dist_entropy)


    # def insert_state_pred(self, state_pred):#, real_state):

    #     self.state_preds.append(state_pred)
    #     # self.real_states.append(real_state)



    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step+1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step+1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            # Discounted future p(x)
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step+1] + self.rewards[step]  













class RolloutStorage_with_var(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.states = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)

        self.value_means = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_logvars = torch.zeros(num_steps + 1, num_processes, 1)

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
        self.value_means = self.value_means.cuda()
        self.value_logvars = self.value_logvars.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, current_state, action, value_mean, value_logvar, reward, mask):
        self.states[step + 1].copy_(current_state)
        self.actions[step].copy_(action)
        self.value_means[step].copy_(value_mean)
        self.value_logvars[step].copy_(value_logvar)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)

    def compute_returns(self, next_value, use_gae, gamma, tau):
        # if use_gae:
        #     self.value_preds[-1] = next_value
        #     gae = 0
        #     for step in reversed(range(self.rewards.size(0))):
        #         delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step+1] - self.value_preds[step]
        #         gae = delta + gamma * tau * self.masks[step+1] * gae
        #         self.returns[step] = gae + self.value_preds[step]
        # else:
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
        self.action_log_probs = []


    def insert(self, step, current_state, action, value_pred, reward, mask, action_logprob):

        self.states.append(current_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.value_preds.append(value_pred)
        self.masks.append(mask)

        # if action_logprob != None:
        self.action_log_probs.append(action_logprob)





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

        # next_value = 0.
        
        self.returns = [next_value] + self.returns  

        for step in reversed(range(len(self.rewards))):
            self.returns = [self.rewards[step] + gamma*self.returns[0]] + self.returns













