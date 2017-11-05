

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import copy
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage







class a2c(object):



    def __init__(self, envs, cuda, num_steps, num_processes, obs_shape, 
                lr, eps, alpha, use_gae, gamma, tau, value_loss_coef, entropy_coef):


        if len(envs.observation_space.shape) == 3:
            actor_critic = CNNPolicy(obs_shape[0], envs.action_space)
        else:
            actor_critic = MLPPolicy(obs_shape[0], envs.action_space)

        if envs.action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = envs.action_space.shape[0]

        if cuda:
            actor_critic.cuda()

        # if args.algo == 'a2c':
        self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps, alpha)


        rollouts = RolloutStorage(num_steps, num_processes, obs_shape, envs.action_space)
        #it has a self.state that is [steps, processes, obs]
        #steps is used to compute expected reward

        if cuda:
            rollouts.cuda()

        self.actor_critic = actor_critic
        self.rollouts = rollouts

        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef


    def act(self, current_state):

        # Sample actions
        value, action = self.actor_critic.act(current_state)
        # make prediction using state that you put into rollouts


        return action, value


    def insert_first_state(self, current_state):

        self.rollouts.states[0].copy_(current_state)
        #set the first state to current state




    def insert_data(self, step, current_state, action, value, reward, masks):

        self.rollouts.insert(step, current_state, action, value, reward, masks)
        # insert all that info into current step
        # not exactly why




    def update(self):




        next_value = self.actor_critic(Variable(self.rollouts.states[-1], volatile=True))[0].data
        # use last state to make prediction of next value



        if hasattr(self.actor_critic, 'obs_filter'):
            self.actor_critic.obs_filter.update(self.rollouts.states[:-1].view(-1, *obs_shape))
        #not sure what this is




        self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau)
        # this computes R =  r + r+ ...+ V(t)  for each step



        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                                                    Variable(self.rollouts.states[:-1].view(-1, *self.obs_shape)), 
                                                    Variable(self.rollouts.actions.view(-1, self.action_shape)))
        # I think this aciton log prob could have been computed and stored earlier 
        # and didnt we already store the value prediction???

        values = values.view(self.num_steps, self.num_processes, 1)
        action_log_probs = action_log_probs.view(self.num_steps, self.num_processes, 1)

        advantages = Variable(self.rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()

        self.optimizer.step()




        self.rollouts.states[0].copy_(self.rollouts.states[-1])
        # the first state is now the last state of the previous 
















