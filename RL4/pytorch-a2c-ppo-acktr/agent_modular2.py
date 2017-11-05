


# take in dict

# add ppo

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

    def __init__(self, envs, hparams):

        self.use_gae = hparams['use_gae']
        self.gamma = hparams['gamma']
        self.tau = hparams['tau']

        self.obs_shape = hparams['obs_shape']
        self.num_steps = hparams['num_steps']
        self.num_processes = hparams['num_processes']
        self.value_loss_coef = hparams['value_loss_coef']
        self.entropy_coef = hparams['entropy_coef']
        self.cuda = hparams['cuda']





        if len(envs.observation_space.shape) == 3:
            actor_critic = CNNPolicy(self.obs_shape[0], envs.action_space)
        else:
            actor_critic = MLPPolicy(self.obs_shape[0], envs.action_space)

        if envs.action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = envs.action_space.shape[0]
        self.action_shape = action_shape

        rollouts = RolloutStorage(self.num_steps, self.num_processes, self.obs_shape, envs.action_space)
        #it has a self.state that is [steps, processes, obs]
        #steps is used to compute expected reward

        if self.cuda:
            actor_critic.cuda()
            rollouts.cuda()


        self.optimizer = optim.RMSprop(params=actor_critic.parameters(), lr=hparams['lr'], eps=hparams['eps'], alpha=hparams['alpha'])

        self.actor_critic = actor_critic
        self.rollouts = rollouts



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






















class ppo(object):


    def __init__(self, envs, hparams):

        self.use_gae = hparams['use_gae']
        self.gamma = hparams['gamma']
        self.tau = hparams['tau']

        self.obs_shape = hparams['obs_shape']
        self.num_steps = hparams['num_steps']
        self.num_processes = hparams['num_processes']
        self.value_loss_coef = hparams['value_loss_coef']
        self.entropy_coef = hparams['entropy_coef']
        self.cuda = hparams['cuda']
        self.ppo_epoch = hparams['ppo_epoch']
        self.batch_size = hparams['batch_size']
        self.clip_param = hparams['clip_param']


        if len(envs.observation_space.shape) == 3:
            actor_critic = CNNPolicy(self.obs_shape[0], envs.action_space)
        else:
            actor_critic = MLPPolicy(self.obs_shape[0], envs.action_space)

        if envs.action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = envs.action_space.shape[0]
        self.action_shape = action_shape

        rollouts = RolloutStorage(self.num_steps, self.num_processes, self.obs_shape, envs.action_space)
        #it has a self.state that is [steps, processes, obs]
        #steps is used to compute expected reward

        if self.cuda:
            actor_critic.cuda()
            rollouts.cuda()


        # self.optimizer = optim.RMSprop(params=actor_critic.parameters(), lr=hparams['lr'], eps=hparams['eps'], alpha=hparams['alpha'])
        self.optimizer = optim.Adam(params=actor_critic.parameters(), lr=hparams['lr'], eps=hparams['eps'])


        self.actor_critic = actor_critic
        self.rollouts = rollouts


        self.old_model = copy.deepcopy(self.actor_critic)



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
            self.actor_critic.obs_filter.update(self.rollouts.states[:-1].view(-1, *self.obs_shape))
        #not sure what this is




        self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau)
        # this computes R =  r + r+ ...+ V(t)  for each step



        advantages = self.rollouts.returns[:-1] - self.rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        self.old_model.load_state_dict(self.actor_critic.state_dict())
        if hasattr(self.actor_critic, 'obs_filter'):
            self.old_model.obs_filter = self.actor_critic.obs_filter

        for _ in range(self.ppo_epoch):
            sampler = BatchSampler(SubsetRandomSampler(range(self.num_processes * self.num_steps)), self.batch_size * self.num_processes, drop_last=False)
            for indices in sampler:
                indices = torch.LongTensor(indices)
                if self.cuda:
                    indices = indices.cuda()
                states_batch = self.rollouts.states[:-1].view(-1, *self.obs_shape)[indices]
                actions_batch = self.rollouts.actions.view(-1, self.action_shape)[indices]
                return_batch = self.rollouts.returns[:-1].view(-1, 1)[indices]

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(Variable(states_batch), Variable(actions_batch))

                _, old_action_log_probs, _ = self.old_model.evaluate_actions(Variable(states_batch, volatile=True), Variable(actions_batch, volatile=True))

                ratio = torch.exp(action_log_probs - Variable(old_action_log_probs.data))
                adv_targ = Variable(advantages.view(-1, 1)[indices])
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                value_loss = (Variable(return_batch) - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss + action_loss - dist_entropy * self.entropy_coef).backward()
                self.optimizer.step()



        self.rollouts.states[0].copy_(self.rollouts.states[-1])
        # the first state is now the last state of the previous 















