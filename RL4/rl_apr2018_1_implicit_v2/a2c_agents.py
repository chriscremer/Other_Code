





import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import copy
# from kfac import KFACOptimizer
# from model import CNNPolicy, MLPPolicy, CNNPolicy_dropout, CNNPolicy2, CNNPolicy_dropout2, CNNPolicy_with_var
from storage import RolloutStorage, RolloutStorage_list, RolloutStorage_with_var

from actor_critic_networks import CNNPolicy, CNNPolicy_trajectory_action_mask








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
        self.opt = hparams['opt']
        self.grad_clip = hparams['grad_clip']



        # Policy and Value network

        # if hparams['dropout'] == True:
        #     print ('CNNPolicy_dropout2')
        #     self.actor_critic = CNNPolicy_dropout2(self.obs_shape[0], envs.action_space)
        # elif len(envs.observation_space.shape) == 3:
        #     print ('CNNPolicy2')
        #     self.actor_critic = CNNPolicy2(self.obs_shape[0], envs.action_space)
        # else:
        #     self.actor_critic = MLPPolicy(self.obs_shape[0], envs.action_space)

        if 'traj_action_mask' in hparams and hparams['traj_action_mask']:
            self.actor_critic = CNNPolicy_trajectory_action_mask(self.obs_shape[0], envs.action_space)
        else:
            self.actor_critic = CNNPolicy(self.obs_shape[0], envs.action_space)



        # Storing rollouts
        self.rollouts = RolloutStorage(self.num_steps, self.num_processes, self.obs_shape, envs.action_space)


        if self.cuda:
            self.actor_critic.cuda()
            self.rollouts.cuda()


        #Optimizer
        if self.opt == 'rms':
            self.optimizer = optim.RMSprop(params=self.actor_critic.parameters(), lr=hparams['lr'], eps=hparams['eps'], alpha=hparams['alpha'])
        elif self.opt == 'adam':
            self.optimizer = optim.Adam(params=self.actor_critic.parameters(), lr=hparams['lr'], eps=hparams['eps'])
        elif self.opt == 'sgd':
            self.optimizer = optim.SGD(params=self.actor_critic.parameters(), lr=hparams['lr'], momentum=hparams['mom'])
        else:
            print ('no opt specified')



        # if envs.action_space.__class__.__name__ == "Discrete":
        #     action_shape = 1
        # else:
        #     action_shape = envs.action_space.shape[0]
        # self.action_shape = action_shape
        self.action_shape = 1
        # if __:
        #     self.deterministic_action = 0
        # else:
        #     self.deterministic_action = 0
        if hparams['gif_'] or hparams['ls_']:
            self.rollouts_list = RolloutStorage_list()

        self.hparams = hparams



    def act(self, current_state):

        # value, action = self.actor_critic.act(current_state)
        # [] [] [P,1] [P]
        value, action, action_log_probs, dist_entropy = self.actor_critic.act(current_state)

        return value, action, action_log_probs, dist_entropy


    def insert_first_state(self, current_state):

        self.rollouts.states[0].copy_(current_state)


    def insert_data(self, step, current_state, action, value, reward, masks, action_log_probs, dist_entropy):#, done):

        self.rollouts.insert(step, current_state, action, value, reward, masks, action_log_probs, dist_entropy)

        if 'traj_action_mask' in self.hparams and self.hparams['traj_action_mask']:
            self.actor_critic.reset_mask(done)


    def update(self):
        

        next_value = self.actor_critic(Variable(self.rollouts.states[-1], volatile=True))[0].data

        self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau)

        # values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
        #                                             Variable(self.rollouts.states[:-1].view(-1, *self.obs_shape)), 
        #                                             Variable(self.rollouts.actions.view(-1, self.action_shape)))


        values = torch.cat(self.rollouts.value_preds, 0).view(self.num_steps, self.num_processes, 1) 
        action_log_probs = torch.cat(self.rollouts.action_log_probs).view(self.num_steps, self.num_processes, 1)
        dist_entropy = torch.cat(self.rollouts.dist_entropy).view(self.num_steps, self.num_processes, 1)


        self.rollouts.value_preds = []
        self.rollouts.action_log_probs = []
        self.rollouts.dist_entropy = []

        advantages = Variable(self.rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()

        self.optimizer.zero_grad()
        cost = action_loss + value_loss*self.value_loss_coef - dist_entropy.mean()*self.entropy_coef
        cost.backward()

        nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.grad_clip)

        self.optimizer.step()









    #avg empowrement rather than avg error
    def update2(self, discrim_error, discrim_error_reverse):
        # discrim_error: [S,P]
        
        # next_value = self.actor_critic(Variable(self.rollouts.states[-1], volatile=True))[0].data


        # next_value, _, _, _ = self.actor_critic.act(Variable(self.rollouts.states[-1], volatile=True), context_onehot)
        # next_value = next_value.data
        # self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau)

        # print (torch.mean(discrim_error, dim=0))


        # print (discrim_error)
        discrim_error_reverse = discrim_error_reverse.view(self.num_steps, self.num_processes, 1)
        action_log_probs = torch.cat(self.rollouts.action_log_probs).view(self.num_steps, self.num_processes, 1)#[S,P,1]
        discrim_error = discrim_error.view(self.num_steps,self.num_processes,1)


        # val_to_maximize = (-discrim_error  + discrim_error_reverse)/2. - action_log_probs.detach()*5. #[S,P,1]

        val_to_maximize = -discrim_error  - action_log_probs.detach() #[S,P,1]


        val_to_maximize = val_to_maximize.view(self.num_steps,self.num_processes) #[S,P]

        discrim_error_unmodified = val_to_maximize.data.clone()
        discrim_error = val_to_maximize.data






        # self.returns[-1] = next_value
        divide_by = torch.ones(self.num_processes).cuda()
        for step in reversed(range(discrim_error.size(0)-1)):
            divide_by += 1
            ttmp = discrim_error_unmodified[step + 1] * self.gamma * torch.squeeze(self.rollouts.masks[step+1])
            discrim_error_unmodified[step] = ttmp + discrim_error_unmodified[step]
            discrim_error[step] = discrim_error_unmodified[step] / divide_by
            divide_by = divide_by * torch.squeeze(self.rollouts.masks[step+1])
        val_to_maximize = Variable(discrim_error.view(self.num_steps,self.num_processes,1))


        # discrim_error = discrim_error.view(self.num_processes*self.num_steps, 1).detach()

        # values = torch.cat(self.rollouts.value_preds, 0).view(self.num_steps, self.num_processes, 1) #[S,P,1]
        # dist_entropy = torch.cat(self.rollouts.dist_entropy).view(self.num_steps, self.num_processes, 1)


        self.rollouts.value_preds = []
        self.rollouts.action_log_probs = []
        self.rollouts.dist_entropy = []
        self.rollouts.state_preds = []

        # advantages = Variable(self.rollouts.returns[:-1]) - values
        # print (values)
        # print (discrim_error_reverse.size())  #[S,P]



        # val_to_maximize = (-discrim_error  + discrim_error_reverse.detach())/2. - action_log_probs.detach()
        # val_to_maximize = discrim_error

        baseline = torch.mean(val_to_maximize)

        advantages = val_to_maximize - baseline  #- values #(-.7)#values
        # value_loss = advantages.pow(2).mean()

        # action_loss = -(advantages.detach() * action_log_probs).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()


        # print (grad_sum)
        # cost = action_loss - dist_entropy.mean()*self.entropy_coef # + value_loss*self.value_loss_coef # - grad_sum*100000.
        cost = action_loss #- dist_entropy.mean()*self.entropy_coef # + value_loss*self.value_loss_coef # - grad_sum*100000.
        # cost = value_loss*self.value_loss_coef - dist_entropy.mean()*self.entropy_coef - grad_sum*500.
        # cost =- grad_sum
            
        self.optimizer.zero_grad()
        cost.backward()

        nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.grad_clip)

        self.optimizer.step()


















    def no_update(self):
        

        next_value = self.actor_critic(Variable(self.rollouts.states[-1], volatile=True))[0].data

        self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau)

        # values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
        #                                             Variable(self.rollouts.states[:-1].view(-1, *self.obs_shape)), 
        #                                             Variable(self.rollouts.actions.view(-1, self.action_shape)))


        values = torch.cat(self.rollouts.value_preds, 0).view(self.num_steps, self.num_processes, 1) 
        action_log_probs = torch.cat(self.rollouts.action_log_probs).view(self.num_steps, self.num_processes, 1)
        dist_entropy = torch.cat(self.rollouts.dist_entropy).view(self.num_steps, self.num_processes, 1)


        self.rollouts.value_preds = []
        self.rollouts.action_log_probs = []
        self.rollouts.dist_entropy = []

        advantages = Variable(self.rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()

        self.optimizer.zero_grad()
        cost = action_loss + value_loss*self.value_loss_coef - dist_entropy.mean()*self.entropy_coef
        cost.backward()

        nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.grad_clip)

        # self.optimizer.step()










