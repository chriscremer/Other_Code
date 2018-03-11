


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import subprocess
import os
import shutil

import imageio

import copy

import csv  


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]


def makedir(dir_, print_=True, rm=False):

    if rm and os.path.exists(dir_):
        shutil.rmtree(dir_)
        os.makedirs(dir_)
        print ('rm dir and made', dir_) 
        
        # if print_:
        #     print ('Made dir', dir_) 
    elif not os.path.exists(dir_):
        os.makedirs(dir_)
        if print_:
            print ('Made dir', dir_) 



def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    # halfwidth = min(31, int(np.ceil(len(x) / 30)))
    halfwidth = min(21, int(np.ceil(len(x) / 20)))

    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]








def get_info(envs, agent, model_dict, total_num_steps, update_current_state, update_rewards):


    # Init state
    state = envs.reset()  # (processes, channels, height, width)
    current_state = torch.zeros(num_processes, *obs_shape)  # (processes, channels*stack, height, width)
    current_state = update_current_state(current_state, state, shape_dim0).type(dtype) #add the new frame, remove oldest, since its a stack
    agent.insert_first_state(current_state) #storage has states: (num_steps + 1, num_processes, *obs_shape), set first step 

    # These are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_processes, 1]) #keeps track of current episode cumulative reward
    final_rewards = torch.zeros([num_processes, 1])

    # num_updates = int(num_frames) // num_steps // num_processes
    # save_interval_num_updates = int(save_interval /num_processes/num_steps)




    #Begin training
    # count =0
    start = time.time()
    start2 = time.time()
    for j in range(num_updates):
        for step in range(num_steps):

            # Act, [P,1], [P], [P,1], [P]
            state_pytorch = Variable(agent.rollouts.states[step])
            value, action, action_log_probs, dist_entropy = agent.act(state_pytorch)#, volatile=True))
            
            # Apply to Environment, S:[P,C,H,W], R:[P], D:[P]
            cpu_actions = action.data.squeeze(1).cpu().numpy() #[P]
            state, reward, done, info = envs.step(cpu_actions) 

            # Record rewards and update state
            reward, masks, final_rewards, episode_rewards, current_state = update_rewards(reward, done, final_rewards, episode_rewards, current_state)
            current_state = update_current_state(current_state, state, shape_dim0)
            agent.insert_data(step, current_state, action.data, value, reward, masks, action_log_probs, dist_entropy, 0) #, done)


        #Optimize agent
        agent.update()  #agent.update(j,num_updates)

        agent.insert_first_state(agent.rollouts.states[-1])



    #Reset the agents rollouts. 
    self.rollouts = RolloutStorage(self.num_steps, self.num_processes, self.obs_shape, hparams['action_space'])

    if self.cuda:
        self.rollouts.cuda()

























































