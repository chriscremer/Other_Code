



import copy
import glob
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from arguments import get_args
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env

from agent_modular2 import a2c
from agent_modular2 import ppo

import argparse
import json

# from arguments import get_args
parser = argparse.ArgumentParser()
parser.add_argument('--m')
args = parser.parse_args()
# print (args.m)


with open(args.m, 'r') as infile:
    model_dict = json.load(infile)

# print (model_dict)


def train(model_dict):

    def update_current_state(current_state, state, shape_dim0):
        state = torch.from_numpy(state).float()
        if num_stack > 1:
            current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
        current_state[:, -shape_dim0:] = state
        return current_state


    def update_rewards(reward, done, final_rewards, episode_rewards, current_state):
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        episode_rewards += reward
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]) #if an env is done
        final_rewards *= masks
        final_rewards += (1 - masks) * episode_rewards
        episode_rewards *= masks
        if cuda:
            masks = masks.cuda()
        if current_state.dim() == 4:
            current_state *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_state *= masks
        return reward, masks, final_rewards, episode_rewards, current_state


    num_frames = model_dict['num_frames']
    cuda = model_dict['cuda']
    num_steps = model_dict['num_steps']
    num_processes = model_dict['num_processes']
    seed = model_dict['seed']
    env = model_dict['env']
    save_dir = model_dict['save_to']
    num_stack = model_dict['num_stack']
    algo = model_dict['algo']
    save_interval = model_dict['save_interval']
    log_interval = model_dict['log_interval']

    # print("#######")
    # print("WARNING: All rewards are clipped so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    # print("#######")


    num_updates = int(num_frames) // num_steps // num_processes

    
    if cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)



    os.environ['OMP_NUM_THREADS'] = '1'

    # Create environments
    envs = SubprocVecEnv([make_env(env, seed, i, save_dir) for i in range(num_processes)])


    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])
    shape_dim0 = envs.observation_space.shape[0]

    model_dict['obs_shape']=obs_shape

    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor


    # Create agent
    if algo == 'a2c':
        agent = a2c(envs, model_dict)
    elif algo == 'ppo':
        agent = ppo(envs, model_dict)

    # #Load model
    # if args.load_path != '':
    #     # agent.actor_critic = torch.load(os.path.join(args.load_path))
    #     agent.actor_critic = torch.load(args.load_path).cuda()
    #     print ('loaded ', args.load_path)

    # Init state
    state = envs.reset()
    current_state = torch.zeros(num_processes, *obs_shape)
    current_state = update_current_state(current_state, state, shape_dim0).type(dtype)
    agent.insert_first_state(current_state)

    # These are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_processes, 1])
    final_rewards = torch.zeros([num_processes, 1])


    #Begin training
    start = time.time()
    for j in range(num_updates):
        for step in range(num_steps):

            # Act
            action, value = agent.act(Variable(agent.rollouts.states[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Observe reward and next state
            state, reward, done, info = envs.step(cpu_actions) # state:[nProcesss, ndims, height, width]

            # Record rewards
            reward, masks, final_rewards, episode_rewards, current_state = update_rewards(reward, done, final_rewards, episode_rewards, current_state)

            # Update state
            current_state = update_current_state(current_state, state, shape_dim0)

            # Agent record step
            agent.insert_data(step, current_state, action.data, value.data, reward, masks)

        #Optimize agent
        agent.update()






        total_num_steps = (j + 1) * num_processes * num_steps

        #Save model
        if total_num_steps % save_interval == 0 and save_dir != "":
            save_path = os.path.join(save_dir, algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            # A really ugly way to save a model to CPU
            save_model = agent.actor_critic
            if cuda:
                save_model = copy.deepcopy(agent.actor_critic).cpu()
            # torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
            save_to=os.path.join(save_path, algo + ".pt")
            torch.save(save_model, save_to)
            print ('saved', save_to)

        #Print updates
        if j % log_interval == 0:
            end = time.time()

            if j % (log_interval*30) == 0:
                print("Upts, n_timesteps, min/med/mean/max, FPS, Time")

            print("{}, {}, {:.1f}/{:.1f}/{:.1f}/{:.1f}, {}, {:.1f}".
                    format(j, total_num_steps,
                           final_rewards.min(),
                           final_rewards.median(),
                           final_rewards.mean(),
                           final_rewards.max(),
                           int(total_num_steps / (end - start)),
                           end - start))



train(model_dict)


# if __name__ == "__main__":
#     main()




