

# pass dict to agent

#confirm ppo works



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
# from visualize import visdom_plot


from agent_modular2 import a2c
from agent_modular2 import ppo




args = get_args()

# assert args.algo in ['a2c', 'ppo', 'acktr']
# if args.algo == 'ppo':
#     assert args.num_processes * args.num_steps % args.batch_size == 0

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


#uncomment this later when you want to save stuff
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    print ('Made dir', args.save_dir)
# else:
#     print ('Dir exists')
#     fadfa


os.environ['OMP_NUM_THREADS'] = '1'


def update_current_state(current_state, state, shape_dim0):
    state = torch.from_numpy(state).float()
    if args.num_stack > 1:
        current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
    current_state[:, -shape_dim0:] = state
    return current_state


def update_rewards(reward, done, final_rewards, episode_rewards, current_state):
    reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
    episode_rewards += reward
    # If done then clean the history of observations.
    # these final rewards are only used for printing. but the mask is used in the storage, dont know why yet
    # oh its just clearing the env that finished, and resetting its episode_reward
    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]) #if an env is done
    final_rewards *= masks
    final_rewards += (1 - masks) * episode_rewards
    episode_rewards *= masks
    if args.cuda:
        masks = masks.cuda()
    if current_state.dim() == 4:
        current_state *= masks.unsqueeze(2).unsqueeze(2)
    else:
        current_state *= masks
    return reward, masks, final_rewards, episode_rewards, current_state







def main():
    print("#######")
    print("WARNING: All rewards are clipped so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")


    # Create environment
    envs = SubprocVecEnv([
        make_env(args.env_name, args.seed, i, args.save_dir)
        for i in range(args.num_processes)
    ])
    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
    shape_dim0 = envs.observation_space.shape[0]

    if args.cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    hparams = {'cuda':args.cuda,
                'num_steps':args.num_steps,
                'num_processes':args.num_processes, 
                'obs_shape':obs_shape,
                'lr':args.lr,
                'eps':args.eps, 
                'alpha':args.alpha,
                'use_gae':args.use_gae, 
                'gamma':args.gamma, 
                'tau':args.tau,
                'value_loss_coef':args.value_loss_coef, 
                'entropy_coef':args.entropy_coef,
                'ppo_epoch': args.ppo_epoch,
                'batch_size': args.batch_size,
                'clip_param':args.clip_param}

    # Create agent
    if args.algo == 'a2c':
        agent = a2c(envs, hparams)
    elif args.algo == 'ppo':
        agent = ppo(envs, hparams)

    #Load model
    if args.load_path != '':
        # agent.actor_critic = torch.load(os.path.join(args.load_path))
        agent.actor_critic = torch.load(args.load_path).cuda()
        print ('loaded ', args.load_path)

    # Init state
    state = envs.reset()
    current_state = torch.zeros(args.num_processes, *obs_shape)
    current_state = update_current_state(current_state, state, shape_dim0).type(dtype)
    agent.insert_first_state(current_state)

    # These are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])


    #Begin training
    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):

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






        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        #Save model
        if total_num_steps % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            # A really ugly way to save a model to CPU
            save_model = agent.actor_critic
            if args.cuda:
                save_model = copy.deepcopy(agent.actor_critic).cpu()
            # torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
            save_to=os.path.join(save_path, args.algo + ".pt")
            torch.save(save_model, save_to)
            print ('saved', save_to)

        #Print updates
        if j % args.log_interval == 0:
            end = time.time()

            if j % (args.log_interval*30) == 0:
                print("Upts, n_timesteps, min/med/mean/max, FPS, Time")

            print("{}, {}, {:.1f}/{:.1f}/{:.1f}/{:.1f}, {}, {:.1f}".
                    format(j, total_num_steps,
                           final_rewards.min(),
                           final_rewards.median(),
                           final_rewards.mean(),
                           final_rewards.max(),
                           int(total_num_steps / (end - start)),
                           end - start))




if __name__ == "__main__":
    main()






