
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
from agent_modular import a2c



args = get_args()

# assert args.algo in ['a2c', 'ppo', 'acktr']
# if args.algo == 'ppo':
#     assert args.num_processes * args.num_steps % args.batch_size == 0

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


#uncomment this later when you want to save stuff
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
    print ('Made dir', args.log_dir)
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
    return reward, masks, final_rewards, episode_rewards 







def main():
    print("#######")
    print("WARNING: All rewards are clipped so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")


    # Create environment
    envs = SubprocVecEnv([
        make_env(args.env_name, args.seed, i, args.log_dir)
        for i in range(args.num_processes)
    ])
    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
    shape_dim0 = envs.observation_space.shape[0]

    if args.cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor


    # Create agent
    agent = a2c(envs, cuda=args.cuda, num_steps=args.num_steps, num_processes=args.num_processes, 
                    obs_shape=obs_shape, lr=args.lr, eps=args.eps, alpha=args.alpha,
                    use_gae=args.use_gae, gamma=args.gamma, tau=args.tau,
                    value_loss_coef=args.value_loss_coef, entropy_coef=args.entropy_coef)


    # Init state
    state = envs.reset()
    current_state = torch.zeros(args.num_processes, *obs_shape)
    current_state = update_current_state(current_state, state, shape_dim0).type(dtype)


    # These are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])


    #Begin training
    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):

            # Act
            action, value = agent.act(Variable(current_state, volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Observe reward and next state
            state, reward, done, info = envs.step(cpu_actions) # state:[nProcesss, ndims, height, width]

            # Record rewards
            reward, masks, final_rewards, episode_rewards = update_rewards(reward, done, final_rewards, episode_rewards, current_state)

            # Update state
            current_state = update_current_state(current_state, state, shape_dim0)

            # Agent record step
            agent.insert_data(step, current_state, action.data, value.data, reward, masks)

        #Optimize agent
        agent.update()








        # #Save model
        # if j % args.save_interval == 0 and args.save_dir != "":
        #     save_path = os.path.join(args.save_dir, args.algo)
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass

        #     # A really ugly way to save a model to CPU
        #     save_model = actor_critic
        #     if args.cuda:
        #         save_model = copy.deepcopy(actor_critic).cpu()
        #     torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        #Print updates
        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median R {:.1f}/{:.1f}, min/max R {:.1f}/{:.1f}".#, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max()))#, -dist_entropy.data[0],
                       # value_loss.data[0], action_loss.data[0]))


if __name__ == "__main__":
    main()






