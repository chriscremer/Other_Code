
# the bug was in the params passed to optimzer. I had eps and alpha reversed


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


# from agent_modular_debug import a2c

from storage import RolloutStorage

import copy
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy

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






# def update_rewards(reward, done, final_rewards, episode_rewards, current_state):
#     reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
#     episode_rewards += reward
#     # If done then clean the history of observations.
#     # these final rewards are only used for printing. but the mask is used in the storage, dont know why yet
#     # oh its just clearing the env that finished, and resetting its episode_reward
#     masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]) #if an env is done
#     final_rewards *= masks
#     final_rewards += (1 - masks) * episode_rewards
#     episode_rewards *= masks
#     if args.cuda:
#         masks = masks.cuda()
#     if current_state.dim() == 4:
#         current_state *= masks.unsqueeze(2).unsqueeze(2)
#     else:
#         current_state *= masks
#     return reward, masks, final_rewards, episode_rewards, current_state







def main():
    print("#######")
    print("WARNING: All rewards are clipped so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'





    print (args.cuda)
    print (args.num_steps)
    print (args.num_processes)
    print (args.lr)
    print (args.eps)
    print (args.alpha)
    print (args.use_gae)
    print (args.gamma)
    print (args.tau)
    print (args.value_loss_coef)
    print (args.entropy_coef)
    # fsdaf





    # Create environment
    envs = SubprocVecEnv([
        make_env(args.env_name, args.seed, i, args.log_dir)
        for i in range(args.num_processes)
    ])
    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if len(envs.observation_space.shape) == 3:
        actor_critic = CNNPolicy(obs_shape[0], envs.action_space)
    else:
        actor_critic = MLPPolicy(obs_shape[0], envs.action_space)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]
    # action_shape = action_shape


    # shape_dim0 = envs.observation_space.shape[0]

    # if args.cuda:
    #     dtype = torch.cuda.FloatTensor
    # else:
    #     dtype = torch.FloatTensor

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
                'entropy_coef':args.entropy_coef}




    # Create agent
    # agent = a2c(envs, hparams)


    # rollouts = RolloutStorage(self.num_steps, self.num_processes, self.obs_shape, envs.action_space)
    #it has a self.state that is [steps, processes, obs]
    #steps is used to compute expected reward

    if args.cuda:
        actor_critic.cuda()
        # rollouts.cuda()
    optimizer = optim.RMSprop(actor_critic.parameters(), hparams['lr'], eps=hparams['eps'], alpha=hparams['alpha'])







    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space)



    # Init state

    current_state = torch.zeros(args.num_processes, *obs_shape)#.type(dtype)
    def update_current_state(state):#, shape_dim0):
        shape_dim0 = envs.observation_space.shape[0]
        state = torch.from_numpy(state).float()
        if args.num_stack > 1:
            current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
        current_state[:, -shape_dim0:] = state
        # return current_state


    state = envs.reset()

    update_current_state(state)#, shape_dim0) 
    # agent.insert_first_state(current_state)
    rollouts.states[0].copy_(current_state)
        #set the first state to current state


    # These are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_state = current_state.cuda()#type(dtype)
        # if args.cuda:
        rollouts.cuda()
    #Begin training
    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):

            # Act
            # action, value = agent.act(Variable(agent.rollouts.states[step], volatile=True))
            value, action = actor_critic.act(Variable(rollouts.states[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Observe reward and next state
            state, reward, done, info = envs.step(cpu_actions) # state:[nProcesss, ndims, height, width]

            # Record rewards
            # reward, masks, final_rewards, episode_rewards, current_state = update_rewards(reward, done, final_rewards, episode_rewards, current_state)
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
            # return reward, masks, final_rewards, episode_rewards, current_state




            # Update state
            update_current_state(state)#, shape_dim0)

            # Agent record step
            # agent.insert_data(step, current_state, action.data, value.data, reward, masks)
            rollouts.insert(step, current_state, action.data, value.data, reward, masks)



        #Optimize agent
        # agent.update()
        next_value = actor_critic(Variable(rollouts.states[-1], volatile=True))[0].data
        # use last state to make prediction of next value



        if hasattr(actor_critic, 'obs_filter'):
            actor_critic.obs_filter.update(rollouts.states[:-1].view(-1, *obs_shape))
        #not sure what this is




        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        # this computes R =  r + r+ ...+ V(t)  for each step



        values, action_log_probs, dist_entropy = actor_critic.evaluate_actions(
                                                    Variable(rollouts.states[:-1].view(-1, *obs_shape)), 
                                                    Variable(rollouts.actions.view(-1, action_shape)))
        # I think this aciton log prob could have been computed and stored earlier 
        # and didnt we already store the value prediction???

        values = values.view(args.num_steps, args.num_processes, 1)
        action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

        advantages = Variable(rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()

        optimizer.zero_grad()
        (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

        optimizer.step()




        rollouts.states[0].copy_(rollouts.states[-1])
        # the first state is now the last state of the previous 









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
            # print("Updates {}, n_timesteps {}, FPS {}, mean/median R {:.1f}/{:.1f}, min/max R {:.1f}/{:.1f}, T:{:.4f}".#, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
            #     format(j, total_num_steps,
            #            int(total_num_steps / (end - start)),
            #            final_rewards.mean(),
            #            final_rewards.median(),
            #            final_rewards.min(),
            #            final_rewards.max(),
            #            end - start))#, -dist_entropy.data[0],
            #            # value_loss.data[0], action_loss.data[0]))

            # print("Upts {}, n_timesteps {}, min/med/mean/max {:.1f}/{:.1f}/{:.1f}/{:.1f}, FPS {}, T:{:.1f}".
            #     format(j, total_num_steps,
            #            final_rewards.min(),
            #            final_rewards.median(),
            #            final_rewards.mean(),
            #            final_rewards.max(),
            #            int(total_num_steps / (end - start)),
            #            end - start))

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






