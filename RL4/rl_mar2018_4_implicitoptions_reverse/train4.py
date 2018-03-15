




import sys
for i in range(len(sys.path)):
    if 'er/Documents' in sys.path[i]:
        sys.path.remove(sys.path[i])#[i]
        break

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


sys.path.insert(0, '../baselines/')
sys.path.insert(0, '../baselines/baselines/common/vec_env')
from subproc_vec_env import SubprocVecEnv

sys.path.insert(0, './utils/')
from envs import make_env, make_env_monitor, make_env_basic


# from agent_modular2 import a2c
# from agent_modular2 import ppo
# from agent_modular2 import a2c_minibatch
# from agent_modular2 import a2c_list_rollout
# from agent_modular2 import a2c_with_var

from a2c_agents import a2c
# from a2c_agents import a2c_over
# from a2c_agents import a2c_under

from discriminator import CNN_Discriminator



from train_utils import do_vid, do_gifs, do_params, do_ls, update_ls_plot, view_reward_episode, do_prob_state
from train_utils3 import do_grad_var, update_grad_plot

sys.path.insert(0, './visualizations/')
from make_plots import make_plots

import argparse
import json
import subprocess


from vae import VAE


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def train(model_dict):

    def update_current_state(current_state, state, channels):
        # current_state: [processes, channels*stack, height, width]
        state = torch.from_numpy(state).float()  # (processes, channels, height, width)
        # if num_stack > 1:
        #first stack*channel-channel frames = last stack*channel-channel , so slide them forward
        current_state[:, :-channels] = current_state[:, channels:] 
        current_state[:, -channels:] = state #last frame is now the new one

        return current_state


    def update_rewards(reward, done, final_rewards, episode_rewards, current_state):
        # Reward, Done: [P], [P]
        # final_rewards, episode_rewards: [P,1]. [P,1]
        # current_state: [P,C*S,H,W]
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float() #[P,1]
        episode_rewards += reward #keeps track of current episode cumulative reward
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]) #[P,1]
        final_rewards *= masks #erase the ones that are done
        final_rewards += (1 - masks) * episode_rewards  #set it to the cumulative episode reward
        episode_rewards *= masks #erase the done ones
        masks = masks.type(dtype) #cuda
        if current_state.dim() == 4:  # if state is a frame/image
            current_state *= masks.unsqueeze(2).unsqueeze(2)  #[P,1,1,1]
        else:
            current_state *= masks   #restart the done ones, by setting the state to zero
        return reward, masks, final_rewards, episode_rewards, current_state



    num_frames = model_dict['num_frames']
    cuda = model_dict['cuda']
    which_gpu = model_dict['which_gpu']
    num_steps = model_dict['num_steps']
    num_processes = model_dict['num_processes']
    seed = model_dict['seed']
    env_name = model_dict['env']
    save_dir = model_dict['save_to']
    num_stack = model_dict['num_stack']
    algo = model_dict['algo']
    save_interval = model_dict['save_interval']
    log_interval = model_dict['log_interval']


    save_params = model_dict['save_params']
    vid_ = model_dict['vid_']
    gif_ = model_dict['gif_']
    ls_ = model_dict['ls_']
    vae_ = model_dict['vae_']
    grad_var_ = model_dict['grad_var_']

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(which_gpu)

    if cuda:
        torch.cuda.manual_seed(seed)
        dtype = torch.cuda.FloatTensor
        model_dict['dtype']=dtype
    else:
        torch.manual_seed(seed)
        dtype = torch.FloatTensor
        model_dict['dtype']=dtype


    # Create environments
    print (num_processes, 'processes')
    monitor_rewards_dir = os.path.join(save_dir, 'monitor_rewards')
    if not os.path.exists(monitor_rewards_dir):
        os.makedirs(monitor_rewards_dir)
        print ('Made dir', monitor_rewards_dir) 
    envs = SubprocVecEnv([make_env(env_name, seed, i, monitor_rewards_dir) for i in range(num_processes)])


    if vid_:
        print ('env for video')
        envs_video = make_env_monitor(env_name, save_dir)

    if gif_:
        print ('env for gif')
        envs_gif = make_env_basic(env_name)

    if ls_:
        print ('env for ls')
        envs_ls = make_env_basic(env_name)

    if vae_:
        print ('env for vae')
        envs_vae = make_env_basic(env_name)

    if grad_var_:
        print ('env for grad_var_')
        envs_grad_var = make_env_basic(env_name)



    obs_shape = envs.observation_space.shape  # (channels, height, width)
    obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])  # (channels*stack, height, width)
    shape_dim0 = envs.observation_space.shape[0]  #channels

    model_dict['obs_shape']=obs_shape
    model_dict['shape_dim0']=shape_dim0
    model_dict['action_size'] = envs.action_space.n
    print (envs.action_space.n, 'actions')

    # next_state_pred_ = 0
    # model_dict['next_state_pred_'] = next_state_pred_
    


    # Create agent
    if algo == 'a2c':
        agent = a2c(envs, model_dict)
        print ('init a2c agent')

    discriminator = CNN_Discriminator(model_dict).cuda()
    print ('init discriminator')

    # elif algo == 'a2c_over':
    #     agent = a2c_over(envs, model_dict)
    #     print ('init a2c_over agent')
    # elif algo == 'a2c_under':
    #     agent = a2c_under(envs, model_dict)
    #     print ('init a2c_under agent')
    # elif algo == 'ppo':
    #     agent = ppo(envs, model_dict)
    #     print ('init ppo agent')
    # elif algo == 'a2c_minibatch':
    #     agent = a2c_minibatch(envs, model_dict)
    #     print ('init a2c_minibatch agent')
    # elif algo == 'a2c_list_rollout':
    #     agent = a2c_list_rollout(envs, model_dict)
    #     print ('init a2c_list_rollout agent')
    # elif algo == 'a2c_with_var':
    #     agent = a2c_with_var(envs, model_dict)
    #     print ('init a2c_with_var agent')
    # elif algo == 'a2c_bin_mask':
    #     agent = a2c_with_var(envs, model_dict)
    #     print ('init a2c_with_var agent')
    # agent = model_dict['agent'](envs, model_dict)

    # #Load model
    # if args.load_path != '':
    #     # agent.actor_critic = torch.load(os.path.join(args.load_path))
    #     agent.actor_critic = torch.load(args.load_path).cuda()
    #     print ('loaded ', args.load_path)




    # see_reward_episode = 0
    # if 'Montez' in env_name and see_reward_episode:
    #     states_list = [[] for i in range(num_processes)]

    # view_reward_episode(model_dict=model_dict, frames=[])
    # dfasddsf

    # if vae_:
    #     vae = VAE()
    #     vae.cuda()





    # Init state
    state = envs.reset()  # (processes, channels, height, width)
    current_state = torch.zeros(num_processes, *obs_shape)  # (processes, channels*stack, height, width)
    current_state = update_current_state(current_state, state, shape_dim0).type(dtype) #add the new frame, remove oldest, since its a stack
    agent.insert_first_state(current_state) #storage has states: (num_steps + 1, num_processes, *obs_shape), set first step 

    # These are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_processes, 1]) #keeps track of current episode cumulative reward
    final_rewards = torch.zeros([num_processes, 1])

    num_updates = int(num_frames) // num_steps // num_processes
    save_interval_num_updates = int(save_interval /num_processes/num_steps)




    #Begin training
    # count =0
    start = time.time()
    start2 = time.time()
    for j in range(num_updates):
        discrim_errors = []
        discrim_errors_reverse = []
        for step in range(num_steps):

            # Act, [P,1], [P], [P,1], [P]
            state_pytorch = Variable(agent.rollouts.states[step])
            value, action, action_log_probs, dist_entropy = agent.act(state_pytorch)#, volatile=True))
            
            # Apply to Environment, S:[P,C,H,W], R:[P], D:[P]
            cpu_actions = action.data.squeeze(1).cpu().numpy() #[P]
            frame, reward, done, info = envs.step(cpu_actions) 


            # current_frame = torch.from_numpy(frame)  #[P,1,84,84]
            current_frame = torch.FloatTensor(frame)  #[P,1,84,84]
            if step ==0:
                prev_frame = torch.FloatTensor(state)  #[P,1,84,84]
            #Pred action and get error
            discrim_error = discriminator.forward(prev_frame, current_frame, action)
            discrim_errors.append(discrim_error)
            
            discrim_error_reverse = discriminator.forward(current_frame, prev_frame, action)
            discrim_errors_reverse.append(discrim_error_reverse)

            # # THIS IS TO SEE PREDICTIONS

            # if step==0:
            #     f =  np.reshape(prev_frame[0].numpy(), [84,84])
            # f =np.concatenate([f,np.reshape(current_frame[0].numpy(),[84,84])], axis=0)

            # # f1 = prev_frame[0].numpy()
            # # f2 = current_frame[0].numpy()
            # # f = np.reshape(np.concatenate([f1,f2], axis=1), [168,84])
            # # print (f.shape)
            # print (cpu_actions[0])
            # # ['NOOP', 'FIRE', 'RIGHT', 'LEFT'] for breakout
            # #for montezuma
            # #['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 
            #     #'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

            # if step ==2:
            #     print (torch.mean(current_frame-prev_frame))
            #     fdafds


            prev_frame = current_frame


            # Record rewards and update state
            reward, masks, final_rewards, episode_rewards, current_state = update_rewards(reward, done, final_rewards, episode_rewards, current_state)
            current_state = update_current_state(current_state, frame, shape_dim0)
            agent.insert_data(step, current_state, action.data, value, reward, masks, action_log_probs, dist_entropy, 0) #, done)


        # print (f.shape)
        # rows = 1
        # cols = 1
        # fig = plt.figure(figsize=(1+cols,5+rows), facecolor='white')
        # ax = plt.subplot2grid((rows,cols), (0,0), frameon=False) #, rowspan=7)
        # ax.imshow(f, cmap=plt.get_cmap('gray'))
        # ax.set_yticks([])
        # ax.set_xticks([])
        # plt.tight_layout()
        # plt.savefig(model_dict['exp_path']+'plot.png')
        # print ('plotted')
        # fadsfad
        



        discrim_errors = torch.stack(discrim_errors)  #[S,P]
        discrim_errors_reverse = torch.stack(discrim_errors_reverse)  #[S,P]

        #Optimize action-predictor
        discriminator.optimize(discrim_errors)

        #Optimize agent
        agent.update2(discrim_errors, discrim_errors_reverse)  #agent.update(j,num_updates)

        agent.insert_first_state(agent.rollouts.states[-1])


        










        # print ('save_interval_num_updates', save_interval_num_updates)
        # print ('num_updates', num_updates)
        # print ('j', j)
        total_num_steps = (j + 1) * num_processes * num_steps
        
        # if total_num_steps % save_interval == 0 and save_dir != "":
        if j % save_interval_num_updates == 0 and save_dir != "" and j != 0:

            #Save model
            if save_params:
                do_params(save_dir, agent, total_num_steps, model_dict)
            #make video
            if vid_:
                do_vid(envs_video, update_current_state, shape_dim0, dtype, agent, model_dict, total_num_steps)
            #make gif
            if gif_:
                do_gifs(envs_gif, agent, model_dict, update_current_state, update_rewards, total_num_steps)
            #make vae prob gif
            if vae_:
                do_prob_state(envs_vae, agent, model_dict, vae, update_current_state, total_num_steps)
            # #make vae prob gif
            # if grad_var_:
            #     do_grad_var(envs_grad_var, agent, model_dict, update_current_state, total_num_steps)

        #Print updates
        if j % log_interval == 0:# and j!=0:
            end = time.time()

            to_print_info_string = "{}, {}, {:.1f}/{:.1f}/{:.1f}/{:.1f}, {}, {:.1f}, {:.2f}, {:.3f}".format(j, total_num_steps,
                                       final_rewards.min(),
                                       final_rewards.median(),
                                       final_rewards.mean(),
                                       final_rewards.max(),
                                       int(total_num_steps / (end - start)),
                                       end - start,
                                       end - start2,
                                       torch.mean(discrim_errors).data.cpu().numpy()[0])

            print(to_print_info_string)


            # if vae_:
            #     elbo =  "{:.2f}".format(elbo.data.cpu().numpy()[0])


            # if next_state_pred_:
            #     state_pred_error_print =  "{:.2f}".format(agent.state_pred_error.data.cpu().numpy()[0])
            #     print(to_print_info_string+' '+state_pred_error_print+' '+elbo)
            #     to_print_legend_string = "Upts, n_timesteps, min/med/mean/max, FPS, total_T, step_T, pred_error, elbo"

            # else:
            # if vae_:
            #     print(to_print_info_string+' '+elbo)
            # else:
            # print(to_print_info_string)


            to_print_legend_string = "Upts, n_timesteps, min/med/mean/max, FPS, total_T, step_T, discrim_E"#, elbo"
            start2 = time.time()

            if j % (log_interval*30) == 0:
            
                if ls_:
                    do_ls(envs_ls, agent, model_dict, total_num_steps, update_current_state, update_rewards)

                # if grad_var_  and j % (log_interval*300) == 0:
                if grad_var_  and j % (log_interval*30) == 0:
                    #writes to file
                    do_grad_var(envs_grad_var, agent, model_dict, total_num_steps, update_current_state, update_rewards)






                # print("Upts, n_timesteps, min/med/mean/max, FPS, Time, Plot updated, LS updated")
                # print(to_print_info_string + ' LS recorded')#, agent.current_lr)
                # else:
                #update plots
                try:
                    if ls_:
                        update_ls_plot(model_dict)

                    # if grad_var_ and j % (log_interval*300) == 0:
                    if grad_var_ and j % (log_interval*30) == 0:
                        update_grad_plot(model_dict)
                        to_print_legend_string += ' grad_var_plot updated '

                    make_plots(model_dict)
                    print(to_print_legend_string + " Plot updated")
                except:
                    raise #pass
                    print(to_print_legend_string + " problem with plot")



    try:
        make_plots(model_dict)
    except:
        print ()
        # pass #raise





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--m')
    args = parser.parse_args()

    #Load model dict 
    with open(args.m, 'r') as infile:
        model_dict = json.load(infile)


    train(model_dict)




