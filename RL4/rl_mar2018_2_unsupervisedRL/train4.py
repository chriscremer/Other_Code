




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

from a2c_agents import a2c_with_context as a2c

from discriminator import CNN_Discriminator #as discriminator


# from a2c_agents import a2c_over
# from a2c_agents import a2c_under



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
    action_space = envs.action_space

    model_dict['action_space']=action_space
    model_dict['obs_shape']=obs_shape
    model_dict['shape_dim0']=shape_dim0

    next_state_pred_ = 0
    model_dict['next_state_pred_'] = next_state_pred_

    n_contexts = 2
    model_dict['n_contexts'] = n_contexts

    

    # Create agent
    # if algo == 'a2c':
    agent = a2c(model_dict)
    print ('init a2c agent')

    discriminator = CNN_Discriminator(num_steps,n_contexts,model_dict).cuda()
    print ('init discriminator')




    # Init state
    state = envs.reset()  # (processes, channels, height, width)
    current_state = torch.zeros(num_processes, *obs_shape)  # (processes, channels*stack, height, width)
    current_state = update_current_state(current_state, state, shape_dim0).type(dtype) #add the new frame, remove oldest, since its a stack
    agent.insert_first_state(current_state) #storage has states: (num_steps + 1, num_processes, *obs_shape), set first step 

    # These are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_processes, 1]) #keeps track of current episode cumulative reward
    final_rewards = torch.zeros([num_processes, 1]) #when episode complete, sotres it here

    num_updates = int(num_frames) // num_steps // num_processes
    save_interval_num_updates = int(save_interval /num_processes/num_steps)


    context_probs = torch.ones(n_contexts) / n_contexts


    #Begin training
    # count =0
    start = time.time()
    start2 = time.time()
    for j in range(num_updates):


        context_np = np.random.choice(n_contexts, num_processes)
        context = torch.from_numpy(context_np).view(num_processes,1)
        context_onehot = torch.FloatTensor(num_processes, n_contexts).zero_()
        context_onehot.scatter_(1, context, 1)  # [P,C]


        list_frames = []
        for step in range(num_steps):

            #Sample context
            # context = torch.unsqueeze(context_probs.multinomial(num_processes), dim=1) # [P,1]

            # print (torch.multinomial.sample(context_probs, num_processes))

            # print (np.random.multinomial(num_processes, [1./n_contexts]*n_contexts))
            # print (np.random.choice(n_contexts, num_processes)) #[1./n_contexts]*n_contexts))


            # Act, [P,1], [P], [P,1], [P]
            state_pytorch = Variable(agent.rollouts.states[step])
            value, action, action_log_probs, dist_entropy = agent.act(state_pytorch, context_onehot)#, volatile=True))

            

            # print (context_np)
            # print (action)

            #ACTIONS 
            #['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 
                #'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

            # TO FIX THE ACTIONS
            # action = action.data
            # # print (action)
            # # fdasf
            # for i in range(len(context_np)):
            #     if context_np[i] == 0:
            #         action[i] = 4
            #     else:
            #         action[i] = 3
            # action = Variable(action)
            # # print (action)
            # # print (action)
            # # fadsf


            # # TO FIX THE ACTIONS 2
            # action = action.data
            # # print (action)
            # # fdasf
            # for i in range(len(action)):
            #     if action[i].cpu().numpy() >= 8:
            #         action[i] = 0
            #     # else:
            #     #     action[i] = 3
            # action = Variable(action)
            # # print (action)
            # # print (action)
            # # fadsf     
            
            # Apply to Environment, S:[P,C,H,W], R:[P], D:[P]
            cpu_actions = action.data.squeeze(1).cpu().numpy() #[P]
            # print (cpu_actions)
            state, reward, done, info = envs.step(cpu_actions) 





            # print (state.shape) #[P,1,84,84]
            list_frames.append(torch.FloatTensor(state))

            # Record rewards and update state
            reward, masks, final_rewards, episode_rewards, current_state = update_rewards(reward, done, final_rewards, episode_rewards, current_state)
            current_state = update_current_state(current_state, state, shape_dim0)
            agent.insert_data(step, current_state, action.data, value, reward, masks, action_log_probs, dist_entropy, 0) #, done)



        #Optimize discriminator
        if j % 2==0:
            discriminator_error = discriminator.update(list_frames, context) #[P]
            if j ==0:
                print ('multiple updates')
                for jj in range(20):
                    discriminator_error = discriminator.update(list_frames, context)
                    # print (torch.mean(discriminator_error).data.cpu().numpy()[0])
                # fasds

        


        grad_sum = agent.actor_critic.graddd(state_pytorch, context_onehot)


        #Optimize agent
        # agent.update(context_onehot, discriminator_error)  #agent.update(j,num_updates)
        agent.update2(context_onehot, discriminator_error, grad_sum)  #agent.update(j,num_updates)


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

            # print (torch.mean(discriminator_error).data.cpu().numpy()[0])
            to_print_info_string = "{}, {}, {:.1f}/{:.1f}/{:.1f}/{:.1f}, {}, {:.1f}, {:.2f}, {:.3f}".format(j, total_num_steps,
                                       final_rewards.min(),
                                       final_rewards.median(),
                                       final_rewards.mean(),
                                       final_rewards.max(),
                                       int(total_num_steps / (end - start)),
                                       end - start,
                                       end - start2,
                                       torch.mean(discriminator_error).data.cpu().numpy()[0])

            print(to_print_info_string)


            to_print_legend_string = "Upts, n_timesteps, min/med/mean/max, FPS, total_T, step_T, D_error"#, elbo"
            start2 = time.time()

            if j % (log_interval*30) == 0:
            
                if ls_:
                    do_ls(envs_ls, agent, model_dict, total_num_steps, update_current_state, update_rewards)

                # if grad_var_  and j % (log_interval*300) == 0:
                if grad_var_  and j % (log_interval*30) == 0:
                    #writes to file
                    do_grad_var(envs_grad_var, agent, model_dict, total_num_steps, update_current_state, update_rewards)



                # # THIS IS TO SEE PREDICTIONS

                # nstep_frames = torch.stack(list_frames)  #[N, P, 1,84,84]  
                # nstep_frames = torch.transpose(nstep_frames, 0,1)
                # nstep_frames = torch.squeeze(nstep_frames) #its [P,N,84,84] so its like a batch of N dimensional images
                # nstep_frames = Variable(nstep_frames).cuda()
                # pred = F.softmax(discriminator.predict(nstep_frames), dim=1)
                # print (pred, context_np)

                # rows = 1
                # cols = 2
                # fig = plt.figure(figsize=(1+cols,15+rows), facecolor='white')
                

                # zero_comp = 0
                # one_comp = 0
                # for ii in range(len(context_np)):

                #     if context_np[ii] == 0 and not zero_comp:
                #         print (ii)
                #         imgg = nstep_frames[ii].view(num_steps*84,84).data.cpu().numpy()


                #         # imgg = nstep_frames[ii].view(num_steps*84//2,84*2)
                #         # # imgg = nstep_frames[ii].view(num_steps*84,84)
                #         # imgg = imgg.data.cpu().numpy()

                #         ax = plt.subplot2grid((rows,cols), (0,0), frameon=False) #, rowspan=7)

                #         # print (imgg.shape)
                #         # plt.imshow(imgg, cmap=plt.get_cmap('gray'))
                #         ax.imshow(imgg, cmap=plt.get_cmap('gray'))
                #         ax.set_yticks([])
                #         # plt.savefig(model_dict['exp_path']+'img0.pdf')
                #         # print (model_dict['exp_path']+'img.png')
                #         zero_comp =1
                #     if context_np[ii] == 1 and not one_comp:
                #         print (ii)
                #         imgg = nstep_frames[ii].view(num_steps*84,84).data.cpu().numpy()

                #         # imgg = nstep_frames[ii].view(num_steps*84//2,84*2)
                #         # # imgg = nstep_frames[ii].view(num_steps*84,84)
                #         # imgg = imgg.data.cpu().numpy()

                #         ax = plt.subplot2grid((rows,cols), (0,1), frameon=False) #, rowspan=7)
                #         # print (imgg.shape)
                #         # plt.imshow(imgg, cmap=plt.get_cmap('gray'))
                #         ax.imshow(imgg, cmap=plt.get_cmap('gray'))
                #         ax.set_yticks([])

                #         # plt.savefig(model_dict['exp_path']+'img1.pdf')
                #         # print (model_dict['exp_path']+'img.png')
                #         one_comp =1
                #     if zero_comp and one_comp:
                #         print ('plotted both')

                #         # imgg = nstep_frames[20].view(num_steps*84,84).data.cpu().numpy()
                #         # plt.imshow(imgg, cmap=plt.get_cmap('gray'))
                #         # plt.savefig(model_dict['exp_path']+'img_20.pdf')


                #         # fdfaa  fig = plt.figure(figsize=(4+cols,1+rows), facecolor='white')

                #         plt.savefig(model_dict['exp_path']+'img_both.pdf')

                #         ffasd








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




