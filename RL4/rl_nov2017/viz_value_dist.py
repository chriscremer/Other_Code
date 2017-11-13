











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

from os.path import expanduser
home = expanduser("~")

sys.path.insert(0, '../baselines/')
sys.path.insert(0, '../baselines/baselines/common/vec_env')
from subproc_vec_env import SubprocVecEnv

from envs import make_env
from envs import make_env_monitor
from envs import make_both_env_types


from agent_modular2 import a2c
from agent_modular2 import ppo
from agent_modular2 import a2c_minibatch



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from make_plots import make_plots

import argparse
import json
import subprocess





def viz(model_dict):

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

    def do_vid():
        n_vids=3
        for i in range(n_vids):
            done=False
            state = envs_video.reset()
            # state = torch.from_numpy(state).float().type(dtype)
            current_state = torch.zeros(1, *obs_shape)
            current_state = update_current_state(current_state, state, shape_dim0).type(dtype)
            # print ('Recording')
            # count=0
            while not done:
                # print (count)
                # count +=1
                # Act
                state_var = Variable(current_state, volatile=True) 
                # print (state_var.size())
                action, value = agent.act(state_var)
                cpu_actions = action.data.squeeze(1).cpu().numpy()

                # Observe reward and next state
                state, reward, done, info = envs_video.step(cpu_actions) # state:[nProcesss, ndims, height, width]
                # state = torch.from_numpy(state).float().type(dtype)
                # current_state = torch.zeros(1, *obs_shape)
                current_state = update_current_state(current_state, state, shape_dim0).type(dtype)
        state = envs_video.reset()
        
        vid_path = save_dir+'/videos/'
        count =0
        for aaa in os.listdir(vid_path):

            if 'openaigym' in aaa and '.mp4' in aaa:
                #os.rename(vid_path+aaa, vid_path+'vid_t'+str(total_num_steps)+'.mp4')
                subprocess.call("(cd "+vid_path+" && mv "+ vid_path+aaa +" "+ vid_path+env_name+'_'+algo+'_vid_t'+str(total_num_steps)+'_'+str(count) +".mp4)", shell=True) 
                count+=1
            if '.json' in aaa:
                os.remove(vid_path+aaa)




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

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(which_gpu)

    
    num_processes = 1
    model_dict['num_processes'] = 1
    
    if cuda:
        torch.cuda.manual_seed(seed)
        dtype = torch.cuda.FloatTensor
    else:
        torch.manual_seed(seed)
        dtype = torch.FloatTensor


    # Create environments
    print (num_processes, 'processes')
    # monitor_rewards_dir = os.path.join(save_dir, 'monitor_rewards')
    # if not os.path.exists(monitor_rewards_dir):
    #     os.makedirs(monitor_rewards_dir)
    #     print ('Made dir', monitor_rewards_dir) 

    monitor_rewards_dir = ''
    envs = SubprocVecEnv([make_env(env_name, seed, i, monitor_rewards_dir) for i in range(num_processes)])


    vid_ = 0
    see_frames = 1

    if vid_:
        print ('env for video')
        envs_video = make_env_monitor(env_name, save_dir)

    obs_shape = envs.observation_space.shape  # (channels, height, width)
    obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])  # (channels*stack, height, width)
    shape_dim0 = envs.observation_space.shape[0]  #channels

    model_dict['obs_shape']=obs_shape


    # Create agent
    if algo == 'a2c':
        agent = a2c(envs, model_dict)
        print ('init a2c agent')
    elif algo == 'ppo':
        agent = ppo(envs, model_dict)
        print ('init ppo agent')
    elif algo == 'a2c_minibatch':
        agent = a2c_minibatch(envs, model_dict)
        print ('init a2c_minibatch agent')
    # agent = model_dict['agent'](envs, model_dict)




    #Load model
    # if args.load_path != '':
        # agent.actor_critic = torch.load(os.path.join(args.load_path))

    # epoch_level = 1e6
    model_params_file = save_dir+ '/model_params/model_params'+str(int(epoch_level))+'.pt'
    agent.actor_critic = torch.load(model_params_file).cuda()
    print ('loaded ', model_params_file)
    # fafdas


    # frame_path = save_dir+'/frames/'
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
        print ('Made dir', frame_path) 




    # Init state
    state = envs.reset()  # (processes, channels, height, width)
    current_state = torch.zeros(num_processes, *obs_shape)  # (processes, channels*stack, height, width)
    current_state = update_current_state(current_state, state, shape_dim0).type(dtype) #add the new frame, remove oldest
    agent.insert_first_state(current_state) #storage has states: (num_steps + 1, num_processes, *obs_shape), set first step 

    # These are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_processes, 1]) #keeps track of current episode cumulative reward
    final_rewards = torch.zeros([num_processes, 1])

    num_updates = int(num_frames) // num_steps // num_processes

    #Begin training
    count =0
    start = time.time()
    for j in range(num_updates):
        for step in range(num_steps):

            # if see_frames:
            #Grayscale
            # save_frame(state, count)




            # #RGB
            # state = envs.render()
            # print(state.shape)
            # fdsafa


            values = []
            actions = []
            for ii in range(100):
                # Act, [P,1], [P,1]
                action, value = agent.act(Variable(agent.rollouts.states[step], volatile=True))
                val = value.data.cpu().numpy()[0][0]
                act_ = action.data.cpu().numpy()[0][0]
                # print ('value', val)
                # print ('action', act_)
                values.append(val)
                actions.append(act_)

            # print ('values', values)
            # print ('actions', actions)

            rows = 1
            cols = 2

            fig = plt.figure(figsize=(8,4), facecolor='white')

            # plot frame
            ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

            state1 = np.squeeze(state[0])
            ax.imshow(state1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.savefig(frame_path+'frame' +str(count)+'.png')
            # print ('saved',frame_path+'frame' +str(count)+'.png')
            # plt.close(fig)


            #plot values histogram
            ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)

            weights = np.ones_like(values)/float(len(values))
            ax.hist(values, 50, range=[0.0, 4.], weights=weights)
            # ax.set_ylim(top=1.)
            ax.set_ylim([0.,1.])

            plt_path = frame_path+'plt' 
            plt.savefig(plt_path+str(count)+'.png')
            print ('saved',plt_path+str(count)+'.png')
            plt.close(fig)
            # fsadf



            count+=1
            if count > 2:
                if done[0] or count > max_frames:
                    ffsdfa





                # action, value = agent.act(Variable(agent.rollouts.states[step], volatile=True))
                # print ('value', value)
                # print ('action', action)

                # action, value = agent.act(Variable(agent.rollouts.states[step], volatile=True))
                # print ('value', value)
                # print ('action', action)


            
            cpu_actions = action.data.squeeze(1).cpu().numpy() #[P]

            # Step, S:[P,C,H,W], R:[P], D:[P]
            state, reward, done, info = envs.step(cpu_actions) 



            # Record rewards
            reward, masks, final_rewards, episode_rewards, current_state = update_rewards(reward, done, final_rewards, episode_rewards, current_state)
            
            # Update state
            current_state = update_current_state(current_state, state, shape_dim0)

            # Agent record step
            agent.insert_data(step, current_state, action.data, value.data, reward, masks)



        # #Optimize agent
        # agent.update()  #agent.update(j,num_updates)
        # agent.insert_first_state(agent.rollouts.states[-1])




        total_num_steps = (j + 1) * num_processes * num_steps

        # #Save model
        # if total_num_steps % save_interval == 0 and save_dir != "":
        #     save_path = os.path.join(save_dir, 'model_params')
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass
        #     # A really ugly way to save a model to CPU
        #     save_model = agent.actor_critic
        #     if cuda:
        #         save_model = copy.deepcopy(agent.actor_critic).cpu()
        #     # torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
        #     # steps_sci_nota = '{e}'.format(total_num_steps)
        #     save_to=os.path.join(save_path, "model_params" + str(total_num_steps)+".pt")
        #     # save_to=os.path.join(save_path, "model_params" + steps_sci_nota+".pt")
        #     torch.save(save_model, save_to)
        #     print ('saved', save_to)

        #     #make video
        #     if vid_:
        #         do_vid()


        # #Print updates
        # if j % log_interval == 0:
        #     end = time.time()

        #     # if j % (log_interval*30) == 0:

        #         # #update plots
        #         # try:
        #         #     make_plots(model_dict)
        #         #     print("Upts, n_timesteps, min/med/mean/max, FPS, Time, Plot updated")
        #         # except:
        #         #     raise
        #         #     print("Upts, n_timesteps, min/med/mean/max, FPS, Time")

        #     print("{}, {}, {:.1f}/{:.1f}/{:.1f}/{:.1f}, {}, {:.1f}".
        #             format(j, total_num_steps,
        #                    final_rewards.min(),
        #                    final_rewards.median(),
        #                    final_rewards.mean(),
        #                    final_rewards.max(),
        #                    int(total_num_steps / (end - start)),
        #                    end - start))#, agent.current_lr)
    
    # try:
    #     make_plots(model_dict)
    # except:
    #     print ()
        # pass #raise





if __name__ == "__main__":


    # parser = argparse.ArgumentParser()
    # parser.add_argument('--m')
    # args = parser.parse_args()

    # for fiels in dir:
    #     if model_dict

    max_frames = 200
    epoch_level = 6e6

    exp_name_ = 'a2c_reg_and_dropout_pong2' #'confirm_a2c_dropout'
    model_name_ =  'a2c_dropout'  # 'a2c'# 
    env_name_ = 'PongNoFrameskip-v4' #'BreakoutNoFrameskip-v4/'
    exp_location = home + '/Documents/tmp/' + exp_name_+'/'
    env_model_seed_path = exp_location +env_name_+'/' +model_name_+'/seed0/'
    model_dict_path = env_model_seed_path + 'model_dict.json'



    #Load model dict 
    with open(model_dict_path, 'r') as infile:
        model_dict = json.load(infile)

    print ('loaded model dict')
    # fafdf

    frame_path = model_dict['save_to']+'/frames_'+ model_name_+ '_'+env_name_+ '_'+ str(int(epoch_level)) +'/'

    viz(model_dict)





