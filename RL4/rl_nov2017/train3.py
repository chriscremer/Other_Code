




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

parser = argparse.ArgumentParser()
parser.add_argument('--m')
args = parser.parse_args()

#Load model dict 
with open(args.m, 'r') as infile:
    model_dict = json.load(infile)


def train(model_dict):

    def update_current_state(current_state, state, channels):
        # current_state: [processes, channels*stack, height, width]
        state = torch.from_numpy(state).float()  # (processes, channels, height, width)
        # if num_stack > 1:
        #first stack*channel-channel frames = last stack*channel-channel , so slide them forward
        current_state[:, :-channels] = current_state[:, channels:] 
        current_state[:, -channels:] = state #last frame is now the new one

        # if see_frames:
        #     #Grayscale
        #     save_frame(state, count)
        #     count+=1
        #     if done[0]:
        #         ffsdfa
        #     #RGB
        #     state = envs.render()
        #     print(state.shape)
        #     fdsafa

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

    def save_frame(state, count):

        frame_path = save_dir+'/frames/'
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
            print ('Made dir', frame_path) 

        state1 = np.squeeze(state[0])
        # print (state1.shape)
        fig = plt.figure(figsize=(4,4), facecolor='white')
        plt.imshow(state1, cmap='gray')
        plt.savefig(frame_path+'frame' +str(count)+'.png')
        print ('saved',frame_path+'frame' +str(count)+'.png')
        plt.close(fig)



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

    

    
    if cuda:
        torch.cuda.manual_seed(seed)
        dtype = torch.cuda.FloatTensor
    else:
        torch.manual_seed(seed)
        dtype = torch.FloatTensor


    # Create environments
    print (num_processes, 'processes')
    monitor_rewards_dir = os.path.join(save_dir, 'monitor_rewards')
    if not os.path.exists(monitor_rewards_dir):
        os.makedirs(monitor_rewards_dir)
        print ('Made dir', monitor_rewards_dir) 
    envs = SubprocVecEnv([make_env(env_name, seed, i, monitor_rewards_dir) for i in range(num_processes)])


    vid_ = 1
    see_frames = 0

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

    # #Load model
    # if args.load_path != '':
    #     # agent.actor_critic = torch.load(os.path.join(args.load_path))
    #     agent.actor_critic = torch.load(args.load_path).cuda()
    #     print ('loaded ', args.load_path)

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
    # count =0
    start = time.time()
    for j in range(num_updates):
        for step in range(num_steps):

            # Act, [P,1], [P,1]
            action, value = agent.act(Variable(agent.rollouts.states[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy() #[P]

            # Step, S:[P,C,H,W], R:[P], D:[P]
            state, reward, done, info = envs.step(cpu_actions) 

            # Record rewards
            reward, masks, final_rewards, episode_rewards, current_state = update_rewards(reward, done, final_rewards, episode_rewards, current_state)
            
            # Update state
            current_state = update_current_state(current_state, state, shape_dim0)

            # Agent record step
            agent.insert_data(step, current_state, action.data, value.data, reward, masks)



        #Optimize agent
        agent.update(j,num_updates)
        agent.insert_first_state(agent.rollouts.states[-1])




        total_num_steps = (j + 1) * num_processes * num_steps

        #Save model
        if total_num_steps % save_interval == 0 and save_dir != "":
            save_path = os.path.join(save_dir, 'model_params')
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            # A really ugly way to save a model to CPU
            save_model = agent.actor_critic
            if cuda:
                save_model = copy.deepcopy(agent.actor_critic).cpu()
            # torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
            # steps_sci_nota = '{e}'.format(total_num_steps)
            save_to=os.path.join(save_path, "model_params" + str(total_num_steps)+".pt")
            # save_to=os.path.join(save_path, "model_params" + steps_sci_nota+".pt")
            torch.save(save_model, save_to)
            print ('saved', save_to)

            #make video
            if vid_:
                do_vid()


        #Print updates
        if j % log_interval == 0:
            end = time.time()

            if j % (log_interval*30) == 0:

                #update plots
                try:
                    make_plots(model_dict)
                    print("Upts, n_timesteps, min/med/mean/max, FPS, Time, Plot updated")
                except:
                    raise
                    print("Upts, n_timesteps, min/med/mean/max, FPS, Time")

            print("{}, {}, {:.1f}/{:.1f}/{:.1f}/{:.1f}, {}, {:.1f}".
                    format(j, total_num_steps,
                           final_rewards.min(),
                           final_rewards.median(),
                           final_rewards.mean(),
                           final_rewards.max(),
                           int(total_num_steps / (end - start)),
                           end - start), agent.current_lr)
    
    try:
        make_plots(model_dict)
    except:
        print ()
        # pass #raise





if __name__ == "__main__":
    train(model_dict)




