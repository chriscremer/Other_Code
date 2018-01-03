



# run a number of episodes storing everything
# once episode complete, compute actual returns
# make figs for each frame
# make gif








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
    model_dict['num_steps'] = max_frames
    num_steps = max_frames
    
    if cuda:
        torch.cuda.manual_seed(seed)
        dtype = torch.cuda.FloatTensor
    else:
        torch.manual_seed(seed)
        dtype = torch.FloatTensor


    # Create environments
    print (num_processes, 'processes')

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


        #         def get_action_meanings(self):
        # return [ACTION_MEANING[i] for i in self._action_set]

            # print (envs.get_action_meanings())

            # print (agent.rollouts.states[step].size())


            

            # print ('values', values)
            # print ('actions', actions)





            # rows = 1
            # cols = 3

            # fig = plt.figure(figsize=(8,4), facecolor='white')

            # # plot frame
            # ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

            # state1 = np.squeeze(state[0])
            # ax.imshow(state1, cmap='gray')
            # ax.set_xticks([])
            # ax.set_yticks([])
            # # ax.savefig(frame_path+'frame' +str(count)+'.png')
            # # print ('saved',frame_path+'frame' +str(count)+'.png')
            # # plt.close(fig)
            # ax.set_title('State',family='serif')





            # #plot values histogram
            # ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)

            # values = []
            # actions = []
            # for ii in range(100):
            #     # Act, [P,1], [P,1]
            #     action, value = agent.act(Variable(agent.rollouts.states[step], volatile=True))
            #     val = value.data.cpu().numpy()[0][0]
            #     act_ = action.data.cpu().numpy()[0][0]
            #     # print ('value', val)
            #     # print ('action', act_)
            #     values.append(val)
            #     actions.append(act_)


            # weights = np.ones_like(values)/float(len(values))
            # ax.hist(values, 50, range=[0.0, 4.], weights=weights)
            # # ax.set_ylim(top=1.)
            # ax.set_ylim([0.,1.])

            # ax.set_title('Value',family='serif')







            # #plot actions
            # ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)

            # action_prob = agent.actor_critic.action_dist(Variable(agent.rollouts.states[step], volatile=True))
            # action_prob = np.squeeze(action_prob.data.cpu().numpy())
            # action_size = envs.action_space.n

            # # print (action_prob.shape)

            # ax.bar(range(action_size), action_prob)

            # ax.set_title('Action',family='serif')
            # # ax.set_xticklabels(['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'])
            # plt.xticks(range(action_size),['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'R_FIRE', 'L_FIRE'], fontsize=6)
            # ax.set_ylim([0.,1.])



            # # print (action_prob)
            # # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
            # # fdsfas

            # plt.tight_layout(pad=3., w_pad=2.5, h_pad=1.0)

            # plt_path = frame_path+'plt' 
            # plt.savefig(plt_path+str(count)+'.png')
            # print ('saved',plt_path+str(count)+'.png')
            # plt.close(fig)
            # # fsadf




            count+=1
            if count % 10 ==0:
                print (count)

            if count > 2:
                if reward.cpu().numpy() > 0:
                    # print (, reward.cpu().numpy(), count)
                    print (done[0],masks.cpu().numpy(), reward.cpu().numpy(),'reward!!', step)
                    print (np.squeeze(agent.rollouts.rewards.cpu().numpy()))
                else:
                    print (done[0],masks.cpu().numpy(), reward.cpu().numpy())


                # if done[0] or count > max_frames:
                if count > max_frames:

                    next_value = agent.actor_critic(Variable(agent.rollouts.states[-1], volatile=True))[0].data
                    agent.rollouts.compute_returns(next_value, agent.use_gae, agent.gamma, agent.tau)

                    rollouts_ =  np.squeeze(agent.rollouts.returns.cpu().numpy())
                    rewards_ =  np.squeeze(agent.rollouts.rewards.cpu().numpy())
                    # rollouts_ =  np.squeeze(agent.rollouts.returns.cpu().numpy())
                    # rollouts_ =  np.squeeze(agent.rollouts.returns.cpu().numpy())


                    for jj in range(len(rollouts_)):

                        print (jj, rollouts_[jj], rewards_[jj])
                    ffsdfa






                # action, value = agent.act(Variable(agent.rollouts.states[step], volatile=True))
                # print ('value', value)
                # print ('action', action)

                # action, value = agent.act(Variable(agent.rollouts.states[step], volatile=True))
                # print ('value', value)
                # print ('action', action)


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


            # print (reward)






        total_num_steps = (j + 1) * num_processes * num_steps





if __name__ == "__main__":


    # parser = argparse.ArgumentParser()
    # parser.add_argument('--m')
    # args = parser.parse_args()

    # for fiels in dir:
    #     if model_dict

    max_frames = 200
    epoch_level = 1e6

    exp_name_ = 'a2c_reg_and_dropout' #'a2c_reg_and_dropout_pong2' #'confirm_a2c_dropout'
    model_name_ =  'a2c_dropout'  # 'a2c'# 
    env_name_ = 'BreakoutNoFrameskip-v4/'# 'PongNoFrameskip-v4' #
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





    dfadf
    #from make_gif.py


    import imageio


    import os
    from os.path import expanduser
    home = expanduser("~")


    # dir_ = home+ '/Downloads/frames_a2c_dropout_6000000'
    dir_ = home+ '/Documents/tmp/a2c_reg_and_dropout_pong2/PongNoFrameskip-v4/a2c_dropout/seed0/frames_a2c_dropout_PongNoFrameskip-v4_6000000'


    print('making gif')
    max_epoch = 0
    for file_ in os.listdir(dir_):
        if 'plt' in file_:
            numb = file_.split('plt')[1].split('.')[0]
            numb = int(numb)
            if numb > max_epoch:
                max_epoch = numb
            # print (numb)
            # fadsfa

    # print ('max_epoch', max_epoch)
    # fadad


    images = []
    # for file_ in os.listdir(dir_):
    for i in range(max_epoch+1):
        # print(file_)
        # fsdfa

        # images.append(imageio.imread(dir_+'/'+file_))
        images.append(imageio.imread(dir_+'/'+'plt'+str(i)+'.png'))

        
    imageio.mimsave(dir_+'/movie.gif', images)
    print ('made gif', dir_+'/movie.gif')






























