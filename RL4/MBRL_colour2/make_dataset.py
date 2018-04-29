

import os, sys
from os.path import expanduser
home = expanduser("~")

for i in range(len(sys.path)):
    if 'er/Documents' in sys.path[i]:
        sys.path.remove(sys.path[i])#[i]
        break

sys.path.insert(0, '../baselines/')

import numpy as np

import torch
from torch.autograd import Variable

import json
import pickle

sys.path.insert(0, './utils/')
from envs import make_env, make_env_monitor, make_env_basic

from actor_critic_networks import CNNPolicy


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



make_ = 0
view_ = 1



exp_name = 'RoadRunner_colour_4'
env_name = 'RoadRunner'
env_name2 = env_name +'NoFrameskip-v4'
exp_dir = home+'/Documents/tmp/' + exp_name



if make_:


    #load experiemetn dict
    print ('load experiment dict')
    dict_location = exp_dir + '/' +env_name+ 'NoFrameskip-v4/A2C/seed0/model_dict.json'
    with open(dict_location, 'r') as outfile:
        exp_dict = json.load(outfile)







    #Init policy , not agent
    print ('init policy')
    policy = CNNPolicy(2*3, 18)   #frames*channels, action size

    #load params
    # param_file = exp_dir + '/' +env_name+ 'NoFrameskip-v4/A2C/seed0/model_params3/model_params2000000.pt'    
    param_file = exp_dir + '/' +env_name+ 'NoFrameskip-v4/A2C/seed0/model_params3/model_params3999840.pt'    
    param_dict = torch.load(param_file)

    policy.load_state_dict(param_dict)
    # policy = torch.load(param_file).cuda()
    print ('loaded params', param_file)
    policy.cuda()









    #init env
    env = make_env_basic(env_name2)

    obs_shape = env.observation_space.shape  # (channels, height, width)
    obs_shape = (obs_shape[0] * exp_dict['num_stack'], *obs_shape[1:])  # (channels*stack, height, width)
    shape_dim0 = env.observation_space.shape[0]  #channels

    n_channels = shape_dim0
    n_stack = exp_dict['num_stack']

    # print (obs_shape)   #6,210,160
    # # print (shape_dim0)  #3
    # dfsfa



    # init states
    def update_current_state(current_state, state, channels):
        # current_state: [processes, channels*stack, height, width]
        state = torch.from_numpy(state).float()  # (processes, channels, height, width)
        # if num_stack > 1:
        #first stack*channel-channel frames = last stack*channel-channel , so slide them forward
        current_state[:, :-channels] = current_state[:, channels:] # first 3 equal last three
        current_state[:, -channels:] = state #last frame is now the new one  , last 3 equal state
        return current_state  #[1,6,210,160]


    state = env.reset() #[3,210,160]
    num_processes = 1
    current_state = torch.zeros(num_processes, *obs_shape)  # (processes, channels*stack, height, width)
    current_state = update_current_state(current_state, state, n_channels).cuda() #.type(dtype) #add the new frame, remove oldest



    # img = state
    # print (np.sum(img))
    # img = np.uint8(current_state[0][3:].cpu().numpy())
    # print (np.sum(img))

    # ffdas

    # img = current_state[0][:3].cpu().numpy()


    # print (img.shape)  #[3,210,160]
    # img = np.rollaxis(img, 1, 0)
    # img = np.rollaxis(img, 2, 1)
    # print (img.shape) #[210,160,3]


    # img2 = np.copy(img)
    # img2 = img

    # #2
    # img2[:,:,0] = img[:,:,1] 
    # img2[:,:,1] = img[:,:,2] 
    # img2[:,:,2] = img[:,:,0] 

    # # 3
    # # print (np.sum(img[:,:,0] ))
    # img2[:,:,0] = img[:,:,1] 
    # # print (np.sum(img[:,:,0] ))
    # # fadsfa
    # img2[:,:,1] = img[:,:,2] 
    # img2[:,:,2] = img[:,:,0] 



    # # 4
    # img2[:,:,0] = img[:,:,2] 
    # img2[:,:,2] = img[:,:,0] 


    # # 5
    # img2[:,:,0] = img[:,:,2] 
    # img2[:,:,1] = img[:,:,0] 
    # img2[:,:,2] = img[:,:,1] 


    # # 6
    # img2[:,:,0] = img[:,:,1] 
    # img2[:,:,1] = img[:,:,2] 
    # img2[:,:,2] = img[:,:,0] 



    # # 7
    # img2[:,:,0] = img[:,:,1] 
    # img2[:,:,1] = img[:,:,0] 
    # img2[:,:,2] = img[:,:,2] 


    # # 8
    # # img2[:,:,0] = img[:,:,2] 
    # img2[:,:,1] = img[:,:,2] 
    # img2[:,:,2] = img[:,:,1] 


    # # 9
    # img2[:,:,0] = img[:,:,1] 
    # img2[:,:,1] = img[:,:,0] 
    # # img2[:,:,2] = img[:,:,1] 


    # plt.imshow(img)
    # save_dir = exp_dir
    # plt_path = save_dir+'/frame1.png'
    # plt.savefig(plt_path)
    # print ('saved viz',plt_path)
    # plt.close()
    # fsdfas



    # state = np.expand_dims(state, 0)
    # print (state.shape)
    # print (current_state.size())  #[1,6,210,160]
    # fad

    # list of lists, where lists are trajectories. trajectories have actinos and states 
    dataset = []
    # tmp_trajs = [[] for x in range(num_processes)]
    tmp_traj = []
    dataset_count = 0
    # done = [0]*num_processes

    done = 0


    print ('Make dataset')
    #run agent, store states, save dataset
    while 1:

        # Act, [P,1], [P], [P,1], [P]
        # value, action = agent.act(Variable(agent.rollouts.states[step], volatile=True))
        value, action, action_log_probs, dist_entropy = policy.act(Variable(current_state))#, volatile=True))
        # print (action_log_probs.size())
        # print (dist_entropy.size())


        # cpu_actions = action.data.cpu().numpy() #[P]
        # print (actions.size())

        # y = torch.LongTensor(batch_size,1).random_() % nb_digits
        # # One hot encoding buffer that you create out of the loop and just keep reusing
        # y_onehot = torch.FloatTensor(batch_size, nb_digits)
        # # In your for loop
        # y_onehot.zero_()
        # y_onehot.scatter_(1, y, 1)

        states_ = current_state.cpu().numpy()  #[P,S,84,84]
        # print (state_t.shape)
        actions_ = action.data.cpu().numpy() #[P,1]

        # print (actions_.shape)

        # fsd
        # print (action)
        # fdsaf


        #store step
        # for proc in range(num_processes):

        #add states
        # state_t = states_[proc]
        # action_t = actions_[proc]
        # tmp_trajs[proc].append([action_t, state_t])
        tmp_traj.append([states_[0], actions_[0]])

        # if done[proc]:
        if done:
            dataset.append(tmp_traj)
            dataset_count += len(tmp_traj)
            print (len(tmp_traj), dataset_count)
            tmp_traj = []
            done = 0
            state = env.reset() 
            current_state = torch.zeros(num_processes, *obs_shape)  # (processes, channels*stack, height, width)
            current_state = update_current_state(current_state, state, shape_dim0).cuda() #.type(dtype) #add the new frame, remove oldest

        else:
            cpu_actions = action.data.squeeze(1).cpu().numpy() #[P]
            # Step, S:[P,C,H,W], R:[P], D:[P]
            state, reward, done, info = env.step(cpu_actions) 
            current_state = update_current_state(current_state, state, shape_dim0).cuda() 


            # for ii in range(len(dataset)):
            #     print (len(dataset[ii]))


        if dataset_count > 10:

            # img = np.uint8( dataset[0][1][0])
            # img = img[3:]
            # print (img.shape)
            # img = np.rollaxis(img, 1, 0)
            # img = np.rollaxis(img, 2, 1)
            # print (img.shape)
            # plt.imshow(img)
            # save_dir = exp_dir
            # plt_path = save_dir+'/frame1.png'
            # plt.savefig(plt_path)
            # print ('saved viz',plt_path)
            # plt.close()
            # fsdfas


            # pickle.dump( dataset, open(home+'/Documents/tmp/breakout_2frames/breakout_trajectories_10000.pkl', "wb" ) )
            pickle.dump( dataset, open(home+'/Documents/tmp/'+exp_name+'/trajectories_10.pkl', "wb" ) )

            print('saved')
            # pickle.save(dataset)
            break








    # print (done)
    # fdsa







# #viz some of the dataset. and some grads of the policy.

# img = dataset[0][0]
# # img = np.reshape(img, [240,320,3])
# img = np.rollaxis(img, 1, 0)
# img = np.rollaxis(img, 2, 1)
# print (img.shape)

# plt.imshow(img)

# save_dir = home+'/Documents/tmp/Doom/'
# plt_path = save_dir+'frmame.png'
# plt.savefig(plt_path)
# print ('saved viz',plt_path)
# plt.close(fig)




if view_:



    print ("Load dataset")
    dir_ = 'RoadRunner_colour_4'
    # dataset = pickle.load( open( home+'/Documents/tmp/breakout_2frames/breakout_trajectories_10000.pkl', "rb" ) )
    dataset = pickle.load( open( home+'/Documents/tmp/'+dir_+'/trajectories_10.pkl', "rb" ) )

    # img = np.uint8( dataset[0][1][0])
    # img = img[3:]
    # print (img.shape)
    # img = np.rollaxis(img, 1, 0)
    # img = np.rollaxis(img, 2, 1)
    # print (img.shape)
    # plt.imshow(img)
    # save_dir = exp_dir
    # plt_path = save_dir+'/frame1.png'
    # plt.savefig(plt_path)
    # print ('saved viz',plt_path)
    # plt.close()
    # fsdfas


    rows = 5
    cols = 1

    # frame_numb = 448


    fig = plt.figure(figsize=(2+cols,4+rows), facecolor='white', dpi = 210*rows)

    for i in range(rows):

        # frame_idx = np.random.randint(0,20)
        frame_idx = i+1

        print (i, frame_idx)

        # frame_idx = frame_numb + i

        # traj_ind = 0
        # start_ind = 2
        # frame = torch.from_numpy(np.array([dataset[traj_ind][start_ind+i][1]])).float()[0].numpy()

        # frame = state_dataset[frame_idx]
        frame = dataset[0][frame_idx][0]

        # frame_pytorch = Variable(torch.from_numpy(np.array([frame])).cuda())
        # mask = model.predict_mask(frame_pytorch)
        # mask = mask.repeat(1,3,1,1)
        # masked_frame = frame * mask.data.cpu().numpy()[0]

        # masked_frame = masked_frame

        # print (frame)
        # fds

        # #scale back up 
        # frame = frame *255.
        # masked_frame = masked_frame *255.


        # print (frame)

        print (frame.shape)  #[6,210,160]
        print (np.max(frame))  #252
        # print (masked_frame.shape)#[6,210,160]
        # print (np.max(masked_frame)) #127


        # need to check sizes, and pot porperly

        frame = frame[:3]
        # frame = frame[0:6:2]

        frame = np.rollaxis(frame, 1, 0)
        frame = np.rollaxis(frame, 2, 1)

        # frame = np.reshape(frame, [2,3,210,160])
        # frame = np.transpose(frame, [0,2,3,1])

        # frame = np.transpose(frame, [1,2,0])

        print (frame.shape)
        # print (frame[0].shape)

        # print (np.concatenate([frame[0], frame[1]] , axis=1).shape)

        # fsdf

        # Plot real frame
        ax = plt.subplot2grid((rows,cols), (i,0), frameon=False)

        # frame = np.concatenate([frame[0], frame[1]] , axis=1)
        # print (frame.shape)
        print ()

        # ax.imshow(state1) #, cmap='gray')
        ax.imshow(np.uint8(frame), interpolation='none') #, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        # ax.text(0.4, 1.04, 'Real Frame', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, family='serif', size=6)
        if i==0:
            ax.text(0.4, 1.04, 'Real Frame', transform=ax.transAxes, family='serif', size=6)


    save_dir = home+'/Documents/tmp/'+dir_
    plt_path = save_dir+'/viz2.png'
    plt.savefig(plt_path)
    print ('saved viz',plt_path)
    plt.close(fig)





print ('Done.')


























