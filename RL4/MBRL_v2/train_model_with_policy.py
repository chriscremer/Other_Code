









import numpy as np

import pickle


import os
from os.path import expanduser
home = expanduser("~")



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# from vae import VAE
from vae_with_policy import VAE

import sys
sys.path.insert(0, './utils/')


# from a2c_agents import a2c
from actor_critic_networks import CNNPolicy

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from train_utils import load_params_v3


dataset = pickle.load( open( home+'/Documents/tmp/breakout_2frames/breakout_trajectories_10000.pkl', "rb" ) )


for ii in range(len(dataset)):
    print (len(dataset[ii]))
    # dataset[ii] = dataset[ii] / 255.


#scale data
for i in range(len(dataset)):
    for t in range(len(dataset[i])):
        dataset[i][t][1] = dataset[i][t][1] / 255.
        # state_dataset.append(dataset[i][t][1]) 


# dataset: trajectories: timesteps: (action,state) state: [2,84,84]

print (len(dataset))
print (len(dataset[ii][0])) # single timepoint
print (dataset[ii][0][0].shape)  #action [1]           a_t+1
print (dataset[ii][0][1].shape)     #state [2,84,84]   s_t


state_dataset = []
for i in range(len(dataset)):
    for t in range(len(dataset[i])):
        state_dataset.append(dataset[i][t][1]) #  /255.)

print (len(state_dataset))



print ('Init Policy')

policy = CNNPolicy(2, 4) #.cuda()
# agent = a2c(model_dict)
# param_file = home+'/Documents/tmp/breakout_2frames/BreakoutNoFrameskip-v4/A2C/seed0/model_params/model_params9999360.pt'

load_policy = 1

if load_policy:
    # param_file = home+'/Documents/tmp/breakout_2frames_leakyrelu2/BreakoutNoFrameskip-v4/A2C/seed0/model_params3/model_params3999840.pt'
    param_file = home+'/Documents/tmp/breakout_2frames_leakyrelu2/BreakoutNoFrameskip-v4/A2C/seed0/model_params3/model_params9999360.pt'    

    param_dict = torch.load(param_file)


    # print (param_dict.keys())
    # for key in param_dict.keys():
    #     print (param_dict[key].size())

    # print (policy.state_dict().keys())
    # for key in policy.state_dict().keys():
    #     print (policy.state_dict()[key].size())


    policy.load_state_dict(param_dict)
    # policy = torch.load(param_file).cuda()
    print ('loaded params', param_file)

policy.cuda()






print('Init VAE')
vae = VAE()
vae.cuda()

save_dir = home+'/Documents/tmp/breakout_2frames_leakyrelu2'

# comment = '_kl'


load_ = 1
train_ = 1
viz_ = 1


if load_:
    load_epoch = 57
    path_to_load_variables = save_dir+'/vae_params'+str(load_epoch)+'_withpolicy.ckpt'
    vae.load_params(path_to_load_variables)

epochs = 57
if load_ and train_:
    path_to_save_variables = save_dir+'/vae_params'+str(epochs+load_epoch)+'_withpolicy.ckpt'
else:
    path_to_save_variables = save_dir+'/vae_params'+str(epochs)+'_withpolicy.ckpt'


if train_:
    vae.train(state_dataset, epochs=epochs, policy=policy)
    vae.save_params(path_to_save_variables)

if viz_:

    if not train_ and not load_:
        vae.load_params(path_to_save_variables)


    # recon = vae.reconstruction([state_dataset[0]])
    # print (recon.shape) #[1,2,84,84]


    rows = 10
    cols = 2

    # traj_ind =  5 # 2 # #0 #
    # start_ind = 7 #2 #7  # #

    # [traj_ind, start_ind]
    to_plot = [[5,7],[2,2],[0,7]]
    for j in range(len(to_plot)):

        traj_ind = to_plot[j][0]
        start_ind = to_plot[j][1]

        fig = plt.figure(figsize=(4+cols,4+rows), facecolor='white')

        for i in range(rows):

            print (i)


            # plot frame
            ax = plt.subplot2grid((rows,cols), (i,0), frameon=False)

            # state1 = np.squeeze(state[0])
            # state1 = np.reshape(dataset[0][0][1], [84,84*2])
            state1 = np.concatenate([dataset[traj_ind][start_ind+i][1][0], dataset[traj_ind][start_ind+i][1][1]] , axis=1)


            # print (np.max(state1))
            # fadsf


            ax.imshow(state1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_title('State '+str(step),family='serif')


            dist_true = vae.get_action_dist([dataset[traj_ind][start_ind+i][1]], policy)[0]
            print (dist_true)


            ax = plt.subplot2grid((rows,cols), (i,1), frameon=False)

            recon = vae.reconstruction([dataset[traj_ind][start_ind+i][1]])[0]
            state1 = np.concatenate([recon[0], recon[1]] , axis=1)

            ax.imshow(state1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])


            dist_recon = vae.get_action_dist([recon], policy)[0]
            print (dist_recon)
            # print (np.mean((dist_recon-dist_true)**2))

            print ('KL:', np.sum((np.log(dist_true) - np.log(dist_recon))*dist_true))
            # print (np.sum((np.log(dist_recon) - np.log(dist_true))*dist_recon) )


            # v1, v2 = vae.get_kl_error([dataset[traj_ind][start_ind+i][1]], policy)
            # print (v1)
            # print (v2)
            # fadf


            print()

            # dfadf



        #plot fig
        # plt.tight_layout(pad=3., w_pad=2.5, h_pad=1.0)
        if load_ and train_:
            plt_path = save_dir+'/viz_mult_recon_'+str(epochs+load_epoch)+'epochs_'+str(traj_ind)+'_'+str(start_ind)+'_withpolicy.png'
        elif train_ and not load_:
            plt_path = save_dir+'/viz_mult_recon_'+str(epochs)+'epochs_'+str(traj_ind)+'_'+str(start_ind)+'_withpolicy.png'
        elif not train_ and not load_:
            plt_path = save_dir+'/viz_mult_recon_'+str(epochs)+'epochs_'+str(traj_ind)+'_'+str(start_ind)+'_withpolicy.png'
        else:
            plt_path = save_dir+'/viz_mult_recon_'+str(load_epoch)+'epochs_'+str(traj_ind)+'_'+str(start_ind)+'_withpolicy.png'

        plt.savefig(plt_path)
        print ('saved viz',plt_path)
        plt.close(fig)



print ('Done.')













