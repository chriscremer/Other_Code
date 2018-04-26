














import numpy as np

import pickle


import os
from os.path import expanduser
home = expanduser("~")



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


# from vae import VAE
# from vae_with_policy import VAE
# from vae_with_policies import VAE
# from vae_with_policies_and_grads import VAE
# from vae_with_policies_and_weighted_likelihood import VAE
# from vae_with_policies_and_grad_prediction import VAE












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










print ('\nInit Policies')


# agent = a2c(model_dict)
# param_file = home+'/Documents/tmp/breakout_2frames/BreakoutNoFrameskip-v4/A2C/seed0/model_params/model_params9999360.pt'

# load_policy = 1

policies = []
policies_dir = home+'/Documents/tmp/multiple_seeds_of_policies/BreakoutNoFrameskip-v4/A2C/'
for f in os.listdir(policies_dir):
    print (f)
    policy = CNNPolicy(2, 4) #.cuda()
    param_file = home+'/Documents/tmp/multiple_seeds_of_policies/BreakoutNoFrameskip-v4/A2C/'+f+'/model_params3/model_params9999360.pt'    
    param_dict = torch.load(param_file)

    policy.load_state_dict(param_dict)
    # policy = torch.load(param_file).cuda()
    print ('loaded params', param_file)
    policy.cuda()


    policies.append(policy)

    #just one for now
    break




# if load_policy:
    # param_file = home+'/Documents/tmp/breakout_2frames_leakyrelu2/BreakoutNoFrameskip-v4/A2C/seed0/model_params3/model_params3999840.pt'



# print (param_dict.keys())
# for key in param_dict.keys():
#     print (param_dict[key].size())

# print (policy.state_dict().keys())
# for key in policy.state_dict().keys():
#     print (policy.state_dict()[key].size())











# print('\nInit VAE')
# vae = VAE()
# vae.cuda()


# # save_dir = home+'/Documents/tmp/breakout_2frames_leakyrelu2'
# # save_dir = home+'/Documents/tmp/multiple_seeds_of_policies/grad_predictor'
# save_dir = home+'/Documents/tmp/multiple_seeds_of_policies/likelihood_weighting'

# # comment = '_kl'


# load_ = 0
# train_ = 1
viz_ = 1


# if load_:
#     load_epoch = 22
#     path_to_load_variables = save_dir+'/vae_params'+str(load_epoch)+'_withpolicy.ckpt'
#     vae.load_params(path_to_load_variables)
#     print()

# epochs = 11 #500# 111 #300
# if load_ and train_:
#     path_to_save_variables = save_dir+'/vae_params'+str(epochs+load_epoch)+'_withpolicy.ckpt'
# else:
#     path_to_save_variables = save_dir+'/vae_params'+str(epochs)+'_withpolicy.ckpt'


# if train_:
#     vae.train(state_dataset, epochs=epochs, policies=policies)
#     vae.save_params(path_to_save_variables)

if viz_:

    # policy = policies[0]

    policy = CNNPolicy(2, 4) #.cuda()

    # if not train_ and not load_:
    #     vae.load_params(path_to_save_variables)


    # recon = vae.reconstruction([state_dataset[0]])
    # print (recon.shape) #[1,2,84,84]

    params = ['2000000', '3999840', '5999680', '7999520','9999360']


    rows = 5 #10  #number of frames
    cols = 7 #5 #7

    # traj_ind =  5 # 2 # #0 #
    # start_ind = 7 #2 #7  # #

    # [traj_ind, start_ind]
    to_plot = [[5,7],[2,2],[0,7]]
    # to_plot = [[5,7]]
    for j in range(len(to_plot)):

        traj_ind = to_plot[j][0]
        start_ind = to_plot[j][1]

        fig = plt.figure(figsize=(8+cols,4+rows), facecolor='white')

        for i in range(rows):

            print (i)

            for pol in range(len(params)):



                #laod a policy
        
                param_file = home+'/Documents/tmp/multiple_seeds_of_policies/BreakoutNoFrameskip-v4/A2C/seed1/model_params3/model_params'+params[pol]+'.pt' #9999360.pt'
                param_dict = torch.load(param_file)
                policy.load_state_dict(param_dict)
                # policy = torch.load(param_file).cuda()
                print ('loaded params', param_file)
                policy.cuda()





                # #PLOT ACTION DIST
                # dist_true = vae.get_action_dist([dataset[traj_ind][start_ind+i][1]], policy)[0]
                # # print (dist_true)

                # recon = vae.reconstruction([dataset[traj_ind][start_ind+i][1]])[0]
                # dist_recon = vae.get_action_dist([recon], policy)[0]
                # # print (dist_recon)

                # # print (np.mean((dist_recon-dist_true)**2))

                # # print ('KL:', np.sum((np.log(dist_true) - np.log(dist_recon))*dist_true))


                # width = .2
                # ax = plt.subplot2grid((rows,cols), (i,0), frameon=False)

                # ax.bar(range(len(dist_true)), dist_true, width=width)

                # ax.bar(np.array(range(len(dist_recon)))+width, dist_recon, width=width)

                # ax.set_xticks(range(len(dist_true)))
                # if i == rows-1:
                #     ax.set_xticklabels([ 'NOOP', 'FIRE', 'RIGHT', 'LEFT'], size=5)
                # else:
                #     ax.set_xticklabels([])

                # ax.yaxis.set_tick_params(labelsize=5)

                # ax.set_ylim([0,.65])
                # # ax.set_ylim(0,.5)

                # # dfadf
                # if i ==0:
                #     ax.text(0.4, 1.04, 'Action Distribution', transform=ax.transAxes, family='serif', size=6)








                # plot frame
                ax = plt.subplot2grid((rows,cols), (i,1), frameon=False)

                # state1 = np.squeeze(state[0])
                # state1 = np.reshape(dataset[0][0][1], [84,84*2])
                state1 = np.concatenate([dataset[traj_ind][start_ind+i][1][0], dataset[traj_ind][start_ind+i][1][1]] , axis=1)

                ax.imshow(state1, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_title('State '+str(step),family='serif')
                ax.text(0.4, 1.04, 'Real', transform=ax.transAxes, family='serif', size=6)





                # print (pol+2)
                #Plot the grad
                ax = plt.subplot2grid((rows,cols), (i,pol+2), frameon=False)

                frame = dataset[traj_ind][start_ind+i][1]  #[2,84,84]
                x = Variable(torch.from_numpy(np.array([frame])).float(), requires_grad=True).cuda()

                # dist = policy.action_dist(x)

                # dist = policy.action_logdist(x)
                # grad = torch.autograd.grad(torch.sum(dist[:,3]), x)[0]

                value = policy.get_value(x)
                grad = torch.autograd.grad(torch.mean(value), x)[0]
                

                grad = grad.data.cpu().numpy()[0] #for the first one in teh batch -> [2,84,84]
                grad = np.abs(grad)

                state1 = np.concatenate([grad[0], grad[1]] , axis=1)
                # ax.imshow(state1, cmap='gray', norm=NoNorm())
                ax.imshow(state1, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])

                ax.text(0.4, 1.04, params[pol], transform=ax.transAxes, family='serif', size=6)








        #plot fig
        # plt.tight_layout(pad=3., w_pad=2.5, h_pad=1.0)
        # if load_ and train_:
        #     plt_path = save_dir+'/viz_mult_recon_'+str(epochs+load_epoch)+'epochs_'+str(traj_ind)+'_'+str(start_ind)+'_withpolicy.png'
        # elif train_ and not load_:
        #     plt_path = save_dir+'/viz_mult_recon_'+str(epochs)+'epochs_'+str(traj_ind)+'_'+str(start_ind)+'_withpolicy.png'
        # elif not train_ and not load_:
        #     plt_path = save_dir+'/viz_mult_recon_'+str(epochs)+'epochs_'+str(traj_ind)+'_'+str(start_ind)+'_withpolicy.png'
        # else:
        #     plt_path = save_dir+'/viz_mult_recon_'+str(load_epoch)+'epochs_'+str(traj_ind)+'_'+str(start_ind)+'_withpolicy.png'

        save_dir = home+'/Documents/tmp/multiple_seeds_of_policies/see_grads'
        plt_path = save_dir+'/viz_grads'+str(j)+'_value.png'
        plt.savefig(plt_path)
        print ('saved viz',plt_path)
        plt.close(fig)

        # fsad



print ('Done.')













