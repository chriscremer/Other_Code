

import numpy as np

import pickle


import os
from os.path import expanduser
home = expanduser("~")



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from vae import VAE


dataset = pickle.load( open( home+'/Documents/tmp/breakout_2frames/breakout_trajectories_10000.pkl', "rb" ) )


for ii in range(len(dataset)):
    print (len(dataset[ii]))




# dataset: trajectories: timesteps: (action,state) state: [2,84,84]

print (len(dataset))
print (len(dataset[ii][0])) # single timepoint
print (dataset[ii][0][0].shape)  #action [1]           a_t+1
print (dataset[ii][0][1].shape)     #state [2,84,84]   s_t


state_dataset = []
for i in range(len(dataset)):
    for t in range(len(dataset[i])):
        state_dataset.append(dataset[i][t][1])

print (len(state_dataset))



print('Init VAE')
vae = VAE()
vae.cuda()

load_ = 1
train_ = 1
viz_ = 1


if load_:
    load_epoch = 50
    path_to_load_variables = home+'/Documents/tmp/breakout_2frames/vae_params'+str(load_epoch)+'.ckpt'
    vae.load_params(path_to_load_variables)

epochs = 100
if load_:
    path_to_save_variables = home+'/Documents/tmp/breakout_2frames/vae_params'+str(epochs+load_epoch)+'.ckpt'
else:
    path_to_save_variables = home+'/Documents/tmp/breakout_2frames/vae_params'+str(epochs)+'.ckpt'


if train_:
    vae.train(state_dataset, epochs=epochs)
    vae.save_params(path_to_save_variables)

if viz_:

    if not train_:

        vae.load_params(path_to_save_variables)

    # recon = vae.reconstruction([state_dataset[0]])
    # print (recon.shape) #[1,2,84,84]


    rows = 10
    cols = 2

    traj_ind = 5
    start_ind = 0

    fig = plt.figure(figsize=(4+cols,4+rows), facecolor='white')

    for i in range(rows):
        # plot frame
        ax = plt.subplot2grid((rows,cols), (i,0), frameon=False)

        # state1 = np.squeeze(state[0])
        # state1 = np.reshape(dataset[0][0][1], [84,84*2])
        state1 = np.concatenate([dataset[traj_ind][start_ind+i][1][0], dataset[traj_ind][start_ind+i][1][1]] , axis=1)

        ax.imshow(state1, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title('State '+str(step),family='serif')


        ax = plt.subplot2grid((rows,cols), (i,1), frameon=False)

        recon = vae.reconstruction([dataset[traj_ind][start_ind+i][1]])[0]
        state1 = np.concatenate([recon[0], recon[1]] , axis=1)

        ax.imshow(state1, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])



    #plot fig
    # plt.tight_layout(pad=3., w_pad=2.5, h_pad=1.0)
    if load_:
        plt_path = home+'/Documents/tmp/breakout_2frames/viz_mult_recon_'+str(epochs+load_epoch)+'epochs.png'
    else:
        plt_path = home+'/Documents/tmp/breakout_2frames/viz_mult_recon_'+str(epochs)+'epochs.png'

    plt.savefig(plt_path)
    print ('saved viz',plt_path)
    plt.close(fig)











