









import numpy as np

import pickle


import os
from os.path import expanduser
home = expanduser("~")



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



dataset = pickle.load( open( home+'/Documents/tmp/breakout_2frames/breakout_trajectories_10000.pkl', "rb" ) )


for ii in range(len(dataset)):
    print (len(dataset[ii]))




# dataset: trajectories: timesteps: (action,state) state: [2,84,84]

print (len(dataset))
print (len(dataset[ii][0])) # single timepoint
print (dataset[ii][0][0].shape)  #action [1]           a_t+1
print (dataset[ii][0][1].shape)     #state [2,84,84]   s_t







# #VIZ one state

# rows = 1
# cols = 1

# fig = plt.figure(figsize=(8,4), facecolor='white')



# # plot frame
# ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

# # state1 = np.squeeze(state[0])
# # state1 = np.reshape(dataset[0][0][1], [84,84*2])
# state1 = np.concatenate([dataset[0][0][1][0], dataset[0][0][1][1]] , axis=1)

# ax.imshow(state1, cmap='gray')
# ax.set_xticks([])
# ax.set_yticks([])
# # ax.set_title('State '+str(step),family='serif')


# #plot fig
# # plt.tight_layout(pad=3., w_pad=2.5, h_pad=1.0)
# plt_path = home+'/Documents/tmp/breakout_2frames/viz2.png'
# plt.savefig(plt_path)
# print ('saved',plt_path)
# plt.close(fig)




#Viz multiple states


rows = 10
cols = 1

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


#plot fig
# plt.tight_layout(pad=3., w_pad=2.5, h_pad=1.0)
plt_path = home+'/Documents/tmp/breakout_2frames/viz_mult.png'
plt.savefig(plt_path)
print ('saved',plt_path)
plt.close(fig)


























