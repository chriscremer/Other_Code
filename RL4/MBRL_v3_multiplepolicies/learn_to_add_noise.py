





# Load dataset
# Load policy/policies
# Make noisy frame
    # As a first step can make noise input independent , so just vairacne params
    # See if it matches the average grad
    # Then make a net that outputs the vars 
# Policies take real and noisy, predict action dists
# Minimize KL and maximize entropy of noise 
# Vizualize images 






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
from actor_critic_networks import CNNPolicy
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from train_utils import load_params_v3
from vae_with_policies import VAE


from matplotlib.colors import NoNorm


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

policy = policies[0]
























































print('\nLearn where to add noise')

save_dir = home+'/Documents/tmp/multiple_seeds_of_policies/noise/'

load_ = 1
save_ = 0


load_val = 5
save_val = 5


if load_:
    logvar = pickle.load( open( save_dir+'logvar_'+str(load_val)+'.pkl', "rb" ) )[0]
    logvar = Variable(torch.from_numpy(logvar).cuda(), requires_grad=True)
    print ('loaded logvar', save_dir+'logvar_'+str(load_val)+'.pkl')
else:
    logvar = Variable(torch.Tensor(1,2,84,84).normal_(-20,.1).cuda(), requires_grad=True)

frame1 = torch.from_numpy(np.array([state_dataset[3]]))

max_val = torch.max(frame1)
# print (torch.max(frame1))
# print (torch.min(frame1))
# fsa

# optimizer = optim.Adam([logvar], lr=.00005) #, weight_decay=.00001)
optimizer = optim.Adam([logvar], lr=.0005) #, weight_decay=.00001)
# optimizer = optim.SGD([logvar], lr=.05)#, momentum=0.1) #, weight_decay=.00001)

init_lr = .001
final_lr = .0001 #.00001

steps = 1#00000
mc_samples = 2
for step in range(steps):

    lr = (step/steps)*(final_lr-init_lr) + init_lr
    optimizer = optim.Adam([logvar], lr=lr)

    optimizer.zero_grad()

    frame = Variable(frame1).cuda()

    losses = []
    for m in range(mc_samples):

        eps = Variable(torch.Tensor(1,2,84,84).normal_(0,1)).cuda()
        noisy_frame = frame + eps*(torch.sqrt(torch.exp(logvar)))

        noisy_frame = torch.clamp(noisy_frame, min=0., max=max_val)

        # noisy_frame = noisy_frame - Variable(torch.min(noisy_frame).data)
        # noisy_frame = noisy_frame / Variable(torch.max(noisy_frame).data)
        # noisy_frame = noisy_frame * max_val

        dist_noise = policy.action_dist(noisy_frame)

        log_dist_noise = policy.action_logdist(noisy_frame)
        log_dist_true = policy.action_logdist(frame)

        action_dist_kl = torch.sum((log_dist_true - log_dist_noise)*torch.exp(log_dist_true), dim=1) #[B]


        action_dist_kl = torch.mean(action_dist_kl) # * 1000.

        # std of 2 is the prior, which is 4 for var, which is log(4) for logvar
        # loss = action_dist_kl + (logvar - torch.log(4))**2
        logvar_dif = torch.mean((logvar - .6)**2) *.0001

        loss = action_dist_kl + logvar_dif

        losses.append(loss)

    loss = torch.stack(losses)
    loss = torch.mean(loss)
    loss.backward()

    # nn.utils.clip_grad_norm(self.parameters(), .5)

    optimizer.step()

    # total_steps+=1

    if step%500==0: # and batch_idx == 0:
        print ('Step: {}/{}'.format(step+1, steps),
            # 'total_epochs {}'.format(total_epochs),
            'Loss:{:.4f}'.format(loss.data[0]),
            # 'logpx:{:.4f}'.format(logpx.data[0]),
            # 'logpz:{:.5f}'.format(logpz.data[0]),
            # 'logqz:{:.5f}'.format(logqz.data[0]),
            'action_kl:{:.4f}'.format(action_dist_kl.data[0]),
            'logvar_dif:{:.4f}'.format(logvar_dif.data[0]),
            'std:{:.4f}'.format(torch.mean(torch.sqrt(torch.exp(logvar))).data[0]),

            'minstd:{:.4f}'.format(torch.min(torch.sqrt(torch.exp(logvar))).data[0]),
            'maxstd:{:.4f}'.format(torch.max(torch.sqrt(torch.exp(logvar))).data[0]),

            'logvar:{:.4f}'.format(torch.mean(logvar).data[0]),
            'lr:{:.5f}'.format(lr),
            )

        # print (torch.sum(frame))










rows = 1
cols = 6


fig = plt.figure(figsize=(6+cols,4+rows), facecolor='white')


traj_ind = 0
start_ind = 2
# frame = torch.from_numpy(np.array([dataset[traj_ind][start_ind+i][1]])).float()[0].numpy()

frame = state_dataset[3]

# print (frame)
# fds


# Plot real frame
ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

state1 = np.concatenate([frame[0], frame[1]] , axis=1)
ax.imshow(state1, cmap='gray')
ax.set_xticks([])
ax.set_yticks([])

# ax.text(0.4, 1.04, 'Real Frame', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, family='serif', size=6)
ax.text(0.4, 1.04, 'Real Frame', transform=ax.transAxes, family='serif', size=6)



#Plot noisy
ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)

eps = torch.Tensor(2,84,84).normal_(0,1).numpy() #.cuda()
noisy_frame = frame + eps*(np.sqrt(np.exp(logvar.data.cpu().numpy()[0])))

noisy_frame = np.clip(noisy_frame, a_min=0., a_max=max_val)
# noisy_frame = noisy_frame - np.min(noisy_frame)
# noisy_frame = noisy_frame / np.max(noisy_frame)

state1 = np.concatenate([noisy_frame[0], noisy_frame[1]] , axis=1)
# ax.imshow(state1, cmap='gray', norm=NoNorm())
ax.imshow(state1, cmap='gray')
ax.set_xticks([])
ax.set_yticks([])
# ax.text(0.4, 1.04, 'Real Frame + Noise', transform=ax.transAxes, family='serif', size=6)
ax.text(0.4, 1.04, 'Real Frame + Noise',  horizontalalignment='center', transform=ax.transAxes, family='serif', size=6)


#Plot logvar
ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)

toplot =logvar.data.cpu().numpy()[0]
state1 = np.concatenate([toplot[0], toplot[1]] , axis=1)
# ax.imshow(state1, cmap='gray', norm=NoNorm())
ax.imshow(state1, cmap='gray')
ax.set_xticks([])
ax.set_yticks([])
ax.text(0.4, 1.04, 'LogVar', transform=ax.transAxes, family='serif', size=6)


#Plot var
ax = plt.subplot2grid((rows,cols), (0,3), frameon=False)

toplot =  np.exp(logvar.data.cpu().numpy()[0])

toplot = toplot - np.min(toplot)
toplot = toplot / np.max(toplot)

state1 = np.concatenate([toplot[0], toplot[1]] , axis=1)
# ax.imshow(state1, cmap='gray', norm=NoNorm())
ax.imshow(state1, cmap='gray')
ax.set_xticks([])
ax.set_yticks([])

ax.text(0.4, 1.04, 'Var', transform=ax.transAxes, family='serif', size=6)



#Plot grads
ax = plt.subplot2grid((rows,cols), (0,4), frameon=False)


x = Variable(torch.from_numpy(np.array([frame])).float(), requires_grad=True).cuda()
dist = policy.action_dist(x)
grad = torch.autograd.grad(torch.sum(dist[:,3]), x)[0]
grad = grad.data.cpu().numpy()[0] #for the first one in teh batch -> [2,84,84]

print (np.max(grad))
print (np.min(grad))

grad = np.abs(grad)

state1 = np.concatenate([grad[0], grad[1]] , axis=1)
# ax.imshow(state1, cmap='gray', norm=NoNorm())
ax.imshow(state1, cmap='gray')
ax.set_xticks([])
ax.set_yticks([])

ax.text(0.4, 1.04, 'Grad of Real', transform=ax.transAxes, family='serif', size=6)






#Plot grads of noisy
ax = plt.subplot2grid((rows,cols), (0,5), frameon=False)


x = Variable(torch.from_numpy(np.array([noisy_frame])).float(), requires_grad=True).cuda()
dist = policy.action_dist(x)
grad = torch.autograd.grad(torch.sum(dist[:,3]), x)[0]
grad = grad.data.cpu().numpy()[0] #for the first one in teh batch -> [2,84,84]

print (np.max(grad))
print (np.min(grad))

grad = np.abs(grad)

state1 = np.concatenate([grad[0], grad[1]] , axis=1)
# ax.imshow(state1, cmap='gray', norm=NoNorm())
ax.imshow(state1, cmap='gray')
ax.set_xticks([])
ax.set_yticks([])

ax.text(0.4, 1.04, 'Grad of Noisy', transform=ax.transAxes, family='serif', size=6)





plt_path = save_dir+'viz_noizy_'+str(traj_ind)+'_'+str(start_ind)+'_3.png'
plt.savefig(plt_path)
print ('saved viz',plt_path)
plt.close(fig)





if save_:
    logvar = logvar.data.cpu().numpy()
    pickle.dump( [logvar], open( save_dir+'logvar_'+str(save_val)+'.pkl', "wb" ) )

    print ('saved logvar', save_dir+'logvar_'+str(save_val)+'.pkl')



print ('Done.')


fsadfa




# dsffsd
















print('\nGet grad on inputs')
# batch_size = 40

# x = state_dataset[0]  #[2,84,84]
# x = Variable(torch.from_numpy(np.array([x])).float(), requires_grad=True).cuda()

# dist = policy.action_dist(x)

# # dist dist.data.cpu().numpy()

# print (dist)


# # print (torch.sum(torch.autograd.grad(torch.sum(dist[:,3]), self.deconv3.weight)[0]))
# print (torch.autograd.grad(torch.sum(dist[:,3]), x)[0])


save_dir = home+'/Documents/tmp/multiple_seeds_of_policies/grads/'

rows = 7
cols = 5

# [traj_ind, start_ind]
# to_plot = [[5,7],[2,2],[0,7]]
to_plot = [[2,0]]
for j in range(len(to_plot)):

    traj_ind = to_plot[j][0]
    start_ind = to_plot[j][1]

    fig = plt.figure(figsize=(6+cols,4+rows), facecolor='white')

    for i in range(rows):

        print (i)

        # Plot real frame
        ax = plt.subplot2grid((rows,cols), (i,0), frameon=False)

        state1 = np.concatenate([dataset[traj_ind][start_ind+i][1][0], dataset[traj_ind][start_ind+i][1][1]] , axis=1)
        ax.imshow(state1, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])


        for a in range(4):

            #Plot grads
            ax = plt.subplot2grid((rows,cols), (i,1+a), frameon=False)
            
            x = Variable(torch.from_numpy(np.array([dataset[traj_ind][start_ind+i][1]])).float(), requires_grad=True).cuda()
            dist = policy.action_dist(x)
            grad = torch.autograd.grad(torch.sum(dist[:,a]), x)[0]
            grad = grad.data.cpu().numpy()[0] #for the first one in teh batch -> [2,84,84]

            grad = np.abs(grad)
            # print (np.max(grad))
            # print (np.min(grad))
            # print (np.mean(grad))
            # fad

            state1 = np.concatenate([grad[0], grad[1]] , axis=1)
            # ax.imshow(state1, cmap='gray', norm=NoNorm())
            ax.imshow(state1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])


    plt_path = save_dir+'viz_grads_'+str(traj_ind)+'_'+str(start_ind)+'_2.png'
    plt.savefig(plt_path)
    print ('saved viz',plt_path)
    plt.close(fig)



print('Done.')
fadsfa



















































