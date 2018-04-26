




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



print ('Init Expert Policy')
expert_policy = CNNPolicy(2, 4) #.cuda()
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

    expert_policy.load_state_dict(param_dict)
    # policy = torch.load(param_file).cuda()
    print ('loaded params', param_file)
expert_policy.cuda()




print ('Init Imitator Policy')
imitator_policy = CNNPolicy(2, 4).cuda()




# def save_params(save_location, model):

#     #saves all params in recommended way

#     save_path = os.path.join(save_dir, 'model_params3')
#     try:
#         os.makedirs(save_path)
#     except OSError:
#         pass

#     save_to=os.path.join(save_path, "model_params" + str(total_num_steps)+".pt")
#     # save_to=os.path.join(save_path, "model_params" + steps_sci_nota+".pt")
#     # torch.save(dict_copy, save_to)
#     torch.save(agent.actor_critic.state_dict(), save_to)
#     print ('saved', save_to)




def train(expert_policy, imitator_policy, train_x, epochs):

    batch_size = 40
    k=1
    display_step = 100 

    train_y = torch.from_numpy(np.zeros(len(train_x)))
    train_x = torch.from_numpy(np.array(train_x)).float().cuda()
    train_ = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(imitator_policy.parameters(), lr=.0005, weight_decay=.00001)

    total_steps = 0
    for epoch in range(epochs):

        for batch_idx, (data, target) in enumerate(train_loader):

            batch = Variable(data)

            
            optimizer.zero_grad()

            log_dist_expert = expert_policy.action_logdist(batch)
            log_dist_imitator = imitator_policy.action_logdist(batch)

            action_dist_kl = torch.sum((log_dist_expert - log_dist_imitator)*torch.exp(log_dist_expert), dim=1) #[B]

            # elbo, logpx, logpz, logqz, action_dist_kl = self.forward(batch, policy, k=k)
            loss = torch.mean(action_dist_kl)

            loss.backward()
            # nn.utils.clip_grad_norm(self.parameters(), .5)
            optimizer.step()

            if total_steps%display_step==0: # and batch_idx == 0:
                print ('Train Epoch: {}/{}'.format(epoch+1, epochs),
                    # 'total_epochs {}'.format(total_epochs),
                    'LL:{:.4f}'.format(loss.data[0])
                    # 'logpx:{:.4f}'.format(logpx.data[0]),
                    # 'logpz:{:.5f}'.format(logpz.data[0]),
                    # 'logqz:{:.5f}'.format(logqz.data[0]),
                    # 'action_kl:{:.4f}'.format(action_dist_kl.data[0])
                    )

            total_steps+=1





epochs = 100

train(expert_policy, imitator_policy, state_dataset, epochs=epochs)


save_to = home+'/Documents/tmp/breakout_2frames_leakyrelu2/imitator_params.ckpt'
torch.save(imitator_policy.state_dict(), save_to)
print ('saved imitator_policy', save_to)

print ('Done.')






















