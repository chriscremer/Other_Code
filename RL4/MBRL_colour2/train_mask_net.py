








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

import imageio


# from vae_with_policies import VAE


# from matplotlib.colors import NoNorm





# dir_ = 'RoadRunner'
dir_ = 'RoadRunner_colour_4'

exp_name = 'RoadRunner_colour_4'
env_name = 'RoadRunner'
env_name2 = env_name +'NoFrameskip-v4'
exp_dir = home+'/Documents/tmp/' + exp_name



print ("Load dataset")
# dataset = pickle.load( open( home+'/Documents/tmp/breakout_2frames/breakout_trajectories_10000.pkl', "rb" ) )
dataset = pickle.load( open( home+'/Documents/tmp/'+dir_+'/trajectories_10000.pkl', "rb" ) )


# dataset: trajectories: timesteps: (action,state) state: [2,84,84]
print (len(dataset))
print (len(dataset[0][0])) # single timepoint
print (dataset[0][0][1].shape)  #action [1]           a_t+1
print (dataset[0][0][0].shape)     #state [2,84,84]   s_t

# fds
# for ii in range(len(dataset)):
#     print (len(dataset[ii]))
    # dataset[ii] = dataset[ii] / 255.

print (np.max(dataset[0][0][0]))

#scale data
for i in range(len(dataset)):
    for t in range(len(dataset[i])):
        dataset[i][t][0] = dataset[i][t][0] / 255.
        # state_dataset.append(dataset[i][t][1]) 
print (np.max(dataset[0][0][0]))

state_dataset = []
for i in range(len(dataset)):
    for t in range(len(dataset[i])):
        state_dataset.append(dataset[i][t][0]) #  /255.)

# for i in range(len(dataset)):
#     print (len(dataset[i]))

print (len(state_dataset))














print ('\nInit Policy')
# # agent = a2c(model_dict)
# # param_file = home+'/Documents/tmp/breakout_2frames/BreakoutNoFrameskip-v4/A2C/seed0/model_params/model_params9999360.pt'
# # load_policy = 1
# policies = []
# # policies_dir = home+'/Documents/tmp/multiple_seeds_of_policies/BreakoutNoFrameskip-v4/A2C/'
# policies_dir = home+'/Documents/tmp/RoadRunner/RoadRunnerNoFrameskip-v4/A2C/'
# for f in os.listdir(policies_dir):

#     print (f)
#     # policy = CNNPolicy(2, 4) #.cuda()
#     policy = CNNPolicy(2, 18) #.cuda()   #num-frames* channels, nyum-actions

#     # param_file = home+'/Documents/tmp/multiple_seeds_of_policies/BreakoutNoFrameskip-v4/A2C/'+f+'/model_params3/model_params9999360.pt'    
#     param_file = home+'/Documents/tmp/RoadRunner/RoadRunnerNoFrameskip-v4/A2C/'+f+'/model_params3/model_params9999360.pt'    
#     param_dict = torch.load(param_file)

#     policy.load_state_dict(param_dict)
#     # policy = torch.load(param_file).cuda()
#     print ('loaded params', param_file)
#     policy.cuda()

#     policies.append(policy)

#     #just one for now
#     break

# policy = policies[0]



policy = CNNPolicy(2*3, 18) #.cuda()   #num-frames* channels, num-actions

# param_file = home+'/Documents/tmp/multiple_seeds_of_policies/BreakoutNoFrameskip-v4/A2C/'+f+'/model_params3/model_params9999360.pt'    
# param_file = home+'/Documents/tmp/RoadRunner/RoadRunnerNoFrameskip-v4/A2C/'+f+'/model_params3/model_params9999360.pt'
param_file = home+'/Documents/tmp/'+exp_name+'/'+env_name+'NoFrameskip-v4/A2C/seed0/model_params3/model_params3999840.pt'

param_dict = torch.load(param_file)
policy.load_state_dict(param_dict)
# policy = torch.load(param_file).cuda()
print ('loaded params', param_file)
policy.cuda()









class MASK_PREDICTOR(nn.Module):
    def __init__(self):
        super(MASK_PREDICTOR, self).__init__()

        # self.x_size = 3072
        # self.x_size = 84
        self.z_size = 100

        # image_channels = 2
        image_channels = 6

        self.act_func = F.leaky_relu# F.relu # # F.tanh ##  F.elu F.softplus

        if image_channels == 6:
            self.intermediate_size = 11264
        else:
            self.intermediate_size = 32*7*7


        #MASK PREDICTOR
        self.conv1_gp = nn.Conv2d(image_channels, 32, 8, stride=4)
        self.conv2_gp = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3_gp = nn.Conv2d(64, 32, 3, stride=1)

        # self.intermediate_size = 32*7*7   #1: (84-(8/2) -> 80, 2: 80-4/2 -> 78, 3: 

        self.fc1_gp = nn.Linear(self.intermediate_size, 200)
        self.fc2_gp = nn.Linear(200, self.z_size)
        self.fc3_gp = nn.Linear(self.z_size, 200)
        self.fc4_gp = nn.Linear(200, self.intermediate_size)

        # self.deconv1 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=1, kernel_size=5, stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)
        self.deconv1_gp = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.deconv2_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, output_padding=1)
        self.deconv3_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=8, stride=4, output_padding=(2,0))

            #out channels is 2 because its one for each frame. 

        params_gp = [list(self.conv1_gp.parameters()) + list(self.conv2_gp.parameters()) + list(self.conv3_gp.parameters()) +
                    list(self.fc1_gp.parameters()) + list(self.fc2_gp.parameters()) + 
                    list(self.fc3_gp.parameters()) + list(self.fc4_gp.parameters()) +
                    list(self.deconv1_gp.parameters()) + list(self.deconv2_gp.parameters()) + list(self.deconv3_gp.parameters())]
                    
                    
        self.optimizer = optim.Adam(params_gp[0], lr=.0001, weight_decay=.00001)




    def predict_mask(self, x):

        # print (x.size())  #[B,32,210,160]


        x = self.act_func(self.conv1_gp(x))  #[B,32,51,39]
        # print (x.size())
        x = self.act_func(self.conv2_gp(x)) #[B,64,24,18]
        # print (x.size())
        x = self.act_func(self.conv3_gp(x)) #[B,32,22,16]
        # print (x.size())
        

        # print (x.size())
        x = x.view(-1, self.intermediate_size)
        h1 = self.act_func(self.fc1_gp(x))
        z = self.fc2_gp(h1)
        z = self.act_func(self.fc3_gp(z)) 
        z = self.act_func(self.fc4_gp(z))  #[B,11264]
        z = z.view(-1, 32, 22, 16)
        z = self.act_func(self.deconv1_gp(z)) #[B,64,24,18]
        # print (z.size())
        z = self.act_func(self.deconv2_gp(z)) #[B,64,51,39]
        # print (z.size())
        grad = self.deconv3_gp(z) # [B,32,210,160]
        # print (grad.size())

        # fdasd
        # x_hat_sigmoid = F.sigmoid(x_hat)
        # fds
        return F.sigmoid(grad)




    def forward(self, frame, policy):
        # x: [B,2,84,84]
        self.B = frame.size()[0]
        # policy = policies[0]

        # print (frame.size())
        # fds

        # print (frame.size())

        #Predict mask
        mask = self.predict_mask(frame)  #[B,2,210,160]
        # print (mask.size())
        

        mask = mask.repeat(1,3,1,1)
        # print (mask.size())
        # fsad
        #frame: [B,6,210,160]
        masked_frame = frame * mask


        log_dist_mask = policy.action_logdist(masked_frame)
        log_dist_true = policy.action_logdist(frame)

        action_dist_kl = torch.sum((log_dist_true - log_dist_mask)*torch.exp(log_dist_true), dim=1) #[B]
        action_dist_kl = torch.mean(action_dist_kl) # * 1000

        mask = mask.view(self.B, -1)
        mask_sum = torch.mean(torch.sum(mask, dim=1)) * .000001

        loss = action_dist_kl + mask_sum

        return loss, action_dist_kl, mask_sum





    def train(self, train_x, epochs, policy):

        batch_size = 32
        # k=1
        display_step = 300 

        # print (np.array(train_x).shape)
        # fds

        train_y = torch.from_numpy(np.zeros(len(train_x)))
        train_x = torch.from_numpy(np.array(train_x)).float().cuda() #/255. #.type(model.dtype)
        train_ = torch.utils.data.TensorDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True)


        total_steps = 0
        for epoch in range(epochs):

            for batch_idx, (data, target) in enumerate(train_loader):

                batch = Variable(data) 

                # print (batch.size())

                self.optimizer.zero_grad()
                loss, action_dist_kl, mask_sum = self.forward(batch, policy)
                loss.backward()
                self.optimizer.step()


                # self.optimizer_gp.zero_grad()
                # loss_gp, true_error, recon_error = self.forward_gp(batch, policies)
                # loss_gp.backward()
                # self.optimizer_gp.step()

                if total_steps%display_step==0: # and batch_idx == 0:
                    print ('Train Epoch: {}/{}'.format(epoch+1, epochs),
                        # 'total_epochs {}'.format(total_epochs),
                        'Loss:{:.4f}'.format(loss.data[0]),
                        # 'logpx:{:.4f}'.format(logpx.data[0]),  
                        'action_dist_kl:{:.4f}'.format(action_dist_kl.data[0]),
                        'mask_sum:{:.4f}'.format(mask_sum.data[0]),
                        # 'mask_min:{:.4f}'.format(true_error.data[0]),
                        # 'mask_max:{:.4f}'.format(recon_error.data[0]),
                        # 'logpz:{:.5f}'.format(logpz.data[0]),
                        # 'logqz:{:.5f}'.format(logqz.data[0]),
                        # 'action_kl:{:.4f}'.format(action_dist_kl.data[0]),
                        # 'act_dif:{:.4f}'.format(act_dif.data[0]),
                        # 'grad_dif:{:.4f}'.format(grad_dif.data[0]),
                        )

                total_steps+=1







































def load_params_v3(save_dir, policy, epochs):
    save_to=os.path.join(save_dir, "model_params" + str(epochs)+".pt")
    # save_path = os.path.join(save_dir, 'model_params3')
    # save_to=os.path.join(save_path, "model_params" + str(total_num_steps)+".pt")
    # save_to=os.path.join(save_path, "model_params" + steps_sci_nota+".pt")
    # torch.save(dict_copy, save_to)
    # torch.save(agent.actor_critic.state_dict(), save_to)
    policy.load_state_dict(torch.load(save_to))
    print ('loaded', save_to)





def save_params_v3(save_dir, policy, epochs):

    save_to=os.path.join(save_dir, "model_params" + str(epochs)+".pt")

    torch.save(policy.state_dict(), save_to)
    print ('saved', save_to)












print('\nLearn where to mask')

train_ = 0
load_ = 0
gif_ = 1

# save_dir = home+'/Documents/tmp/multiple_seeds_of_policies/mask/'
# save_dir = home+'/Documents/tmp/RoadRunner/mask/'
# save_dir = exp_dir+'_mask' +'/'+env_name+'NoFrameskip-v4'+'/mask/'
save_dir = exp_dir+'_mask/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    # if print_
    print ('Made dir', save_dir) 

# load_ = 0
# save_ = 0

# load_val = 5
# save_val = 5


model = MASK_PREDICTOR()
model.cuda()


epochs = 200
# load_params_v3(save_dir, model, epochs=50)


if train_:

    model.train(state_dataset, epochs=epochs, policy=policy)
    save_params_v3(save_dir, model, epochs=epochs)




if load_:
    load_params_v3(save_dir, model, epochs=epochs)


    rows = 5
    cols = 3

    # frame_numb = 448


    fig = plt.figure(figsize=(2+cols,4+rows), facecolor='white', dpi=210*rows)

    for i in range(rows):

        frame_idx = np.random.randint(0,10000)

        print (i, frame_idx)

        # frame_idx = frame_numb + i

        # traj_ind = 0
        # start_ind = 2
        # frame = torch.from_numpy(np.array([dataset[traj_ind][start_ind+i][1]])).float()[0].numpy()

        frame = state_dataset[frame_idx]

        frame_pytorch = Variable(torch.from_numpy(np.array([frame])).cuda())
        mask = model.predict_mask(frame_pytorch)
        mask = mask.repeat(1,3,1,1)

        masked_frame = frame * mask.data.cpu().numpy()[0]
        # masked_frame = masked_frame

        # print (frame)
        # fds

        #scale back up 
        frame = frame *255.
        masked_frame = masked_frame *255.


        # print (frame)

        print (frame.shape)  #[6,210,160]
        print (np.max(frame))  #252
        print (masked_frame.shape)#[6,210,160]
        print (np.max(masked_frame)) #127


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

        # state1 = np.concatenate([frame[0], frame[1]] , axis=1)

        # ax.imshow(state1) #, cmap='gray')
        ax.imshow(np.uint8(frame)) #, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        # ax.text(0.4, 1.04, 'Real Frame', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, family='serif', size=6)
        if i==0:
            ax.text(0.4, 1.04, 'Real Frame', transform=ax.transAxes, family='serif', size=6)





        #Plot masked
        ax = plt.subplot2grid((rows,cols), (i,1), frameon=False)

        masked_frame = masked_frame[:3]
        masked_frame = np.rollaxis(masked_frame, 1, 0)
        masked_frame = np.rollaxis(masked_frame, 2, 1)
        # masked_frame = masked_frame.data.cpu().numpy()[0]
        # state1 = np.concatenate([masked_frame[0], masked_frame[1]] , axis=1)
        # ax.imshow(state1, cmap='gray', norm=NoNorm())
        ax.imshow(np.uint8(masked_frame))#, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.text(0.4, 1.04, 'Real Frame + Noise', transform=ax.transAxes, family='serif', size=6)
        if i==0:
            ax.text(0.4, 1.04, 'Real Frame * Mask',  horizontalalignment='center', transform=ax.transAxes, family='serif', size=6)







        #Plot mask
        ax = plt.subplot2grid((rows,cols), (i,2), frameon=False)

        mask = mask.data.cpu().numpy()[0]
        mask = mask[:3]
        mask = np.rollaxis(mask, 1, 0)
        mask = np.rollaxis(mask, 2, 1)
        mask = mask[:,:,0]
        # state1 = np.concatenate([mask[0], mask[1]] , axis=1)
        # ax.imshow(state1, cmap='gray', norm=NoNorm())
        ax.imshow(mask, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        if i==0:
            ax.text(0.4, 1.04, 'Mask', transform=ax.transAxes, family='serif', size=6)









        # #Plot var
        # ax = plt.subplot2grid((rows,cols), (0,3), frameon=False)

        # toplot =  np.exp(mask.data.cpu().numpy()[0])

        # toplot = toplot - np.min(toplot)
        # toplot = toplot / np.max(toplot)

        # state1 = np.concatenate([toplot[0], toplot[1]] , axis=1)
        # # ax.imshow(state1, cmap='gray', norm=NoNorm())
        # ax.imshow(state1, cmap='gray')
        # ax.set_xticks([])
        # ax.set_yticks([])

        # ax.text(0.4, 1.04, 'Var', transform=ax.transAxes, family='serif', size=6)



        # #Plot grads
        # ax = plt.subplot2grid((rows,cols), (0,4), frameon=False)


        # x = Variable(torch.from_numpy(np.array([frame])).float(), requires_grad=True).cuda()
        # dist = policy.action_dist(x)
        # grad = torch.autograd.grad(torch.sum(dist[:,3]), x)[0]
        # grad = grad.data.cpu().numpy()[0] #for the first one in teh batch -> [2,84,84]

        # print (np.max(grad))
        # print (np.min(grad))

        # grad = np.abs(grad)

        # state1 = np.concatenate([grad[0], grad[1]] , axis=1)
        # # ax.imshow(state1, cmap='gray', norm=NoNorm())
        # ax.imshow(state1, cmap='gray')
        # ax.set_xticks([])
        # ax.set_yticks([])

        # ax.text(0.4, 1.04, 'Grad of Real', transform=ax.transAxes, family='serif', size=6)






        # #Plot grads of noisy
        # ax = plt.subplot2grid((rows,cols), (0,5), frameon=False)


        # x = Variable(torch.from_numpy(np.array([noisy_frame])).float(), requires_grad=True).cuda()
        # dist = policy.action_dist(x)
        # grad = torch.autograd.grad(torch.sum(dist[:,3]), x)[0]
        # grad = grad.data.cpu().numpy()[0] #for the first one in teh batch -> [2,84,84]

        # print (np.max(grad))
        # print (np.min(grad))

        # grad = np.abs(grad)

        # state1 = np.concatenate([grad[0], grad[1]] , axis=1)
        # # ax.imshow(state1, cmap='gray', norm=NoNorm())
        # ax.imshow(state1, cmap='gray')
        # ax.set_xticks([])
        # ax.set_yticks([])

        # ax.text(0.4, 1.04, 'Grad of Noisy', transform=ax.transAxes, family='serif', size=6)





    # plt_path = save_dir+'viz_mask_'+str(traj_ind)+'_'+str(start_ind)+'_5.png'
    plt_path = save_dir+'viz.png'
    plt.savefig(plt_path)
    print ('saved viz',plt_path)
    plt.close(fig)










if gif_:



    n_frames = 200

    print (len(dataset[0]))

    load_params_v3(save_dir, model, epochs=200)


    # frames_path = save_dir+'gif/'
    # images = []
    # for i in range(n_frames):
    #     images.append(imageio.imread(frames_path+'frame'+str(i)+'.png'))
        
    # # gif_path_this = gif_epoch_path + str(j) + '.gif'
    # gif_path_this = frames_path+ 'first.gif'
    # imageio.mimsave(gif_path_this, images)
    # print ('made gif', gif_path_this)

    # dfad

    gif_dir = save_dir+'gif3/'


    

    cols  =3
    rows = 1

    for i in range(n_frames):

        print (i)

        plt_path = gif_dir+'frame'+str(i)+'.png'

        fig = plt.figure(figsize=(2+cols,2+rows), facecolor='white', dpi=210*rows)

        


        frame = dataset[0][200+i][0]
            
        frame_pytorch = Variable(torch.from_numpy(np.array([frame])).cuda())
        mask = model.predict_mask(frame_pytorch)
        mask = mask.repeat(1,3,1,1)

        masked_frame = frame * mask.data.cpu().numpy()[0]
        # masked_frame = masked_frame

        # print (frame)
        # fds

        #scale back up 
        frame = frame *255.
        masked_frame = masked_frame *255.

        frame = frame[:3]
        # frame = frame[0:6:2]

        frame = np.rollaxis(frame, 1, 0)
        frame = np.rollaxis(frame, 2, 1)


        # Plot real frame
        ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

        # state1 = np.concatenate([frame[0], frame[1]] , axis=1)

        # ax.imshow(state1) #, cmap='gray')
        ax.imshow(np.uint8(frame)) #, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        # ax.text(0.4, 1.04, 'Real Frame', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, family='serif', size=6)
        if i==0:
            ax.text(0.4, 1.04, 'Real Frame', transform=ax.transAxes, family='serif', size=6)
        else:
            ax.text(0.4, 1.04, str(i), transform=ax.transAxes, family='serif', size=6)





        #Plot masked
        ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)

        masked_frame = masked_frame[:3]
        masked_frame = np.rollaxis(masked_frame, 1, 0)
        masked_frame = np.rollaxis(masked_frame, 2, 1)
        # masked_frame = masked_frame.data.cpu().numpy()[0]
        # state1 = np.concatenate([masked_frame[0], masked_frame[1]] , axis=1)
        # ax.imshow(state1, cmap='gray', norm=NoNorm())
        ax.imshow(np.uint8(masked_frame))#, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.text(0.4, 1.04, 'Real Frame + Noise', transform=ax.transAxes, family='serif', size=6)
        if i==0:
            ax.text(0.4, 1.04, 'Real Frame * Mask',  horizontalalignment='center', transform=ax.transAxes, family='serif', size=6)







        #Plot mask
        ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)

        mask = mask.data.cpu().numpy()[0]
        mask = mask[:3]
        mask = np.rollaxis(mask, 1, 0)
        mask = np.rollaxis(mask, 2, 1)
        mask = mask[:,:,0]
        # state1 = np.concatenate([mask[0], mask[1]] , axis=1)
        # ax.imshow(state1, cmap='gray', norm=NoNorm())
        ax.imshow(mask, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        if i==0:
            ax.text(0.4, 1.04, 'Mask', transform=ax.transAxes, family='serif', size=6)





        # plt_path = save_dir #+'viz.png'
        plt.savefig(plt_path)
        print ('saved viz',plt_path)
        plt.close(fig)



    print('Making Gif')
    # frames_path = save_dir+'gif/'
    images = []
    for i in range(n_frames):
        images.append(imageio.imread(gif_dir+'frame'+str(i)+'.png'))
        
    # gif_path_this = gif_epoch_path + str(j) + '.gif'
    gif_path_this = gif_dir+ 'first.gif'
    imageio.mimsave(gif_path_this, images)
    print ('made gif', gif_path_this)






# if save_:
#     mask = mask.data.cpu().numpy()
#     pickle.dump( [mask], open( save_dir+'mask_'+str(save_val)+'.pkl', "wb" ) )

#     print ('saved mask', save_dir+'mask_'+str(save_val)+'.pkl')



print ('Done.')


# fsadfa




# # dsffsd
















# print('\nGet grad on inputs')
# # batch_size = 40

# # x = state_dataset[0]  #[2,84,84]
# # x = Variable(torch.from_numpy(np.array([x])).float(), requires_grad=True).cuda()

# # dist = policy.action_dist(x)

# # # dist dist.data.cpu().numpy()

# # print (dist)


# # # print (torch.sum(torch.autograd.grad(torch.sum(dist[:,3]), self.deconv3.weight)[0]))
# # print (torch.autograd.grad(torch.sum(dist[:,3]), x)[0])


# save_dir = home+'/Documents/tmp/multiple_seeds_of_policies/grads/'

# rows = 7
# cols = 5

# # [traj_ind, start_ind]
# # to_plot = [[5,7],[2,2],[0,7]]
# to_plot = [[2,0]]
# for j in range(len(to_plot)):

#     traj_ind = to_plot[j][0]
#     start_ind = to_plot[j][1]

#     fig = plt.figure(figsize=(6+cols,4+rows), facecolor='white')

#     for i in range(rows):

#         print (i)

#         # Plot real frame
#         ax = plt.subplot2grid((rows,cols), (i,0), frameon=False)

#         state1 = np.concatenate([dataset[traj_ind][start_ind+i][1][0], dataset[traj_ind][start_ind+i][1][1]] , axis=1)
#         ax.imshow(state1, cmap='gray')
#         ax.set_xticks([])
#         ax.set_yticks([])


#         for a in range(4):

#             #Plot grads
#             ax = plt.subplot2grid((rows,cols), (i,1+a), frameon=False)
            
#             x = Variable(torch.from_numpy(np.array([dataset[traj_ind][start_ind+i][1]])).float(), requires_grad=True).cuda()
#             dist = policy.action_dist(x)
#             grad = torch.autograd.grad(torch.sum(dist[:,a]), x)[0]
#             grad = grad.data.cpu().numpy()[0] #for the first one in teh batch -> [2,84,84]

#             grad = np.abs(grad)
#             # print (np.max(grad))
#             # print (np.min(grad))
#             # print (np.mean(grad))
#             # fad

#             state1 = np.concatenate([grad[0], grad[1]] , axis=1)
#             # ax.imshow(state1, cmap='gray', norm=NoNorm())
#             ax.imshow(state1, cmap='gray')
#             ax.set_xticks([])
#             ax.set_yticks([])


#     plt_path = save_dir+'viz_grads_'+str(traj_ind)+'_'+str(start_ind)+'_2.png'
#     plt.savefig(plt_path)
#     print ('saved viz',plt_path)
#     plt.close(fig)



# print('Done.')
# fadsfa



















































