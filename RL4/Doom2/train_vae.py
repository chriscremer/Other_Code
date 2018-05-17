
#Train vae on doom dataset

from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


import time


# from tqdm import trange

# import skimage.color, skimage.transform

from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import imageio
import os
import math
import pickle

def preprocess_action(action):
    #onehot
    n_actions = 4
    B = action.size()[0]
    y_onehot = torch.FloatTensor(B, n_actions).zero_().cuda()
    y_onehot.scatter_(1, action.view(B,1).long(), 1)
    return y_onehot
 

def preprocess(img):
    # numpy to pytoch, float, scale, cuda, downsample
    # img = torch.from_numpy(img).cuda().float() / 255.
    # img = (img.float() / 255.).cuda()
    img = img.cuda().float() / 255.
    # img = F.avg_pool2d(img, kernel_size=2, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
    # print (img.shape) #[3,240,320]
    # fsdaf
    return img 






class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        image_channels = 3
        self.z_size = 100

        self.act_func = F.leaky_relu# F.relu # # F.tanh ##  F.elu F.softplus
        self.intermediate_size = 1120 # 1728 #1536

        #ENCODER
        self.conv1_gp = nn.Conv2d(image_channels, 32, kernel_size=10, stride=5)
        self.conv2_gp = nn.Conv2d(32, 64, kernel_size=7, stride=4)
        self.conv3_gp = nn.Conv2d(64, 32, kernel_size=3, stride=2)
        self.fc1_gp = nn.Linear(self.intermediate_size, 200)
        self.fc2_gp = nn.Linear(200, self.z_size)


        #DECODER
        self.fc3_gp = nn.Linear(self.z_size, 200)
        self.fc4_gp = nn.Linear(200, self.intermediate_size)
        self.deconv1_gp = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)#, output_padding=(0,1))
        self.deconv2_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=7, stride=4)#, output_padding=(3,3))
        self.deconv3_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=10, stride=5)#, output_padding=(4,4))


        params_gp = [list(self.conv1_gp.parameters()) + list(self.conv2_gp.parameters()) + list(self.conv3_gp.parameters()) +
                    list(self.fc1_gp.parameters()) + list(self.fc2_gp.parameters()) + 
                    list(self.fc3_gp.parameters()) + list(self.fc4_gp.parameters()) +
                    list(self.deconv1_gp.parameters()) + list(self.deconv2_gp.parameters()) + list(self.deconv3_gp.parameters())]
         

                    
        self.optimizer = optim.Adam(params_gp[0], lr=.0001, weight_decay=.0000001)


    def encode(self, x):
        B = x.shape[0]
        # print (x.size())  #[B,32,480,640]
        x = self.act_func(self.conv1_gp(x))  #[B,32,59,79]
        # print (x.size())
        x = self.act_func(self.conv2_gp(x)) #[B,64,13,18]
        # print (x.size())
        x = self.act_func(self.conv3_gp(x)) #[B,32,5,7]
        # print (x.size())
        # fds
        x = x.view(B, self.intermediate_size)
        x = self.act_func(self.fc1_gp(x))
        z = self.fc2_gp(x)

        return z



    def decode(self, z):
        B = z.size()[0]
        z = self.act_func(self.fc3_gp(z)) 
        z_pre = self.act_func(self.fc4_gp(z))  #[B,11264]
        # print (z_pre.shape)
        z = z_pre.view(B, 32, 5, 7)
        z = self.act_func(self.deconv1_gp(z)) #[B,64,13,18]
        # print (z.size())
        z = self.act_func(self.deconv2_gp(z)) #[B,64,59,79]
        # print (z.size())
        x = self.deconv3_gp(z) # [B,1,452,580]
        # print (x.size())
        # fdf


        # z = self.act_func(self.fc_isterm1(z_pre)) 
        # isTerm = F.sigmoid(self.fc_isterm2(z))


        # return F.sigmoid(z)
        return x#, isTerm



    def forward(self, x):
        # x: [B,2,84,84]
        B = x.size()[0]

        z = self.encode(x) 
        # za = torch.cat((z1.detach(),a),1) #[B,z+a]
        # z2_prior = self.transition(za)
        # z2 = self.encode(s2)
        recon = self.decode(z) 

        # z_loss = torch.mean(z2**2) * .001

        recon_loss = F.binary_cross_entropy_with_logits(input=recon, target=x) #, reduce=False)

        loss = recon_loss #+ tran_loss + terminal_loss + z_loss

        return loss, recon_loss#, tran_loss, terminal_loss, z_loss











    def train2(self, epochs, trainingset, validationset, save_dir, start_epoch):

        batch_size = 32

        display_step = int(len(trainingset) / 32)  # 200 
        
        loss_list = []
        valid_loss_list = []
        total_steps = 0
        epoch_time = 0.0


        training_dataloader = DataLoader(trainingset, batch_size=batch_size, shuffle=True, num_workers=2)
        validation_dataloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=2)

        for epoch in range(epochs):

            start_time = time.time()
            for i_batch, batch in enumerate(training_dataloader):

                batch = preprocess(batch) #.cuda()

                self.optimizer.zero_grad()
                loss, recon_loss = self.forward(batch) #, DQN)
                loss.backward()
                self.optimizer.step()

                if total_steps%display_step==0: # and batch_idx == 0:

                    for i_batch, batch in enumerate(validation_dataloader):
                        batch = preprocess(batch) 
                        valid_loss, valid_recon_loss = self.forward(batch)
                        break 


                    print ('Train Epoch: {}/{}'.format(epoch+start_epoch, epochs+start_epoch),
                        'N_Steps:{:5d}'.format(total_steps),
                        'T:{:.2f}'.format(epoch_time),
                        'Loss:{:.4f}'.format(loss.data.item()),
                        'Recon:{:.4f}'.format(recon_loss.data.item()),
                        'ValidLoss:{:.4f}'.format(valid_loss.data.item()),
                        # 'tran:{:.4f}'.format(tran_loss.data.item()), 
                        # 'term:{:.4f}'.format(terminal_loss.data.item()),
                        )
                    if total_steps!=0:
                        loss_list.append(loss.data.item())
                        valid_loss_list.append(valid_loss.data.item())

                total_steps+=1

            epoch_time = time.time() - start_time

            if epoch % 10==0 and epoch > 2:
                #Save params
                save_params_v3(save_dir=save_dir, model=self, epochs=epoch+start_epoch)

                if len(loss_list) > 7:
                    #plot the training curve
                    plt.plot(loss_list[2:], label='Train')
                    plt.plot(valid_loss_list[2:], label='Valid')
                    # save_dir = home+'/Documents/tmp/Doom/'
                    plt_path = save_dir+'training_plot.png'
                    plt.legend()
                    plt.savefig(plt_path)
                    print ('saved training plot',plt_path)
                    plt.close()









def load_params_v3(save_dir, model, epochs):
    save_to=os.path.join(save_dir, "MP_params" + str(epochs)+".pt")
    model.load_state_dict(torch.load(save_to))
    print ('loaded', save_to)





def save_params_v3(save_dir, model, epochs):
    save_to=os.path.join(save_dir, "MP_params" + str(epochs)+".pt")
    torch.save(model.state_dict(), save_to)
    print ('saved', save_to)



































if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1' # '0' #,1'

    resolution = (240,320) # (480, 640)

    epochs = 500

    save_dir =  home+'/Documents/tmp/Doom2/' 
    load_dataset_from = save_dir+'doom_dataset_10000.pkl'

    exp_path = save_dir+'vae_with_valid/'

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        print ('Made dir', exp_path) 


    run_what = 'train'
    # run_what = 'viz'

    if run_what == 'train':
        train_dkf_ = 1
        viz_ = 0
    else:
        train_dkf_ = 0
        viz_ = 1  

    start_epoch = 0 #500





    print ("Load dataset")
    dataset = pickle.load( open( load_dataset_from, "rb" ) )
    print ('numb trajectories', len(dataset))
    # print ([len(x) for x in dataset])
    print ('Avg len', np.mean([len(x) for x in dataset]), np.mean([len(x) for x in dataset])*12)
    #Put all trajectories into one list
    dataset_states = []
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset_states.append(dataset[i][j][0])
    
    training_set = dataset_states[:9000]
    validation_set = dataset_states[9000:]


    class DoomDataset(Dataset):

        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    training_set = DoomDataset(data=training_set)
    validation_set = DoomDataset(data=validation_set)
    print ('training set:', len(training_set))
    print ('validation set:', len(validation_set))
    print()
    





    print("Initializing VAE...")
    vae = VAE()
    if start_epoch>0:
        load_params_v3(save_dir=exp_path, model=vae, epochs=start_epoch)
        # load_params_v3(save_dir=exp_path + 'blur_v2_moretrainin500/', model=vae, epochs=start_epoch)
    vae.cuda()
    # n_gpus = 2
    # print (torch.cuda.device_count())
    # fad
    # MP = torch.nn.DataParallel(MP, device_ids=range(n_gpus)).cuda()
    # MP = torch.nn.DataParallel(MP, device_ids=[1])
    # print (MP)
    print("VAE initialized\n")
    









    if train_dkf_:

        print("Training")
        vae.train2(epochs=epochs, trainingset=training_set, validationset=validation_set,
                     save_dir=exp_path, start_epoch=start_epoch)

        save_params_v3(save_dir=exp_path, model=vae, epochs=epochs+start_epoch)

















    if viz_:

        # load_from = save_dir+'doom_dataset_10000.pkl'


        # print ("Load dataset")
        # # dir_ = 'RoadRunner_colour_4'
        # # dataset = pickle.load( open( home+'/Documents/tmp/breakout_2frames/breakout_trajectories_10000.pkl', "rb" ) )
        # dataset = pickle.load( open( load_from, "rb" ) )

        # print ('numb trajectories', len(dataset))
        # print ([len(x) for x in dataset])
        # print ('Avg len', np.mean([len(x) for x in dataset]), np.mean([len(x) for x in dataset])*12)
        # print()



        rows = 1
        cols = 1

        traj_numb = 100
        print ('traj', str(traj_numb), 'len', str(len(dataset[traj_numb])))

        gif_dir = exp_path + 'view_traj_recon'+str(traj_numb)+'/'


        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
            print ('Made dir', gif_dir) 
        else:
            print (gif_dir, 'exists')
            # fasdf 

        max_count = 100000 #1000
        # count = 0    

        cols = 2
        rows = 1

        traj = dataset[traj_numb]
        for i in range(len(traj)):

            s = traj[i][0]
            a = traj[i][1]
            t = traj[i][2]

            plt_path = gif_dir+'frame'+str(i)+'.png'
            # fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=640*rows)
            fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=150)


            frame = np.rollaxis(s, 1, 0)
            frame = np.rollaxis(frame, 2, 1)

            s1 = preprocess(torch.from_numpy(np.array([s])))
            recon = F.sigmoid(vae.decode(vae.encode( s1)))
            recon = recon.data.cpu().numpy()[0]

            recon = np.rollaxis(recon, 1, 0)
            recon = np.rollaxis(recon, 2, 1)






            #Plot Frame
            ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.1, 1.04, 'Real Frame '+str(i)+'  a:'+str(a)+'  t:'+str(t), transform=ax.transAxes, family='serif', size=6)

            #Plot recon
            ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)
            ax.imshow(recon)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.1, 1.04, 'Recon', transform=ax.transAxes, family='serif', size=6)


            # plt.tight_layout()
            plt.savefig(plt_path)
            print ('saved viz',plt_path)
            plt.close(fig)

            # frame_count+=1

            # count+=1



        # print ('game over', game.is_episode_finished() )
        # print (count)
        # print (len(frames))


        print('Making Gif')
        # frames_path = save_dir+'gif/'
        images = []
        for i in range(len(traj)):
            images.append(imageio.imread(gif_dir+'frame'+str(i)+'.png'))

        gif_path_this = gif_dir+ 'gif_'+str(traj_numb)+'.gif'
        # imageio.mimsave(gif_path_this, images)
        imageio.mimsave(gif_path_this, images, duration=.1)
        print ('made gif', gif_path_this)






    print ('Done.')
































































