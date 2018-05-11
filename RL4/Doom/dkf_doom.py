

#Train dkf on doom

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
from torch.autograd import Variable



import time


from tqdm import trange

import skimage.color, skimage.transform

from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import imageio
import os
import math


# def preprocess(img):
#     img = img.astype(np.float32)
#     img = img / 255.
#     return img

# def preprocess_pytorch(img):
#     img = img.float()
#     img = img / 255.
#     return img

def preprocess_action(action):
    #onehot
    n_actions = 4
    B = action.size()[0]
    y_onehot = torch.FloatTensor(B, n_actions).zero_().cuda()
    y_onehot.scatter_(1, action.view(B,1).long(), 1)
    return y_onehot
 
def preprocess2(img):
    # numpy to pytoch, float, scale, cuda, downsample
    img = torch.from_numpy(img).cuda().float() / 255.
    img = F.avg_pool2d(img, kernel_size=2, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
    # print (img.shape) #[3,240,320]
    # fsdaf
    return img 



class ReplayMemory:
    def __init__(self, capacity):
        channels = 3
        state_shape = (capacity, channels, resolution[0], resolution[1])

        self.s1 = torch.zeros(state_shape, dtype=torch.uint8).cuda()
        self.s2 = torch.zeros(state_shape, dtype=torch.uint8).cuda()
        self.a = torch.zeros(capacity, dtype=torch.uint8).cuda()
        self.r = torch.zeros(capacity, dtype=torch.uint8).cuda()
        self.isterminal = torch.zeros(capacity, dtype=torch.uint8).cuda()
        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, :] = s1
        self.s2[self.pos, :, :, :] = s2
        self.a[self.pos] = action
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)




class DQN(nn.Module):
    def __init__(self, n_channels, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, 10, stride=5)
        self.conv2 = nn.Conv2d(32, 64, 7, stride=4)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=2)
        #Conv reminder: (w-k)/s + 1
        # (240-k)/s +1
        self.act_func = F.leaky_relu
        self.intermediate_size = 1120

        self.fc1 = nn.Linear(self.intermediate_size, 100)
        self.fc2 = nn.Linear(100, action_size)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=.0001)

    def forward(self, x):
        B = x.shape[0]
        # print (x.shape)
        x = self.act_func(self.conv1(x))
        # print (x.shape)
        x = self.act_func(self.conv2(x))
        # print (x.shape)
        x = self.act_func(self.conv3(x))
        # print (x.shape)
        # print (B, self.intermediate_size)
        # fadf
        x = x.view(B, self.intermediate_size)
        x = self.act_func(self.fc1(x))
        return self.fc2(x)







class DKF(nn.Module):
    def __init__(self):
        super(DKF, self).__init__()

        image_channels = 3
        self.z_size = 100
        self.action_size = 4

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


        #TRANSITION
        self.fc_tran1 = nn.Linear(self.z_size+self.action_size, 200)
        self.fc_tran2 = nn.Linear(200, self.z_size)

        #IsTerminal
        self.fc_isterm1 = nn.Linear(self.intermediate_size, 50)
        self.fc_isterm2 = nn.Linear(50, 1)

        params_gp = [list(self.conv1_gp.parameters()) + list(self.conv2_gp.parameters()) + list(self.conv3_gp.parameters()) +
                    list(self.fc1_gp.parameters()) + list(self.fc2_gp.parameters()) + 
                    list(self.fc3_gp.parameters()) + list(self.fc4_gp.parameters()) +
                    list(self.deconv1_gp.parameters()) + list(self.deconv2_gp.parameters()) + list(self.deconv3_gp.parameters()),
                    list(self.fc_tran1.parameters()) + list(self.fc_tran2.parameters()) +
                    list(self.fc_isterm1.parameters()) + list(self.fc_isterm2.parameters())]
         

                    
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
        h1 = self.act_func(self.fc1_gp(x))
        z = self.fc2_gp(h1)

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


        z = self.act_func(self.fc_isterm1(z_pre)) 
        isTerm = F.sigmoid(self.fc_isterm2(z))



        # return F.sigmoid(z)
        return x, isTerm



    def transition(self, za):

        #Resnet style
        z = self.act_func(self.fc_tran1(za))
        z = self.fc_tran2(z)
        # z = z + z_1 
        return z     



    def forward(self, s1, s2, a, isterminal):
        # x: [B,2,84,84]
        B = s1.size()[0]

        z1 = self.encode(s1) 
        za = torch.cat((z1.detach(),a),1) #[B,z+a]
        z2_prior = self.transition(za)
        z2 = self.encode(s2)
        s2_recon, isTerm = self.decode(z2) 

        z_loss = torch.mean(z2**2) * .001

        recon_loss = 10. * F.binary_cross_entropy_with_logits(input=s2_recon, target=s2) #, reduce=False)
        # recon_loss = recon_loss.view(B,-1)
        # recon_loss = torch.mean(torch.sum(recon_loss, dim=1))

        # tran_loss = torch.mean(torch.sum((z2.detach()-z2_prior)**2, dim=1))
        tran_loss = torch.mean((z2.detach()-z2_prior)**2)

        # if self.tmp:
        #     print (z2)
        #     print (z2_prior)
        #     print (tran_loss)

        # fsa
        # fads
        # tran_loss = torch.mean((z2-z2_prior)**2)

        # print (isTerm)
        # print (isterminal)
        # print (isTerm.shape)
        # print (isterminal.shape)
        # print (torch.sum((isTerm-isterminal)**2, dim=1).shape)
        # print (torch.sum((isTerm-isterminal)**2, dim=1))
        # fdsa

        # terminal_loss = torch.mean(torch.sum((isTerm-isterminal)**2, dim=1))
        terminal_loss = torch.mean((isTerm-isterminal)**2)

        loss = recon_loss + tran_loss + terminal_loss + z_loss

        return loss, recon_loss, tran_loss, terminal_loss, z_loss











    def train2(self, epochs, DQN, memory, game):
    # def train2(self, epochs, DQN, game):

        batch_size = 32
        
        replay_memory_size = 1000
        play_steps = 100
        learning_steps = 200

        frame_repeat = 12
        display_step = 200 
        
        self.tmp = 0


        loss_list = []
        total_steps = 0
        epoch_time = 0.0
        
        for epoch in range(epochs):

            start_time = time.time()

            game.new_episode()

            for play_step in range(play_steps):

                s1 = game.get_state().screen_buffer  #[3,480,640] uint8
                s1 = preprocess2(s1) #[3,240,320]
                s1 = s1.view([1, 3, resolution[0], resolution[1]])

                if not dqn_needed:
                # if DQN_loadfile=='':
                    action_i = np.random.randint(0,4)
                    action_cpu = actions[action_i]
                    action_i = torch.from_numpy(np.array([action_i]))
                else:
                    q = DQN[0](s1)
                    q_val, action_i = torch.max(q, 1)
                    action_cpu = actions[action_i.data.cpu().numpy()[0]]

                s2 = game.get_state().screen_buffer #if not isterminal else None
                s2 = preprocess2(s2) #[3,240,320]
                s2 = s2.view([1, 3, resolution[0], resolution[1]])

                reward = game.make_action(action_cpu, frame_repeat) / float(frame_repeat)
                isterminal = game.is_episode_finished()*1 #converts bool to int


                s1_uint8 = (s1*255.).type(torch.uint8)
                s2_uint8 = (s2*255.).type(torch.uint8)
                action_uint8 = (action_i).type(torch.uint8)
                isterminal_uint8 = torch.from_numpy(np.array([isterminal])).type(torch.uint8)
                reward_uint8 = torch.from_numpy(np.array([reward])).type(torch.uint8)

                memory.add_transition(s1_uint8, action_uint8, s2_uint8, isterminal_uint8, reward_uint8)

                if game.is_episode_finished():
                    # score = game.get_total_reward()
                    game.new_episode()

            
                    



            for learning_step in range(learning_steps):

                # if learning_step%10==0:
                #     print (learning_step)

                if total_steps%display_step==0:
                    self.tmp=1

                idxs = sample(range(0, memory.size), batch_size)
                s1 = memory.s1[idxs].float()   /255.    
                s2 = memory.s2[idxs].float()    /255.    

                # print (torch.max(s1))
                # print (torch.min(s1))
                # fsda
                a = memory.a[idxs]        
                isterminal = memory.isterminal[idxs]  

                self.optimizer.zero_grad()
                loss, recon_loss, tran_loss, terminal_loss, z_loss = self.forward(s1, 
                                                                        s2, 
                                                                        preprocess_action(a), 
                                                                        isterminal.float().view(batch_size,1)) #, DQN)
                loss.backward()
                self.optimizer.step()

                if total_steps%display_step==0: # and batch_idx == 0:
                    print ('Train Epoch: {}/{}'.format(epoch, epochs),
                        'epoch_time:{:.2f}'.format(epoch_time),
                        'Loss:{:.4f}'.format(loss.data.item()),
                        'recon:{:.4f}'.format(recon_loss.data.item()),
                        'tran:{:.4f}'.format(tran_loss.data.item()), 
                        'term:{:.4f}'.format(terminal_loss.data.item()),
                        'z:{:.4f}'.format(z_loss.data.item()),
                        )
                    if total_steps!=0:
                        loss_list.append(loss.data.item())

                self.tmp=0

                total_steps+=1

            epoch_time = time.time() - start_time

                
            if epoch % 10==0 and epoch > 2:
                #Save params
                save_params_v3(save_dir=exp_path_2, model=self, epochs=epoch+start_epoch)

                if len(loss_list) > 7:
                    #plot the training curve
                    plt.plot(loss_list[2:])
                    # save_dir = home+'/Documents/tmp/Doom/'
                    plt_path = exp_path_2+'training_plot.png'
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
    replay_memory_size = 1000

    exp_path = home + '/Documents/tmp/Doom/'


    # exp_path_2 = exp_path + 'blur_v2_clampblur/'
    # exp_path_2 = exp_path + 'blur_v2_moretrainin500_upto1500/'
    exp_path_2 = exp_path + 'doom_dkf_resized_z100/'
    # exp_path_2 = exp_path + '2DQNs_biasframe_v4/'


    if not os.path.exists(exp_path_2):
        os.makedirs(exp_path_2)
        print ('Made dir', exp_path_2) 
    # else:
    #     print ('exists')
    #     fasdf 

    # maskpredictor_savefile = exp_path + 'maskpredictor_5.pth'
    # maskpredictor_savefile = exp_path_2 + 'MP_params'+str(epochs)+'.pth'
    # maskpredictor_loadfile = exp_path + 'first_4.pth'

    run_what = 'train'
    # run_what = 'viz'

    if run_what == 'train':
        train_dkf_ = 1
        viz_ = 0
    else:
        train_dkf_ = 0
        viz_ = 1  

    start_epoch = 0 #160 #100 # 100 #200 #10 # 460 #0 # 110 # 0 # 270 #1000 # 570 #500




    # Create Doom instance
    print("Initializing doom...")
    # config_file_path = "../../scenarios/simpler_basic.cfg"
    # config_file_path = "../../scenarios/rocket_basic.cfg"
    # config_file_path = "../../scenarios/basic.cfg"
    config_file_path = home + "/ViZDoom/scenarios/take_cover.cfg"
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    # game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.\n")
    



    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    print (game.get_available_buttons()) #_size()
    # fasf
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    print ('actions', actions)
    # fadsfsa




    
    dqn_needed = 0
    if dqn_needed:
        print("\nInitializing DQNs...")
        # #trained for 500
        # DQN_loadfile = exp_path + 'first_4.pth'
        # # if load_model:
        # print("Loading model from: ", DQN_loadfile)
        # DQN = torch.load(DQN_loadfile)
        # DQN.cuda()

        # #trained for 2000, new seed
        # DQN_loadfile = exp_path + 'training_2/DQN_params_1999.pth'
        # # if load_model:
        # print("Loading model from: ", DQN_loadfile)
        # DQN2 = torch.load(DQN_loadfile)
        # DQN2.cuda()

        # DQNs =[DQN, DQN2]


        #this is the one with the smaller stride
        # DQN_loadfile = exp_path + 'training_3/DQN_params_999.pth'
        DQN_loadfile = ''
        if DQN_loadfile != '':
            # if load_model:
            print("Loading model from: ", DQN_loadfile)
            # DQN3 = torch.load(DQN_loadfile)
        else:
            DQN3 = DQN(n_channels=3, action_size=4)
        DQN3.cuda()
        DQNs =[DQN3]
        print("DQNs initialized\n")

    else:
        DQNs =[]
        print("\nDQN not required")
    





    print("Initializing DKF...")
    dkf = DKF()
    if start_epoch>0:
        load_params_v3(save_dir=exp_path_2, model=dkf, epochs=start_epoch)
        # load_params_v3(save_dir=exp_path + 'blur_v2_moretrainin500/', model=vae, epochs=start_epoch)
    dkf.cuda()
    # n_gpus = 2

    # print (torch.cuda.device_count())
    # fads

    # MP = torch.nn.DataParallel(MP, device_ids=range(n_gpus)).cuda()
    # MP = torch.nn.DataParallel(MP, device_ids=[1])
    # print (MP)
    print("DKF initialized\n")
    









    if train_dkf_:


        print("Initializing Buffer...")
        memory = ReplayMemory(capacity=replay_memory_size)
        print("Buffer initialized\n")


        print("Training")
        # MP.module.train2(epochs=epochs, DQN=DQNs, memory=memory, game=game)
        dkf.train2(epochs=epochs, DQN=DQNs, memory=memory, game=game)

        save_params_v3(save_dir=exp_path_2, model=dkf, epochs=epochs+start_epoch)

















    if viz_:

        # load_params_v3(save_dir=exp_path, model=MP, epochs=20)




        numb = start_epoch
        gif_dir = exp_path_2 + 'gif_frames'+str(numb)+'/'


        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
            print ('Made dir', gif_dir) 
        else:
            print (gif_dir, 'exists')
            # fasdf 

        max_count = 100 #1000
        count = 0
        frame_count = 0
        

        cols = 2
        rows = 1


        # frames = []

        game.new_episode()
        while not game.is_episode_finished() and count<max_count:

            # print (count)

            frame = game.get_state().screen_buffer #[3,480,640] uint8

            s1 = preprocess2(frame) #[3,240,320] float, cuda
            s1 = s1.view([1, 3, resolution[0], resolution[1]])

            if count %12==0:

                if not dqn_needed:
                    # if DQN_loadfile=='':
                    action_i = np.random.randint(0,4)
                    action_cpu = actions[action_i]
                    action_i = torch.from_numpy(np.array([action_i]))
                else:
                    q = DQN[0](s1)
                    q_val, action_i = torch.max(q, 1)
                    action_cpu = actions[action_i.data.cpu().numpy()[0]]


                # state = preprocess(frame)
                # state = state.reshape([1, 3, resolution[0], resolution[1]])
                # state = torch.from_numpy(state).cuda()
                # # a = get_best_action(state) #scalar
                # q = DQNs[0](Variable(state))
                # m, index = torch.max(q, 1)
                # a = index.data.cpu().numpy()[0]


            reward = game.make_action(action_cpu, 1)

            if count %4==0:

                plt_path = gif_dir+'frame'+str(frame_count)+'.png'
                # fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=640*rows)
                fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=150)



                # prepro_frame = torch.from_numpy(preprocess(np.array([frame]))).cuda()
                # s1 = preprocess2(frame) #[3,240,320]
                # s1 = s1.view([1, 3, resolution[0], resolution[1]])
                z1 = dkf.encode(s1) 
                s2_recon, isTerm = dkf.decode(z1) 
                recon = F.sigmoid(s2_recon)
                recon = recon.data.cpu().numpy()[0]
                recon = np.rollaxis(recon, 1, 0)
                recon = np.rollaxis(recon, 2, 1)

                frame = s1.data.cpu().numpy()[0] #[3,240,320]
                frame = np.rollaxis(frame, 1, 0)
                frame = np.rollaxis(frame, 2, 1)

                # print (np.max(recon))
                # print (np.min(recon))

                # print (np.max(frame))
                # print (np.min(frame))
                # fasdf


                #Plot Frame
                ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

                ax.imshow(frame)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.35, 1.04, 'Real Frame '+str(count), transform=ax.transAxes, family='serif', size=6)





                #Plot Recon
                ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)
                # ax.imshow(np.uint8(masked_frame*255.))
                ax.imshow(recon)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.4, 1.04, 'Reconstruction', transform=ax.transAxes, family='serif', size=6)




                # fsadfs
                # plt.tight_layout()
                plt.savefig(plt_path)
                print ('saved viz',plt_path)
                plt.close(fig)


                frame_count+=1

            count+=1

        

        print ('game over', game.is_episode_finished() )
        print (count)
        # print (len(frames))


        print('Making Gif')
        # frames_path = save_dir+'gif/'
        images = []
        for i in range(frame_count):
            images.append(imageio.imread(gif_dir+'frame'+str(i)+'.png'))

        gif_path_this = gif_dir+ 'gif_dkf'+str(numb)+'.gif'
        # imageio.mimsave(gif_path_this, images)
        imageio.mimsave(gif_path_this, images, duration=.1)
        print ('made gif', gif_path_this)
        fds
































































