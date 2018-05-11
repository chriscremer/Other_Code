


#Train vae on doom

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


# Converts and down-samples the input image
def preprocess(img):
    img = img.astype(np.float32)
    img = img / 255.
    return img

def preprocess_pytorch(img):
    img = img.float()
    img = img / 255.
    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 3
        state_shape = (capacity, channels, resolution[0], resolution[1])

        self.s1 = torch.zeros(state_shape, dtype=torch.uint8).cuda()

        # self.s1 = torch.zeros(state_shape).cuda()


        self.s2 = torch.zeros(state_shape, dtype=torch.uint8).cuda()
        self.a = torch.zeros(capacity, dtype=torch.uint8).cuda()
        self.r = torch.zeros(capacity, dtype=torch.uint8).cuda()
        self.isterminal = torch.zeros(capacity, dtype=torch.uint8).cuda()
        self.capacity = capacity
        self.size = 0
        self.pos = 0

    # def add_s1(self, s1):
    #     self.s1[self.pos, :, :, :] = s1

    #     self.pos = (self.pos + 1) % self.capacity
    #     self.size = min(self.size + 1, self.capacity)

    def add_transition(self, s1, action, s2, isterminal, reward):
        # self.s1[self.pos, 0, :, :] = s1
        self.s1[self.pos, :, :, :] = torch.from_numpy(s1).cuda()
        action = np.array([action])
        self.a[self.pos] = torch.from_numpy(action).cuda()
        if not isterminal:
            # self.s2[self.pos, 0, :, :] = s2
            self.s2[self.pos, :, :, :] = torch.from_numpy(s2).cuda()

        isterminal = np.array([isterminal]).astype(np.uint8)
        self.isterminal[self.pos] = torch.from_numpy(isterminal).cuda()

        reward = np.array([reward])
        self.r[self.pos] = torch.from_numpy(reward).cuda()

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)




class Net(nn.Module):
    def __init__(self, n_channels, action_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, 12, stride=8)
        self.conv2 = nn.Conv2d(32, 64, 8, stride=4)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=2)

        self.act_func = F.leaky_relu
        self.intermediate_size = 1536

        self.fc1 = nn.Linear(self.intermediate_size, 100)
        self.fc2 = nn.Linear(100, action_size)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

    def forward(self, x):
        x = self.act_func(self.conv1(x))
        x = self.act_func(self.conv2(x))
        x = self.act_func(self.conv3(x))
        x = x.view(-1, self.intermediate_size)
        x = self.act_func(self.fc1(x))
        return self.fc2(x)






class BLUR_PREDICTOR(nn.Module):
    def __init__(self):
        super(BLUR_PREDICTOR, self).__init__()

        image_channels = 3
        self.z_size = 100

        self.act_func = F.leaky_relu# F.relu # # F.tanh ##  F.elu F.softplus
        self.intermediate_size = 1536

        #MASK PREDICTOR
        self.conv1_gp = nn.Conv2d(image_channels, 32, 12, stride=8)
        self.conv2_gp = nn.Conv2d(32, 64, 8, stride=4)
        self.conv3_gp = nn.Conv2d(64, 32, 3, stride=2)

        self.fc1_gp = nn.Linear(self.intermediate_size, 200)
        self.fc2_gp = nn.Linear(200, self.z_size)
        self.fc3_gp = nn.Linear(self.z_size, 200)
        self.fc4_gp = nn.Linear(200, self.intermediate_size)

        # self.deconv1_gp = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)#, output_padding=(0,1))
        # self.deconv2_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=8, stride=2)#, output_padding=(3,3))
        # self.deconv3_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=12, stride=4)#, output_padding=(4,4))
       
        self.deconv1_gp = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, output_padding=(0,1))
        self.deconv2_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=8, stride=4, output_padding=(3,3))
        self.deconv3_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=12, stride=8, output_padding=(4,4))



        params_gp = [list(self.conv1_gp.parameters()) + list(self.conv2_gp.parameters()) + list(self.conv3_gp.parameters()) +
                    list(self.fc1_gp.parameters()) + list(self.fc2_gp.parameters()) + 
                    list(self.fc3_gp.parameters()) + list(self.fc4_gp.parameters()) +
                    list(self.deconv1_gp.parameters()) + list(self.deconv2_gp.parameters()) + list(self.deconv3_gp.parameters())]
         

                    
        self.optimizer = optim.Adam(params_gp[0], lr=.0001, weight_decay=.0000001)




    def predict_precision(self, x):
        # print (x.size())  #[B,32,480,640]
        x = self.act_func(self.conv1_gp(x))  #[B,32,59,79]
        # print (x.size())
        x = self.act_func(self.conv2_gp(x)) #[B,64,13,18]
        # print (x.size())
        x = self.act_func(self.conv3_gp(x)) #[B,32,6,8]
        # print (x.size())
        # print()
        

        # print (x.size())
        x = x.view(-1, self.intermediate_size)
        h1 = self.act_func(self.fc1_gp(x))
        z = self.fc2_gp(h1)
        z = self.act_func(self.fc3_gp(z)) 
        z = self.act_func(self.fc4_gp(z))  #[B,11264]
        z = z.view(-1, 32, 6, 8)
        z = self.act_func(self.deconv1_gp(z)) #[B,64,13,18]
        # print (z.size())
        z = self.act_func(self.deconv2_gp(z)) #[B,64,59,79]
        # print (z.size())
        z = self.deconv3_gp(z) # [B,1,452,580]
        # print (z.size())
        # fdf

        return F.sigmoid(z)







    def blur_frame(self,frame):


        aaa = 0

        if aaa ==0:

            if torch.max(frame) > 1.:
                print ('DDDDDDDD')
                print (torch.max(frame).data.cpu().numpy())
                fasdf

            K = 21 #11 #21
            padding = 10# 5
            filter_weights = torch.ones(1,1,K,K).cuda()

            filter_weights = filter_weights / K**2
            frame_c0 = frame[:,0].unsqueeze(1)
            # print (frame_c0.shape)
            frame_c0 = F.conv2d(input=frame_c0, weight=filter_weights, bias=None, padding=padding, stride=1, dilation=1)
            # print (frame_c0.size())
            # print ('Output: [B,outC,outH,outW]')
            # print ()

            # print (torch.max(frame_c0).data.cpu().numpy())

            frame_c1 = frame[:,1].unsqueeze(1)
            frame_c1 = F.conv2d(input=frame_c1, weight=filter_weights, bias=None, padding=padding, stride=1, dilation=1)

            # print (torch.max(frame_c1).data.cpu().numpy())


            frame_c2 = frame[:,2].unsqueeze(1)
            frame_c2 = F.conv2d(input=frame_c2, weight=filter_weights, bias=None, padding=padding, stride=1, dilation=1)

            # print (torch.max(frame_c2).data.cpu().numpy())
            # fdsfa

            blurred_image = [frame_c0, frame_c1, frame_c2]
            blurred_image = torch.stack(blurred_image, dim=1)

            # print (blurred_image.shape)

            blurred_image = blurred_image.squeeze(dim=2)  #[B,3,480,640]

            # blurred_image = blurred_image / torch.max(blurred_image)
            blurred_image = torch.clamp(blurred_image, max=1.0)

            # print (torch.max(blurred_image).data.cpu().numpy())
            # fas

        else:
            blurred_image = torch.zeros(frame.size()[0],3,480,640).cuda()

        return blurred_image








    def forward(self, frame, DQNs):
        # x: [B,2,84,84]
        self.B = frame.size()[0]

        blurred_frame = self.blur_frame(frame)

        #Predict mask
        blur_weighting = self.predict_precision(frame)  #[B,1,480,640]
        blur_weighting = blur_weighting.repeat(1,3,1,1)

        mixed_frame = frame * blur_weighting + (1.-blur_weighting)*blurred_frame


        difs= []
        for i in range(len(DQNs)):
            q_mask = DQNs[i](mixed_frame)
            q_real = DQNs[i](frame)

            dif = torch.mean((q_mask-q_real)**2)  #[B,A]
            difs.append(dif)

        difs = torch.stack(difs)
        dif = torch.mean(difs)


        blur_weighting = blur_weighting.view(self.B, -1)
        mask_sum = torch.mean(torch.sum(blur_weighting, dim=1)) * .0000001

        loss = dif + mask_sum

        return loss, dif, mask_sum



    def train2(self, epochs, DQN, memory, game):

        batch_size = 32
        
        replay_memory_size = 1000
        play_steps = 100
        learning_steps = 200

        frame_repeat = 12
        display_step = 200 
        



        loss_list = []
        total_steps = 0
        epoch_time = 0.0
        
        for epoch in range(epochs):

            start_time = time.time()

            # print("Playing...")
            game.new_episode()
            # game_step_count = 0
            # game_steps = []
            
            for play_step in range(play_steps):

                s1 = game.get_state().screen_buffer  #[3,480,640] uint8
                # a = get_eps_action(epoch, actions, preprocess(s1))
                s1 = s1.reshape([1, 3, resolution[0], resolution[1]])
                state = torch.from_numpy(preprocess(s1)).cuda()
                # best_action_index = get_best_action(state)
                # print (state.size())
                # fasdf
                q = DQN[0](Variable(state))
                val, index = torch.max(q, 1)
                index = index.data.cpu().numpy()[0]
                action = actions[index]

                reward = game.make_action(action, frame_repeat) / float(frame_repeat)
                isterminal = game.is_episode_finished()
                # game_step_count +=1
                s2 = game.get_state().screen_buffer if not isterminal else None
                memory.add_transition(s1, index, s2, isterminal, reward)

                if game.is_episode_finished():
                    # score = game.get_total_reward()
                    # train_scores.append(score)
                    # train_episodes_finished += 1
                    # game_steps.append(game_step_count)
                    game.new_episode()
                    # game_step_count = 0

            
                    



            for learning_step in range(learning_steps):

                # if learning_step%10==0:
                #     print (learning_step)

                idxs = sample(range(0, memory.size), batch_size)
                s1 = memory.s1[idxs]        

                batch = Variable(preprocess_pytorch(s1))

                self.optimizer.zero_grad()
                loss, dif, prec_sum= self.forward(batch, DQN)
                loss.backward()
                self.optimizer.step()

                if total_steps%display_step==0: # and batch_idx == 0:
                    print ('Train Epoch: {}/{}'.format(epoch, epochs),
                        'epoch_time:{:.2f}'.format(epoch_time),
                        'Loss:{:.4f}'.format(loss.data.item()),
                        'dif:{:.4f}'.format(dif.data.item()),
                        'prec_sum:{:.4f}'.format(prec_sum.data.item()),
                        )

                    if total_steps!=0:
                        loss_list.append(loss.data.item())


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











class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        image_channels = 3
        self.z_size = 100

        self.act_func = F.leaky_relu# F.relu # # F.tanh ##  F.elu F.softplus
        self.intermediate_size = 1536

        #MASK PREDICTOR
        self.conv1_gp = nn.Conv2d(image_channels, 32, 12, stride=8)
        self.conv2_gp = nn.Conv2d(32, 64, 8, stride=4)
        self.conv3_gp = nn.Conv2d(64, 32, 3, stride=2)

        self.fc1_gp = nn.Linear(self.intermediate_size, 200)
        self.fc2_gp = nn.Linear(200, self.z_size)
        self.fc3_gp = nn.Linear(self.z_size, 200)
        self.fc4_gp = nn.Linear(200, self.intermediate_size)

        # self.deconv1_gp = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)#, output_padding=(0,1))
        # self.deconv2_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=8, stride=2)#, output_padding=(3,3))
        # self.deconv3_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=12, stride=4)#, output_padding=(4,4))
       
        self.deconv1_gp = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, output_padding=(0,1))
        self.deconv2_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=8, stride=4, output_padding=(3,3))
        self.deconv3_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=12, stride=8, output_padding=(4,4))



        params_gp = [list(self.conv1_gp.parameters()) + list(self.conv2_gp.parameters()) + list(self.conv3_gp.parameters()) +
                    list(self.fc1_gp.parameters()) + list(self.fc2_gp.parameters()) + 
                    list(self.fc3_gp.parameters()) + list(self.fc4_gp.parameters()) +
                    list(self.deconv1_gp.parameters()) + list(self.deconv2_gp.parameters()) + list(self.deconv3_gp.parameters())]
         

                    
        self.optimizer = optim.Adam(params_gp[0], lr=.0001, weight_decay=.0000001)




    def reconstruct(self, x):
        # print (x.size())  #[B,32,480,640]
        x = self.act_func(self.conv1_gp(x))  #[B,32,59,79]
        # print (x.size())
        x = self.act_func(self.conv2_gp(x)) #[B,64,13,18]
        # print (x.size())
        x = self.act_func(self.conv3_gp(x)) #[B,32,6,8]
        # print (x.size())
        # print()
        

        # print (x.size())
        x = x.view(-1, self.intermediate_size)
        h1 = self.act_func(self.fc1_gp(x))
        z = self.fc2_gp(h1)
        z = self.act_func(self.fc3_gp(z)) 
        z = self.act_func(self.fc4_gp(z))  #[B,11264]
        z = z.view(-1, 32, 6, 8)
        z = self.act_func(self.deconv1_gp(z)) #[B,64,13,18]
        # print (z.size())
        z = self.act_func(self.deconv2_gp(z)) #[B,64,59,79]
        # print (z.size())
        z = self.deconv3_gp(z) # [B,1,452,580]
        # print (z.size())
        # fdf

        # return F.sigmoid(z)
        return z





    def forward(self, frame): #, DQNs):
        # x: [B,2,84,84]
        self.B = frame.size()[0]

        recon = self.reconstruct(frame)  #[B,3,480,640]

        loss = F.binary_cross_entropy_with_logits(input=recon, target=frame)

        return loss #, dif, mask_sum











    def train2(self, epochs, DQN, memory, game, blurnet):
    # def train2(self, epochs, DQN, game):

        batch_size = 32
        
        replay_memory_size = 1000
        play_steps = 100
        learning_steps = 200

        frame_repeat = 12
        display_step = 200 
        



        loss_list = []
        total_steps = 0
        epoch_time = 0.0
        
        for epoch in range(epochs):

            start_time = time.time()

            # print("Playing...")
            game.new_episode()
            # game_step_count = 0
            # game_steps = []
            
            for play_step in range(play_steps):

                s1 = game.get_state().screen_buffer  #[3,480,640] uint8
                # a = get_eps_action(epoch, actions, preprocess(s1))
                s1 = s1.reshape([1, 3, resolution[0], resolution[1]])

                state = torch.from_numpy(preprocess(s1)).cuda().float()




                # blurred_frame = blurnet.blur_frame(state)
                # blur_weighting = blurnet.predict_precision(state)  #[B,1,480,640]
                # blur_weighting = blur_weighting.repeat(1,3,1,1)        
                # mixed_frame = state * blur_weighting + (1.-blur_weighting)*blurred_frame
                # s1 = mixed_frame #.data.cpu().numpy





                # best_action_index = get_best_action(state)
                # print (state.size())
                # fasdf
                q = DQN[0](Variable(state))
                val, index = torch.max(q, 1)
                index = index.data.cpu().numpy()[0]
                action = actions[index]

                reward = game.make_action(action, frame_repeat) / float(frame_repeat)
                isterminal = game.is_episode_finished()
                # game_step_count +=1
                s2 = game.get_state().screen_buffer if not isterminal else None

                memory.add_transition(s1, index, s2, isterminal, reward)
                # memory.add_s1(s1)

                if game.is_episode_finished():
                    # score = game.get_total_reward()
                    # train_scores.append(score)
                    # train_episodes_finished += 1
                    # game_steps.append(game_step_count)
                    game.new_episode()
                    # game_step_count = 0

            
                    



            for learning_step in range(learning_steps):

                # if learning_step%10==0:
                #     print (learning_step)

                idxs = sample(range(0, memory.size), batch_size)
                s1 = memory.s1[idxs]        

                # batch = s1 #Variable(preprocess_pytorch(s1))
                batch = Variable(preprocess_pytorch(s1))

                blurred_frame = blurnet.blur_frame(batch)
                blur_weighting = blurnet.predict_precision(batch)  #[B,1,480,640]
                blur_weighting = blur_weighting.repeat(1,3,1,1)        
                mixed_frame = batch * blur_weighting + (1.-blur_weighting)*blurred_frame
                batch = mixed_frame #.data.cpu().numpy



                self.optimizer.zero_grad()
                loss = self.forward(batch) #, DQN)
                loss.backward()
                self.optimizer.step()

                if total_steps%display_step==0: # and batch_idx == 0:
                    print ('Train Epoch: {}/{}'.format(epoch, epochs),
                        'epoch_time:{:.2f}'.format(epoch_time),
                        'Loss:{:.4f}'.format(loss.data.item()),
                        # 'dif:{:.4f}'.format(dif.data.item()),
                        # 'prec_sum:{:.4f}'.format(prec_sum.data.item()),
                        )

                    if total_steps!=0:
                        loss_list.append(loss.data.item())


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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'# '1' # '0' #,1'

    resolution = (480, 640)

    epochs = 500
    replay_memory_size = 1000

    exp_path = home + '/Documents/tmp/Doom/'


    # exp_path_2 = exp_path + 'blur_v2_clampblur/'
    # exp_path_2 = exp_path + 'blur_v2_moretrainin500_upto1500/'
    exp_path_2 = exp_path + 'doom_vae_withblur/'
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




    # train_maskpredictor_ = 0
    train_vae_ = 0
    viz_ = 1

    start_epoch = 140 # 270 #1000 # 570 #500




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




    print("Initializing DQNs...")

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
    DQN_loadfile = exp_path + 'training_3/DQN_params_999.pth'
    # if load_model:
    print("Loading model from: ", DQN_loadfile)
    DQN3 = torch.load(DQN_loadfile)
    DQN3.cuda()
    DQNs =[DQN3]

    print("DQNs initialized\n")
    





    print("Initializing VAE...")
    vae = VAE()
    if start_epoch>0:
        load_params_v3(save_dir=exp_path_2, model=vae, epochs=start_epoch)
        # load_params_v3(save_dir=exp_path + 'blur_v2_moretrainin500/', model=vae, epochs=start_epoch)
    vae.cuda()
    # n_gpus = 2

    # print (torch.cuda.device_count())
    # fads

    # MP = torch.nn.DataParallel(MP, device_ids=range(n_gpus)).cuda()
    # MP = torch.nn.DataParallel(MP, device_ids=[1])
    # print (MP)
    print("VAE initialized\n")
    



    print("Initializing BlurNet...")
    MP = BLUR_PREDICTOR()
    if start_epoch>0:
        # load_params_v3(save_dir=exp_path_2, model=MP, epochs=start_epoch)
        load_params_v3(save_dir=exp_path + 'blur_v2_moretrainin500_upto1500/', model=MP, epochs=1360)
    MP.cuda()
    # n_gpus = 2

    # print (torch.cuda.device_count())
    # fads

    # MP = torch.nn.DataParallel(MP, device_ids=range(n_gpus)).cuda()
    # MP = torch.nn.DataParallel(MP, device_ids=[1])
    # print (MP)
    print("BlurNet initialized\n")
    







    if train_vae_:


        print("Initializing Buffer...")
        memory = ReplayMemory(capacity=replay_memory_size)
        print("Buffer initialized\n")


        print("Training")
        # MP.module.train2(epochs=epochs, DQN=DQNs, memory=memory, game=game)
        vae.train2(epochs=epochs, DQN=DQNs, memory=memory, game=game, blurnet=MP)

        save_params_v3(save_dir=exp_path_2, model=vae, epochs=epochs+start_epoch)

















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

        max_count = 300 #1000
        count = 0
        frame_count = 0
        

        cols = 2
        rows = 1


        # frames = []

        game.new_episode()
        while not game.is_episode_finished() and count<max_count:

            # print (count)

            frame = game.get_state().screen_buffer #[3,480,640] uint8

            if count %12==0:
                state = preprocess(frame)
                state = state.reshape([1, 3, resolution[0], resolution[1]])
                state = torch.from_numpy(state).cuda()
                # a = get_best_action(state) #scalar
                q = DQNs[0](Variable(state))
                m, index = torch.max(q, 1)
                a = index.data.cpu().numpy()[0]


            reward = game.make_action(actions[a], 1)

            if count %4==0:

                # #Get grad of real
                # x = Variable(torch.from_numpy(np.array([preprocess(frame)])).float(), requires_grad=True).cuda()
                # # print (x.size())  #[1,3,480,640]
                # q = DQNs[0](x) #[1,A]
                # m, index = torch.max(q, 1)
                # val = q[:,index]
                # grad = torch.autograd.grad(val, x)[0]
                # grad = grad.data.cpu().numpy()[0] #for the first one in teh batch -> [2,84,84]
                # grad = np.abs(grad)  #[3,480,640]
                # # print (grad.shape)
                # grad = np.rollaxis(grad, 1, 0)
                # grad = np.rollaxis(grad, 2, 1)
                # grad = np.mean(grad, 2) #[480,640]



                # #Get blur weighting
                # x = Variable(torch.from_numpy(np.array([preprocess(frame)])).float()).cuda()
                # # print (x.size())  #[1,3,480,640]
                # # fdsfas

                # blur_weighting = MP.predict_precision(x) #[1,1,480,640]


                # # # Blur frame
                # # K = 11 #21
                # # padding = 5
                # # filter_weights = torch.ones(3,3,K,K).cuda()
                # # filter_weights = filter_weights / K**2
                # # blurred_frame = F.conv2d(x, filter_weights, padding=padding, stride=1)

                # # print (x.shape)
                # blurred_frame = MP.blur_frame(x) #[1,3,480,640]

                # # print (blurred_frame.shape)
                # # fsda

                # # q_b = DQNs[0](blurred_frame.unsqueeze(0)) #[1,A]
                # q_b = DQNs[0](blurred_frame) #[1,A]

                # #Get q of blurred






                # # blurred_frame = F.upsample(input=blurred_frame, size=(480,640), mode='bilinear')
                # # print (blurred_frame)
                # # fds


                # # masked_frame = x * mask
                # mixed_frame = x * blur_weighting + (1.-blur_weighting)* blurred_frame












                # # masked_frame = x * mask
                # # masked_frame = x * mask + (1.-mask)* F.sigmoid(MP.bias_frame.bias_frame)
                # masked_frame = mixed_frame


                # # print (masked_frame.shape)
                # # print (blurred_frame.shape)
                # # fds


                # #Get grad of masked
                # # x = Variable(torch.from_numpy(np.array([masked_frame)])).float(), requires_grad=True).cuda()
                # # print (x.size())  #[1,3,480,640]
                # q_m = DQNs[0](masked_frame) #[1,A]
                # m, index = torch.max(q_m, 1)
                # val = q_m[:,index]
                # grad_m = torch.autograd.grad(val, masked_frame)[0]
                # grad_m = grad_m.data.cpu().numpy()[0] #for the first one in teh batch -> [2,84,84]
                # grad_m = np.abs(grad_m)  #[3,480,640]
                # # print (grad.shape)
                # grad_m = np.rollaxis(grad_m, 1, 0)
                # grad_m = np.rollaxis(grad_m, 2, 1)
                # grad_m = np.mean(grad_m, 2) #[480,640]

                # masked_frame = masked_frame.squeeze() #[3,480.640]
                # masked_frame = masked_frame.data.cpu().numpy()
                # masked_frame = np.rollaxis(masked_frame, 1, 0)
                # masked_frame = np.rollaxis(masked_frame, 2, 1)

                # # mask = mask.squeeze()
                # # mask = mask.data.cpu().numpy()#[0] #for the first one in teh batch -> [2,84,84]



                # frames.append(frame)
                plt_path = gif_dir+'frame'+str(frame_count)+'.png'
                # fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=640*rows)
                fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=150)


                #Get recon
                # print (preprocess(np.array([frame])).shape)  #[1,3,480,640]

                prepro_frame = torch.from_numpy(preprocess(np.array([frame]))).cuda()
                recon = F.sigmoid(vae.reconstruct(prepro_frame))

                recon = recon.data.cpu().numpy()[0]
                recon = np.rollaxis(recon, 1, 0)
                recon = np.rollaxis(recon, 2, 1)
                # print (recon.shape)

                # dfasf
                # F.sigmoid(vae.reconstruct(torch.from_numpy(np.array([frame)]))))





                #Plot Frame
                ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
                frame = np.rollaxis(frame, 1, 0)
                frame = np.rollaxis(frame, 2, 1)
                # print (np.max(frame))
                # print (np.min(frame))
                # print (np.mean(frame))
                # print ()
                ax.imshow(frame) #, cmap='gray')
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






                # #Plot Blur Weighting
                # ax = plt.subplot2grid((rows,cols), (1,2), frameon=False)
                # # ax.imshow(np.uint8(mask), cmap='gray')

                # # print (np.max(mask))
                # # print (np.min(mask))
                # # print (np.mean(mask))
                # # print ()
                # # print (blur_weighting.data.cpu().numpy().shape)
                # blur_weighting = blur_weighting.data.cpu().numpy()[0][0]
                # # fds
                # ax.imshow(blur_weighting, cmap='gray')
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax.text(0.4, 1.04, 'Blur Weighting', transform=ax.transAxes, family='serif', size=6)







                # #Plot Blurred frame
                # ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)

                # # print (blurred_frame.shape) #[1,3,.,.]
                # # fads
                # blurred_frame = blurred_frame.data.cpu().numpy()[0]

                # # bias_frame = bias_frame.squeeze() 

                # # bias_frame = bias_frame.data.cpu().numpy() *255.
                # blurred_frame = np.rollaxis(blurred_frame, 1, 0)
                # blurred_frame = np.rollaxis(blurred_frame, 2, 1)

                # ax.imshow(blurred_frame) #, cmap='gray')
                # # ax.imshow(bias_frame) #, cmap='gray')
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax.text(0.4, 1.04, 'Blurred Frame', transform=ax.transAxes, family='serif', size=6)





                # #Plot Grad Real
                # ax = plt.subplot2grid((rows,cols), (1,0), frameon=False)
                # ax.imshow(grad, cmap='gray')
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax.text(0.35, 1.04, 'Grad of Real', transform=ax.transAxes, family='serif', size=6)





                # #Plot Grad Masked
                # ax = plt.subplot2grid((rows,cols), (1,1), frameon=False)
                # ax.imshow(grad_m, cmap='gray')
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax.text(0.3, 1.04, 'Grad of Mixed', transform=ax.transAxes, family='serif', size=6)




                # #Plot Q values
                # width = .2
                # ax = plt.subplot2grid((rows,cols), (1,1), frameon=False)

                # q = q.squeeze().data.cpu().numpy()
                # q_m = q_m.squeeze().data.cpu().numpy()
                # q_b = q_b.squeeze().data.cpu().numpy()

                # ax.bar(range(len(q)), q, width=width)

                # ax.bar(np.array(range(len(q_m)))+width, q_m, width=width)
                # ax.bar(np.array(range(len(q_b)))+width+width, q_b, width=width)


                # ax.set_xticks(range(len(q_m)))
                # # if i == rows-1:
                # ax.set_xticklabels([ 'NOOP', 'RIGHT', 'LEFT', 'NOOP'], size=5)
                # # else:
                # #     ax.set_xticklabels([])
                # ax.yaxis.set_tick_params(labelsize=5)
                # ax.xaxis.set_tick_params(labelsize=5)
                # ax.set_ylim([0,40.])

                # ax.text(0.34, .9, 'Q Values', transform=ax.transAxes, family='serif', size=6)






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

        gif_path_this = gif_dir+ 'gif_vae'+str(numb)+'.gif'
        # imageio.mimsave(gif_path_this, images)
        imageio.mimsave(gif_path_this, images, duration=.1)
        print ('made gif', gif_path_this)
        fds
































































