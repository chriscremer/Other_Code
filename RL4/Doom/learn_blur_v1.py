


# Learn to blur the image while keeping the action the same

#the idea was to output variance of gaussian filter for each pixel
# then use a locally connected net to compute the blurred pixel
# so the net could learn how much blur to apply to each pixel


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
        self.s2 = torch.zeros(state_shape, dtype=torch.uint8).cuda()
        self.a = torch.zeros(capacity, dtype=torch.uint8).cuda()
        self.r = torch.zeros(capacity, dtype=torch.uint8).cuda()
        self.isterminal = torch.zeros(capacity, dtype=torch.uint8).cuda()
        self.capacity = capacity
        self.size = 0
        self.pos = 0

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







# def conv2d_local(input, weight, bias=None, padding=0, stride=1, dilation=1):
#     if input.dim() != 4:
#         raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
#     if weight.dim() != 6:
#         # outH x outW x outC x inC x kH x kW
#         raise NotImplementedError("Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))
 
#     outH, outW, outC, inC, kH, kW = weight.size()
#     kernel_size = (kH, kW)
 
#     # N x [inC * kH * kW] x [outH * outW]
#     cols = F.unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
#     cols = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)
 
#     out = torch.matmul(cols, weight.view(outH * outW, outC, inC * kH * kW).permute(0, 2, 1))
#     out = out.view(cols.size(0), outH, outW, outC).permute(0, 3, 1, 2)
 
#     if bias is not None:
#         out = out + bias.expand_as(out)
#     return out



def batch_conv2d_local(input, weight, bias=None, padding=0, stride=1, dilation=1):
    if input.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
    if weight.dim() != 7:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 7D weight Tensors supported (got {}D)".format(weight.dim()))
 
    B, outH, outW, outC, inC, kH, kW = weight.size()
    kernel_size = (kH, kW)
 

    print(input.shape)
    # N x [inC * kH * kW] x [outH * outW]
    cols = F.unfold(input, kernel_size, dilation=dilation, padding=padding, stride=2)

    # print(cols.shape) 2764800
    # print(cols.shape) stride=2 691200

    fasdf




    # cols: [B,inC*kH*kW,outH*outW] = [32,1*5*5,480*640]
    cols = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)
    # cols: [B,outH*outW,1,inC*kH*kW] = [32,307200,1,25]
    # print (cols.shape)

    # weight: [B,outH*outW,outC,inC*kH*kW] = [32,307200,1,25]
    weight = weight.view(B, outH * outW, outC, inC * kH * kW)
    # print (weight.shape)
    weight = weight.permute(0, 1, 3, 2)
    # weight: [B,outH*outW,inC*kH*kW,outC] = [32,307200,25,1]

    #  [32,307200,1,25] * [32,307200,25,1] -> [32,307200,1,1]
    out = torch.matmul(cols, weight)
    # print (out.shape)
    out = out.view(cols.size(0), outH, outW, outC).permute(0, 3, 1, 2)
    #out: [32,1,480,640]
    # print (out.shape)
    # fad
    if bias is not None:
        out = out + bias.expand_as(out)
    return out






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

        return torch.exp(z)


    def generate_gaussian_filters(self, precisions):

        H = resolution[0]
        W = resolution[1]
        # stride = 1
        # padding = 2
        # dilation = 1
        # K = 5

        stride = 1
        padding = 1
        dilation = 1
        K = 3


        outH = int(math.floor((H + 2 * padding - dilation * (K - 1) - 1) / stride + 1))
        outW = int(math.floor((W + 2 * padding - dilation * (K - 1) - 1) / stride + 1))
        # # outH = int(np.floor((H + (2*padding) - (dilation*(K-1)-1) /stride) +1))
        # print ('stride',stride)
        # print ('padding',padding)
        # print ('dilation',dilation)
        # print ('K',K)
        # # print ()
        # print ('outH',outH) #480
        # print ('outW',outW) #640


        # prec = np.ones([outH,outW,1])
        x = np.linspace(-1, 1, K) #[k]
        x = np.reshape(x, [1,1,K])
        x = Variable(torch.from_numpy(x).float().cuda())

        precisions = precisions.view(-1, outH, outW, 1)

        filters = -precisions*(x**2)
        filters = torch.exp(filters)
        # filters = np.reshape(filters, [outH,outW,K,1]) * np.reshape(filters, [outH,outW,1,K])
        filters = filters.view(-1,outH,outW,K,1) * filters.view(-1,outH,outW,1,K) 
        # filters = np.reshape(filters, [outH,outW,1,1,K,K])  #
        filters = filters.view(-1,outH,outW,1,1,K,K) 

        # print (filters.shape)
        # print ('[B,M,N,1,1,K,K]')  
        # print ('[B,M,N,outC,inC,K,K]') 
        # print ()
        sum_ = torch.sum(filters, dim=6, keepdim=True)
        sum_ = torch.sum(sum_, dim=5, keepdim=True)
        filters = filters / sum_
        # filters = filters / np.mean(filters, axis=4, keepdims=1)
        # print (filters.shape)
        # fas
        # filters = torch.from_numpy(filters).float()

        # print (filters.size())
        # fasd

        # [B,outH,outN,1,1,K,K]
        return filters




    def forward(self, frame, DQNs):
        # x: [B,2,84,84]
        self.B = frame.size()[0]

        H = resolution[0]
        W = resolution[1]
        # stride = 1
        # padding = 2
        # dilation = 1
        # K = 5

        stride = 1
        padding = 1
        dilation = 1
        K = 3

        #Predict lowres
        precisions = self.predict_precision(frame)  #[B,1,480,680]
        # print (precisions.size())

        filters = self.generate_gaussian_filters(precisions)
        # print (filters.size())

        # print()
        # print ('First channel')
        frame_c0 = frame[:,0]
        frame_c0 = torch.unsqueeze(frame_c0, dim=1)
        # # frame_c0 = torch.unsqueeze(frame_c0, dim=0)
        # print (frame_c0.shape) #[B,1,H,W] = [B,C,H,W]
        # print ('Input: [B,C,H,W]')
        # fds

        # print ('Local Conv')
        # frame_c0 = torch.from_numpy(frame_c0).float()

        frame_c0 = batch_conv2d_local(input=frame_c0, weight=filters, bias=None, padding=padding, stride=1, dilation=1)

        # print (frame_c0.size())
        # print ('Output: [B,outC,outH,outW]')
        # print ()



        # print ('Second channel')
        frame_c1 = frame[:,1]
        frame_c1 = torch.unsqueeze(frame_c1, dim=1)
        frame_c1 = batch_conv2d_local(input=frame_c1, weight=filters, bias=None, padding=padding, stride=1, dilation=1)


        # print ('Third channel')
        frame_c2 = frame[:,2]
        frame_c2 = torch.unsqueeze(frame_c2, dim=1)
        frame_c2 = batch_conv2d_local(input=frame_c2, weight=filters, bias=None, padding=padding, stride=1, dilation=1)



        blurred_images = [frame_c0, frame_c1, frame_c2]
        blurred_images = torch.stack(blurred_images, dim=1)
        # print (blurred_images.size())
        blurred_images = blurred_images.view(-1,3,480,640)
        # print (blurred_images.size())
        # fdadsf



        # #VIEW IMAGE
        # blurred_image = blurred_images[0]
        # frame = blurred_image.data.cpu().numpy()
        # frame = np.rollaxis(frame, 1, 0)
        # frame = np.rollaxis(frame, 2, 1)
        # # print (frame.shape)
        # # fsd
        # plt.imshow(frame)
        # # save_dir = home+'/Documents/tmp/Doom/'
        # plt_path = home+'/Documents/tmp/Doom/temp_frmae_blurred_v2.png' 
        # plt.tight_layout()
        # plt.savefig(plt_path)
        # print ('saved viz',plt_path)
        # plt.close()




        # fdsaf




        # fasd



        # # upsample
        # # highdim = F.upsample(input=frame[:,:,:400,:200], size=(480,640), mode='bilinear')
        # highdim = F.upsample(input=lowres, size=(480,640), mode='bilinear')
        # # highdim = F.upsample(input=lowres, size=(480,640), mode='nearest')




        # #plot
        # highdim = highdim.data.cpu().numpy()[0]
        # highdim = np.rollaxis(highdim, 1, 0)
        # highdim = np.rollaxis(highdim, 2, 1)
        # print (highdim.shape)
        
        # plt.imshow(highdim)
        # # save_dir = home+'/Documents/tmp/Doom/'
        # plt_path = exp_path_2+'training_plot.png'
        # plt.savefig(plt_path)
        # print ('saved viz',plt_path)
        # plt.close()
        # fd
        # fsdaf
        
        # mask = mask.repeat(1,3,1,1)

        # masked_frame = frame * mask


        # bias_frame = Variable(torch.ones(1,3,480,640).cuda())  *  F.sigmoid(self.bias_frame.bias_frame)
        
        # masked_frame = frame * mask + (1.-mask)*bias_frame
        # masked_frame = highdim


        difs= []
        for i in range(len(DQNs)):
            q_mask = DQNs[i](blurred_images)
            # val, index = torch.max(q_mask, 1)
            # q_mask = q_mask[:,index]

            q_real = DQNs[i](frame)
            # val, index = torch.max(q_real, 1)
            # q_real = q_real[:,index]

            dif = torch.mean((q_mask-q_real)**2)  #[B,A]
            difs.append(dif)

        difs = torch.stack(difs)
        dif = torch.mean(difs)


        # mask = mask.view(self.B, -1)
        # mask_sum = torch.mean(torch.sum(mask, dim=1)) * .0000001


        prec_sum = torch.mean(torch.sum(precisions, dim=1)) * .01


        

        # loss = dif + mask_sum

        loss = dif + prec_sum

        return loss, dif, prec_sum





    def train2(self, epochs, DQN, memory, game):

        batch_size = 32
        
        replay_memory_size = 1000
        play_steps = 100
        learning_steps = 200

        frame_repeat = 12
        display_step = 2#0#0 
        
        loss_list = []
        total_steps = 0
        for epoch in range(epochs):

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
                    print ('Train Epoch: {}/{}'.format(epoch+1, epochs),
                        'Loss:{:.4f}'.format(loss.data.item()),
                        'dif:{:.4f}'.format(dif.data.item()),
                        'prec_sum:{:.4f}'.format(prec_sum.data.item()),
                        )

                    if total_steps!=0:
                        loss_list.append(loss.data.item())


                total_steps+=1

                
            if epoch % 50==0:
                #Save params
                save_params_v3(save_dir=exp_path_2, model=self, epochs=epoch+start_epoch)

                if len(loss_list) > 10:
                    #plot the training curve
                    plt.plot(loss_list[3:])
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    resolution = (480, 640)

    epochs = 300
    replay_memory_size = 1000

    exp_path = home + '/Documents/tmp/Doom/'


    exp_path_2 = exp_path + 'blur_v1/'
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
    train_blur_net_ = 1
    viz_ = 0


    start_epoch = 0




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
    





    print("Initializing BlurNet...")
    MP = BLUR_PREDICTOR()
    if start_epoch>0:
        load_params_v3(save_dir=exp_path_2, model=MP, epochs=start_epoch)
    MP.cuda()
    # n_gpus = 2

    # print (torch.cuda.device_count())
    # fads

    # MP = torch.nn.DataParallel(MP, device_ids=range(n_gpus)).cuda()
    # MP = torch.nn.DataParallel(MP, device_ids=[1])
    # print (MP)
    print("BlurNet initialized\n")
    









    if train_blur_net_:


        print("Initializing Buffer...")
        memory = ReplayMemory(capacity=replay_memory_size)
        print("Buffer initialized\n")


        print("Training")
        # MP.module.train2(epochs=epochs, DQN=DQNs, memory=memory, game=game)
        MP.train2(epochs=epochs, DQN=DQNs, memory=memory, game=game)

        save_params_v3(save_dir=exp_path_2, model=MP, epochs=epochs+start_epoch)

















    if viz_:

        # load_params_v3(save_dir=exp_path, model=MP, epochs=20)




        numb = 1
        gif_dir = exp_path_2 + 'gif_frames'+str(numb)+'/'


        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
            print ('Made dir', gif_dir) 
        else:
            print (gif_dir, 'exists')
            # fasdf 

        max_count = 1000
        count = 0
        frame_count = 0
        

        cols = 3
        rows = 2


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

                #Get grad of real
                x = Variable(torch.from_numpy(np.array([preprocess(frame)])).float(), requires_grad=True).cuda()
                # print (x.size())  #[1,3,480,640]
                q = DQNs[0](x) #[1,A]
                m, index = torch.max(q, 1)
                val = q[:,index]
                grad = torch.autograd.grad(val, x)[0]
                grad = grad.data.cpu().numpy()[0] #for the first one in teh batch -> [2,84,84]
                grad = np.abs(grad)  #[3,480,640]
                # print (grad.shape)
                grad = np.rollaxis(grad, 1, 0)
                grad = np.rollaxis(grad, 2, 1)
                grad = np.mean(grad, 2) #[480,640]



                #Get mask
                x = Variable(torch.from_numpy(np.array([preprocess(frame)])).float()).cuda()
                # print (x.size())  #[1,3,480,640]
                lowres = MP.predict_lowres(x) #[1,1,480,640]
                highdim = F.upsample(input=lowres, size=(480,640), mode='bilinear')


                # masked_frame = x * mask
                # masked_frame = x * mask + (1.-mask)* F.sigmoid(MP.bias_frame.bias_frame)
                masked_frame = highdim


                #Get grad of masked
                # x = Variable(torch.from_numpy(np.array([masked_frame)])).float(), requires_grad=True).cuda()
                # print (x.size())  #[1,3,480,640]
                q_m = DQNs[0](masked_frame) #[1,A]
                m, index = torch.max(q_m, 1)
                val = q_m[:,index]
                grad_m = torch.autograd.grad(val, masked_frame)[0]
                grad_m = grad_m.data.cpu().numpy()[0] #for the first one in teh batch -> [2,84,84]
                grad_m = np.abs(grad_m)  #[3,480,640]
                # print (grad.shape)
                grad_m = np.rollaxis(grad_m, 1, 0)
                grad_m = np.rollaxis(grad_m, 2, 1)
                grad_m = np.mean(grad_m, 2) #[480,640]




                masked_frame = masked_frame.squeeze() #[3,480.640]
                masked_frame = masked_frame.data.cpu().numpy()
                masked_frame = np.rollaxis(masked_frame, 1, 0)
                masked_frame = np.rollaxis(masked_frame, 2, 1)

                # mask = mask.squeeze()
                # mask = mask.data.cpu().numpy()#[0] #for the first one in teh batch -> [2,84,84]


                frame = np.rollaxis(frame, 1, 0)
                frame = np.rollaxis(frame, 2, 1)
                # frames.append(frame)

                plt_path = gif_dir+'frame'+str(frame_count)+'.png'
                # fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=640*rows)
                fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=150)

                #Plot Frame
                ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

                # print (np.max(frame))
                # print (np.min(frame))
                # print (np.mean(frame))
                # print ()

                ax.imshow(frame) #, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.35, 1.04, 'Real Frame '+str(count), transform=ax.transAxes, family='serif', size=6)



                #Plot Masked
                ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)
                # ax.imshow(np.uint8(masked_frame))

                # masked_frame = masked_frame *255.

                # print (np.max(masked_frame))
                # print (np.min(masked_frame))
                # print (np.mean(masked_frame))
                # print ()


                # ax.imshow(masked_frame)
                ax.imshow(np.uint8(masked_frame*255.))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.4, 1.04, 'Masked', transform=ax.transAxes, family='serif', size=6)




                # #Plot Mask
                # ax = plt.subplot2grid((rows,cols), (1,2), frameon=False)
                # # ax.imshow(np.uint8(mask), cmap='gray')

                # # print (np.max(mask))
                # # print (np.min(mask))
                # # print (np.mean(mask))
                # # print ()

                # ax.imshow(mask, cmap='gray')
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax.text(0.4, 1.04, 'Mask', transform=ax.transAxes, family='serif', size=6)




                # #Plot Bias Frame
                # ax = plt.subplot2grid((rows,cols), (1,2), frameon=False)

                # bias_frame = Variable(torch.ones(1,3,480,640).cuda())  *  F.sigmoid(MP.bias_frame.bias_frame)
                # bias_frame = bias_frame.squeeze() 

                # # bias_frame = F.sigmoid(MP.bias_frame.bias_frame).squeeze() #[1,3,.,.]

                # # print (torch.sum(bias_frame))
                # # bias_frame2 = F.sigmoid(MP.bias_frame.bias_frame).squeeze() #[1,3,.,.]
                # # print (torch.sum(bias_frame2))

                # # fsdfda


                # bias_frame = bias_frame.data.cpu().numpy() *255.
                # bias_frame = np.rollaxis(bias_frame, 1, 0)
                # bias_frame = np.rollaxis(bias_frame, 2, 1)

                # # print (np.max(bias_frame))
                # # print (np.min(bias_frame))
                # # print (np.mean(bias_frame))
                # # print ()

                # ax.imshow(np.uint8(bias_frame)) #, cmap='gray')
                # # ax.imshow(bias_frame) #, cmap='gray')
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax.text(0.4, 1.04, 'Bias Frame', transform=ax.transAxes, family='serif', size=6)





                #Plot Grad Real
                ax = plt.subplot2grid((rows,cols), (1,0), frameon=False)
                ax.imshow(grad, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.35, 1.04, 'Grad of Real', transform=ax.transAxes, family='serif', size=6)





                #Plot Grad Masked
                ax = plt.subplot2grid((rows,cols), (1,1), frameon=False)
                ax.imshow(grad_m, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.3, 1.04, 'Grad of Masked', transform=ax.transAxes, family='serif', size=6)




                #Plot Q values
                width = .2
                ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)

                q = q.squeeze().data.cpu().numpy()
                q_m = q_m.squeeze().data.cpu().numpy()

                ax.bar(range(len(q)), q, width=width)

                ax.bar(np.array(range(len(q_m)))+width, q_m, width=width)

                ax.set_xticks(range(len(q_m)))
                # if i == rows-1:
                ax.set_xticklabels([ 'NOOP', 'RIGHT', 'LEFT', 'NOOP'], size=5)
                # else:
                #     ax.set_xticklabels([])
                ax.yaxis.set_tick_params(labelsize=5)
                ax.xaxis.set_tick_params(labelsize=5)
                ax.set_ylim([0,40.])

                ax.text(0.34, .9, 'Q Values', transform=ax.transAxes, family='serif', size=6)






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

        gif_path_this = gif_dir+ 'gif_mask'+str(numb)+'.gif'
        # imageio.mimsave(gif_path_this, images)
        imageio.mimsave(gif_path_this, images, duration=.1)
        print ('made gif', gif_path_this)
        fds





























































