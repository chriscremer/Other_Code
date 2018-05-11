





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





# def get_best_action(state):
#     q = model(Variable(state))
#     m, index = torch.max(q, 1)
#     action = index.data.cpu().numpy()[0]
#     return action

# def get_eps_action(epoch, actions, s1):
#     # With probability eps make a random action.
#     eps = exploration_rate(epoch)
#     if random() <= eps:
#         a = randint(0, len(actions) - 1)
#     else:
#         # Choose the best action according to the network.
#         s1 = s1.reshape([1, 3, resolution[0], resolution[1]])
#         s1 = torch.from_numpy(s1).cuda()
#         a = get_best_action(s1)
#     return a

# def exploration_rate(epoch):
#     """# Define exploration rate change over time"""
#     start_eps = 1.0
#     end_eps = .02 #0.1
#     const_eps_epochs = 0.1 * epochs  # 10% of learning time
#     eps_decay_epochs = 0.6 * epochs  # 60% of learning time
#     if epoch < const_eps_epochs:
#         return start_eps
#     elif epoch < eps_decay_epochs:
#         # Linear decay
#         return start_eps - (epoch - const_eps_epochs) / \
#                            (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
#     else:
#         return end_eps




# def learn_from_memory():
#     """ Learns from a single transition (making use of replay memory).
#     s2 is ignored if s2_isterminal """

#     # Get a random minibatch from the replay memory and learns from it.
#     if memory.size > batch_size:

#         idxs = sample(range(0, memory.size), batch_size)
#         s1 = memory.s1[idxs]        
#         a = memory.a[idxs]  #[B]
#         s2 = memory.s2[idxs]
#         isterminal = memory.isterminal[idxs]
#         r = memory.r[idxs]

#         s1 = preprocess_pytorch(s1)
#         s2 = preprocess_pytorch(s2)
#         a = a.int()
#         r = r.float()
#         isterminal = isterminal.float()
        
#         q2 = model.forward(Variable(s2)).data #.cpu().numpy()  #[B,4]
#         q2 = torch.max(q2, dim=1)[0]  #[B]

#         target_q = r + discount_factor * (1 - isterminal) * q2   #[B]

#         s1, target_q = Variable(s1), Variable(target_q)

#         q1_pred = model(s1)

#         a = a.long()#.cuda()
#         a = a.view(batch_size,1)

#         q1_pred = torch.gather(q1_pred, 1, a).squeeze()

#         loss = torch.mean((q1_pred-target_q)**2)

#         # compute gradient and do SGD step
#         model.optimizer.zero_grad()
#         loss.backward()
#         model.optimizer.step()

#     return loss
















class BIAS_FRAME(nn.Module):


    def __init__(self):
        super(BIAS_FRAME, self).__init__()

        # self.bias_frame = Variable(torch.empty(1, 3, 480, 640).uniform_(0, 1), requires_grad=True)

        # self.bias_frame = nn.Parameter(torch.empty(1, 3, 480, 640).uniform_(-1, 1))
        self.bias_frame = nn.Parameter(torch.empty(1,3,1,1).uniform_(-1, 1))

        # self.optimizer = optim.Adam([self.bias_frame], lr=.0001)









class MASK_PREDICTOR(nn.Module):
    def __init__(self):
        super(MASK_PREDICTOR, self).__init__()

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

        self.deconv1_gp = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, output_padding=(0,1))
        self.deconv2_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=8, stride=4, output_padding=(3,3))
        self.deconv3_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=12, stride=8, output_padding=(4,4))



        self.bias_frame = BIAS_FRAME()



        # print (self.conv1_gp.parameters())
        # print (self.bias_frame)
        # # print (self.bias_frame.parameters())

        # print (self)


        params_gp = [list(self.bias_frame.parameters()) + list(self.conv1_gp.parameters()) + list(self.conv2_gp.parameters()) + list(self.conv3_gp.parameters()) +
                    list(self.fc1_gp.parameters()) + list(self.fc2_gp.parameters()) + 
                    list(self.fc3_gp.parameters()) + list(self.fc4_gp.parameters()) +
                    list(self.deconv1_gp.parameters()) + list(self.deconv2_gp.parameters()) + list(self.deconv3_gp.parameters())]
         

                    
        self.optimizer = optim.Adam(params_gp[0], lr=.0001, weight_decay=.0000001)




    def predict_mask(self, x):
        # print (x.size())  #[B,32,480,640]
        x = self.act_func(self.conv1_gp(x))  #[B,32,59,79]
        # print (x.size())
        x = self.act_func(self.conv2_gp(x)) #[B,64,13,18]
        # print (x.size())
        x = self.act_func(self.conv3_gp(x)) #[B,32,6,8]
        # print (x.size())
        

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




    def forward(self, frame, DQNs):
        # x: [B,2,84,84]
        self.B = frame.size()[0]

        #Predict mask
        mask = self.predict_mask(frame)  #[B,2,210,160]
        # print (mask.size())
        
        mask = mask.repeat(1,3,1,1)

        # masked_frame = frame * mask


        bias_frame = Variable(torch.ones(1,3,480,640).cuda())  *  F.sigmoid(self.bias_frame.bias_frame)
        
        masked_frame = frame * mask + (1.-mask)*bias_frame

        # print (torch.max(frame))
        # print (torch.min(frame))
        # print (torch.mean(frame))
        # print (torch.max(mask))
        # print (torch.min(mask))
        # print (torch.mean(mask))
        # print (torch.max(masked_frame))
        # print (torch.min(masked_frame))
        # print (torch.mean(masked_frame))
        # fsaf

        difs= []
        for i in range(len(DQNs)):
            q_mask = DQNs[i](masked_frame)
            q_real = DQNs[i](frame)
            dif = torch.mean((q_mask-q_real)**2)  #[B,A]
            difs.append(dif)

        difs = torch.stack(difs)
        dif = torch.mean(difs)


        mask = mask.view(self.B, -1)
        mask_sum = torch.mean(torch.sum(mask, dim=1)) * .0000001

        loss = dif + mask_sum

        return loss, dif, mask_sum





    def train(self, epochs, DQN, memory, game):

        batch_size = 32
        
        replay_memory_size = 1000
        play_steps = 100
        learning_steps = 200

        frame_repeat = 12
        display_step = 200 
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
                m, index = torch.max(q, 1)
                index = index.data.cpu().numpy()[0]
                action = actions[index]

                reward = game.make_action(action, frame_repeat) /12.
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

                idxs = sample(range(0, memory.size), batch_size)
                s1 = memory.s1[idxs]        

                batch = Variable(preprocess_pytorch(s1))

                self.optimizer.zero_grad()
                loss, dif, mask_sum = self.forward(batch, DQN)
                loss.backward()
                self.optimizer.step()

                if total_steps%display_step==0: # and batch_idx == 0:
                    print ('Train Epoch: {}/{}'.format(epoch+1, epochs),
                        'Loss:{:.4f}'.format(loss.data[0]),
                        'dif:{:.4f}'.format(dif.data[0]),
                        'mask_sum:{:.4f}'.format(mask_sum.data[0]),
                        )
                    if total_steps!=0:
                        loss_list.append(loss.data.item())
                total_steps+=1

            if epoch % 100==0 and epoch !=0:
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    resolution = (480, 640)

    epochs = 300
    replay_memory_size = 1000

    exp_path = home + '/Documents/tmp/Doom/'


    exp_path_2 = exp_path + '2DQNs_biasframe_v3_2/'


    if not os.path.exists(exp_path_2):
        os.makedirs(exp_path_2)
        print ('Made dir', exp_path_2) 
    # else:
    #     print ('exists')
    #     fasdf 

    # maskpredictor_savefile = exp_path + 'maskpredictor_5.pth'
    # maskpredictor_savefile = exp_path_2 + 'MP_params'+str(epochs)+'.pth'
    # maskpredictor_loadfile = exp_path + 'first_4.pth'




    train_maskpredictor_ = 0
    viz_ = 1


    start_epoch = 300 #501




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

    #this is the one with the smaller stride
    DQN_loadfile = exp_path + 'training_3/DQN_params_999.pth'
    # if load_model:
    print("Loading model from: ", DQN_loadfile)
    DQN3 = torch.load(DQN_loadfile)
    DQN3.cuda()

    DQNs =[DQN3]
    print("DQNs initialized\n")
    





    print("Initializing MaskPredictor...")
    MP = MASK_PREDICTOR()
    if start_epoch>0:
        load_params_v3(save_dir=exp_path_2, model=MP, epochs=start_epoch)
    MP.cuda()
    print("MaskPredictor initialized\n")









    if train_maskpredictor_:


        print("Initializing Buffer...")
        memory = ReplayMemory(capacity=replay_memory_size)
        print("Buffer initialized\n")


        print("Training")
        MP.train(epochs=epochs, DQN=DQNs, memory=memory, game=game)

        save_params_v3(save_dir=exp_path_2, model=MP, epochs=epochs+start_epoch)

















    if viz_:

        # load_params_v3(save_dir=exp_path, model=MP, epochs=20)




        numb = start_epoch
        gif_dir = exp_path_2 + 'gif_frames'+str(numb)+'/'


        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
            print ('Made dir', gif_dir) 
        else:
            print ('exists')
            # fasdf 

        max_count = 200 #1000
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
                mask = MP.predict_mask(x) #[1,1,480,640]

                # masked_frame = x * mask
                masked_frame = x * mask + (1.-mask)* F.sigmoid(MP.bias_frame.bias_frame)


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

                mask = mask.squeeze()
                mask = mask.data.cpu().numpy()#[0] #for the first one in teh batch -> [2,84,84]


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




                #Plot Mask
                ax = plt.subplot2grid((rows,cols), (1,2), frameon=False)
                # ax.imshow(np.uint8(mask), cmap='gray')

                # print (np.max(mask))
                # print (np.min(mask))
                # print (np.mean(mask))
                # print ()

                ax.imshow(mask, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.4, 1.04, 'Mask', transform=ax.transAxes, family='serif', size=6)




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






















































