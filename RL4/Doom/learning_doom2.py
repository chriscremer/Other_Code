






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
    # img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    # img = img.astype(np.float16)
    # print (img.shape) #(3, 480, 640)
    # print (np.max(img)) #215
    # print (np.min(img)) #0
    # fads
    img = img / 255.

    return img



def preprocess_pytorch(img):

    # img = img.astype(torch.float32)
    img = img.float()
    img = img / 255.
    return img





# class ReplayMemory:
#     def __init__(self, capacity):
#         channels = 3
#         state_shape = (capacity, channels, resolution[0], resolution[1])
#         self.s1 = np.zeros(state_shape, dtype=np.float32)
#         self.s2 = np.zeros(state_shape, dtype=np.float32)
#         self.a = np.zeros(capacity, dtype=np.int32)
#         self.r = np.zeros(capacity, dtype=np.float32)
#         self.isterminal = np.zeros(capacity, dtype=np.float32)

#         self.capacity = capacity
#         self.size = 0
#         self.pos = 0

#     def add_transition(self, s1, action, s2, isterminal, reward):
#         # self.s1[self.pos, 0, :, :] = s1
#         self.s1[self.pos, :, :, :] = s1
#         self.a[self.pos] = action
#         if not isterminal:
#             # self.s2[self.pos, 0, :, :] = s2
#             self.s2[self.pos, :, :, :] = s2
#         self.isterminal[self.pos] = isterminal
#         self.r[self.pos] = reward

#         self.pos = (self.pos + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)



    # def get_sample(self, sample_size):
    #     i = sample(range(0, self.size), sample_size)
    #     # print (i)
    #     # i = np.random.choice(self.size, sample_size)
    #     # print (i)
    #     # fadsf
    #     return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


    # def get_sample(self, idxs):
    #     # i = sample(range(0, self.size), sample_size)
    #     # print (i)
    #     # i = np.random.choice(self.size, sample_size)
    #     # print (i)
    #     # fadsf
    #     return self.s1[idxs], self.a[idxs], self.s2[idxs], self.isterminal[idxs], self.r[idxs]




# class ReplayMemory:
#     def __init__(self, capacity):
#         channels = 3
#         state_shape = (capacity, channels, resolution[0], resolution[1])
#         self.s1 = np.zeros(state_shape, dtype=np.uint8)
#         self.s2 = np.zeros(state_shape, dtype=np.uint8)
#         self.a = np.zeros(capacity, dtype=np.uint8)
#         self.r = np.zeros(capacity, dtype=np.uint8)
#         self.isterminal = np.zeros(capacity, dtype=np.uint8)

#         self.capacity = capacity
#         self.size = 0
#         self.pos = 0

#     def add_transition(self, s1, action, s2, isterminal, reward):
#         # self.s1[self.pos, 0, :, :] = s1
#         self.s1[self.pos, :, :, :] = s1
#         self.a[self.pos] = action
#         if not isterminal:
#             # self.s2[self.pos, 0, :, :] = s2
#             self.s2[self.pos, :, :, :] = s2
#         self.isterminal[self.pos] = isterminal
#         self.r[self.pos] = reward

#         self.pos = (self.pos + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)




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

        # print (action.dtype)
        # fsad
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
        # self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        # self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)

        self.conv1 = nn.Conv2d(n_channels, 32, 12, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 8, stride=4)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=2)

        self.act_func = F.leaky_relu

        self.intermediate_size = 7488

        self.fc1 = nn.Linear(self.intermediate_size, 100)
        self.fc2 = nn.Linear(100, action_size)

        self.criterion = nn.MSELoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=.000001)

    def forward(self, x):
        x = self.act_func(self.conv1(x))
        x = self.act_func(self.conv2(x))
        x = self.act_func(self.conv3(x))
        x = x.view(-1, self.intermediate_size)
        x = self.act_func(self.fc1(x))
        return self.fc2(x)






# def learn(s1, target_q):
#     s1 = torch.from_numpy(s1)
#     target_q = torch.from_numpy(target_q)
#     s1, target_q = Variable(s1), Variable(target_q)
#     output = model(s1)
#     loss = criterion(output, target_q)
#     # compute gradient and do SGD step
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return loss

# def get_q_values(state):
#     # state = torch.from_numpy(state).cuda()
#     state = Variable(state)
#     return model(state)

def get_best_action(state):
    # q = get_q_values(state)
    q = model(Variable(state))
    m, index = torch.max(q, 1)
    # action = index.data.numpy()[0]
    action = index.data.cpu().numpy()[0]
    return action

def get_eps_action(epoch, actions, s1):
    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        s1 = s1.reshape([1, 3, resolution[0], resolution[1]])
        s1 = torch.from_numpy(s1).cuda()
        a = get_best_action(s1)
    return a

def exploration_rate(epoch):
    """# Define exploration rate change over time"""
    start_eps = 1.0
    end_eps = .02 #0.1
    const_eps_epochs = 0.1 * epochs  # 10% of learning time
    eps_decay_epochs = 0.6 * epochs  # 60% of learning time

    # return .02

    if epoch < const_eps_epochs:
        return start_eps
    elif epoch < eps_decay_epochs:
        # Linear decay
        return start_eps - (epoch - const_eps_epochs) / \
                           (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
    else:
        return end_eps




def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:

        idxs = sample(range(0, memory.size), batch_size)

        
        # s1, a, s2, isterminal, r = memory.get_sample(batch_size)
        # s1, a, s2, isterminal, r = memory.get_sample(idxs)

        # time_start = time()
        s1 = memory.s1[idxs]
        # time_elapsed_1 = time() - time_start

        
        a = memory.a[idxs]  #[B]
        # print (a.shape)
        # fsdf
        

        # time_start = time()
        s2 = memory.s2[idxs]
        # time_elapsed_2 = time() - time_start
        

        # time_start = time()
        isterminal = memory.isterminal[idxs]
        r = memory.r[idxs]
        # time_elapsed_3 = time() - time_start


        s1 = preprocess_pytorch(s1)
        s2 = preprocess_pytorch(s2)
        # a = a.astype(np.int32)
        a = a.int()
        # r = r.astype(np.float32)
        r = r.float()
        # isterminal = isterminal.astype(np.float32)
        isterminal = isterminal.float()
        




        
        # q = get_q_values(s2).data.numpy()
        # q = get_q_values(s2).data.cpu().numpy()  #[B,4]
        # q2 = np.max(q, axis=1)  #[B]

        # q2 = get_q_values(s2).data #.cpu().numpy()  #[B,4]
        q2 = model.forward(Variable(s2)).data #.cpu().numpy()  #[B,4]
        q2 = torch.max(q2, dim=1)[0]  #[B]


        # print (q2.shape)
        # fasdf

        # target_q = get_q_values(s1).data.numpy()
        # target_q = get_q_values(s1).data.cpu().numpy()  #[B,4]

        target_q = r + discount_factor * (1 - isterminal) * q2   #[B]

        # print (isterminal)
        # print (r)
        # fsd


        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        # target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2

        # print (target_q.shape)  #[B,4]

        

        # learn(s1, target_q)

        # s1 = torch.from_numpy(s1).cuda()
        # target_q = torch.from_numpy(target_q).cuda()

        # print (s1)
        # print (target_q)

        s1, target_q = Variable(s1), Variable(target_q)


        # print (a.shape)
        q1_pred = model(s1)
        # print(q1_pred.size())

        # q1_pred = q1_pred.gather(1, action.unsqueeze(1))
        # q1_pred = torch.gather(q1_pred, ) q1_pred.gather(1, action.unsqueeze(1))


        # a = torch.from_numpy(a).long().cuda()
        a = a.long()#.cuda()

        a = a.view(batch_size,1)
        # q1_pred = torch.index_select(q1_pred, 1, a)
        q1_pred = torch.gather(q1_pred, 1, a).squeeze()

        # q1_pred = q1_pred.view(batch_size)


        # print(q1_pred.size())
        # print(target_q.size())
        # fdsdafs


        # print (q1_pred)
        # print (target_q)


        # loss = model.criterion(q1_pred, target_q)
        loss = torch.mean((q1_pred-target_q)**2)

        # compute gradient and do SGD step
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        # print (loss)
        # fds
        


    # return time_elapsed_1, time_elapsed_2, time_elapsed_3
    return loss



# def perform_learning_step(epoch):
#     """ Makes an action according to eps-greedy policy, observes the result
#     (next state, reward) and learns from the transition"""

#     s1 = preprocess(game.get_state().screen_buffer)
#     # With probability eps make a random action.
#     a = get_eps_action(epoch, actions, s1)
#     reward = game.make_action(actions[a], frame_repeat)
#     isterminal = game.is_episode_finished()
#     s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

#     # Remember the transition that was just experienced.
#     memory.add_transition(s1, a, s2, isterminal, reward)

#     learn_from_memory()


# # Creates and initializes ViZDoom environment.
# def initialize_vizdoom(config_file_path):
#     print("Initializing doom...")
#     game = DoomGame()
#     game.load_config(config_file_path)
#     game.set_window_visible(False)


#     game.set_mode(Mode.PLAYER)
#     game.set_screen_format(ScreenFormat.GRAY8)
#     game.set_screen_resolution(ScreenResolution.RES_640X480)


#     game.init()
#     print("Doom initialized.")
#     return game
















if __name__ == '__main__':



    # Q-learning settings
    # learning_rate = 0.00025
    learning_rate = 0.00001
    discount_factor = 0.99
    epochs = 501 # 2001
    learning_steps_per_epoch = 200# 200 # 500 # 50 #100 # 200 #0
    replay_memory_size = 1000 # 500 # 1000 #0
    play_steps = 500# 100 #0

    # NN learning settings
    batch_size = 32 #64

    # Training regime
    test_episodes_per_epoch = 10

    # Other parameters
    frame_repeat = 12
    # resolution = (30, 45)
    # resolution = (60, 90)
    resolution = (480, 640)
    episodes_to_watch = 10

    # model_savefile = "./model-doom.pth"
    save_path = home + '/Documents/tmp/Doom/'
    # model_loadfile = save_path + 'first_4.pth'
    model_loadfile = save_path + 'training_3/DQN_params_999.pth'
    model_savefile_pre = save_path + 'training_3/DQN_params_' #+str(epochs)+'.pth'

    save_model = 0 # 1#True
    load_model = 1
    # skip_learning = False
    train_ = 0
    test_ = 0
    gif_ = 0
    make_gif_ = 0 
    gif_with_grad_ = 1



    seed = 2
    torch.cuda.manual_seed(seed)

    # Configuration file path
    # config_file_path = "../../scenarios/simpler_basic.cfg"
    # config_file_path = "../../scenarios/rocket_basic.cfg"
    # config_file_path = "../../scenarios/basic.cfg"
    config_file_path = home + "/ViZDoom/scenarios/take_cover.cfg"


    # Create Doom instance
    # game = initialize_vizdoom(config_file_path)
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    # game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    



    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    print ('actions', actions)




    print("Initializing Model...")
    if load_model:
        print("Loading model from: ", model_loadfile)
        model = torch.load(model_loadfile)
    else:
        model = Net(n_channels=3, action_size=len(actions))

    model.cuda()
    print("Model initialized")
    
    





    if train_:


        print("Initializing Buffer...")
        # Create replay memory which will store the transitions
        memory = ReplayMemory(capacity=replay_memory_size)
        print("Buffer initialized")



        test_errors = []

        print("Starting the training!")
        time_start = time()
        for epoch in range(epochs):

            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Playing...")
            game.new_episode()
            game_step_count = 0
            game_steps = []
            
            # for play_step in trange(play_steps, leave=False):
            for play_step in range(play_steps):#, leave=False):

                # # img = game.get_state().screen_buffer
                # img = preprocess(game.get_state().screen_buffer)
                # print (img.shape) #480,640
                # # fadsfsd

                # # # img = np.reshape(img, [240,320,3])
                # # img = np.rollaxis(img, 1, 0)
                # # img = np.rollaxis(img, 2, 1)
                # # print (img.shape)

                # plt.imshow(img, cmap='gray')

                # save_dir = home+'/Documents/tmp/Doom/'
                # plt_path = save_dir+'frmame2.png'
                # plt.savefig(plt_path)
                # print ('saved viz',plt_path)
                # plt.close()

                # fdas

                # perform_learning_step(epoch)
                s1 = game.get_state().screen_buffer  #[3,480,640] uint8
                # print (s1.shape)
                # print (s1.dtype)
                # fdsa
                # s1 = preprocess(game.get_state().screen_buffer)


                # With probability eps make a random action.
                a = get_eps_action(epoch, actions, preprocess(s1))
                reward = game.make_action(actions[a], frame_repeat) /12.
                isterminal = game.is_episode_finished()
                game_step_count +=1

                # s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None
                s2 = game.get_state().screen_buffer if not isterminal else None

                # Remember the transition that was just experienced.
                memory.add_transition(s1, a, s2, isterminal, reward)





                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    train_episodes_finished += 1
                    game_steps.append(game_step_count)
                    game.new_episode()
                    game_step_count = 0
                    

            print("%d training episodes played." % train_episodes_finished, 'avg ep length', np.mean(game_steps))
            print ('Buffer size', memory.size, 'Exp', exploration_rate(epoch))
            train_scores = np.array(train_scores)



            print("Training...")
            # t1=[]
            # t2=[]
            # t3=[]
            losses = []
            # for learning_step in trange(learning_steps_per_epoch, leave=True):
            for learning_step in range(learning_steps_per_epoch): #, leave=True):

                # time_elapsed_1, time_elapsed_2, time_elapsed_3 = learn_from_memory()
                loss = learn_from_memory()
                # t1.append(time_elapsed_1)
                # t2.append(time_elapsed_2)
                # t3.append(time_elapsed_3)
                losses.append(loss.data.cpu().numpy())



            # print ('t1', np.mean(t1), np.max(t1), np.min(t1))
            # print ('t2', np.mean(t2), np.max(t2), np.min(t2))
            # print ('t3', np.mean(t3), np.max(t3), np.min(t3))
            print ('Loss', np.mean(losses), np.max(losses), np.min(losses))
            

            print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), \
                  "max: %.1f," % train_scores.max())








            print("\nTesting...")
            test_episode = []
            test_scores = []
            # for test_episode in trange(test_episodes_per_epoch, leave=False):
            for test_episode in range(test_episodes_per_epoch): #, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    # state = state.reshape([1, 1, resolution[0], resolution[1]])
                    state = state.reshape([1, 3, resolution[0], resolution[1]])
                    state = torch.from_numpy(state).cuda()
                    best_action_index = get_best_action(state)

                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f +/- %.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            test_errors.append(test_scores.mean())




            if save_model and ((epoch+1) %200)==0:
                model_savefile = model_savefile_pre + str(epoch) + '.pth'
                print("Saving the network weigths to:", model_savefile) 
                torch.save(model, model_savefile)

            # print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

        plt.plot(test_errors)
        save_dir = home+'/Documents/tmp/Doom/'
        plt_path = save_dir+'plot.png'
        plt.savefig(plt_path)
        print ('saved viz',plt_path)
        plt.close()

        game.close()







    if test_:



        for i in range(10):

            print("\nTesting...")
            test_episode = []
            test_scores = []
            # for test_episode in trange(test_episodes_per_epoch, leave=False):
            for test_episode in range(test_episodes_per_epoch): #, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    # state = state.reshape([1, 1, resolution[0], resolution[1]])
                    state = state.reshape([1, 3, resolution[0], resolution[1]])
                    state = torch.from_numpy(state).cuda()
                    best_action_index = get_best_action(state)

                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f +/- %.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())





















    if gif_:
        # print("======================================")
        # print("Training finished. It's time to watch!")

        # Reinitialize the game with window visible
        # game.set_window_visible(True)
        # game.set_mode(Mode.ASYNC_PLAYER)
        # game.init()

        # print("Initializing doom...")
        # game = DoomGame()
        # game.load_config(config_file_path)
        # game.set_window_visible(False)
        # game.set_mode(Mode.PLAYER)
        # # game.set_screen_format(ScreenFormat.GRAY8)
        # game.set_screen_resolution(ScreenResolution.RES_640X480)
        # game.init()
        # print("Doom initialized.")


        numb = 16
        gif_dir = save_path + 'gif'+str(numb)+'/'


        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
            print ('Made dir', gif_dir) 
        else:
            print ('exists')
            # fasdf

        n_frames = 1000
        count = 0

        cols  =1
        rows = 1


        frames = []

        game.new_episode()
        while not game.is_episode_finished() and count<n_frames:

            print (count)


            frame = game.get_state().screen_buffer #[3,480,640] uint8
            # f1 = np.copy(frame)

            if count %12==0:
                state = preprocess(frame)
                state = state.reshape([1, 3, resolution[0], resolution[1]])
                state = torch.from_numpy(state).cuda()
                a = get_best_action(state) #scalar

            # reward = game.make_action(actions[a], frame_repeat) #/12.
            # game.set_action(actions[a])
            # for _ in range(frame_repeat):
                # game.set_action(actions[a])
                # game.advance_action()

            # reward = game.make_action(actions[a], 1)

            # for _ in range(frame_repeat):

            reward = game.make_action(actions[a], 1)
            # frame = game.get_state().screen_buffer

            if count %4==0:
                frame = np.rollaxis(frame, 1, 0)
                frame = np.rollaxis(frame, 2, 1)
                frames.append(frame)

                # if game.is_episode_finished() :
                #     break

            # reward = game.make_action(actions[a], 4) #/12.
            # isterminal = game.is_episode_finished()
            # frame = game.get_state().screen_buffer #[3,480,640] uint8

            # frame = f1
            # frame = game.get_state().screen_buffer 

            # if count ==0 :
                


            # frame = np.rollaxis(frame, 1, 0)
            # frame = np.rollaxis(frame, 2, 1)
            # frames.append(frame)

            # # aaa = np.copy(frame)
            # # print (np.sum(frame) , frame.shape, frame.dtype)
            # # print (frame)

            # frames.append(frame)

            # else:
            #     frames.append(aaa)

            # if np.sum(frame) == 0:
            #     flfjads



            # plt_path = gif_dir+'frame'+str(count)+'.png'
            # fig = plt.figure(figsize=(3+cols,3+rows), facecolor='white', dpi=640*rows)

            # ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

            # frame = np.rollaxis(frame, 1, 0)
            # frame = np.rollaxis(frame, 2, 1)

            # ax.imshow(np.uint8(frame)) #, cmap='gray')
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.text(0.4, 1.04, str(count), transform=ax.transAxes, family='serif', size=6)

            # plt.savefig(plt_path)
            # print ('saved viz',plt_path)
            # plt.close(fig)

            count+=1

        

        print ('game over', game.is_episode_finished() )
        print (count)
        print (len(frames))

        gif_path_this = gif_dir+ 'first99'+str(numb)+'.gif'
        imageio.mimsave(gif_path_this, frames)
        print ('made gif', gif_path_this)
        fdsf



            # state = state.reshape([1, 1, resolution[0], resolution[1]])
            # best_action_index = get_best_action(state)

            # # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            # game.set_action(actions[best_action_index])

            # for _ in range(frame_repeat):
            #     game.advance_action()

        # Sleep between episodes
        # sleep(1.0)

        score = game.get_total_reward()
        print("Total score: ", score)


        print('Making Gif')
        # frames_path = save_dir+'gif/'
        images = []
        for i in range(count-1):
            images.append(imageio.imread(gif_dir+'frame'+str(i)+'.png'))
 


        # gif_path_this = gif_epoch_path + str(j) + '.gif'
        gif_path_this = gif_dir+ 'first'+str(numb)+'.gif'
        imageio.mimsave(gif_path_this, images)
        print ('made gif', gif_path_this)
















    if make_gif_:

        numb = 8
        gif_dir = save_path + 'gif'+str(numb)+'/'


        count  = 10
        print('Making Gif')
        # frames_path = save_dir+'gif/'
        images = []
        for i in range(count-1):
            images.append(imageio.imread(gif_dir+'frame'+str(i)+'.png'))
 
        print (images[2].shape)
        print (images[0].shape)
        fasd

        imageio.imwrite(gif_dir+'picture_out.png', images[2])
        # imageio.imwrite(gif_dir+'picture_out.jpg', images[2][:,:,:3])
        fsdaf


        # gif_path_this = gif_epoch_path + str(j) + '.gif'
        gif_path_this = gif_dir+ 'first'+str(numb)+'.gif'
        imageio.mimsave(gif_path_this, images)
        print ('made gif', gif_path_this)






























    if gif_with_grad_:


        numb = 4
        gif_dir = save_path + 'gif_and_grad_29'+str(numb)+'/'


        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
            print ('Made dir', gif_dir) 
        else:
            print ('exists')
            # fasdf 

        max_count = 1000
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
                a = get_best_action(state) #scalar

            reward = game.make_action(actions[a], 1)

            if count %4==0:

                #Get grad
                x = Variable(torch.from_numpy(np.array([preprocess(frame)])).float(), requires_grad=True).cuda()
                # print (x.size())  #[1,3,480,640]
                q = model(x) #[1,A]
                m, index = torch.max(q, 1)
                val = q[:,index]
                grad = torch.autograd.grad(val, x)[0]
                grad = grad.data.cpu().numpy()[0] #for the first one in teh batch -> [2,84,84]
                grad = np.abs(grad)  #[3,480,640]
                # print (grad.shape)
                grad = np.rollaxis(grad, 1, 0)
                grad = np.rollaxis(grad, 2, 1)
                grad = np.mean(grad, 2) #[480,640]


                frame = np.rollaxis(frame, 1, 0)
                frame = np.rollaxis(frame, 2, 1)
                # frames.append(frame)

                plt_path = gif_dir+'frame'+str(frame_count)+'.png'
                # fig = plt.figure(figsize=(3+cols,3+rows), facecolor='white', dpi=640*rows)
                fig = plt.figure(figsize=(3+cols,3+rows), facecolor='white', dpi=150)

                #Plot Frame
                ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
                ax.imshow(frame) #, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.4, 1.04, str(count), transform=ax.transAxes, family='serif', size=6)


                #Plot Grad
                ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)
                ax.imshow(grad, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.text(0.4, 1.04, 'Grad of Real', transform=ax.transAxes, family='serif', size=6)


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

        gif_path_this = gif_dir+ 'gif_and_grad'+str(numb)+'.gif'
        imageio.mimsave(gif_path_this, images)
        print ('made gif', gif_path_this)
        fds








    















    # load_params_v3(save_dir, model, epochs=200)


    # frames_path = save_dir+'gif/'
    # images = []
    # for i in range(n_frames):
    #     images.append(imageio.imread(frames_path+'frame'+str(i)+'.png'))
        
    # # gif_path_this = gif_epoch_path + str(j) + '.gif'
    # gif_path_this = frames_path+ 'first.gif'
    # imageio.mimsave(gif_path_this, images)
    # print ('made gif', gif_path_this)

    # # dfad

    # gif_dir = save_dir+'gif3/'


    

    # cols  =3
    # rows = 1

    # for i in range(n_frames):

    #     print (i)

    #     plt_path = gif_dir+'frame'+str(i)+'.png'

    #     fig = plt.figure(figsize=(2+cols,2+rows), facecolor='white', dpi=210*rows)

        


    #     frame = dataset[0][200+i][0]
            
    #     frame_pytorch = Variable(torch.from_numpy(np.array([frame])).cuda())
    #     mask = model.predict_mask(frame_pytorch)
    #     mask = mask.repeat(1,3,1,1)

    #     masked_frame = frame * mask.data.cpu().numpy()[0]
    #     # masked_frame = masked_frame

    #     # print (frame)
    #     # fds

    #     #scale back up 
    #     frame = frame *255.
    #     masked_frame = masked_frame *255.

    #     frame = frame[:3]
    #     # frame = frame[0:6:2]

    #     frame = np.rollaxis(frame, 1, 0)
    #     frame = np.rollaxis(frame, 2, 1)


    #     # Plot real frame
    #     ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

    #     # state1 = np.concatenate([frame[0], frame[1]] , axis=1)

    #     # ax.imshow(state1) #, cmap='gray')
    #     ax.imshow(np.uint8(frame)) #, cmap='gray')
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    #     # ax.text(0.4, 1.04, 'Real Frame', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, family='serif', size=6)
    #     if i==0:
    #         ax.text(0.4, 1.04, 'Real Frame', transform=ax.transAxes, family='serif', size=6)
    #     else:
    #         ax.text(0.4, 1.04, str(i), transform=ax.transAxes, family='serif', size=6)





    #     #Plot masked
    #     ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)

    #     masked_frame = masked_frame[:3]
    #     masked_frame = np.rollaxis(masked_frame, 1, 0)
    #     masked_frame = np.rollaxis(masked_frame, 2, 1)
    #     # masked_frame = masked_frame.data.cpu().numpy()[0]
    #     # state1 = np.concatenate([masked_frame[0], masked_frame[1]] , axis=1)
    #     # ax.imshow(state1, cmap='gray', norm=NoNorm())
    #     ax.imshow(np.uint8(masked_frame))#, cmap='gray')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     # ax.text(0.4, 1.04, 'Real Frame + Noise', transform=ax.transAxes, family='serif', size=6)
    #     if i==0:
    #         ax.text(0.4, 1.04, 'Real Frame * Mask',  horizontalalignment='center', transform=ax.transAxes, family='serif', size=6)







    #     #Plot mask
    #     ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)

    #     mask = mask.data.cpu().numpy()[0]
    #     mask = mask[:3]
    #     mask = np.rollaxis(mask, 1, 0)
    #     mask = np.rollaxis(mask, 2, 1)
    #     mask = mask[:,:,0]
    #     # state1 = np.concatenate([mask[0], mask[1]] , axis=1)
    #     # ax.imshow(state1, cmap='gray', norm=NoNorm())
    #     ax.imshow(mask, cmap='gray')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     if i==0:
    #         ax.text(0.4, 1.04, 'Mask', transform=ax.transAxes, family='serif', size=6)





    #     # plt_path = save_dir #+'viz.png'
    #     plt.savefig(plt_path)
    #     print ('saved viz',plt_path)
    #     plt.close(fig)



    # print('Making Gif')
    # # frames_path = save_dir+'gif/'
    # images = []
    # for i in range(n_frames):
    #     images.append(imageio.imread(gif_dir+'frame'+str(i)+'.png'))
        
    # # gif_path_this = gif_epoch_path + str(j) + '.gif'
    # gif_path_this = gif_dir+ 'first.gif'
    # imageio.mimsave(gif_path_this, images)
    # print ('made gif', gif_path_this)












