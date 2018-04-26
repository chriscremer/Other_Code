
#this is on python3 , not anaconda and not 2.7

from vizdoom import *
import random
import time

import numpy as np

from os.path import expanduser
home = expanduser("~")


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



game = DoomGame()
game.load_config(home + "/ViZDoom/scenarios/take_cover.cfg")
game.set_window_visible(False)
# game.load_config("vizdoom/scenarios/basic.cfg")
game.init()

shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]

episodes = 10
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        # print (state.shape)
        img = state.screen_buffer
        print (img.shape)

        # img = np.reshape(img, [240,320,3])
        img = np.rollaxis(img, 1, 0)
        img = np.rollaxis(img, 2, 1)
        print (img.shape)

        plt.imshow(img)

        save_dir = home+'/Documents/tmp/Doom/'
        plt_path = save_dir+'frmame.png'
        plt.savefig(plt_path)
        print ('saved viz',plt_path)
        plt.close(fig)

        fdas

        misc = state.game_variables
        reward = game.make_action(random.choice(actions))
        print ("\treward:", reward)
        time.sleep(0.02)
    print ("Result:", game.get_total_reward())
    time.sleep(2)

afdads




import gym
env = gym.make('DoomTakeCover-v0')
env.reset()
for _ in range(100): # run for 1000 steps
    # env.render()
    action = env.action_space.sampe() # pick a random action
    env.step(action) # take action

    print (action)











import gym
import gym_pull
gym_pull.pull('github.com/ppaquette/gym-super-mario')        # Only required once, envs will be loaded with import gym_pull afterwards
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')



fsdas




import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler







import gym
# env = gym.make('DoomTakeCover-v0')
# env = gym.make('FetchReach-v0')
env = gym.make('CarRacing-v0')
env.reset()
for _ in range(100): # run for 1000 steps
    # env.render()
    action = env.action_space.sampe() # pick a random action
    env.step(action) # take action

    print (action)

fdsfa



v
import retro
env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')


fdsa



import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler






