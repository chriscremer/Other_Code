
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
env = gym.make('CarRacing-v0')
env.reset()
for _ in range(100): # run for 1000 steps
    # env.render()
    action = env.action_space.sampe() # pick a random action
    env.step(action) # take action

    print (action)




