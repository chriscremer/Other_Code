

import os
from os.path import expanduser
home = expanduser("~")


import gym


#STEPS
# see old code that did pg on this.
# get episode score
# set up similar to train4
# get PG working
# add Q net that tries to predict r
# then use Q for PG, see if it works 
# store a dataste to make it more efficient 







# env_name = home + '/Documents/tmp/' + 'BreakoutNoFrameskip-v4'
# env_name = home + '/Documents/tmp/' + 'CartPole-v1'
env_name =  'CartPole-v1'


env = gym.make(env_name) 
env.reset()
for _ in range(200):
    # env.render()
    state, reward, done, info = env.step(env.action_space.sample()) 
    print (state, reward, done, info )

print ('Done.')


























