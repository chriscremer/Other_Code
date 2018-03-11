

import os
from os.path import expanduser
home = expanduser("~")

import numpy as np

import gym
from collections import deque

from NN import NN

#STEPS
# see old code that did pg on this.
# get episode score
# set up similar to train4
# get PG working
# add Q net that tries to predict r
# then use Q for PG, see if it works 
# store a dataste to make it more efficient 

#DONE
#set up a NN to predict R + Q

#NOW
# either use dataset or train a policy using Q


#oh btw this is a discrete task, not continuos



# env_name = home + '/Documents/tmp/' + 'BreakoutNoFrameskip-v4'
# env_name = home + '/Documents/tmp/' + 'CartPole-v1'
env_name =  'CartPole-v0'
env = gym.make(env_name) 

model = NN()


MAX_EPISODES = 10000
MAX_STEPS    = 200

episode_history = deque(maxlen=100)
for i_episode in range(MAX_EPISODES):

  s_t = env.reset()
  total_rewards = 0

  for t in range(MAX_STEPS):
    # env.render()
    # action = pg_reinforce.sampleAction(state[np.newaxis,:])

    rand = np.random.rand()
    if rand > .9:
        a_t = env.action_space.sample()
    else:
        data_x_0 = np.array([np.concatenate([s_t,np.array([0.])])])
        data_x_1 = np.array([np.concatenate([s_t,np.array([1.])])])
        q0 = model.forward(data_x_0).data.numpy()
        q1 = model.forward(data_x_1).data.numpy()
        if q0 > q1:
            a_t = 0
        else:
            a_t = 1




    # print (t, 'action', action, env.action_space)
    # afadsf

    s_tp1, reward, done, _ = env.step(a_t)

    # print (r_t)

    # print t+1, 'state', next_state, 'reward', reward, 'done', done

    total_rewards += reward
    # r_t = 0 if done else 0.1 # normalize reward
    if done:
        r_t = 0
    else:
        r_t = reward
    # pg_reinforce.storeRollout(state, action, reward)




    # print (s_t, a_t)

    if t % 2 ==0:
        data_x = np.array([np.concatenate([s_t,np.array([a_t])])])
        # print (data_x.shape)  #[B,X+A]

        input_xa = np.array([np.concatenate([s_tp1,np.array([env.action_space.sample()])])])
        Q_tp1 = model.forward(input_xa).data.numpy()
        if not done:
            R_and_Q = r_t + Q_tp1 
        else:
            R_and_Q = np.reshape(np.array([r_t]), [1,1])

        # print (R_and_Q)
        
        model.train(data_x, R_and_Q)

    # if t == 20:
    #     print (R_and_Q, model.forward(data_x).data.numpy())

    



    s_t = s_tp1
    # state = next_state
    if done: break

  # fadsafa

  # pg_reinforce.updateModel()

  episode_history.append(total_rewards)
  mean_rewards = np.mean(episode_history)

  if i_episode % 5 == 0:
    # print("Episode {}".format(i_episode))
    # print("Finished after {} timesteps".format(t+1))
    # print("Reward for this episode: {}".format(total_rewards))
    # print("Average reward for last 100 episodes: {}".format(mean_rewards))
    print('Episode', i_episode, 'Return', total_rewards, 'Last 100 Avg', mean_rewards)
    if mean_rewards >= 195.0 and len(episode_history) >= 100:
        print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
        break



print ('Done.')










