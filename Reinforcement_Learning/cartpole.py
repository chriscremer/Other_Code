

import numpy as np
import tensorflow as tf
import gym
from collections import deque

from policy_gradient_reinforce import REINFORCE





env = gym.make('CartPole-v0')

state_dim   = env.observation_space.shape[0]
num_actions = env.action_space.n
network_architecture = dict(    policy_net=[20],
                                n_input=state_dim, 
                                n_actions=num_actions)

pg_reinforce = REINFORCE(network_architecture)
pg_reinforce.initialize_sess()

MAX_EPISODES = 10000
MAX_STEPS    = 200

episode_history = deque(maxlen=100)
for i_episode in xrange(MAX_EPISODES):

    # initialize
    state = env.reset()
    total_rewards = 0

    states = []
    actions = []
    rewards = []

    for t in xrange(MAX_STEPS):

        # env.render()


        action = pg_reinforce.sampleAction([state])        

        next_state, reward, done, _ = env.step(action)

        total_rewards += reward

        reward = -10 if done else 0.1

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        

        if done: break

    pg_reinforce.update_policy(states, actions, rewards)

    episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)


    print("Episode {}".format(i_episode))
    print("Finished after {} timesteps".format(t+1))
    # print("Reward for this episode: {}".format(total_rewards))
    print("Average reward for last 100 episodes: {}".format(mean_rewards))
    if mean_rewards >= 195.0 and len(episode_history) >= 100:
        print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
        break
