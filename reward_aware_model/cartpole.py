


import numpy as np
import gym




env = gym.make('CartPole-v0')


# env = gym.make('MountainCar-v0')



obs_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

print (obs_dim)
print (num_actions)
print()




#Make dataset
# print 'Making dataset'
# dataset = []
MAX_EPISODES = 100
MAX_STEPS    = 200

# episode_history = deque(maxlen=100)
for i_episode in range(MAX_EPISODES):

    if i_episode %10 == 0:
        print (i_episode)

    # initialize
    obs = env.reset()
    # total_rewards = 0

    observations = []
    actions = []
    rewards = []

    for t in range(MAX_STEPS):


        # env.render()
        # action = pg_reinforce.sampleAction([state])  
        action = np.random.randint(num_actions)
        obs, reward, done, _ = env.step(action)

        # print reward

        # total_rewards += reward

        reward = -10 if done else 0.1
        one_hot_action = np.zeros((num_actions))
        one_hot_action[action] = 1.

        observations.append(obs)
        actions.append(one_hot_action)
        rewards.append(reward)
  
        if done: break

    # print len(observations)

    for i in range(1, len(rewards)):
        rewards[-i-1] = rewards[-i-1] + rewards[-i]

    for i in range(len(rewards)):
        rewards[i] = [rewards[i]]
    # rewards = np.array(rewards) - np.mean(rewards)

#     # print rewards
#     dataset.append([observations, actions, rewards])

# print 'Dataset size:' +str(len(dataset))
# # print len(dataset[0][0])
# # print len(dataset[1][0])













