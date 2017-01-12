



import numpy as np
import tensorflow as tf
import gym
from collections import deque

from policy_gradient_reinforce import REINFORCE
from DKF6 import DKF

from os.path import expanduser
home = expanduser("~")



env = gym.make('CartPole-v0')

state_dim   = env.observation_space.shape[0]
num_actions = env.action_space.n
network_architecture = dict(    policy_net=[20],
                                n_input=state_dim, 
                                n_actions=num_actions)

pg_reinforce = REINFORCE(network_architecture)
pg_reinforce.initialize_sess()

MAX_EPISODES = 100
MAX_STEPS    = 200

dataset = []

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

    dataset.append([states, actions, rewards])

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






print 'Train model of dynamics'

#TODO: -cahgne reward to discounted reward
#      - add reward to observations
#      - modify dkf to handle the different observations


# dataset = np.array(dataset)
# print dataset.shape
# fdsfa



# calc discounted reward
# pad the dataset so theyre all the same length 
# add reward to observation

# actually I dont think I should calc discounted rewards, because its dependent on the policy

# discount_factor = .99
# for episode in range(len(dataset)):

#     episode_rewards = dataset[episode][2]
#     discounted_rewards = np.zeros(len(episode_rewards))
#     r=0
#     for t in reversed(xrange(len(episode_rewards))):
#         # future discounted reward from now on
#         r = episode_rewards[t] + discount_factor * r
#         discounted_rewards[t] = r

#     dataset[episode][2] = discounted_rewards


for episode in range(len(dataset)):

    # append reward to observation, this is what the model will be predicting
    dataset[episode].append([])
    for t in range(len(dataset[episode][0])):
        
        dataset[episode][3].append(list(dataset[episode][0][t]) +  [dataset[episode][2][t]])

    # print dataset[episode][0]
    # print dataset[episode][2]
    # print dataset[episode][3]

#convert actions to one hot
for episode in range(len(dataset)):

    dataset[episode].append([])
    for t in range(len(dataset[episode][1])):

        action = dataset[episode][1][t]

        actions_ = np.zeros((num_actions))
        actions_[action] = 1
        dataset[episode][4].append(actions_)


# pad the episodes
for episode in range(len(dataset)):

    while(len(dataset[episode][3]) != MAX_STEPS):

        dataset[episode][3].append([0.]*5)
        dataset[episode][4].append([0.]*num_actions)


save_to = home + '/data/' #for boltz
# save_to = home + '/Documents/tmp/' # for mac

training_steps = 1000
n_input = 5 #4 for state and 1 for reward
n_time_steps = MAX_STEPS
n_particles = 3
batch_size = 4
# path_to_load_variables=save_to + 'dkf_ball_vars.ckpt'
path_to_load_variables=''
# path_to_save_variables=save_to + 'dkf_ball_vars2.ckpt'
path_to_save_variables=''

train = 1
test = 1


def get_data():

    n_episodes = len(dataset)
    #select a random episode
    epi = np.random.randint(0,n_episodes/2)

    sequence = dataset[epi][3]
    actions = dataset[epi][4]

    sequence = np.array(sequence)
    actions =  np.array(actions)
    # actions = np.reshape(actions, [-1,1])

    return sequence, actions

def get_data_test():

    n_episodes = len(dataset)
    #select a random episode
    epi = np.random.randint(n_episodes/2, n_episodes)

    sequence = dataset[epi][3]
    actions = dataset[epi][4]

    sequence = np.array(sequence)
    actions =  np.array(actions)
    # actions = np.reshape(actions, [-1,1])
    return sequence, actions



network_architecture = \
    dict(   encoder_net=[20],
            decoder_net=[20],
            trans_net=[20],
            n_input=n_input,
            n_z=4,  
            n_actions=num_actions) 

print 'Initializing model..'
dkf = DKF(network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=batch_size, n_time_steps=n_time_steps, n_particles=n_particles)


if train:
    print 'Training'
    dkf.train(get_data=get_data, steps=training_steps, display_step=20, path_to_load_variables=path_to_load_variables, path_to_save_variables=path_to_save_variables)


if test:
    print 'Test'
    dkf.test(get_data=get_data_test, steps=100, display_step=1, path_to_load_variables='', path_to_save_variables='')





#Ok now train policy

# start by producing one sequence. 
# so give 0 observation to model, show it to policy, get action, apply action, get observation.


#give observation net 0 observation and 0 z, *im not sure if it evers sees that ,so not sure if its valid
# itll predict a observation
# give it policy
# get action
# give it to model
# generate next state
# give state and observation to make the next observation



# so need ot make a method to call frmo dkf


obs = dkf.get_current_emission(np.zeros([1,4]), np.zeros([1,5]))

mean = obs[0]
log_var = obs[1]

print obs
print mean.shape
print log_var.shape

mean_2 = np.reshape(mean, [5])
action = pg_reinforce.sampleAction([mean_2[:4]]) 

actions_ = np.zeros((num_actions))
actions_[action] = 1
actions_ = np.reshape(actions_, [1,2])

# print obs
# print obs.shape

state = dkf.get_next_state(np.zeros([1,4]), actions_, mean)

state_mean = state[0]
state_log_var = state[1]

print state
print state_mean.shape
print state_log_var.shape

print 


for t in range(30):



    obs = dkf.get_current_emission(state_mean, mean)
    mean = obs[0]
    log_var = obs[1]

    mean_2 = np.reshape(mean, [5])
    action = pg_reinforce.sampleAction([mean_2[:4]]) 

    actions_ = np.zeros((num_actions))
    actions_[action] = 1
    actions_ = np.reshape(actions_, [1,2])

    state = dkf.get_next_state(np.zeros([1,4]), actions_, mean)
    state_mean = state[0]
    state_log_var = state[1]


    print t
    print mean_2
    print action
    print 



print 'Lets train the policy'

for e in range(1000):

    if e % 50 == 0:
        print e

    states = []
    actions = []
    rewards = []

    # obs = dkf.get_current_emission(np.zeros([1,4]), np.zeros([1,5]))
    obs = dkf.get_current_emission(np.random.normal(size=[1,4]), np.random.normal(size=[1,5]))


    mean = obs[0]
    log_var = obs[1]
    mean_2 = np.reshape(mean, [5])
    action = pg_reinforce.sampleAction([mean_2[:4]]) 
    actions_ = np.zeros((num_actions))
    actions_[action] = 1
    actions_ = np.reshape(actions_, [1,2])
    state = dkf.get_next_state(np.zeros([1,4]), actions_, mean)
    state_mean = state[0]
    state_log_var = state[1]

    states.append(mean_2[:4])
    actions.append(action)
    rewards.append(mean_2[-1])

    for t in range(50):

        obs = dkf.get_current_emission(state_mean, mean)
        mean = obs[0]
        log_var = obs[1]

        mean_2 = np.reshape(mean, [5])
        action = pg_reinforce.sampleAction([mean_2[:4]]) 

        actions_ = np.zeros((num_actions))
        actions_[action] = 1
        actions_ = np.reshape(actions_, [1,2])

        state = dkf.get_next_state(np.zeros([1,4]), actions_, mean)
        state_mean = state[0]
        state_log_var = state[1]

        states.append(mean_2[:4])
        actions.append(action)
        rewards.append(mean_2[-1])

        # print t
        # print mean_2
        # print action
        # print 


    pg_reinforce.update_policy(states, actions, rewards)



print 'Test policy'


state_dim   = env.observation_space.shape[0]
num_actions = env.action_space.n
network_architecture = dict(    policy_net=[20],
                                n_input=state_dim, 
                                n_actions=num_actions)

pg_reinforce = REINFORCE(network_architecture)
pg_reinforce.initialize_sess()

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

    dataset.append([states, actions, rewards])

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

























