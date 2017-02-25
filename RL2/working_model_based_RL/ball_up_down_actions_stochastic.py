
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import scipy.misc

from os.path import expanduser
home = expanduser("~")






#Make sequential data where a ball moves up and down a 1d vector
#Allows me to see the uncertainty in future predicitons 
# see fig 4 https://arxiv.org/pdf/1603.06277v3.pdf
# but ill make it wider than a single vector


import numpy as np

import matplotlib.pyplot as plt

import scipy.misc

from os.path import expanduser
home = expanduser("~")





# def get_sequence(n_timesteps = 100, vector_height = 30, ball_speed = 1, direction = 1):

#     # position = int(vector_height / 2)
#     position = np.random.randint(0, vector_height)

#     sequence = []
#     for t in range(n_timesteps):

#         position += ball_speed*direction

#         if position < 0:
#             position =0
#             direction *= -1
#         if position >= vector_height-1:
#             position = vector_height-1
#             direction *= -1

#         state = np.zeros([vector_height])
#         state[position] = 1.
#         sequence.append(state)

#     # [timesteps, vector_height]
#     sequence = np.array(sequence)
#     return sequence


# def get_sequence():

#     n_timesteps = 20
#     obs_height = 30
#     obs_width = 3
#     obs_shape = [obs_height, obs_width]
#     ball_speed = 1
#     direction = 1
#     position = int(obs_height / 2)


#     for t in range(n_timesteps):

#         action = np.random.randint(0,3)
#         action_mat = np.zeros([3,obs_width])
#         action_mat[action] = 1

#         action -= 1

#         # if action == -1:
#         direction = action

#         position += ball_speed*direction

#         if position < 0:
#             position =0
#             direction *= -1
#         if position >= obs_height-1:
#             position = obs_height-1
#             direction *= -1

#         obs = np.zeros(obs_shape)
#         obs[position] = 1.

#         if t==0:
#             concat_timesteps = obs
#             concat_actions = action_mat
#         else:
#             concat_timesteps = np.concatenate((concat_timesteps, obs), axis=1)
#             concat_actions = np.concatenate((concat_actions, action_mat), axis=1)

#     # print concat_timesteps.shape
#     # print concat_actions.shape

#     return concat_timesteps, concat_actions



def get_sequence(n_timesteps, obs_height, obs_width):

    obs_shape = [obs_height, obs_width]
    ball_speed = 1
    direction = 1
    # position = int(obs_height / 2)
    position = np.random.randint(0,obs_height)

    stochasticity = .0001


    sequence_obs =[]
    sequence_actions =[]
    for t in range(n_timesteps):

        action = np.random.randint(0,3)
        action_mat = np.zeros([3])
        action_mat[action] = 1

        action -= 1
        direction = action

        #STOCHASTICITY
        val = np.random.rand()
        if direction == 1:
            if val<stochasticity:
                direction = -1
        elif direction == -1:
            if val<stochasticity:
                direction = 1

        position += ball_speed*direction

        if position < 0:
            position =0
            direction *= -1
        if position >= obs_height-1:
            position = obs_height-1
            direction *= -1

        obs = np.zeros(obs_shape)
        obs[position] = 1.
        #make in to vector
        obs = np.reshape(obs, [-1])

        # if t==0:
        #     concat_timesteps = obs
        #     concat_actions = action_mat
        # else:
        #     concat_timesteps = np.concatenate((concat_timesteps, obs), axis=1)
        #     concat_actions = np.concatenate((concat_actions, action_mat), axis=1)

        sequence_obs.append(obs)
        sequence_actions.append(action_mat)

    # print concat_timesteps.shape
    # print concat_actions.shape

    return sequence_obs, sequence_actions








def get_result_of_action(prev_position, current_action, obs_height, obs_width):

    #int action
    action = np.argmax(current_action)
    action -= 1

    #STOCHASTICITY
    val = np.random.rand()
    if action == 1:
        if val<.2:
            action = -1
    elif action == -1:
        if val<.2:
            action = 1


    position = prev_position+action

    if position < 0:
        position =0
    if position >= obs_height-1:
        position = obs_height-1

    obs_shape = [obs_height, obs_width]
    obs = np.zeros(obs_shape)
    obs[position] = 1.
    #make in to vector
    obs = np.reshape(obs, [-1])


    return obs, position




if __name__ == "__main__":


    # save_to = home + '/data/' #for boltz
    save_to = home + '/Documents/tmp/' # for mac


    n_timesteps = 20
    obs_height = 30
    obs_width = 3
    obs_shape = [obs_height, obs_width]
    ball_speed = 1
    direction = 1
    position = int(obs_height / 2)



    #Initialize vector
    # bouncy_ball = np.zeros([obs_height,1])

    #Randomly start the ball at a spot and a direction

    #Depending on speed, move the ball in the next vector

    # sequence = []

    for t in range(n_timesteps):

        action = np.random.randint(0,3)
        action_mat = np.zeros([3,obs_width])
        action_mat[action] = 1

        action -= 1

        # if action == -1:
        direction = action

        position += ball_speed*direction

        if position < 0:
            position =0
            direction *= -1
        if position >= obs_height-1:
            position = obs_height-1
            direction *= -1


        obs = np.zeros(obs_shape)
        obs[position] = 1.

        # sequence.append(obs)


        if t==0:
            concat_timesteps = obs
            concat_actions = action_mat
        else:
            concat_timesteps = np.concatenate((concat_timesteps, obs), axis=1)
            concat_actions = np.concatenate((concat_actions, action_mat), axis=1)

    # sequence = np.array(sequence).T
    print concat_timesteps.shape
    print concat_actions.shape

    #Continue for a number of steps

    # scipy.misc.imsave(save_to +'outfile.jpg', concat_timesteps)
    # print 'saved'


    # fig = plt.figure(figsize=(8, 6))
    fig = plt.figure()

    # fig, (ax1, ax2) = plt.subplots(2, 1)

    # #nrows, ncols, plot_number
    # plt.subplot(4,1,1)
    # plt.imshow(concat_actions, vmin=0, vmax=1, cmap="gray", shape=concat_actions.shape)
    # plt.ylabel('Actions')
    # plt.yticks([])


    # # ax = plt.subplot(3,1,2)
    # plt.subplot(2,1,2)
    # plt.imshow(concat_timesteps, vmin=0, vmax=1, cmap="gray", shape=concat_timesteps.shape)
    # plt.ylabel('Observations')
    # plt.yticks([])



    G = gridspec.GridSpec(4, 1)

    axes_1 = plt.subplot(G[0, :])
    plt.imshow(concat_actions, vmin=0, vmax=1, cmap="gray", shape=concat_actions.shape)
    plt.ylabel('Actions')
    plt.yticks([])

    axes_2 = plt.subplot(G[1:, :])
    plt.imshow(concat_timesteps, vmin=0, vmax=1, cmap="gray", shape=concat_timesteps.shape)
    plt.ylabel('Observations')
    plt.yticks([])


    # plt.tight_layout()
    plt.show()
    # fig.savefig('yourfilename.png')

    #Visualize the concatenation of all timesteps. 



    #Also it would be really cool to see it learn in real time. 
    # So after each time step, show the predictions .









