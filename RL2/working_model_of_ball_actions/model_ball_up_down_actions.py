

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import scipy.misc

from os.path import expanduser
home = expanduser("~")

import ball_up_down_actions as buda
from DKF_9feb2017 import DKF


if __name__ == "__main__":

    #Define the sequence
    n_timesteps = 10
    obs_height = 15
    obs_width = 2

    def get_data():
        sequence_obs, sequence_actions = buda.get_sequence(n_timesteps=n_timesteps, obs_height=obs_height, obs_width=obs_width)
        return np.array(sequence_obs), np.array(sequence_actions)


    #Specify where to save stuff

    # save_to = home + '/data/' #for boltz
    save_to = home + '/Documents/tmp/' # for mac
    # path_to_load_variables=save_to + 'dkf_ball_vars.ckpt'
    path_to_load_variables=''
    path_to_save_variables=save_to + 'bouncy_ball.ckpt'
    # path_to_save_variables=''



    #Define the model setup

    training_steps = 8000
    n_input = obs_height * obs_width
    n_time_steps = n_timesteps
    batch_size = 5

    train = 0
    visualize=1

    network_architecture = \
        dict(   encoder_net=[100,100],
                decoder_net=[100,100],
                trans_net=[100,100],
                n_input=n_input,
                n_z=20,  
                n_actions=3) 





    if train==1:

        print 'Initializing model..'
        dkf = DKF(network_architecture, batch_size=batch_size)


        print 'Training'
        dkf.train(get_data=get_data, steps=training_steps, display_step=20, 
                    path_to_load_variables=path_to_load_variables, 
                    path_to_save_variables=path_to_save_variables)


    if visualize==1:

        print 'Visualizing'
        def get_data():
            sequence_obs, sequence_actions = buda.get_sequence(n_timesteps=30, obs_height=obs_height, obs_width=obs_width)
            return np.array(sequence_obs), np.array(sequence_actions)
        #get actions and frames
        #give model all actions and only some frames

        print 'Initializing model..'
        batch_size = 1
        dkf = DKF(network_architecture, batch_size=batch_size)

        real_sequence, actions, real_and_gen = dkf.test(get_data=get_data, path_to_load_variables=path_to_save_variables)

        print real_sequence.shape
        print actions.shape
        print real_and_gen.shape



        fig = plt.figure()

        G = gridspec.GridSpec(7, 3)

        axes_1 = plt.subplot(G[0, 1])
        plt.imshow(actions.T, vmin=0, vmax=1, cmap="gray")
        plt.ylabel('Actions')
        plt.yticks([])

        axes_2 = plt.subplot(G[1:4, :])
        plt.imshow(real_sequence.T, vmin=0, vmax=1, cmap="gray")
        plt.ylabel('Observations')
        plt.yticks([])
        plt.xticks([])

        axes_3 = plt.subplot(G[4:7, :])
        plt.imshow(real_and_gen.T, vmin=0, vmax=1, cmap="gray")
        plt.ylabel('Predictions')
        plt.yticks([])


        # plt.tight_layout()
        plt.show()


