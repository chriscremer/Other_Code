

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import scipy.misc

from os.path import expanduser
home = expanduser("~")

import ball_up_down_actions_stochastic as buda
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
    n_particles = 4

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
        dkf = DKF(network_architecture, batch_size=batch_size, n_particles=n_particles)


        print 'Training'
        dkf.train(get_data=get_data, steps=training_steps, display_step=20, 
                    path_to_load_variables=path_to_load_variables, 
                    path_to_save_variables=path_to_save_variables)


    if visualize==1:

        print 'Visualizing'
        viz_timesteps = 40
        viz_n_particles = 3
        viz_batch_size = 1

        def get_data():
            sequence_obs, sequence_actions = buda.get_sequence(n_timesteps=viz_timesteps, obs_height=obs_height, obs_width=obs_width)
            return np.array(sequence_obs), np.array(sequence_actions)
        #get actions and frames
        #give model all actions and only some frames

        print 'Initializing model..'
        
        dkf = DKF(network_architecture, batch_size=viz_batch_size, n_particles=viz_n_particles)


        # need to change it so i get mulitple trajectories , one for each particle

        real_sequence, actions, real_and_gen = dkf.test(get_data=get_data, path_to_load_variables=path_to_save_variables)

        print real_sequence.shape #[T,X]
        print actions.shape #[T,A]
        print real_and_gen.shape #[P, T, X]


        fig = plt.figure(figsize=(6, 8))

        per_traj = 3
        offset = per_traj+2 #2 for the actions, per_traj for real
        G = gridspec.GridSpec(per_traj*(n_particles+1)+offset, 3) # +1 for avg

        axes_1 = plt.subplot(G[0:2, 1])
        plt.imshow(actions.T, vmin=0, vmax=1, cmap="gray")
        plt.ylabel('Actions', size=10)
        plt.yticks([])
        plt.xticks(size=7)

        axes_2 = plt.subplot(G[2:offset, :])
        plt.imshow(real_sequence.T, vmin=0, vmax=1, cmap="gray")
        plt.ylabel('True Trajectory', size=10)
        plt.yticks([])
        plt.xticks([])

        avg_traj = np.zeros((obs_width*obs_height, viz_timesteps))
        for p in range(len(real_and_gen)):

            plt.subplot(G[offset+(p*per_traj):offset+(p*per_traj)+per_traj, :])
            plt.imshow(real_and_gen[p].T, vmin=0, vmax=1, cmap="gray")
            plt.ylabel('Trajectory ' + str(p), size=10)
            plt.yticks([])
            plt.xticks([])

            avg_traj += real_and_gen[p].T 

            if p == len(real_and_gen)-1:
                plt.subplot(G[offset+((p+1)*per_traj):offset+((p+1)*per_traj)+per_traj, :])
                plt.imshow(avg_traj/ float(len(real_and_gen)), vmin=0, vmax=1, cmap="gray")
                plt.ylabel('Avg', size=10)
                plt.yticks([])
                plt.xticks(size=7)


        # plt.tight_layout()
        plt.show()


