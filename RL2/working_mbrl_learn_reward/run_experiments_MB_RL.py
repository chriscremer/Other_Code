

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import scipy.misc

from os.path import expanduser
home = expanduser("~")

import ball_up_down_actions_stochastic as buda
# from DKF_9feb2017 import DKF
from model_based_RL import MB_RL




#now it must learn the reward function
# so thats a change to the model, ie the observation net outputs the reward prediction
# and the policy objective is through that net now.



if __name__ == "__main__":



    #Which task to run
    train_both = 0
    train_model = 0
    train_policy = 0
    visualize = 1



    #Define the sequence
    n_timesteps = 40
    obs_height = 15
    obs_width = 2
    training_steps = 10000
    display_step = 20
    n_input = obs_height * obs_width
    z_size = 20
    n_actions = 3
    reward_size = 1
    batch_size = 10
    n_particles = 1


    #Specify where to save stuff

    # save_to = home + '/data/' #for boltz
    save_to = home + '/Documents/tmp/' # for mac

    model_path_to_load_variables=save_to + 'mb_rl_model_nosto.ckpt'
    # model_path_to_load_variables=''
    model_path_to_save_variables=save_to + 'mb_rl_model_nosto.ckpt'
    # model_path_to_save_variables=''

    policy_path_to_load_variables=save_to + 'mb_rl_policy_nosto.ckpt'
    # policy_path_to_load_variables=''
    policy_path_to_save_variables=save_to + 'mb_rl_policy_nosto.ckpt'
    # path_to_save_variables=''

    #Tensorboard path
    tb_path =save_to + 'tb_info'



    def get_data():
        sequence_obs, sequence_actions, sequence_rewards = buda.get_sequence(n_timesteps=n_timesteps, obs_height=obs_height, obs_width=obs_width)
        return np.array(sequence_obs), np.array(sequence_actions), np.reshape(np.array(sequence_rewards), [-1,1])


    model_architecture = \
        dict(   encoder_net=[100,100],
                decoder_net=[100,100],
                trans_net=[100,100],
                n_input=n_input,
                n_z=z_size,  
                n_actions=n_actions,
                reward_size=reward_size) 

    policy_architecture = \
        dict(   policy_net=[100,100],
                z_size=z_size, 
                action_size=n_actions,
                input_size=n_input,
                reward_size=reward_size) 



    if train_both==1:

        # print 'Initializing model..'
        # dkf = DKF(network_architecture, batch_size=batch_size, n_particles=n_particles)


        # print 'Training'
        # dkf.train(get_data=get_data, steps=training_steps, display_step=20, 
        #             path_to_load_variables=path_to_load_variables, 
        #             path_to_save_variables=path_to_save_variables)


        mb_rl = MB_RL(model_architecture=model_architecture, 
                        policy_architecture=policy_architecture, 
                        batch_size=batch_size, n_particles=n_particles, n_timesteps=n_timesteps,
                        model_path_to_load_variables=model_path_to_load_variables,
                        model_path_to_save_variables=model_path_to_save_variables,
                        policy_path_to_load_variables=policy_path_to_load_variables,
                        policy_path_to_save_variables=policy_path_to_save_variables,
                        tb_path=tb_path)

        print 'Training both'
        mb_rl.train_both(get_data=get_data, steps=training_steps, display_step=display_step)






    if train_model==1:

        mb_rl = MB_RL(model_architecture=model_architecture, 
                        policy_architecture=policy_architecture, 
                        batch_size=batch_size, n_particles=n_particles, n_timesteps=n_timesteps,
                        model_path_to_load_variables=model_path_to_load_variables,
                        model_path_to_save_variables=model_path_to_save_variables,
                        policy_path_to_load_variables=policy_path_to_load_variables,
                        policy_path_to_save_variables=policy_path_to_save_variables,
                        tb_path=tb_path)

        print 'Training model'
        mb_rl.train_model(get_data=get_data, steps=training_steps, display_step=display_step)




    if train_policy==1:

        mb_rl = MB_RL(model_architecture=model_architecture, 
                        policy_architecture=policy_architecture, 
                        batch_size=batch_size, n_particles=n_particles, n_timesteps=n_timesteps,
                        model_path_to_load_variables=model_path_to_load_variables,
                        model_path_to_save_variables=model_path_to_save_variables,
                        policy_path_to_load_variables=policy_path_to_load_variables,
                        policy_path_to_save_variables=policy_path_to_save_variables,
                        tb_path=tb_path)

        print 'Training policy'
        mb_rl.train_policy(get_data=get_data, steps=training_steps, display_step=display_step)








    if visualize==1:

        print 'Testing model and policy'

        viz_timesteps = n_timesteps
        viz_n_particles = 3
        viz_batch_size = 1

        def get_data():
            sequence_obs, sequence_actions, sequence_rewards = buda.get_sequence(n_timesteps=viz_timesteps, obs_height=obs_height, obs_width=obs_width)
            return np.array(sequence_obs), np.array(sequence_actions), np.reshape(np.array(sequence_rewards), [-1,1])

        mb_rl = MB_RL(model_architecture=model_architecture, 
                policy_architecture=policy_architecture, 
                batch_size=viz_batch_size, n_particles=viz_n_particles, n_timesteps=viz_timesteps,
                model_path_to_load_variables=model_path_to_save_variables,
                model_path_to_save_variables='',
                policy_path_to_load_variables=policy_path_to_save_variables,
                policy_path_to_save_variables='',
                tb_path=tb_path)







        print 'Visualizing model predictions'


        policy_gen_traj = mb_rl.viz_traj_of_policy(n_timesteps=viz_timesteps)

        # print policy_gen_traj.shape [T,X]
        # fsd


        real_sequence, actions, real_and_gen, mean_traj = mb_rl.test_model(get_data=get_data)

        print real_sequence.shape #[T,X]
        print actions.shape #[T,A]
        print real_and_gen.shape #[P, T, X]
        print mean_traj.shape  #[P, T, X]

        #Get return for the trajectories
        real_return = buda.get_return(real_sequence)
        print real_return






        print 'Visualizing policy performance'
        # def get_action_result():
        #     sequence_obs, sequence_actions = buda.get_result_of_action(prez_position, current_action, obs_height, obs_width)
        #     return np.array(sequence_obs), np.array(sequence_actions) 


        seq = mb_rl.test_policy(get_action_results=buda.get_result_of_action, n_timesteps=viz_timesteps, obs_height=obs_height, obs_width=obs_width)

        print seq.shape  #[T,X]
        #plot it

        # fig = plt.figure(figsize=(6, 8))

        # per_traj = 3
        # offset = per_traj+2 #2 for the actions, per_traj for real
        # G = gridspec.GridSpec(per_traj*(viz_n_particles+1)+offset, 3) # +1 for avg


        # axes_2 = plt.subplot(G[2:offset, :])
        # plt.imshow(seq.T, vmin=0, vmax=1, cmap="gray")
        # plt.ylabel('Policy', size=10)
        # plt.yticks([])
        # plt.xticks(size=7)
        # plt.show()


        #PLOT IT ALL


        fig = plt.figure(figsize=(6, 8))

        per_traj = 3
        offset = per_traj+2 #2 for the actions, per_traj for real (ie the 2 is for the actions, the pertraj is for the real trajectory)
        G = gridspec.GridSpec(per_traj*(viz_n_particles+1+1+1+1)+offset, 3) # +1 for avg, +1 for policy, +1 for mean traj, +1 view of policy gen

        axes_1 = plt.subplot(G[0:2, 1])
        plt.imshow(actions.T, vmin=0, vmax=1, cmap="gray")
        plt.ylabel('Actions', size=10)
        plt.yticks([])
        plt.xticks(size=7)

        axes_2 = plt.subplot(G[2:offset, :])
        plt.imshow(real_sequence.T, vmin=0, vmax=1, cmap="gray")
        plt.ylabel('True Traj', size=10)
        plt.yticks([])
        plt.xticks([])
        axes_2.annotate(str(real_return), xy=(1, 1), xytext=(1.5,.5),size=10, xycoords='axes fraction')


        avg_traj = np.zeros((obs_width*obs_height, viz_timesteps))
        for p in range(len(real_and_gen)):

            axes_p = plt.subplot(G[offset+(p*per_traj):offset+(p*per_traj)+per_traj, :])
            plt.imshow(real_and_gen[p].T, vmin=0, vmax=1, cmap="gray")
            plt.ylabel('Traj ' + str(p), size=10)
            plt.yticks([])
            plt.xticks([])
            return_p = buda.get_return(real_and_gen[p])
            axes_p.annotate(str(return_p), xy=(1, 1), xytext=(1.5,.5),size=10, xycoords='axes fraction')

            avg_traj += real_and_gen[p].T 

            if p == len(real_and_gen)-1:
                axes_avg = plt.subplot(G[offset+((p+1)*per_traj):offset+((p+1)*per_traj)+per_traj, :])
                plt.imshow(avg_traj/ float(len(real_and_gen)), vmin=0, vmax=1, cmap="gray")
                plt.ylabel('Avg', size=10)
                plt.yticks([])
                plt.xticks([])
                # return_p = mb_rl.get_return(real_and_gen[p])
                axes_avg.annotate('max is '+str(7*viz_timesteps), xy=(1, 1), xytext=(1.5,.5),size=10, xycoords='axes fraction')


                axes_mean = plt.subplot(G[offset+((p+2)*per_traj):offset+((p+2)*per_traj)+per_traj, :])
                plt.imshow(mean_traj[0].T, vmin=0, vmax=1, cmap="gray")
                plt.ylabel('Mean Traj ', size=10)
                plt.yticks([])
                plt.xticks([])
                return_p = buda.get_return(mean_traj[0])
                axes_mean.annotate(str(return_p), xy=(1, 1), xytext=(1.5,.5),size=10, xycoords='axes fraction')


                axes_policy = plt.subplot(G[offset+((p+3)*per_traj):offset+((p+3)*per_traj)+per_traj, :])
                plt.imshow(seq.T, vmin=0, vmax=1, cmap="gray")
                plt.ylabel('Policy', size=10)
                plt.yticks([])                
                plt.xticks(size=7)
                return_p = buda.get_return(seq)
                axes_policy.annotate(str(return_p), xy=(1, 1), xytext=(1.5,.5),size=10, xycoords='axes fraction')


                axes_policy = plt.subplot(G[offset+((p+4)*per_traj):offset+((p+4)*per_traj)+per_traj, :])
                plt.imshow(policy_gen_traj.T, vmin=0, vmax=1, cmap="gray")
                plt.ylabel('Policy View', size=10)
                plt.yticks([])                
                plt.xticks(size=7)
                return_p = buda.get_return(policy_gen_traj)
                axes_policy.annotate(str(return_p), xy=(1, 1), xytext=(1.5,.5),size=10, xycoords='axes fraction')



        # plt.tight_layout()
        plt.show()























