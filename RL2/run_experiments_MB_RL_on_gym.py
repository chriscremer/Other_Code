
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import scipy.misc

from os.path import expanduser
home = expanduser("~")

# import ball_up_down_actions_stochastic as buda
# from DKF_9feb2017 import DKF
from model_based_RL import MB_RL

import gym

import pickle

import random


# from skimage.color import rgb2grey

from PIL import Image

#these wont work on boltz, so just comment out
# import cv2  #for viz
# from skimage.measure import block_reduce


#now we'll see if this MB-RL works on Open AI gym


if __name__ == "__main__":

    #Which task to run
    make_data = 0
    train_both = 0
    train_model = 1
    train_policy = 0
    # train_policy_using_policy = 1
    visualize = 0
    run_gym = 0
    viz2 = 0

    # save_to = home + '/data/' #for boltz
    save_to = home + '/Documents/tmp/' # for mac












    if make_data ==1:

        #GYM environment

        env = gym.make('CartPole-v0')
        # env = gym.make('MountainCar-v0')

        # obs_dim = env.observation_space.shape[0]
        obs_dim = 24*56
        print 'obs dims ' + str(obs_dim)

        num_actions = env.action_space.n
        print 'action dims ' + str(num_actions)


        #Make dataset
        print 'Making dataset'
        MAX_EPISODES = 20
        MAX_STEPS    = 200
        dataset = []
        lenghts = []
        for i_episode in xrange(MAX_EPISODES):
            if i_episode %10==0:
                print i_episode

            obs = env.reset()

            observations = []
            actions = []
            rewards = []
            for t in xrange(MAX_STEPS):

                rgb_array = env.render(mode='rgb_array') 

                # image = rgb_array[150:350,100:500,:] #.shape 200, 400, 3
                image = rgb_array[180:300,160:440,:]  #200 300 3

                # image = np.reshape(image, [-1])

                # print image.shape
                image = block_reduce(image, block_size=(5, 5, 1), func=np.max) #now its 40 60 3 = 7200
                # print image.shape


                image = Image.fromarray(image, 'RGB').convert('L')

                image = np.array(image)
                image_shape = image.shape
                image = np.reshape(image, [image_shape[0], image_shape[1], 1]) #24, 56, 1

                image = image / 255.

                # image = np.array(image)
                # print image.shape
                # fasfs

                #show pil grey image, dont have the third dimension, image is 2d array
                # image = Image.fromarray(image, 'L')
                # image.show()
                # fsd

                # show rgb array
                # plt.imshow(image) 
                # plt.show()
                # fsdaf

                action = np.random.randint(num_actions)
                obs, reward, done, _ = env.step(action)

                reward = -10 if done else 0.1
                one_hot_action = np.zeros((num_actions))
                one_hot_action[action] = 1.

                observations.append(image)
                actions.append(one_hot_action)
                rewards.append(reward)

                if done: break


            # for i in range(1, len(rewards)):
            #     rewards[-i-1] = rewards[-i-1] + rewards[-i]

            for i in range(len(rewards)):
                rewards[i] = [rewards[i]]
            # rewards = np.array(rewards) - np.mean(rewards)

            # print rewards
            dataset.append([observations, actions, rewards])
            lenghts.append(len(observations))

        print 'Dataset size:' +str(len(dataset))
        print 'Average length:' + str(np.mean(lenghts))
        print 'Min length:' + str(np.min(lenghts))
        print 'Max length:' + str(np.max(lenghts))

        # print len(dataset[0][0])
        # print len(dataset[1][0])

            
        with open(save_to+'cartpole_data_validation.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        print 'saved data to: ' +save_to+'cartpole_data_validation.pkl'
        fsaf




    #load data
    print 'loading data'
    with open(save_to+'cartpole_data3.pkl', 'rb') as f:
        dataset = pickle.load(f)
    len_dataset = len(dataset)
    # dataset = np.array(dataset) cant because len of seqs are dif
    print 'loaded data from: ' +save_to+'cartpole_data3.pkl', len_dataset

    with open(save_to+'cartpole_data_validation.pkl', 'rb') as f:
        validation_set = pickle.load(f)
    print 'loaded data from: ' +save_to+'cartpole_data_validation.pkl', len(validation_set)


    # print dataset[0][2]
    # # for i in range(len(dataset[0][0][0])):#.shape
    # #     print dataset[0][0][0][i]

    # fadsf


    #Define the sequence
    n_timesteps = 30 #for simulated trajs
    # obs_height = 15
    # obs_width = 2
    training_steps = 100000
    display_step = 20
    n_input = 1344
    z_size = 20
    n_actions = 2
    reward_size = 1
    batch_size = 1
    n_particles = 1


    #Specify where to save stuff

    # save_to = home + '/data/' #for boltz
    save_to = home + '/Documents/tmp/' # for mac

    model_path_to_load_variables=save_to + 'mb_rl_model_cartpole_pixels6.ckpt'
    # model_path_to_load_variables=''
    model_path_to_save_variables=save_to + 'mb_rl_model_cartpole_pixels7.ckpt'
    # model_path_to_save_variables=''

    policy_path_to_load_variables=save_to + 'mb_rl_policy_cartpole_pixels6.ckpt'
    # policy_path_to_load_variables=''
    policy_path_to_save_variables=save_to + 'mb_rl_policy_cartpole_pixels7.ckpt'
    # path_to_save_variables=''

    #Tensorboard path
    tb_path =save_to + 'tb_info'



    # def get_data():
    #     sequence_obs, sequence_actions, sequence_rewards = buda.get_sequence(n_timesteps=n_timesteps, obs_height=obs_height, obs_width=obs_width)
    #     return np.array(sequence_obs), np.array(sequence_actions), np.reshape(np.array(sequence_rewards), [-1,1])

    def get_data(valid=-1, step=-1):

        if valid != -1:
            ind = valid
            return np.array(validation_set[ind][0]), np.array(validation_set[ind][1]), np.array(validation_set[ind][2])
        
        elif step != -1:
            ind = step % len_dataset
            if ind == 0:
                #shuffle
                random.shuffle(dataset)

            return np.array(dataset[ind][0]), np.array(dataset[ind][1]), np.array(dataset[ind][2])

        else:
            ind = np.random.randint(len(dataset))
            #return a sequence from the dataset. obs, actions, rewards
            return np.array(dataset[ind][0]), np.array(dataset[ind][1]), np.array(dataset[ind][2])



    model_architecture = \
        dict(   encoder_net=[100,100],
                decoder_net=[100,100],
                trans_net=[20,20],
                n_input=n_input,
                n_z=z_size,  
                n_actions=n_actions,
                reward_size=reward_size) 

    policy_architecture = \
        dict(   policy_net=[20,20],
                z_size=z_size, 
                action_size=n_actions,
                input_size=n_input,
                reward_size=reward_size) 


    #dont need this, just putting here for now
    # n_timesteps = 0


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
        mb_rl.train_policy(steps=training_steps, display_step=display_step)





    # if train_policy_using_policy == 1:

    #     mb_rl = MB_RL(model_architecture=model_architecture, 
    #                     policy_architecture=policy_architecture, 
    #                     batch_size=batch_size, n_particles=n_particles, n_timesteps=n_timesteps,
    #                     model_path_to_load_variables=model_path_to_load_variables,
    #                     model_path_to_save_variables=model_path_to_save_variables,
    #                     policy_path_to_load_variables=policy_path_to_load_variables,
    #                     policy_path_to_save_variables=policy_path_to_save_variables,
    #                     tb_path=tb_path)

    #     print 'Training policy with policy'
    #     mb_rl.train_policy(steps=training_steps, display_step=display_step)






    if visualize==1:
        #this shows what the generated obs look like
        # and can help for other debugging


        # print 'Testing model and policy'
        print 'Viz'


        viz_timesteps = 10
        viz_n_particles = 1
        viz_batch_size = 1

        # def get_data():
        #     sequence_obs, sequence_actions, sequence_rewards = buda.get_sequence(n_timesteps=viz_timesteps, obs_height=obs_height, obs_width=obs_width)
        #     return np.array(sequence_obs), np.array(sequence_actions), np.reshape(np.array(sequence_rewards), [-1,1])

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

        for frame in policy_gen_traj:
            print '1'
            cv2.imshow('Video', np.reshape(frame, [24, 56]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # for t in range(len(policy_gen_traj)):

        #     image = np.reshape(policy_gen_traj[t], [24, 56]) * 255.
        #     image = Image.fromarray(image, 'L')
        #     image.show()


        fsadf


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
















    if run_gym == 1:
        #this shows how well the policy does in the environment

        def convert_image(image):
            # - crop
            # - downsample
            # - greyscale
            # - bernoulli

            # image = rgb_array[150:350,100:500,:] #.shape 200, 400, 3
            image = image[180:300,160:440,:]  #200 300 3
            # image = np.reshape(image, [-1])
            # print image.shape
            image = block_reduce(image, block_size=(5, 5, 1), func=np.max) #now its 40 60 3 = 7200
            # print image.shape
            image = Image.fromarray(image, 'RGB').convert('L')
            image = np.array(image)
            image_shape = image.shape
            image = np.reshape(image, [image_shape[0], image_shape[1], 1]) #24, 56, 1
            image = image / 255.

            image = np.reshape(image, [-1]) #1


            return image



        mb_rl = MB_RL(model_architecture=model_architecture, 
                policy_architecture=policy_architecture, 
                batch_size=1, n_particles=1, n_timesteps=1,
                model_path_to_load_variables=model_path_to_save_variables,
                model_path_to_save_variables='',
                policy_path_to_load_variables=policy_path_to_save_variables,
                policy_path_to_save_variables='',
                tb_path=tb_path)


        env = gym.make('CartPole-v0')

        MAX_EPISODES = 100
        MAX_STEPS    = 200

        running_timesteps_mean = 0
        count = 0

        # episode_history = deque(maxlen=100)
        for i_episode in xrange(MAX_EPISODES):

            # if i_episode %10 == 0:
            #     print i_episode

            # initialize
            obs = env.reset()
            
            obs = env.render(mode='rgb_array') 
            obs = convert_image(obs)

            obs_shape = obs.shape
            obs_dims = np.prod(obs_shape)
            # print 'obs dims ' + str(obs_dim)
            num_actions = env.action_space.n
            # print 'action dims ' + str(num_actions)


            # get first state, using recognitino net, given current obs and zero state and random action
            state = mb_rl.sess.run(mb_rl.model.z_mean_, feed_dict={mb_rl.model.current_observation: [obs], mb_rl.model.prev_z_: [np.zeros(z_size)], mb_rl.model.current_a_: [np.zeros((num_actions))]})

            for t in xrange(MAX_STEPS):

                # env.render()


                # rgb_array = env.render(mode='rgb_array') 
                # from matplotlib import pyplot as PLT
                # PLT.imshow(rgb_array[100:500,5:200,:])
                # PLT.show()



                # PLT.imshow(rgb_array[10:50,5:20,:])
                # PLT.show()

                # print rgb_array.shape

                # fasdfa


                # action = pg_reinforce.sampleAction([state])  

                #random action
                # action_vec = np.zeros((num_actions))
                # action_vec[np.random.randint(num_actions)] = 1
                # action_vec = [action_vec]

                action_vec = mb_rl.sess.run(mb_rl.policy.action_, feed_dict={mb_rl.policy.state_: state})

                # print action_
                # print action_.shape #[1,2]
                # fadsf

                action_int = np.argmax(action_vec[0])

                action_one_hot = np.zeros((num_actions))
                action_one_hot[action_int] = 1
                # print 
                # print a
                # print np.argmax(action_)

                

                # print a
                # print np.argmax(action_)
                # print

                obs, reward, done, _ = env.step(action_int)
                # obs, reward, done, _ = env.step(np.random.randint(num_actions))

                obs = env.render(mode='rgb_array') 
                obs = convert_image(obs)


                state = mb_rl.sess.run(mb_rl.model.z_mean_, feed_dict={mb_rl.model.current_observation: [obs], mb_rl.model.prev_z_: state, mb_rl.model.current_a_: [action_one_hot]})



                # print reward

                # total_rewards += reward

                # reward = -10 if done else 0.1
                # one_hot_action = np.zeros((num_actions))
                # one_hot_action[action] = 1.

                # observations.append(obs)
                # actions.append(one_hot_action)
                # rewards.append(reward)

                # print obs

                # state = next_state
                

                if done: break

            

            if t ==0:
                count =1.
                running_timesteps_mean = t
            else:
                count += 1.
                running_timesteps_mean = ((count-1)*running_timesteps_mean + t) / count

            print i_episode, t, running_timesteps_mean

            # for i in range(1, len(rewards)):
            #     rewards[-i-1] = rewards[-i-1] + rewards[-i]

            # for i in range(len(rewards)):
            #     rewards[i] = [rewards[i]]
            # rewards = np.array(rewards) - np.mean(rewards)

        #     # print rewards
        #     dataset.append([observations, actions, rewards])

        # print 'Dataset size:' +str(len(dataset))
        # # print len(dataset[0][0])
        # # print len(dataset[1][0])

        print 'Done'















    if viz2 ==1:
        #this compares the model to the real env by showing the predicted obs
        #also prints the reward and predicted reward

        # so run environment with rand actions, save actions and obs
        # use actions on simulated model, save obs

        # show the obs




        def convert_image(image):
            # - crop
            # - downsample
            # - greyscale
            # - bernoulli

            # image = rgb_array[150:350,100:500,:] #.shape 200, 400, 3
            image = image[180:300,160:440,:]  #200 300 3
            # image = np.reshape(image, [-1])
            # print image.shape
            image = block_reduce(image, block_size=(5, 5, 1), func=np.max) #now its 40 60 3 = 7200
            # print image.shape
            image = Image.fromarray(image, 'RGB').convert('L')
            image = np.array(image)
            image_shape = image.shape
            image = np.reshape(image, [image_shape[0], image_shape[1], 1]) #24, 56, 1
            image = image / 255.

            image = np.reshape(image, [-1]) #1

            return image



        mb_rl = MB_RL(model_architecture=model_architecture, 
                policy_architecture=policy_architecture, 
                batch_size=1, n_particles=1, n_timesteps=1,
                model_path_to_load_variables=model_path_to_save_variables,
                model_path_to_save_variables='',
                policy_path_to_load_variables=policy_path_to_save_variables,
                policy_path_to_save_variables='',
                tb_path=tb_path)


        env = gym.make('CartPole-v0')

        num_actions = env.action_space.n
        # print 'action dims ' + str(num_actions)

        MAX_EPISODES = 1
        MAX_STEPS    = 200

        # episode_history = deque(maxlen=100)
        for i_episode in xrange(MAX_EPISODES):

            # if i_episode %10 == 0:
            #     print i_episode

            # initialize
            obs = env.reset()
            
            obs = env.render(mode='rgb_array') 
            obs = convert_image(obs)

            obs_shape = obs.shape
            obs_dims = np.prod(obs_shape)
            # print 'obs dims ' + str(obs_dim)

            # get first state, using recognitino net, given current obs and zero state and random action
            state = mb_rl.sess.run(mb_rl.model.z_mean_, feed_dict={mb_rl.model.current_observation: [obs], mb_rl.model.prev_z_: [np.zeros(z_size)], mb_rl.model.current_a_: [np.zeros(num_actions)]})


            real_obs = []
            real_rewards = []
            sim_obs = []
            sim_rewards = []

            for t in xrange(MAX_STEPS):

                action_ = np.random.randint(num_actions)
                # action_ = mb_rl.sess.run(mb_rl.policy.action_, feed_dict={mb_rl.policy.state_: state})
                a = np.zeros((num_actions))
                a[action_] = 1

                #Apply action to env
                obs, reward, done, _ = env.step(action_)

                reward = -10 if done else 0.1

                #Apply action to model
                state = mb_rl.sess.run(mb_rl.model.next_state_mean, feed_dict={mb_rl.model.prev_z_: state, mb_rl.model.current_a_: [a]})

                #Get obs of env
                obs = env.render(mode='rgb_array') 
                real_ob = convert_image(obs)


                #Get obs from model
                sim_ob, sim_reward = mb_rl.sess.run(mb_rl.model.current_emission, feed_dict={mb_rl.model.current_z_: state})

                # print reward
                # print sim_reward

                real_obs.append(real_ob)
                real_rewards.append(reward)
                sim_obs.append(np.reshape(sim_ob, [-1]))
                sim_rewards.append(sim_reward)



                if done: break

            print t
            # fsdafsd

        # for i in range(1, len(real_rewards)):
        #     real_rewards[-i-1] = real_rewards[-i-1] + real_rewards[-i]

        # for i in range(len(real_rewards)):
        #     real_rewards[i] = [real_rewards[i]]

        for i in range(len(real_rewards)):
            print real_rewards[i], sim_rewards[i]



        #show what I got
        print np.array(real_obs).shape
        print np.array(sim_obs).shape

        for t in range(len(real_obs)):

            frame = np.reshape(real_obs[t], [24, 56])

            if t == 0:
                all_times = frame
            else:
                all_times = np.concatenate([all_times, frame], axis=0)

        for t in range(len(sim_obs)):

            frame = np.reshape(sim_obs[t], [24, 56])

            if t == 0:
                all_times2 = frame
            else:
                all_times2 = np.concatenate([all_times2, frame], axis=0)

        all_frames = np.concatenate([all_times, all_times2], axis=1)


        scipy.misc.imsave(save_to +'outfile.jpg', all_frames)
        print 'saved'





        print 'Done'


















    print 'Done everything'














