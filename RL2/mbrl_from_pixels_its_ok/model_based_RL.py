

#This will combine the model and the policy.

#Two main functions
    # -train model + policy
    # -test model + policy



import numpy as np
import tensorflow as tf

from DKF_9feb2017_with_policy import DKF as model_
from policy_network import Policy as policy_

import scipy.misc
import matplotlib.pyplot as plt


class MB_RL():


    def __init__(self, model_architecture, policy_architecture, batch_size, n_particles, n_timesteps,
                        model_path_to_load_variables, model_path_to_save_variables,
                        policy_path_to_load_variables, policy_path_to_save_variables,
                        tb_path):


        self.batch_size = batch_size

        #Build Graph
            # - define all the vars
            # - start session
            # - initialize vars or load them
            # - later: save vars

        #Define model 
        print ('Defining model..')
        self.model = model_(model_architecture, batch_size=batch_size, n_particles=n_particles)

        #Define policy
        print ('Defining policy..')
        self.policy = policy_(policy_architecture, model=self.model, batch_size=batch_size, n_particles=n_particles, n_timesteps=n_timesteps)

        #Start session
        self.sess = tf.Session()

        
        #For tensorboard
        # train_writer = tf.summary.FileWriter(tb_path, self.sess.graph)
        writer = tf.summary.FileWriter(tb_path, graph=tf.get_default_graph())



        #Init the optimizer params, Im not if this resets all the other params. need to check by loading params
        self.sess.run(tf.global_variables_initializer())

        #Initialize vars or load them
        #Model
        print ('Initializing model..')
        saver = tf.train.Saver(self.model.params_dict)
        if model_path_to_load_variables == '':
            self.sess.run(tf.variables_initializer(self.model.params_list))
        else:
            saver.restore(self.sess, model_path_to_load_variables)
            print ('loaded model variables ' + model_path_to_load_variables)

        #Policy
        print( 'Initializing policy..')
        saver = tf.train.Saver(self.policy.params_dict)
        if policy_path_to_load_variables == '':
            self.sess.run(tf.variables_initializer(self.policy.params_list))
        else:
            saver.restore(self.sess, policy_path_to_load_variables)
            print ('loaded policy variables ' + policy_path_to_load_variables)



        self.model_path_to_save_variables = model_path_to_save_variables
        self.policy_path_to_save_variables = policy_path_to_save_variables

        print ('Init Complete')




    def train_both(self, get_data, steps=1000, display_step=10):

        for step in range(steps):

            batch = []
            batch_actions = []
            batch_rewards = []
            while len(batch) != self.batch_size:

                sequence, actions, rewards =get_data()
                batch.append(sequence)
                batch_actions.append(actions)
                batch_rewards.append(rewards)


            # self.train_model(batch, batch_actions)
            _ = self.sess.run(self.model.optimizer, feed_dict={self.model.x: batch, self.model.actions: batch_actions, self.model.rewards: batch_rewards})

            # self.train_policy()
            _ = self.sess.run(self.policy.optimizer)

            # # Display
            if step % display_step == 0:

                elbo, p1,p2,p3 = self.sess.run([self.model.elbo, self.model.log_p_x_final, self.model.log_p_z_final, self.model.log_q_z_final], feed_dict={self.model.x: batch, self.model.actions: batch_actions, self.model.rewards: batch_rewards})

                j_eqn = self.sess.run(self.policy.objective)

                print ("Step:", '%04d' % (step), "elbo=", "{:.5f}".format(elbo), 'px', p1, 'pz', p2, 'qz', p3, '   J', j_eqn)




        #Save parameters
        #Model
        saver = tf.train.Saver(self.model.params_dict)
        if self.model_path_to_save_variables != '':
            saver.save(self.sess, self.model_path_to_save_variables)
            print ('Saved variables to ' + self.model_path_to_save_variables)
        #Policy
        saver = tf.train.Saver(self.policy.params_dict)
        if self.policy_path_to_save_variables != '':
            saver.save(self.sess, self.policy_path_to_save_variables)
            print ('Saved variables to ' + self.policy_path_to_save_variables)

        print ('Done training')




    def train_model(self, get_data, steps=1000, display_step=10):

        # print 'aaaaa'
        best_mean_elbo=-1
        for step in range(steps):

            batch = []
            batch_actions = []
            batch_rewards = []
            while len(batch) != self.batch_size:

                sequence, actions, rewards = get_data()
                batch.append(sequence)
                batch_actions.append(actions)
                batch_rewards.append(rewards)
                # print 'bbbbb'


            # print 'cccccc'
            # self.train_model(batch, batch_actions)

            # elbo, p1,p2,p3 = self.sess.run([self.model.elbo, self.model.log_p_x_final, self.model.log_p_z_final, self.model.log_q_z_final], feed_dict={self.model.x: batch, self.model.actions: batch_actions, self.model.rewards: batch_rewards})
            # print "Step:", '%04d' % (step), "elbo=", "{:.5f}".format(elbo), 'px', p1, 'pz', p2, 'qz', p3 #'   J', j_eqn



            # grads, vars_ = self.sess.run([self.model.grads, self.model.params_dict], feed_dict={self.model.x: batch, self.model.actions: batch_actions, self.model.rewards: batch_rewards})
            # names = self.model.grad_names

            # bins=[.0001, .001, .01, .1, 1., 10.]
            # print bins

            # for i in range(len(grads)):

            #     grads__ = np.array(grads[i][0]) #100, 1344
            #     var__ = np.array(vars_[names[i]])  #100, 1344

            #     print grads__.shape
            #     print var__.shape

            #     # if names[i] == 'decoder_weights_out_mean': 

            #     print names[i]
            #     print 'grads'
            #     counts,edges = np.histogram(grads__, bins=bins)
            #     print counts
                

            #     print 'vars'
            #     counts,edges = np.histogram(var__, bins=bins)
            #     print counts
            #     # print edges
            #     print 
            #         # print np.array(grads[i]).shape 
            #         # print 'vars grads'
            #         # print np.array(vars_[names[i]]).shape
            #         # print np.array(grads[i]).shape
            #         # for j in range(len(vars_[names[i]])):
            #         #     print vars_[names[i]][j] , grads[i][j]
            #         # # print vars_[names[i]]   
            #         # # print 'grads'
            #         # # print grads[i]
            #         # print 
            # fdsf



            _ = self.sess.run(self.model.optimizer, feed_dict={self.model.x: batch, self.model.actions: batch_actions, self.model.rewards: batch_rewards})


            # elbo, p1,p2,p3 = self.sess.run([self.model.elbo, self.model.log_p_x_final, self.model.log_p_z_final, self.model.log_q_z_final], feed_dict={self.model.x: batch, self.model.actions: batch_actions, self.model.rewards: batch_rewards})
            # print "Step:", '%04d' % (step), "elbo=", "{:.5f}".format(elbo), 'px', p1, 'pz', p2, 'qz', p3 #'   J', j_eqn


            # self.train_policy()
            # _ = self.sess.run(self.policy.optimizer)
            # print 'dddddd'

            # # Display
            if step % display_step == 0:

                elbo, p1,p2,p3 = self.sess.run([self.model.elbo, self.model.log_p_x_final, self.model.log_p_z_final, self.model.log_q_z_final], feed_dict={self.model.x: batch, self.model.actions: batch_actions, self.model.rewards: batch_rewards})

                # j_eqn = self.sess.run(self.policy.objective)

                print ("Step:", '%04d' % (step), "elbo=", "{:.5f}".format(elbo), 'px', p1, 'pz', p2, 'qz', p3 )#'   J', j_eqn)


            #Check validation 
            if step % 5000 == 0:
                elbos = []
                #size of validation set
                for i in range(20):

                    batch = []
                    batch_actions = []
                    batch_rewards = []
                    while len(batch) != self.batch_size:
                        sequence, actions, rewards = get_data(valid=i)
                        batch.append(sequence)
                        batch_actions.append(actions)
                        batch_rewards.append(rewards)


                    elbo, p1,p2,p3 = self.sess.run([self.model.elbo, self.model.log_p_x_final, self.model.log_p_z_final, self.model.log_q_z_final], feed_dict={self.model.x: batch, self.model.actions: batch_actions, self.model.rewards: batch_rewards})

                    elbos.append(elbo)

                mean_elbo = np.mean(elbos)
                print ('Validation', mean_elbo, np.std(elbos))

                if mean_elbo > best_mean_elbo or best_mean_elbo==-1:
                    best_mean_elbo = mean_elbo
                    #save model
                    saver = tf.train.Saver(self.model.params_dict)
                    if self.model_path_to_save_variables != '':
                        saver.save(self.sess, self.model_path_to_save_variables)
                        print ('Saved variables to ' + self.model_path_to_save_variables)
                else:
                    print ('worse, best is', best_mean_elbo)



        elbos = []
        #size of validation set
        for i in range(20):

            batch = []
            batch_actions = []
            batch_rewards = []
            while len(batch) != self.batch_size:
                sequence, actions, rewards = get_data(valid=i)
                batch.append(sequence)
                batch_actions.append(actions)
                batch_rewards.append(rewards)

            elbo, p1,p2,p3 = self.sess.run([self.model.elbo, self.model.log_p_x_final, self.model.log_p_z_final, self.model.log_q_z_final], feed_dict={self.model.x: batch, self.model.actions: batch_actions, self.model.rewards: batch_rewards})
            elbos.append(elbo)

        mean_elbo = np.mean(elbos)
        print ('Validation', mean_elbo, np.std(elbos))

        if mean_elbo > best_mean_elbo or best_mean_elbo==-1:
            best_mean_elbo = mean_elbo
            #save model
            saver = tf.train.Saver(self.model.params_dict)
            if self.model_path_to_save_variables != '':
                saver.save(self.sess, self.model_path_to_save_variables)
                print ('Saved variables to ' + self.model_path_to_save_variables)
        else:
            print ('worse, best is', best_mean_elbo)

        #Save parameters
        # #Model
        # saver = tf.train.Saver(self.model.params_dict)
        # if self.model_path_to_save_variables != '':
        #     saver.save(self.sess, self.model_path_to_save_variables)
        #     print 'Saved variables to ' + self.model_path_to_save_variables
        # Policy
        saver = tf.train.Saver(self.policy.params_dict)
        if self.policy_path_to_save_variables != '':
            saver.save(self.sess, self.policy_path_to_save_variables)
            print ('Saved variables to ' + self.policy_path_to_save_variables)

        print ('Done training model\n')
        return


    def train_policy(self, steps=1000, display_step=10):

        #will have to call the model
        # _ = self.sess.run(self.policy.optimizer)

        for step in range(steps):


            # batch = []
            # batch_actions = []
            # batch_rewards = []
            # while len(batch) != self.batch_size:

            #     sequence, actions, rewards =get_data()
            #     batch.append(sequence)
            #     batch_actions.append(actions)
            #     batch_rewards.append(rewards)


            # self.train_model(batch, batch_actions)
            # _ = self.sess.run(self.model.optimizer, feed_dict={self.model.x: batch, self.model.actions: batch_actions})

            # self.train_policy()


            _ = self.sess.run(self.policy.optimizer)


            # # Display
            if step % display_step == 0:

                # elbo, p1,p2,p3 = self.sess.run([self.model.elbo, self.model.log_p_x_final, self.model.log_p_z_final, self.model.log_q_z_final], feed_dict={self.model.x: batch, self.model.actions: batch_actions})

                j_eqn = self.sess.run(self.policy.objective)

                print ("Step:", '%04d' % (step), ' J', j_eqn)


                # rew = self.sess.run(self.policy.get_r)

                # print rew.shape
                # print rew



                # obs_for_train = self.sess.run(self.policy.obs_for_train)

                # print obs_for_train.shape
                # # print rew

                # for t in range(len(obs_for_train)):

                #     print t
                #     print np.reshape(obs_for_train[t][0], [15,2])



                # fsa



            # #get gradients
            # grad = self.sess.run(self.policy.grad)
            # for l in range(len(grad)):
            #     # print np.array(grad).shape
            #     if step % display_step == 0:
            #         print grad[l].shape

            #     if grad[l].any() > 9999:
            #         print grad[l].shape
            #         print 'too big'
            #     if grad[l].any() < -9999:
            #         print grad[l].shape
            #         print 'too small'


            # abs_grad = np.abs(grad)
            # print abs_grad.any().shape
            # if abs_grad.any() > 9999:
            #     print 'thing'
            #     fsdf
            #     print grad
            # else:
            #     print 'all good'


        #Save parameters
        #Model
        saver = tf.train.Saver(self.model.params_dict)
        if self.model_path_to_save_variables != '':
            saver.save(self.sess, self.model_path_to_save_variables)
            print( 'Saved variables to ' + self.model_path_to_save_variables)
        #Policy
        saver = tf.train.Saver(self.policy.params_dict)
        if self.policy_path_to_save_variables != '':
            saver.save(self.sess, self.policy_path_to_save_variables)
            print ('Saved variables to ' + self.policy_path_to_save_variables)

        print( 'Done training policy\n')

        return






    # def test_both(self):

    #     return 


    def test_model(self, get_data):

        #get actions and frames
        #give model all actions and only some frames
        # so its two steps, 
            # run normally while there are given frames, take last state
            # give last state and generate next states and obs
        #append gen to real 


        batch = []
        batch_actions = []
        batch_rewards = []
        while len(batch) != self.batch_size:

            sequence, actions, rewards=get_data()
            batch.append(sequence)
            batch_actions.append(actions)
            batch_rewards.append(rewards)

        #chop up the sequence, only give it first 3 frames
        n_time_steps_given = 3
        given_sequence = []
        given_actions = []
        given_rewards = []
        hidden_sequence = []
        hidden_actions = []
        hidden_rewards = []
        for b in range(len(batch)):
            given_sequence.append(batch[b][:n_time_steps_given])
            given_actions.append(batch_actions[b][:n_time_steps_given])
            given_rewards.append(batch_rewards[b][:n_time_steps_given])
            #and the ones that are not used
            hidden_sequence.append(batch[b][n_time_steps_given:])
            hidden_actions.append(batch_actions[b][n_time_steps_given:])
            hidden_rewards.append(batch_rewards[b][n_time_steps_given:])

            
        #Get states [T,B,PZ+3]
        current_state = self.sess.run(self.model.particles_and_logprobs, feed_dict={self.model.x: given_sequence, self.model.actions: given_actions, self.model.rewards: given_rewards})
        # Unpack, get states
        current_state = current_state[n_time_steps_given-1][0][:self.model.z_size*self.model.n_particles]

        #Step 2: Predict future states and decode to frames

        # print np.array(hidden_actions).shape #[B,leftovertime, A]
        current_state = [current_state] #so it fits batch
        
        # [TL, P, B, X]
        obs = self.model.predict_future(self.sess, current_state, hidden_actions) 
        
        # [T-TL, X]
        # given_s = list(given_sequence[0])
        # [P, T-TL, X]
        real_and_gen = []
        for p in range(self.model.n_particles):
            real_and_gen.append(list(given_sequence[0]))

        # [P, T-TL, X]
        # print np.array(real_and_gen).shape
        # fdsf

        for obs_t in range(len(obs)):
            # print 'obs_t', obs_t
            for p in range(len(obs[obs_t])):
                # print 'p', p

                obs_t_p = np.reshape(obs[obs_t][p][0], [-1])

                real_and_gen[p].append(obs_t_p)


        #[T,P,X]
        real_and_gen = np.array(real_and_gen)
        # print real_and_gen.shape #[T,X]
        

        real_sequence = np.array(batch[0])
        actions = np.array(batch_actions[0])





        # gettign the mean trajectory
        obs = self.model.predict_future_only_mean(self.sess, current_state, hidden_actions) 

        real_and_gen2 = []
        for p in range(self.model.n_particles):
            real_and_gen2.append(list(given_sequence[0]))
        # [P, T-TL, X]
        # print np.array(real_and_gen).shape
        for obs_t in range(len(obs)):
            # print 'obs_t', obs_t
            for p in range(len(obs[obs_t])):
            # p=0 #since all particles should be the same
                obs_t_p = np.reshape(obs[obs_t][p][0], [-1])
                real_and_gen2[p].append(obs_t_p)

        #[T,P,X]
        real_and_gen2 = np.array(real_and_gen2)



        return real_sequence, actions, real_and_gen, real_and_gen2










    def test_policy(self, get_action_results, n_timesteps, obs_height, obs_width):


        frames = []
        position = 4

        state = np.zeros([1,self.policy.z_size], dtype='float32')

        for t in range(n_timesteps):

            #predict action
            action_ = self.sess.run(self.policy.action_, feed_dict={self.policy.state_: state})
            
            # print action_
            action = np.reshape(action_, [self.policy.action_size])


            #get result of action
            obs, position = get_action_results(prev_position=position, current_action=action, obs_height=obs_height, obs_width=obs_width)
            frames.append(obs)
            obs = np.reshape(obs, [1,self.model.input_size])

            #encode obs
            state = self.sess.run(self.model.z_mean_, feed_dict={self.model.current_observation: obs, self.model.prev_z_: state, self.model.current_a_: action_})


        return np.array(frames)



    # def get_return(self, trajectory):
    #     '''
    #     traj: [T,X]
    #     '''

    #     return_ = 0

    #     for t in range(len(trajectory)):

    #         reward = 

    #         reward = self.sess.run(self.policy.reward, feed_dict={self.policy.obs: [trajectory[t]]})

    #         #show frame and reward
    #         # print trajectory[t].shape
    #         # print reward
    #         # print np.reshape(trajectory[t],[15,2])
    #         # print

    #         # scipy.misc.imshow(np.reshape(trajectory[t], [15,2]))

    #         # fig = plt.figure()
    #         # plt.imshow(np.reshape(trajectory[t],[15,2]), vmin=0, vmax=1, cmap="gray")
    #         # plt.show()

    #         return_ += reward

    #     return return_[0]



    def viz_traj_of_policy(self, n_timesteps):

        obs = []

        state = np.zeros([1,self.policy.z_size], dtype='float32')

        for t in range(n_timesteps):

            print (t)

            #predict action
            action_ = self.sess.run(self.policy.action_, feed_dict={self.policy.state_: state})
            # action_ = [[1, 0]]

            print ('action ' + str(action_))
            print ('state ' + str(state))


            prev_z_and_current_a = np.concatenate((state, action_), axis=1) #[B,ZA]

            # [B,Z]
            prior_mean, prior_log_var = self.sess.run(self.model.next_state, feed_dict={self.model.prev_z_and_current_a_: prev_z_and_current_a})

            #sample new state
            sample = self.sess.run(self.model.sample, feed_dict={self.model.prior_mean_: prior_mean, self.model.prior_logvar_: prior_log_var})

            x_mean, r_mean = self.sess.run(self.model.current_emission, feed_dict={self.model.current_z_: sample})
            
            print (x_mean.shape, r_mean)

            


            obs.append(np.reshape(x_mean, [-1]))

            state = sample


            #get result of action
            # obs, position = get_action_results(prev_position=position, current_action=action, obs_height=obs_height, obs_width=obs_width)
            # frames.append(obs)
            # obs = np.reshape(obs, [1,self.model.input_size])

            #encode obs
            # state = self.sess.run(self.model.z_mean_, feed_dict={self.model.current_observation: obs, self.model.prev_z_: state, self.model.current_a_: action_})




        return np.array(obs)














