

#This will combine the model and the policy.

#Two main functions
    # -train model + policy
    # -test model + policy



import numpy as np
import tensorflow as tf

from DKF_9feb2017_with_policy import DKF as model_
from policy_network import Policy as policy_



class MB_RL():


    def __init__(self, model_architecture, policy_architecture, batch_size, n_particles, n_timesteps,
                        model_path_to_load_variables, model_path_to_save_variables,
                        policy_path_to_load_variables, policy_path_to_save_variables):


        self.batch_size = batch_size

        #Build Graph
            # - define all the vars
            # - start session
            # - initialize vars or load them
            # - later: save vars

        #Define model 
        print 'Defining model..'
        self.model = model_(model_architecture, batch_size=batch_size, n_particles=n_particles)

        #Define policy
        print 'Defining policy..'
        self.policy = policy_(policy_architecture, model=self.model, batch_size=batch_size, n_particles=n_particles, n_timesteps=n_timesteps)

        #Start session
        self.sess = tf.Session()

        

        #Init the optimizer params, Im not if this resets all the other params. need to check by loading params
        self.sess.run(tf.global_variables_initializer())

        #Initialize vars or load them
        #Model
        print 'Initializing model..'
        saver = tf.train.Saver(self.model.params_dict)
        if model_path_to_load_variables == '':
            self.sess.run(tf.variables_initializer(self.model.params_list))
        else:
            saver.restore(self.sess, model_path_to_load_variables)
            print 'loaded model variables ' + model_path_to_load_variables

        #Policy
        print 'Initializing policy..'
        saver = tf.train.Saver(self.policy.params_dict)
        if policy_path_to_load_variables == '':
            self.sess.run(tf.variables_initializer(self.policy.params_list))
        else:
            saver.restore(self.sess, policy_path_to_load_variables)
            print 'loaded policy variables ' + policy_path_to_load_variables



        self.model_path_to_save_variables = model_path_to_save_variables
        self.policy_path_to_save_variables = policy_path_to_save_variables

        print 'Init Complete'




    def train_both(self, get_data, steps=1000, display_step=10):

        for step in range(steps):

            batch = []
            batch_actions = []
            while len(batch) != self.batch_size:

                sequence, actions=get_data()
                batch.append(sequence)
                batch_actions.append(actions)


            self.train_model(batch, batch_actions)

            self.train_policy()

            # # Display
            if step % display_step == 0:

                elbo, p1,p2,p3 = self.sess.run([self.model.elbo, self.model.log_p_x_final, self.model.log_p_z_final, self.model.log_q_z_final], feed_dict={self.model.x: batch, self.model.actions: batch_actions})

                j_eqn = self.sess.run(self.policy.objective)

                print "Step:", '%04d' % (step), "elbo=", "{:.5f}".format(elbo), 'px', p1, 'pz', p2, 'qz', p3, '   J', j_eqn


        #Save parameters
        #Model
        saver = tf.train.Saver(self.model.params_dict)
        if self.model_path_to_save_variables != '':
            saver.save(self.sess, self.model_path_to_save_variables)
            print 'Saved variables to ' + self.model_path_to_save_variables
        #Policy
        saver = tf.train.Saver(self.policy.params_dict)
        if self.policy_path_to_save_variables != '':
            saver.save(self.sess, self.policy_path_to_save_variables)
            print 'Saved variables to ' + self.policy_path_to_save_variables

        print 'Done training'




    def train_model(self, batch, batch_actions):

        _ = self.sess.run(self.model.optimizer, feed_dict={self.model.x: batch, self.model.actions: batch_actions})

        return


    def train_policy(self):

        #will have to call the model

        _ = self.sess.run(self.policy.optimizer)

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
        while len(batch) != self.batch_size:

            sequence, actions=get_data()
            batch.append(sequence)
            batch_actions.append(actions)

        #chop up the sequence, only give it first 3 frames
        n_time_steps_given = 3
        given_sequence = []
        given_actions = []
        hidden_sequence = []
        hidden_actions = []
        for b in range(len(batch)):
            given_sequence.append(batch[b][:n_time_steps_given])
            given_actions.append(batch_actions[b][:n_time_steps_given])
            #and the ones that are not used
            hidden_sequence.append(batch[b][n_time_steps_given:])
            hidden_actions.append(batch_actions[b][n_time_steps_given:])
            
        #Get states [T,B,PZ+3]
        current_state = self.sess.run(self.model.particles_and_logprobs, feed_dict={self.model.x: given_sequence, self.model.actions: given_actions})
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



        return real_sequence, actions, real_and_gen










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














