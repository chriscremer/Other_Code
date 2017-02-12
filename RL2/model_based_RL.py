

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


        #Build Graph
            # - define all the vars
            # - start session
            # - initialize vars or load them
            # - later: save vars

        #Define model 
        print 'Initializing model..'
        self.model = model_(model_architecture, batch_size=batch_size, n_particles=n_particles)

        #Define policy
        print 'Initializing policy..'
        self.policy = policy_(policy_architecture, model=self.model, batch_size=batch_size, n_particles=n_particles, n_timesteps=n_timesteps)

        #Start session
        self.sess = tf.Session()

        #Initialize vars or load them
        #Model
        saver = tf.train.Saver(self.model.param_dict)
        if model_path_to_load_variables == '':
            self.sess.run(tf.variables_initializer(self.model.var_list))
        else:
            saver.restore(self.sess, model_path_to_load_variables)
            print 'loaded model variables ' + model_path_to_load_variables
        #Policy
        saver = tf.train.Saver(self.policy.param_dict)
        if policy_path_to_load_variables == '':
            self.sess.run(tf.variables_initializer(self.policy.var_list))
        else:
            saver.restore(self.sess, policy_path_to_load_variables)
            print 'loaded policy variables ' + policy_path_to_load_variables
        

        self.model_path_to_save_variables = model_path_to_save_variables
        self.policy_path_to_save_variables = policy_path_to_save_variables






    def train_both(self, get_data, steps=1000, display_step=10):

        for step in range(steps):

            batch = []
            batch_actions = []
            while len(batch) != self.batch_size:

                sequence, actions=get_data()
                batch.append(sequence)
                batch_actions.append(actions)


            self.train_model()

            self.train_policy()

            # # Display
            if step % display_step == 0:
                print step

            #     cost = self.sess.run(self.elbo, feed_dict={self.x: batch, self.actions: batch_actions})
            #     cost = -cost #because I want to see the NLL

            #     p1,p2,p3 = self.sess.run([self.log_p_x_final, self.log_p_z_final, self.log_q_z_final ], feed_dict={self.x: batch, self.actions: batch_actions})


            #     print "Step:", '%04d' % (step+1), "cost=", "{:.5f}".format(cost), p1, p2, p3


        #Save parameters
        #Model
        saver = tf.train.Saver(self.model.param_dict)
        if self.model_path_to_save_variables != '':
            saver.save(self.sess, self.model_path_to_save_variables)
            print 'Saved variables to ' + self.model_path_to_save_variables
        #Policy
        saver = tf.train.Saver(self.policy.param_dict)
        if self.policy_path_to_save_variables != '':
            saver.save(self.sess, self.policy_path_to_save_variables)
            print 'Saved variables to ' + self.policy_path_to_save_variables

        print 'Done training'




    def train_model(self):

        _ = self.sess.run(self.model.optimizer, feed_dict={self.x: batch, self.actions: batch_actions})

        return


    def train_policy(self):

        #will have to call the model

        _ = self.sess.run(self.policy.optimizer, feed_dict={self.x: batch, self.actions: batch_actions})

        return






    def test_both(self):

        return 


    def test_model(self):

        return


    def test_policy(self):


        return














