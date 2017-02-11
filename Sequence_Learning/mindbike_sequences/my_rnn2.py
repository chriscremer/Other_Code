

#including batches in this one


import numpy as np
import tensorflow as tf
# import random
# import math
# from os.path import expanduser
# home = expanduser("~")
# import imageio

# from ball_sequence import make_ball_gif


class RNN():

    def __init__(self, input_size, action_size, state_size, output_size, max_sequence_length, batch_size=5):
        
        tf.reset_default_graph()

        self.transfer_fct = tf.nn.softplus #tf.nn.relu
        self.learning_rate = 0.001
        self.reg_param = .00001

        self.input_size = input_size
        self.action_size = action_size
        self.state_size = state_size
        self.output_size = input_size
        self.batch_size = batch_size
        self.pred_hidden_layer = 100
        self.max_sequence_length = max_sequence_length
        
        # Graph Input 
        self.frames = tf.placeholder(tf.float32, [None, None, self.input_size])  #[B,t,X]
        self.actions_with_frames = tf.placeholder(tf.float32, [None, None, self.action_size]) # [B,t,A]
        self.actions_without_frames = tf.placeholder(tf.float32, [None, None, self.action_size]) # [B,T-t,A]
        self.n_frames = tf.placeholder(tf.int32, []) #t
        self.all_frames = tf.placeholder(tf.float32, [None, None, self.input_size]) #[B,T,X]
        
        #Variables
        self._initialize_weights()

        #Predictions
        self.predictions = self.feedforward(self.frames, self.actions_with_frames, self.actions_without_frames)

        #Objective
        self.reconst_loss = self.calc_error(sequence=self.all_frames, predictions=self.predictions)
        self.cost = self.reconst_loss + (self.reg_param * self.l2_regularization())

        #Optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(self.cost)


    def _initialize_weights(self):

        def xavier_init(fan_in, fan_out, constant=1): 
            """ Xavier initialization of network weights"""
            # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
            high = constant*np.sqrt(6.0/(fan_in + fan_out))
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

        # self.fc1_weights = tf.Variable(tf.truncated_normal([self.flatten_len1, self.fc1_output_len],stddev=0.1))
        # self.fc1_biases = tf.Variable(tf.truncated_normal([self.fc1_output_len], stddev=0.1))

        # self.forget_gate_variables = tf.Variable(xavier_init(self.action_size+self.state_size+self.input_size, self.state_size))
        # self.forget_gate_variables_b = tf.Variable(tf.zeros([self.state_size]))
        # self.forget_gate_variables2 = tf.Variable(tf.truncated_normal([self.state_size, self.state_size],stddev=0.1))
        # self.forget_gate_variables2_b = tf.Variable(tf.truncated_normal([self.state_size], stddev=0.1))

        self.update_gate_variables = tf.Variable(xavier_init(self.action_size+self.state_size+self.input_size, self.pred_hidden_layer))
        self.update_gate_variables_b = tf.Variable(tf.zeros([self.pred_hidden_layer]))
        self.update_gate_variables2 = tf.Variable(xavier_init(self.pred_hidden_layer, self.state_size))
        self.update_gate_variables2_b = tf.Variable(tf.zeros([self.state_size]))

        self.pred_update_vars = tf.Variable(xavier_init(self.action_size+self.state_size+self.input_size, self.pred_hidden_layer))
        self.pred_update_vars_b = tf.Variable(tf.zeros([self.pred_hidden_layer]))
        self.pred_update_vars2 = tf.Variable(xavier_init(self.pred_hidden_layer, self.output_size))
        self.pred_update_vars2_b = tf.Variable(tf.zeros([self.output_size]))


    def l2_regularization(self):

        vars_ = [self.update_gate_variables2, self.update_gate_variables2_b,
                    self.update_gate_variables, self.update_gate_variables_b,
                    self.pred_update_vars, self.pred_update_vars_b,
                    self.pred_update_vars2, self.pred_update_vars2_b]

        sum_ = 0
        for var in vars_:
                sum_ += tf.reduce_sum(tf.square(var))
        return sum_


    def update_state(self, action, prev_frame, prev_state):
        '''
        action [A]
        prev_frame [X]
        prev_state [Z]
        '''

        #Concat all inputs
        inputs = tf.concat(1, [tf.concat(1, [action, prev_frame]), prev_state])
        # inputs = tf.reshape(inputs, [1,-1])

        # #Forget gate
        # forget = tf.matmul(inputs, self.forget_gate_variables) + self.forget_gate_variables_b
        # # forget = tf.nn.relu(forget)
        # # forget = tf.matmul(forget, self.forget_gate_variables2) + self.forget_gate_variables2_b
        # forget = tf.sigmoid(forget)
        # state = prev_state * forget

        #Update gate
        update = tf.matmul(inputs, self.update_gate_variables) + self.update_gate_variables_b
        update = self.transfer_fct(update)
        update = tf.matmul(update, self.update_gate_variables2) + self.update_gate_variables2_b
        # update = self.transfer_fct(update)
        # remember = 1. - forget
        # state = prev_state + (remember * update)
        # state = prev_state + update

        state = update

        return state

    def predict_frame(self, prev_frame, state, current_action):
        '''
        #TODO
        '''

        #Concat all inputs
        inputs = tf.concat(1, [tf.concat(1, [state, prev_frame]), current_action])
        # inputs = tf.reshape(inputs, [1,-1])


        frame = tf.matmul(inputs, self.pred_update_vars) + self.pred_update_vars_b
        frame = self.transfer_fct(frame)
        frame = tf.matmul(frame, self.pred_update_vars2) + self.pred_update_vars2_b

        # frame = tf.reshape(frame, [-1])

        return frame



    def fn(self, previous_output, current_input):
        '''
        current_input is current frame and action
        previous_output is frame_prediction,prev_frame,state

        Use prev_frame to instead of frame_prediction to make next prediction
        frame_prediction is there to calc the error later
        #TODO shapes
        '''

        #Unpack all arguments (init_state, init_prev_frame, init_frame_pred)
        prev_state = tf.slice(previous_output, [0,0], [self.batch_size, self.state_size])
        prev_frame = tf.slice(previous_output, [0, self.state_size], [self.batch_size, self.input_size])
        # prev_frame_pred = tf.slice(previous_output, [0, self.state_size+self.input_size], [self.batch_size, self.input_size])

        #Unpack input (frame, action)
        frame = tf.slice(current_input, [0, 0], [self.batch_size, self.input_size])
        action = tf.slice(current_input, [0, self.input_size], [self.batch_size, self.action_size])

        #Update state and predict next frame
        state = self.update_state(action, prev_frame, prev_state)
        frame_t_pred = self.predict_frame(prev_frame, state, action)

        #Pack results
        pack_output = tf.concat(1, [tf.concat(1, [state, frame]), frame_t_pred])

        return pack_output


    def fn2(self, previous_output, current_input):
        '''
        current_input is action
        previous_output is frame_prediction,state
        #TODOshapes
        '''

        #Unpack all arguments
        prev_state = tf.slice(previous_output, [0,0], [self.batch_size, self.state_size])
        prev_frame = tf.slice(previous_output, [0, self.state_size], [self.batch_size, self.input_size])

        action = tf.slice(current_input, [0, 0], [self.batch_size, self.action_size])

        #Update state and predict next frame
        state = self.update_state(action, prev_frame, prev_state)
        frame_t_pred = self.predict_frame(prev_frame, state, action)

        #Pack results
        pack_output = tf.concat(1, [state, frame_t_pred])

        return pack_output




    def feedforward(self, frames, actions_with_frames, actions_without_frames):
        '''
        sequence will be t length (betw 1 and T)
        actions will be T length

        output is T-t, so the ones it had to predict

        # frames: [B, t, X]
        # actions_with_frames: [B, t, A]
        # actions_without_frames: [B, T-t, A]

        ''' 
        #[t,B,X]
        frames = tf.transpose(frames, [1,0,2])
        actions_with_frames = tf.transpose(actions_with_frames, [1,0,2])
        actions_without_frames = tf.transpose(actions_without_frames, [1,0,2])



        #Concat frames with actions [t, B, X+A]
        frames_actions = tf.concat(2, [frames, actions_with_frames])

        init_state = tf.zeros([self.batch_size, self.state_size])
        init_prev_frame = tf.zeros([self.batch_size, self.input_size])
        init_frame_pred = tf.zeros([self.batch_size, self.input_size])
        initializer = tf.concat(1, [tf.concat(1, [init_state, init_prev_frame]), init_frame_pred])
        #Go over given sequence, predicting next frame. [t, Z+X+X]
        outputs = tf.scan(self.fn, frames_actions, initializer=initializer)




        #Get last state, and last frame
        last_output = tf.slice(outputs, [self.n_frames-1,0,0], [1,self.batch_size, self.state_size+self.input_size+self.input_size])
        last_output = tf.reshape(last_output, [self.batch_size, -1])
 
        prev_state = tf.slice(last_output, [0,0], [self.batch_size, self.state_size])
        prev_frame = tf.slice(last_output, [0, self.state_size], [self.batch_size, self.input_size])
        initializer = tf.concat(1, [prev_state, prev_frame])
        # Make future predictions  [T-t, Z+X]
        outputs2 = tf.scan(self.fn2, actions_without_frames, initializer=initializer)




        # Get only the predictions
        outputs = tf.slice(outputs, [0,0,self.state_size+self.input_size], [self.n_frames, self.batch_size, self.input_size])
        outputs2 = tf.slice(outputs2, [0,0,self.state_size], [self.max_sequence_length-self.n_frames, self.batch_size, self.input_size])

        # Concate outputs, should be length T or maybe T-1
        output = tf.concat(0, [outputs, outputs2])
        return output



    def calc_error(self, sequence, predictions):
        '''
        sequence: []
        predictions: []
        '''
        sequence = tf.transpose(sequence, [1,0,2])


        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, targets=sequence)
        loss = tf.reduce_mean(loss)

        return loss




                

    def train(self, frames, actions, steps=1000, display_step=10, path_to_load_variables='', path_to_save_variables=''):


        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            self.sess.run(tf.global_variables_initializer())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables


        n_sequences = len(frames)
        n_timesteps = len(frames[0])

        # Training cycle
        for step in range(steps):

            #Randomly select a timestep to predict after
            # timestep = np.random.randint(low=1, high=n_timesteps-1)
            timestep = 18


            batch_frames = []
            batch_actions_with_frames = []
            batch_actions_without_frames = []
            batch_all_frames = []

            while len(batch_frames) != self.batch_size:

                #Randomly select a sequence
                index = np.random.randint(low=0, high=n_sequences)

                frames_ = frames[index][:timestep]
                actions_with_frames = actions[index][:timestep]
                actions_without_frames = actions[index][timestep:]
                all_frames = frames[index]

                batch_frames.append(frames_)
                batch_actions_with_frames.append(actions_with_frames)
                batch_actions_without_frames.append(actions_without_frames)
                batch_all_frames.append(all_frames)

            _ = self.sess.run(self.optimizer, feed_dict={self.frames: batch_frames, 
                                                    self.actions_with_frames: batch_actions_with_frames, 
                                                    self.actions_without_frames: batch_actions_without_frames,
                                                    self.n_frames: timestep,
                                                    self.all_frames: batch_all_frames})

            # Display
            if step % display_step == 0:

                reconst_loss = self.sess.run(self.reconst_loss, feed_dict={self.frames: batch_frames, 
                                                    self.actions_with_frames: batch_actions_with_frames, 
                                                    self.actions_without_frames: batch_actions_without_frames,
                                                    self.n_frames: timestep,
                                                    self.all_frames: batch_all_frames})

                print "Step:", '%04d' % (step), "reconst_loss=", str(reconst_loss)

        if path_to_save_variables != '':
            saver.save(self.sess, path_to_save_variables)
            print 'Saved variables to ' + path_to_save_variables

        print 'Done training'



    def test(self, frames, actions, path_to_load_variables=''):

        saver = tf.train.Saver()
        self.sess = tf.Session()

        if path_to_load_variables == '':
            self.sess.run(tf.global_variables_initializer())
        else:
            #Load variables
            saver.restore(self.sess, path_to_load_variables)
            print 'loaded variables ' + path_to_load_variables


        n_sequences = len(frames)
        n_timesteps = len(frames[0])

        #Randomly select a timestep to predict after
        # timestep = np.random.randint(low=1, high=n_timesteps-1)
        timestep = n_timesteps /2


        batch_frames = []
        batch_actions_with_frames = []
        batch_actions_without_frames = []
        batch_all_frames = []

        while len(batch_frames) != self.batch_size:

            #Randomly select a sequence
            index = np.random.randint(low=0, high=n_sequences)

            frames_ = frames[index][:timestep]
            actions_with_frames = actions[index][:timestep]
            actions_without_frames = actions[index][timestep:]
            all_frames = frames[index]

            batch_frames.append(frames_)
            batch_actions_with_frames.append(actions_with_frames)
            batch_actions_without_frames.append(actions_without_frames)
            batch_all_frames.append(all_frames)


        # Save a sequence
        seq = self.sess.run(tf.sigmoid(self.predictions), feed_dict={self.frames: batch_frames, 
                                            self.actions_with_frames: batch_actions_with_frames, 
                                            self.actions_without_frames: batch_actions_without_frames,
                                            self.n_frames: timestep,
                                            self.all_frames: batch_all_frames})
        keep_seq = batch_all_frames

        print 'saw ' + str(timestep) + ' timesteps'

        return seq, keep_seq
















