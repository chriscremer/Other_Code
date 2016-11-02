



import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import imageio



#The game is to move to the top left of the grid
class game():


    def __init__(self, f_height=30, f_width=30, ball_size=5):

        self.f_height = f_height
        self.f_width = f_width
        self.ball_size = ball_size

        values= np.reshape(np.array(range(1,31)[::-1]), [30,1])
        self.values_grid = np.dot(values, values.T)
        self.values_grid = self.values_grid + (np.identity(30)*100)

        # [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]]

        self.pos = [f_height/2, f_width/2]

    def get_pos_value(self):

        return self.values_grid[self.pos[0], self.pos[1]]

    def get_current_frame(self):

        new_frame = np.zeros([self.f_height,self.f_width])
        new_frame[self.pos[0]:self.pos[0]+self.ball_size, self.pos[1]:self.pos[1]+self.ball_size] = 1.
        return new_frame

    def reset_frame(self):
        self.pos = [self.f_height/2, self.f_width/2]

    def move_ball(self, action):

        #up 0
        #down 1
        #right 2
        #left 3

        if action == 0:
            if self.pos[0] - 1 >= 0:
                self.pos[0] = self.pos[0] - 1
        elif action == 1:
            if self.pos[0] + 1 < self.f_height-self.ball_size:
                self.pos[0] = self.pos[0] + 1
        elif action == 2:
            if self.pos[1] + 1 < self.f_width-self.ball_size:
                self.pos[1] = self.pos[1] + 1
        elif action == 3:
            if self.pos[1] - 1 >= 0:
                self.pos[1] = self.pos[1] - 1

        new_frame = np.zeros([self.f_height,self.f_width])
        new_frame[self.pos[0]:self.pos[0]+self.ball_size, self.pos[1]:self.pos[1]+self.ball_size] = 1.

        return new_frame


class predict_next_move_net():

    def __init__(self, batch_size=1):

        self.graph=tf.Graph()

        self.batch_size = batch_size

        self.input_size = 900
        self.fc1_output_len = 100
        self.fc2_output_len = 20
        self.n_classes = 4

        self.lr = .0001
        self.mom =  .01
        self.lmbd = .0001

        # self.path_to_load_variables = home+'/Documents/tmp/rnn_sequence4.ckpt'
        self.path_to_load_variables = ''
        self.path_to_save_variables = home+'/Documents/tmp/rnn_sequence1.ckpt'
        
        with self.graph.as_default():

            #PLACEHOLDERS
            self.input = tf.placeholder("float", shape=[self.batch_size, 900])
            # self.target = tf.placeholder("float", shape=[self.batch_size, 4])
            self.target = tf.placeholder("int32", shape=[self.batch_size])

            self.reward_dif = tf.placeholder("float", shape=[self.batch_size])
       
            self.fc1_weights = tf.Variable(tf.truncated_normal([self.input_size, self.fc1_output_len],stddev=0.1))
            self.fc1_biases = tf.Variable(tf.truncated_normal([self.fc1_output_len], stddev=0.1))
       
            self.fc2_weights = tf.Variable(tf.truncated_normal([self.fc1_output_len, self.fc2_output_len],stddev=0.1))
            self.fc2_biases = tf.Variable(tf.truncated_normal([self.fc2_output_len], stddev=0.1))

            self.fc3_weights = tf.Variable(tf.truncated_normal([self.fc2_output_len, self.n_classes],stddev=0.1))
            self.fc3_biases = tf.Variable(tf.truncated_normal([self.n_classes], stddev=0.1))

            #MODEL
            self.logits = self.feedforward(self.input)

            # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.target))
            self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.target))

            self.weight_decay = self.lmbd * (tf.nn.l2_loss(self.fc1_weights) +
                                            tf.nn.l2_loss(self.fc1_biases) +
                                            tf.nn.l2_loss(self.fc2_weights) +
                                            tf.nn.l2_loss(self.fc2_biases) +
                                            tf.nn.l2_loss(self.fc3_weights) +
                                            tf.nn.l2_loss(self.fc3_biases))

            self.cost = (self.cross_entropy  *self.reward_dif)  + self.weight_decay


            self.actual_output = tf.sigmoid(self.logits)

            #TRAIN
            self.opt = tf.train.MomentumOptimizer(self.lr,self.mom)
            self.grads_and_vars = self.opt.compute_gradients(self.cost)
            self.train_opt = self.opt.apply_gradients(self.grads_and_vars)





    def feedforward(self, sample):

        layer_1 = tf.nn.relu(tf.add(tf.matmul(sample, self.fc1_weights), self.fc1_biases))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.fc2_weights), self.fc2_biases))
        output = tf.add(tf.matmul(layer_2, self.fc3_weights), self.fc3_biases)

        return output



    def fit(self, session, input_, target_, reward_dif_):

        if session == None:
            with self.graph.as_default():
                saver = tf.train.Saver()
                # with tf.Session() as sess:
                sess = tf.Session(graph=self.graph)
                sess.run(tf.initialize_all_variables())
        else:
            sess = session


        for step in range(len(input_)):
            # if len(input_) > 100:
            #     index_ = random.randint(0,len(input_)-1)

            #     feed_dict={self.input: [input_[index_]], self.target: [target_[index_]], self.reward_dif: [reward_dif_[step]]}
            #     _ = sess.run(self.train_opt, feed_dict=feed_dict)

            # else:
            #     feed_dict={self.input: [input_[step]], self.target: [target_[step]], self.reward_dif: [reward_dif_[step]]}
            #     _ = sess.run(self.train_opt, feed_dict=feed_dict)

            index_ = random.randint(0,len(input_)-1)
            feed_dict={self.input: [input_[index_]], self.target: [target_[index_]], self.reward_dif: [reward_dif_[index_]]}


            # if step %10 == 0: 


            # print target_[index_], reward_dif_[index_]
            # print 'before', sess.run(tf.nn.softmax(self.logits), feed_dict=feed_dict)  
            # print 'before', sess.run([self.cross_entropy, self.weight_decay], feed_dict=feed_dict) 
            _ = sess.run(self.train_opt, feed_dict=feed_dict)  
            # print 'after', sess.run([self.cross_entropy, self.weight_decay], feed_dict=feed_dict) 
            # print 'after', sess.run(tf.nn.softmax(self.logits), feed_dict=feed_dict)
            # print


            # else:
            #     _ = sess.run(self.train_opt, feed_dict=feed_dict) 


            # _ = sess.run(self.train_opt, feed_dict=feed_dict)      



            if step > 0:
                break
                        
        return sess




    def predict(self, session, samples, path_to_load_variables):

        if session == None:

            with self.graph.as_default():

                saver = tf.train.Saver()

                sess = tf.Session(graph=self.graph)

                # with tf.Session() as sess:

                if path_to_load_variables == '':
                    sess.run(tf.initialize_all_variables())
                else:
                    saver.restore(sess, path_to_load_variables)
                    print 'loaded variables ' + path_to_load_variables

        else:
            sess = session



        feed_dict={self.input: samples}
        
        ff = sess.run(tf.nn.softmax(self.logits), feed_dict=feed_dict)

        ff= ff[0]
        
        # print ff
        # return ff, sess
        # ff = ff - np.max(ff)


        # return np.exp(ff)/np.sum(np.exp(ff)), sess

        return ff, sess














if __name__ == "__main__":

    batch_size = 1
    game = game()
    model = predict_next_move_net(batch_size=1)
    sess1=None
    path_to_load_variables=''

    frame_dataset = []
    action_dataset = []
    reward_dif_dataset = []


    for i in range(2000):

        #Make the dataset by predicting a move given the current frame
        frame = game.get_current_frame()
        frame = np.reshape(frame, [900])
        value = game.get_pos_value()

        #exploration vs exploitation
        if i%5 ==0 or i%4==0 or i%6 ==0 or i%7==0 or i%8 ==0 or i%9==0 or i%3==0 or i%2==0:
        # if 0:
            action = np.random.choice(4, 1, p=[.25, .25, .25, .25])[0]
            # print i, 'sampled'
            # action_vec = np.zeros([4])
            # action_vec[action] = 1
        else:
            prediction, sess1 = model.predict(sess1, [frame], path_to_load_variables)
            # print i, prediction
            action = np.random.choice(4, 1, p=prediction)[0]
            # action = np.argmax(prediction)
            # action_vec = np.zeros([4])
            # action_vec[action] = 1


        # print action
        game.move_ball(action)
        new_value = game.get_pos_value()
        value_dif = new_value #- value

        frame_dataset.append(frame)
        action_dataset.append(action)
        reward_dif_dataset.append(value_dif)

        #Train on the current dataset
        # if len(frame_dataset) > 20:

            # print frame_dataset
            # # print
            # print action_dataset
            # print
            # print reward_dif_dataset
            # fsafs
            # print 'fitting'
        model.fit(sess1, frame_dataset, action_dataset, reward_dif_dataset)

        frame_dataset = []
        action_dataset = []
        reward_dif_dataset = []

        #Reset every 10
        if i%10 == 0:
            game.reset_frame()

        if i%100 == 0:
            print i


    #LOOK AT RESULTS



    # w = sess1.run(model.fc1_weights)
    # print w.shape
    # print np.sum(np.abs(w))
    # fsdafaf

    frames = []
    game.reset_frame()
    print 'test'
    for i in range(20):
        frame = game.get_current_frame()
        frame = np.reshape(frame, [900])
        prediction, sess1 = model.predict(sess1, [frame],  path_to_load_variables)
        print prediction
        action = np.argmax(prediction)
        print action
        game.move_ball(action)
        frame = game.get_current_frame()
        frames.append(frame)

    together = np.array(frames)
    for i in range(len(together)): 
        together[i] = together[i] * (255. / np.max(together[i]))
        together[i] = together[i].astype('uint8')
    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/frames_gif.gif", together, 'GIF', **kargs)
    print 'saved frame'

    fsdfas



    # together = np.array([frame])
    # for i in range(len(together)): 
    #     together[i] = together[i] * (255. / np.max(together[i]))
    #     together[i] = together[i].astype('uint8')
    # kargs = { 'duration': .6 }
    # imageio.mimsave(home+"/Downloads/frame_gif.gif", together, 'GIF', **kargs)
    # print 'saved frame'

    # print game.return_pos_value()

    # game.move_ball(3)
    # game.move_ball(3)
    # game.move_ball(3)
    # game.move_ball(3)
    # game.move_ball(3)
    # game.move_ball(3)

    # frame = game.return_current_frame()
    # together = np.array([frame])
    # for i in range(len(together)): 
    #     together[i] = together[i] * (255. / np.max(together[i]))
    #     together[i] = together[i].astype('uint8')
    # kargs = { 'duration': .6 }
    # imageio.mimsave(home+"/Downloads/frame2_gif.gif", together, 'GIF', **kargs)
    # print 'saved frame'
    # print game.return_pos_value()

    # print 'DONE'
    # dfasfasdsad



    
    # print prediction[0][0]


    input_ = []
    target_ = []
    for i in range(batch_size):
        input_.append(frame)
        target_.append([1,0,0,0])

    model.fit(input_, target_)


    fasdfas






######################################################################
#TRAIN

# model = Predict_Next_Frame(batch_size=5)
# start_time = time.time()
# model.fit()
# print time.time() - start_time
# print 'DONE'
# fsdfa



######################################################################
#DEBUG


# model = RNN(batch_size=1)
# sess = None
# seq = make_ball_gif()
# for i in range(len(seq)):
#   seq[i] = seq[i] / np.max(seq[i])

# prediction, sess = model.test_run(samples=[seq], session=sess, path_to_load_variables=home+'/Documents/tmp/rnn_sequence.ckpt')
# print np.array(prediction).shape
# prediction = np.reshape(prediction, [8,10])
# print prediction

# fasfd


######################################################################
#TEST


# # #For gettign test error and visualizing errors
# import cv2
# # path_to_visualize_pics_dir = home+'/Documents/Viewnyx_data/cleaned_vids/valid/'
# path_to_visualize_pics_dir = home+'/Documents/Viewnyx_data/cleaned_vids2/test/'

# # path_to_visualize_pics_dir = home+'/storage/viewnyx/cleaned_vids/valid/'


# reader = vid_reader.Video_Reader2(path_to_visualize_pics_dir)
# model = v2_RNN(batch_size=1)
# correct = 0
# sess = None
# for i in range(100):
#   print i
#   samp, label = reader.get_next_vid_and_label()
#   # print samp.shape

#   prediction, sess = model.predict(samples=samp, session=sess)
#   # print label
#   # print prediction[0][0], label

#   #COUNT
#   if np.argmax(prediction[0][0]) == label:
#       print prediction[0][0], label, 'yes!'
#       correct+=1
#   else:
#       print prediction[0][0], label, 'no.'
#       #VIEW
#       for frame in samp:
#           cv2.imshow('Video', frame)
#           cv2.waitKey(0)
#           cv2.destroyAllWindows()





model = Predict_Next_Frame(batch_size=1)
sess = None
seq = make_ball_gif()
seq2 = list(seq)
#scale to 1
for i in range(len(seq2)):
    seq2[i] = seq2[i] / np.max(seq2[i])

batch=[]
#first two frames
batch.append(seq2[0:2:1])

# labels.append(seq[2])


prediction, sess = model.predict(samples=batch, session=sess, path_to_load_variables=home+'/Documents/tmp/rnn_sequence4.ckpt')
prediction = np.array(prediction)

n_frames=3
f_height=14
f_width=14
ball_size=2

prediction = np.reshape(prediction, [f_height,f_width,1])


# prediction = np.reshape(prediction, [n_frames,f_height,f_width,1])#.astype('uint8')
# prediction = np.reshape(prediction, [f_height,f_width,1])#.astype('uint8')

# print np.reshape(prediction[2], [f_height,f_width])
# print 
# print np.reshape(prediction[3], [f_height,f_width])

# #for multiple frmaes
# for i in range(len(prediction)):

#   #logsumexp
#   prediction[i] = prediction[i] - np.max(prediction[i])
#   #sigmoid
#   prediction[i]= np.exp(prediction[i])/np.sum(np.exp(prediction[i]))
#   #scale for uint8
#   prediction[i]= prediction[i] * (255. / np.max(prediction[i]))
#   if np.max(seq[i]) > 0:
#       seq[i] = seq[i] * (255. / np.max(seq[i]))


#logsumexp
prediction = prediction - np.max(prediction)
#sigmoid
prediction= np.exp(prediction)/np.sum(np.exp(prediction))
#scale for uint8
prediction= prediction * (255. / np.max(prediction))
prediction = prediction.astype('uint8')

# if np.max(seq[i]) > 0:
#   seq[i] = seq[i] * (255. / np.max(seq[i]))




kargs = { 'duration': .5 }
imageio.mimsave(home+"/Downloads/real_gif.gif", seq, 'GIF', **kargs)

seq[2] = prediction
imageio.mimsave(home+"/Downloads/pred_gif.gif", seq, 'GIF', **kargs)



print 'DONE'













