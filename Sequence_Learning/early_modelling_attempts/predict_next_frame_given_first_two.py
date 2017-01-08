
#this works
# addign conv layer doenst work


import numpy as np
from os.path import expanduser
home = expanduser("~")
import imageio
import random

import tensorflow as tf
import time


######################################################################
#DATA
######################################################################
def make_ball_gif(n_frames=3, f_height=14, f_width=14, ball_size=2):
    
    row = random.randint(0,f_height-ball_size-1)
    # speed = random.randint(1,9)
    speed = random.randint(1,3)
    # speed= 1
    
    gif = []
    for i in range(n_frames):

        hot = np.zeros([f_height,f_width])
        if i*speed+ball_size >= f_width:
        	hot[row:row+ball_size:1,f_width-ball_size:f_width+ball_size:1] = 255.
        else:
        	hot[row:row+ball_size:1,i*speed:i*speed+ball_size:1] = 255.
        gif.append(hot.astype('uint8'))

    gif = np.array(gif)
    gif = np.reshape(gif, [n_frames,f_height,f_width,1])
    return gif

# gif = make_ball_gif()
# gif = np.array(gif)
# print gif.shape
# kargs = { 'duration': .5 }
# imageio.mimsave(home+"/Downloads/mygif2.gif", gif, 'GIF', **kargs)
# print 'saved'



######################################################################
#MODEL
######################################################################


class Predict_Next_Frame():

	def __init__(self, batch_size=1):

		self.graph=tf.Graph()

		self.batch_size = batch_size
		self.n_time_steps = 2
		self.image_height = 14
		self.image_width = 14
		self.n_channels = 1

		
		
		self.filter_height = 2
		self.filter_width = 2
		self.filter_out_channels1 = 4

		self.flatten_len1 = self.image_height * self.image_width
		self.fc1_output_len = 100
		self.pred_hidden_layer = 100
		self.n_classes = self.image_height * self.image_width
		self.state_size = 100

		self.lr = .01
		self.mom =  .01
		self.lmbd = .0000000001

		# self.path_to_train_dir = home+'/storage/viewnyx/cleaned_vids2/train/'
		# self.path_to_valid_dir = home+'/storage/viewnyx/cleaned_vids2/valid/'
		# # self.path_to_load_variables = home+'/storage/viewnyx/model_variables/v2_rnn_vars_8frames_3.ckpt' #rao
		# self.path_to_load_variables = home+'/Documents/Viewnyx_data/model_variables/from_rao/v2_rnn_vars_pro_3.ckpt' #mac


		# self.path_to_load_variables = home+'/Documents/tmp/rnn_sequence4.ckpt'
		self.path_to_load_variables = ''
		self.path_to_save_variables = home+'/Documents/tmp/rnn_sequence3.ckpt'
		
		with self.graph.as_default():

			#PLACEHOLDERS
			self.input = tf.placeholder("float", shape=[self.batch_size, self.n_time_steps, self.image_height, self.image_width, self.n_channels])
			self.target = tf.placeholder("float", shape=[self.batch_size,  self.image_height, self.image_width, self.n_channels])

			#VARIABLES
			# self.conv1_weights = tf.Variable(tf.truncated_normal([self.filter_height, self.filter_width, self.n_channels, self.filter_out_channels1], stddev=0.1))
			# self.conv1_biases = tf.Variable(tf.truncated_normal([self.filter_out_channels1], stddev=0.1))
			
			self.fc1_weights = tf.Variable(tf.truncated_normal([self.flatten_len1, self.fc1_output_len],stddev=0.1))
			self.fc1_biases = tf.Variable(tf.truncated_normal([self.fc1_output_len], stddev=0.1))

			self.forget_gate_variables = tf.Variable(tf.truncated_normal([self.fc1_output_len+self.state_size, self.state_size],stddev=0.1))
			self.forget_gate_variables_b = tf.Variable(tf.truncated_normal([self.state_size], stddev=0.1))
			self.forget_gate_variables2 = tf.Variable(tf.truncated_normal([self.state_size, self.state_size],stddev=0.1))
			self.forget_gate_variables2_b = tf.Variable(tf.truncated_normal([self.state_size], stddev=0.1))

			self.update_gate_variables = tf.Variable(tf.truncated_normal([self.fc1_output_len+self.state_size, self.state_size],stddev=0.1))
			self.update_gate_variables_b = tf.Variable(tf.truncated_normal([self.state_size], stddev=0.1))
			self.update_gate_variables2 = tf.Variable(tf.truncated_normal([self.state_size, self.state_size],stddev=0.1))
			self.update_gate_variables2_b = tf.Variable(tf.truncated_normal([self.state_size], stddev=0.1))

			self.pred_update_vars = tf.Variable(tf.truncated_normal([self.state_size, self.pred_hidden_layer],stddev=0.1))
			self.pred_update_vars_b = tf.Variable(tf.truncated_normal([self.pred_hidden_layer], stddev=0.1))
			self.pred_update_vars2 = tf.Variable(tf.truncated_normal([self.pred_hidden_layer, self.n_classes],stddev=0.1))
			self.pred_update_vars2_b = tf.Variable(tf.truncated_normal([self.n_classes], stddev=0.1))


			#MODEL
			self.logits = self.feedforward(self.input)
			self.weight_decay = self.lmbd * (tf.nn.l2_loss(self.fc1_weights) +
											tf.nn.l2_loss(self.fc1_biases) +
											tf.nn.l2_loss(self.pred_update_vars) +
											tf.nn.l2_loss(self.pred_update_vars2) +
											tf.nn.l2_loss(self.pred_update_vars_b) +
											tf.nn.l2_loss(self.pred_update_vars2_b))
		

			# self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.target))
			self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.target))

			self.cost = self.cross_entropy + self.weight_decay


			self.actual_output = tf.sigmoid(self.logits)


			# self.distorted_input = tf.image.random_contrast(tf.image.random_brightness(self.input,max_delta=63),lower=0.2, upper=1.8)
			# self.logits2 = self.feedforward(self.distorted_input)
			# self.cross_entropy2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits2, self.target))
			# self.cost2 = self.cross_entropy + self.weight_decay

			#TRAIN
			self.opt = tf.train.MomentumOptimizer(self.lr,self.mom)
			self.grads_and_vars = self.opt.compute_gradients(self.cost)
			self.train_opt = self.opt.apply_gradients(self.grads_and_vars)


			# self.test3 = self.test(self.input)


	# def whitten_batch(self, batch):

	# 	images = []
	# 	for i in range(self.batch_size):

	# 		image = tf.slice(batch, [i,0,0,0], [1,480,640,1])
	# 		image = tf.reshape(image, [480,640,1])
	# 		image = tf.image.per_image_whitening(image)
	# 		image = tf.reshape(image, [1,480,640,1])
	# 		images.append(image)

	# 	return tf.concat(0, images)


	def state_update(self, conv_image, state):

		#Forget gate
		forget = tf.matmul(tf.concat(1, [conv_image, state]), self.forget_gate_variables) + self.forget_gate_variables_b
		forget = tf.nn.relu(forget)
		forget = tf.matmul(forget, self.forget_gate_variables2) + self.forget_gate_variables2_b
		forget = tf.sigmoid(forget)
		state = state * forget

		#Update gate
		update = tf.matmul(tf.concat(1, [conv_image, state]), self.update_gate_variables) + self.update_gate_variables_b
		update = tf.nn.relu(update)
		update = tf.matmul(update, self.update_gate_variables2) + self.update_gate_variables2_b
		update = tf.nn.relu(update)
		change = 1. - forget
		state = state + (change * update)

		return state


	def prediction_update(self, state):

		out = tf.matmul(state, self.pred_update_vars) + self.pred_update_vars_b
		out = tf.nn.relu(out)
		out = tf.matmul(out, self.pred_update_vars2) + self.pred_update_vars2_b

		return out


	# def conv_feedforward(self, conv_input):

	# 	# conv_input = self.whitten_batch(conv_input)

	# 	# out = tf.nn.conv2d(conv_input,self.conv1_weights,strides=[1, 2, 2, 1],padding='VALID')
	# 	# out = tf.nn.relu(tf.nn.bias_add(out, self.conv1_biases))

	# 	out = tf.reshape(conv_input, [self.batch_size, -1])
	# 	out = tf.nn.relu(tf.matmul(out, self.fc1_weights) + self.fc1_biases)




		# out = tf.nn.max_pool(out,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
		# out = tf.nn.local_response_normalization(out)

		# out = tf.nn.conv2d(out,self.conv2_weights,strides=[1, 2, 2, 1],padding='VALID')
		# out = tf.nn.relu(tf.nn.bias_add(out, self.conv2_biases))
		# out = tf.nn.max_pool(out,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
		# out = tf.nn.local_response_normalization(out)


		# out = tf.reshape(out, [-1, 1])
		# out = tf.nn.dropout(out, self.keep_prob)
		# return out

	def feedforward(self, sample):

		outputs = []
		state = tf.zeros([self.batch_size,self.state_size])
		# pred = tf.zeros([self.batch_size, self.n_time_steps, self.image_height, self.image_width, self.n_channels])

		for time_step in range(self.n_time_steps):

			frame = tf.slice(sample, [0,time_step,0,0,0], [self.batch_size, 1,self.image_height,self.image_width,1])
			frame = tf.reshape(frame, [self.batch_size,self.image_height,self.image_width,1])

			# frame = tf.nn.conv2d(frame,self.conv1_weights,strides=[1, 2, 2, 1],padding='VALID')
			# frame = tf.nn.relu(tf.nn.bias_add(frame, self.conv1_biases))

			frame_vector = tf.reshape(frame, [self.batch_size, -1])
			frame_vector = tf.nn.relu(tf.matmul(frame_vector, self.fc1_weights) + self.fc1_biases)

			state = self.state_update(frame_vector, state)
			pred = tf.reshape(self.prediction_update(state), [self.batch_size, self.image_height,self.image_width,1])

			outputs.append(pred)

		# return outputs[-1]
		# return [outputs]

		# # out = tf.reshape(sample, [self.batch_size, -1])
		# out = tf.nn.relu(tf.matmul(out, self.fc1_weights) + self.fc1_biases)
		# out = tf.matmul(out, self.pred_update_vars) + self.pred_update_vars_b
		# out = tf.nn.relu(out)
		# out = tf.matmul(out, self.pred_update_vars2) + self.pred_update_vars2_b
		# out = tf.reshape(out, [self.batch_size, self.image_height, self.image_width, self.n_channels])
		# return out
		return outputs[-1]


	# def test(self, sample):

	# 	outputs = []
	# 	state = tf.zeros([self.batch_size,self.state_size])
	# 	# pred = tf.zeros([self.batch_size, self.n_time_steps, self.image_height, self.image_width, self.n_channels])

	# 	for time_step in range(self.n_time_steps):

	# 		# conv_output = self.conv_feedforward(tf.reshape(sample[time_step, :, :, :], [1,480,640,1]))
	# 		conv_output = self.conv_feedforward(tf.reshape(tf.slice(sample, [0,time_step,0,0,0], [self.batch_size, 1,self.image_height,10,1]), [self.batch_size,self.image_height,10,1]))

	# 		# cell_output, state = self.cell_feedforward(conv_output, state)
	# 		state = self.state_update(conv_output, state)
	# 		pred = self.prediction_update(conv_output, state)
	# 		pred = tf.reshape(pred, [self.image_height,10,1])

	# 		outputs.append(state)

	# 	# return outputs[-1]
	# 	return [outputs]


	# def test_run(self, samples, session, path_to_load_variables):

	# 	if session == None:

	# 		with self.graph.as_default():

	# 			saver = tf.train.Saver()

	# 			sess = tf.Session(graph=self.graph)

	# 			# with tf.Session() as sess:

	# 			if path_to_load_variables == '':
	# 				sess.run(tf.initialize_all_variables())
	# 			else:
	# 				saver.restore(sess, path_to_load_variables)
	# 				print 'loaded variables ' + path_to_load_variables

	# 	else:
	# 		sess = session



	# 	feed_dict={self.input: samples}
		
	# 	# ff = sess.run([self.logits], feed_dict=feed_dict)
	# 	ff = sess.run(self.test3, feed_dict=feed_dict)

	# 	return ff, sess





	def fit(self):

		# reader = vid_reader.Video_Reader2(self.path_to_train_dir)
		with self.graph.as_default():
			saver = tf.train.Saver()

			best_error = -1

			with tf.Session() as sess:

				if self.path_to_load_variables == '':
					sess.run(tf.initialize_all_variables())
				else:
					saver.restore(sess, self.path_to_load_variables)
					print 'loaded variables ' + self.path_to_load_variables



				last_20_ces = [99]*20
				last_ce_mean = 99


				for step in range(99999):

					batch = []
					labels = []
					# for i in range(self.batch_size):
					while len(batch) != self.batch_size:
						# samp, label = reader.get_rand_vid_and_label()

						# if len(samp) != 8:
						# 	print 'WHAT!'
						# 	continue

						# distorted_samp = []
						# for j in range(len(samp)):

						# 	distorted_image = tf.image.random_contrast(tf.image.random_brightness(samp[j],max_delta=63),lower=0.2, upper=1.8)
						# 	distorted_samp.append(distorted_image)
						seq=make_ball_gif()

						for i in range(len(seq)):
							seq[i] = seq[i] / np.max(seq[i])


						# seq1=list(seq)
						# seq1.pop(-1)
						batch.append(seq[0:2:1])

						# seq2=list(seq)
						# seq2.pop(0)
						# seq2.append(np.zeros((self.image_height,self.image_width,1)))
						labels.append(seq[2])

					# batch = np.array(batch)
					# print batch.shape
					# fdsfadsa


					feed_dict={self.input: batch, self.target: labels}

					# _ = sess.run(self.train_opt, feed_dict=feed_dict)
					
					# ce, ff = sess.run([self.cross_entropy, self.logits], feed_dict=feed_dict)
					# _ = sess.run([self.train_opt], feed_dict=feed_dict)
					# ce2, ff = sess.run([self.cross_entropy, self.logits], feed_dict=feed_dict)
					# print step, ce2, ce-ce2, best_error


					# print len(batch)
					# print batch[0].shape
					# fsdaas

					# _, ce = sess.run([self.train_opt, self.cross_entropy], feed_dict=feed_dict)
					if step % 50 == 0:
						ce = sess.run(self.cross_entropy, feed_dict=feed_dict)
						print step, ce

					_ = sess.run(self.train_opt, feed_dict=feed_dict)

					if step % 50 == 0:
						ce = sess.run(self.cross_entropy, feed_dict=feed_dict)
						print step, ce
						print

					# if ce < .01:
					# 	break

					if step % 50 == 0:
						# index_ = step % 20
						# last_20_ces[index_] = ce
						# mean_ce = np.mean(last_20_ces)
						if ce < .01:
							break

					if step % 1000 == 0:
						print step
						act_out = sess.run(self.actual_output, feed_dict=feed_dict)
						print act_out
						print 
						print batch
						print 
						print labels


				saver.save(sess, self.path_to_save_variables)
				print 'Saved variables to ' + self.path_to_save_variables			

					# batch_ce = sess.run(self.cross_entropy, feed_dict=feed_dict)
					# ces.append(batch_ce)
					# if batch_ce > last_ce_mean-.15:
					# _ = sess.run(self.train_opt, feed_dict=feed_dict)
						# print step, batch_ce, best_error
					# else:
						# print step, batch_ce, 'skip'


					# if step % 50 == 0:

					# 	print 'Calculating validation error'
					# 	#Get validation error
					# 	errors = []
					# 	accs = []

					# 	reader2 = vid_reader.Video_Reader2(self.path_to_valid_dir)
					# 	for v_samp in range(100/self.batch_size):
					# 	# for v_samp in range(12):

					# 		batch=[]
					# 		labels=[]
					# 		# for i in range(self.batch_size):
					# 		while len(batch) != self.batch_size:
					# 			samp, label = reader2.get_next_vid_and_label()
					# 			if len(samp) != 8:
					# 				print 'WHAT!2'
					# 				# i = i-1
					# 				continue
					# 			batch.append(samp)
					# 			labels.append(label)

					# 		# batch = np.array(batch)
					# 		# print batch.shape

					# 		feed_dict={self.input: batch, self.target: labels, self.keep_prob: 1.}
					# 		ce, ff = sess.run([self.cross_entropy, self.logits], feed_dict=feed_dict)
					# 		errors.append(ce)
					# 		accs.append(np.argmax(ff, axis=1)==labels)

					# 	valid_error = np.mean(errors)

					# 	print 'Validation error is ' + str(valid_error) + ' acc= ' + str(np.mean(accs))

					# 	if valid_error < best_error or best_error == -1:
					# 		best_error = valid_error
					# 		saver.save(sess, self.path_to_save_variables)
					# 		print 'Saved variables to ' + self.path_to_save_variables
					# 	else:
					# 		self.lr = self.lr / 10
					# 		if self.lr < .0000000001:
					# 			self.lr = .0000000001
					# 		print 'lr is ' + str(self.lr)


					# 	last_ce_mean = np.mean(ces)
					# 	print 'ce mean = ', last_ce_mean
					# 	ces = []


	def predict(self, samples, session, path_to_load_variables):

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
		
		ff = sess.run([self.logits], feed_dict=feed_dict)

		return ff, sess


		# return np.exp(ff)/np.sum(np.exp(ff)), sess



######################################################################
#TRAIN

model = Predict_Next_Frame(batch_size=5)
start_time = time.time()
model.fit()
print time.time() - start_time
print 'DONE'
fsdfa



######################################################################
#DEBUG


# model = RNN(batch_size=1)
# sess = None
# seq = make_ball_gif()
# for i in range(len(seq)):
# 	seq[i] = seq[i] / np.max(seq[i])

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
# 	print i
# 	samp, label = reader.get_next_vid_and_label()
# 	# print samp.shape

# 	prediction, sess = model.predict(samples=samp, session=sess)
# 	# print label
# 	# print prediction[0][0], label

# 	#COUNT
# 	if np.argmax(prediction[0][0]) == label:
# 		print prediction[0][0], label, 'yes!'
# 		correct+=1
# 	else:
# 		print prediction[0][0], label, 'no.'
# 		#VIEW
# 		for frame in samp:
# 			cv2.imshow('Video', frame)
# 			cv2.waitKey(0)
# 			cv2.destroyAllWindows()





model = Predict_Next_Frame(batch_size=1)
sess = None
seq = make_ball_gif()
for i in range(len(seq)):
	seq[i] = seq[i] / np.max(seq[i])


prediction, sess = model.predict(samples=[seq], session=sess, path_to_load_variables=home+'/Documents/tmp/rnn_sequence.ckpt')
print np.array(prediction).shape


n_frames=4
f_height=5
f_width=6
ball_size=2

prediction = np.reshape(prediction, [n_frames,f_height,f_width,1])#.astype('uint8')

print np.reshape(prediction[2], [f_height,f_width])
print 
print np.reshape(prediction[3], [f_height,f_width])


for i in range(len(prediction)):

	prediction[i] = prediction[i] - np.max(prediction[i])
	prediction[i]= np.exp(prediction[i])/np.sum(np.exp(prediction[i]))
	prediction[i]= prediction[i] * (255. / np.max(prediction[i]))
	if np.max(seq[i]) > 0:
		seq[i] = seq[i] * (255. / np.max(seq[i]))


print np.reshape(prediction[2], [f_height,f_width])
print 
print np.reshape(prediction[3], [f_height,f_width])



prediction = prediction.astype('uint8')

print prediction[2]





kargs = { 'duration': .5 }
imageio.mimsave(home+"/Downloads/real_gif.gif", seq, 'GIF', **kargs)
imageio.mimsave(home+"/Downloads/pred_gif.gif", prediction, 'GIF', **kargs)



print 'DONE'





