#VAE

import numpy as np
import tensorflow as tf

# from os.path import expanduser
# home = expanduser("~")
# from PIL import Image
# import pickle
# import random

class VAE():

	def __init__(self, train=None, valid=None, path_to_load_variables='', path_to_save_variables):

		self.graph=tf.Graph()

		self.train_data = train
		self.valid_data = valid

		self.path_to_load_variables = home+'/Documents/Viewnyx_data/model_variables/from_rao/v3_vars_5.ckpt'
		self.path_to_save_variables = home+'/storage/viewnyx/model_variables/v3_vars_6.ckpt'
		
		#MODEL SPECIFICATION
		self.image_height = 240
		self.image_width = 320
		self.n_channels = 1

		self.filter_height = 25
		self.filter_width = 25
		self.filter_out_channels1 = 20

		self.filter2_height = 10
		self.filter2_width = 10
		self.filter2_out_channels = 20

		self.flatten_len1 = 560
		self.fc1_output_len = 100
		self.pred_hidden_layer = 20
		self.n_classes = 3

		#HYPERPARAMETERS
		self.lr = .0001
		self.mom =  .1
		self.lmbd = .0001
		self.batch_size = 1

		
		with self.graph.as_default():

			#PLACEHOLDERS
			self.input = tf.placeholder("float", shape=[None, self.image_height, self.image_width, self.n_channels])
			self.target = tf.placeholder("int32", shape=[self.batch_size])

			#VARIABLES
			self.conv1_weights = tf.Variable(tf.truncated_normal([self.filter_height, self.filter_width, self.n_channels, self.filter_out_channels1], stddev=0.1))
			self.conv1_biases = tf.Variable(tf.truncated_normal([self.filter_out_channels1], stddev=0.1))
			self.conv2_weights = tf.Variable(tf.truncated_normal([self.filter2_height, self.filter2_width, self.filter_out_channels1, self.filter2_out_channels], stddev=0.1))
			self.conv2_biases = tf.Variable(tf.truncated_normal([self.filter2_out_channels], stddev=0.1))
			
			self.fc1_weights = tf.Variable(tf.truncated_normal([self.flatten_len1, self.fc1_output_len],stddev=0.1))
			self.fc1_biases = tf.Variable(tf.truncated_normal([self.fc1_output_len], stddev=0.1))

			self.fc2_weights = tf.Variable(tf.truncated_normal([self.fc1_output_len, self.pred_hidden_layer],stddev=0.1))
			self.fc2_biases = tf.Variable(tf.truncated_normal([self.pred_hidden_layer], stddev=0.1))

			self.fc3_weights = tf.Variable(tf.truncated_normal([self.pred_hidden_layer, self.n_classes],stddev=0.1))
			self.fc3_biases = tf.Variable(tf.truncated_normal([self.n_classes], stddev=0.1))

			#MODEL
			self.logits = self.feedforward(self.input)
			self.weight_decay = self.lmbd * (tf.nn.l2_loss(self.fc1_weights) +
											tf.nn.l2_loss(self.fc1_biases) +
											tf.nn.l2_loss(self.fc2_weights) +
											tf.nn.l2_loss(self.fc2_biases) +
											tf.nn.l2_loss(self.fc3_weights) +
											tf.nn.l2_loss(self.fc3_biases))

			self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.target))
			self.cost = self.cross_entropy + self.weight_decay

			self.distorted_input = tf.image.random_contrast(tf.image.random_brightness(self.input,max_delta=63),lower=0.2, upper=1.8)
			self.logits2 = self.feedforward(self.distorted_input)
			self.cross_entropy2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits2, self.target))
			self.cost2 = self.cross_entropy2 + self.weight_decay

			#TRAIN
			self.opt = tf.train.MomentumOptimizer(self.lr,self.mom)
			self.grads_and_vars = self.opt.compute_gradients(self.cost2)
			self.train_opt = self.opt.apply_gradients(self.grads_and_vars)


	def whitten_batch(self, batch):

		images = []
		for i in range(self.batch_size):

			image = tf.slice(batch, [i,0,0,0], [1,self.image_height,self.image_width,self.n_channels])
			image = tf.reshape(image, [self.image_height,self.image_width,self.n_channels])
			image = tf.image.per_image_whitening(image)
			image = tf.reshape(image, [1,self.image_height,self.image_width,self.n_channels])
			images.append(image)

		return tf.concat(0, images)


	def feedforward(self, out):

		out = self.whitten_batch(out)

		out = tf.nn.conv2d(out,self.conv1_weights,strides=[1, 4, 4, 1],padding='VALID')
		out = tf.nn.relu(tf.nn.bias_add(out, self.conv1_biases))
		out = tf.nn.max_pool(out,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
		out = tf.nn.local_response_normalization(out)

		out = tf.nn.conv2d(out,self.conv2_weights,strides=[1, 2, 2, 1],padding='VALID')
		out = tf.nn.relu(tf.nn.bias_add(out, self.conv2_biases))
		out = tf.nn.max_pool(out,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
		out = tf.nn.local_response_normalization(out)

		out = tf.reshape(out, [self.batch_size, -1])
		out = tf.nn.relu(tf.matmul(out, self.fc1_weights) + self.fc1_biases)
		out = tf.nn.relu(tf.matmul(out, self.fc2_weights) + self.fc2_biases)
		out = tf.matmul(out, self.fc3_weights) + self.fc3_biases

		return out




	def fit(self):

		saver = tf.train.Saver()
		best_error = -1

		with tf.Session() as sess:

			if self.path_to_load_variables == '':
				sess.run(tf.initialize_all_variables())
			else:
				saver.restore(sess, self.path_to_load_variables)
				print 'loaded variables ' + self.path_to_load_variables

			ces = []
			last_ce_mean = .9

			for step in range(99999):

				batch = []
				labels = []
				# for i in range(self.batch_size):
				while len(batch) != self.batch_size:

					iii = random.randint(0, len(self.train_data)-1)
					samp, label = self.train_data[iii]
					batch.append(samp)
					labels.append(label)

				feed_dict={self.input: batch, self.target: labels}
				

				batch_ce = sess.run(self.cross_entropy, feed_dict=feed_dict)
				ces.append(batch_ce)
				if batch_ce > last_ce_mean-.3:
					_ = sess.run(self.train_opt, feed_dict=feed_dict)
					if step % 50 == 0:
						print step, batch_ce, best_error
				else:
					if step % 50 == 0:
						print step, batch_ce, 'skip'

				# _, ce = sess.run([self.train_opt, self.cross_entropy], feed_dict=feed_dict)

				# if step % 50 == 0:
				# 	print step, ce, best_error
				# 	# print np.around(sess.run(self.logits, feed_dict=feed_dict), decimals=2)
				# 	# print labels


				if step % 200 == 0:

					print 'Calculating validation error'
					#Get validation error
					errors = []
					accs = []

					iii = 0
					for v_samp in range(len(self.valid_data)/self.batch_size):
					# for v_samp in range(12):

						batch=[]
						labels=[]
						# for i in range(self.batch_size):
						
						while len(batch) != self.batch_size:
							samp, label = self.valid_data[iii]

							batch.append(samp)
							labels.append(label)
							iii +=1


						feed_dict={self.input: batch, self.target: labels}
						ce, ff = sess.run([self.cross_entropy, self.logits], feed_dict=feed_dict)
						errors.append(ce)
						accs.append(np.argmax(ff, axis=1)==labels)

					valid_error = np.mean(errors)

					print 'Validation error is ' + str(valid_error) + ' acc= ' + str(np.mean(accs))

					if valid_error < best_error or best_error == -1:
						best_error = valid_error
						saver.save(sess, self.path_to_save_variables)
						print 'Saved variables to ' + self.path_to_save_variables
					else:
						self.lr = self.lr / 10
						if self.lr < .0000000001:
							self.lr = .0000000001
						print 'lr is ' + str(self.lr)

					last_ce_mean = np.mean(ces)
					print 'ce mean = ', last_ce_mean
					ces = []


				if step % 10000 == 0:
					saver.save(sess, home+'/storage/viewnyx/model_variables/v5_vars_backup.ckpt')
					print 'Saved variables to ' + home+'/storage/viewnyx/model_variables/v5_vars_backup.ckpt'



	def predict(self, samples, session=None):

		if session == None:

			with self.graph.as_default():
				saver = tf.train.Saver()
				sess = tf.Session(graph=self.graph)
				
				# with tf.Session() as sess:
				if self.path_to_load_variables == '':
					sess.run(tf.initialize_all_variables())
				else:
					saver.restore(sess, self.path_to_load_variables)
					print 'loaded variables ' + self.path_to_load_variables

		else:
			sess = session

		feed_dict={self.input: samples}
		ff = sess.run([self.logits], feed_dict=feed_dict)

		return np.exp(ff)/np.sum(np.exp(ff)), sess








if __name__ == "__main__":

	# #For training
	# print 'Loading data'
	# with open(home + '/storage/viewnyx/20160724_v3_train.pkl', 'rb') as f:
	# 	data = pickle.load(f)
	# # with open(home + '/Documents/Viewnyx_data/20160513_data.pkl', 'rb') as f:
	# # 	data = pickle.load(f)
	# X = np.array(data[0])
	# y = np.array(data[1]).astype(int)
	# print X.shape, y.shape







