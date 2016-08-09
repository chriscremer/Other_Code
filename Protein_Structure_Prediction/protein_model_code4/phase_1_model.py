



import numpy as np
import tensorflow as tf
import math
import json
import random




class Expert_Learner():
	'''
	Learn the weights and biases of an expert. 
	'''

	def __init__(self, n_aa=22, tol=2., batch_size=5, lr=.01, mom=.01, lmbd=.1, w=None, b=None):

		self.n_aa = n_aa #number of types of amino acids
		self.D = 3 #Dimensions of space
		self.tol =  tol
		self.batch_size = batch_size #batch size

		#Hyperparameters
		self.lr = lr #learning rate divided by batch size
		self.mom = mom #momentum
		self.lmbd = lmbd #weight decay

		#Parameters	
		if w == None: 
			self.w = tf.Variable(tf.random_normal([n_aa, n_aa], stddev=0.1))
		else: 
			self.w = w
		if b == None:
			self.b = tf.Variable(tf.random_normal([n_aa], stddev=0.1))
		else: 
			self.b = b

		
	def fit(self, train_data, valid_data):
		'''
		Learn the expert
		'''

		input_tensor = tf.placeholder(tf.float32, shape=[None, self.n_aa])
		target = tf.placeholder(tf.float32, shape=[None, self.n_aa])

		prediction = self.feed_forward(input_tensor)

		cross_entropy = -tf.reduce_sum(target*tf.log(prediction))
		weight_decay = self.lmbd * tf.reduce_sum(tf.square(self.w))
		cost_function = cross_entropy + weight_decay

		CE2 = -tf.reduce_sum(target*tf.log(self.feed_forward2(input_tensor)))

		train_op = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.mom).minimize(cost_function)

		previous_mean_valid_error = -1

		sess = tf.Session()
		sess.run(tf.initialize_all_variables())

		for samp in range(len(train_data[0])):

			# print '\nSamp ' + str(samp)

			X_samp = np.reshape(train_data[0][samp], (1,self.n_aa))
			y_samp = np.reshape(train_data[1][samp], (1,self.n_aa))

			_, cost, ce, wd = sess.run([train_op, cost_function, cross_entropy, weight_decay], 
				feed_dict={input_tensor: X_samp, target: y_samp})
			# print str(samp) + ' cost: ' + str(cost) + ' CE: ' + str(ce) + ' WD: ' + str(wd)

			if samp % 5000 == 0:
				valid_costs = []
				for valid_samp in range(len(valid_data[0])):
					X_samp = np.reshape(valid_data[0][valid_samp], (1,self.n_aa))
					y_samp = np.reshape(valid_data[1][valid_samp], (1,self.n_aa))
					ce = sess.run(cross_entropy, feed_dict={input_tensor: X_samp, target: y_samp})
					valid_costs.append(ce)
				mean_valid_error = np.mean(valid_costs)
				# print mean_valid_error


				valid_costs2 = []
				for valid_samp in range(len(valid_data[0])):
					X_samp = np.reshape(valid_data[0][valid_samp], (1,self.n_aa))
					y_samp = np.reshape(valid_data[1][valid_samp], (1,self.n_aa))
					ce = sess.run(CE2, feed_dict={input_tensor: X_samp, target: y_samp})
					valid_costs2.append(ce)
				mean_valid_error2 = np.mean(valid_costs2)
				# print 'CE2 ' + str(mean_valid_error2)


				if previous_mean_valid_error - mean_valid_error < self.tol and previous_mean_valid_error != -1:
					# print 'Tol reached'
					break
				previous_mean_valid_error = mean_valid_error

		#Convert it to arrays before ending
		# print 'Went to samp ' + str(samp)
		self.w = sess.run(self.w)
		self.b = sess.run(self.b)


	def feed_forward(self, input1):

		return tf.nn.softmax(tf.matmul(input1, self.w) + self.b)



	def feed_forward2(self, input1):

		return tf.nn.softmax(tf.matmul(input1, self.w))














