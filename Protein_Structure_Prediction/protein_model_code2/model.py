



import numpy as np
import tensorflow as tf
import math
import json
import random


class Position_Encoder():
	'''
	Learn the 3D coordinates for the columns of a MSA
	'''

	def __init__(self, L=None, n_aa=22, tol=2., B=20, K=5, lr=.01, mom=.01, nu=.1, lmbd=.1, rho=.01, sigma_repel=.01, sigma_adj=.1, w=None, b=None, coordinates=[]):

		self.L = L
		self.n_aa = n_aa
		self.D = 3 #Dimensions of space
		self.tol =  tol
		self.B = B #batch size

		#Hyperparameters
		self.K = K #length of encoding
		self.lr = lr #learning rate divided by batch size
		self.mom = mom #momentum
		self.nu = nu #weight of repel all cost
		self.lmbd = lmbd #weight of adjacent distance cost
		self.rho = rho #weight decay
		self.sigma_repel = sigma_repel
		self.sigma_adj = sigma_adj
		self.sigma_gate = 1. #start at 10 then go to 1000
		self.help_predict = 1000. #start at 10 then go to 1000

		#Parameters
		
		if w == None: 
			self.w = tf.Variable(tf.random_normal([L, L, n_aa, n_aa], stddev=0.01))
		else: 
			self.w = w
		if b == None:
			self.b = tf.Variable(tf.random_normal([L, L, n_aa], stddev=0.01))
		else: 
			self.b = b
		if coordinates == []:
			self.coordinates = tf.Variable(tf.random_normal([self.D, L, 1], stddev=0.01))
		else:
			coords = np.reshape(coordinates.T, [self.D,L,1])
			coords = np.float32(coords)
			self.coordinates = tf.Variable(coords)

		#FOR EMBEDDING
		# self.G = tf.Variable(tf.random_normal([L,L,K,K], stddev=0.01))
		# self.embedding = tf.Variable(tf.random_normal([n_aa, K], stddev=0.01))
		# self.decoding = tf.Variable(tf.random_normal([K, n_aa], stddev=0.01))

		# self.bbb = tf.Variable([[1,2,3],[2,1,5],[6,3,4]])

		
	def fit(self, X_train, X_valid):
		'''
		Input should be ints
		'''

		input_tensor = tf.placeholder(tf.float32, shape=[self.L, self.L, self.n_aa, 1])
		target = tf.placeholder(tf.float32, shape=[self.L, self.n_aa, 1])

		sigma_gate_holder = tf.placeholder(tf.float32, shape=[])
		help_predict_holder = tf.placeholder(tf.float32, shape=[])

		# #BATCHES
		# input_tensor = tf.placeholder(tf.float32, shape=[self.B, self.L, self.L, self.n_aa, 1])
		# target = tf.placeholder(tf.float32, shape=[self.B, self.L, self.n_aa, 1])


		gating_network = self.gate(sigma_gate_holder)
		prediction = self.feed_forward(input_tensor, gating_network)

		cross_entropy = help_predict_holder*(-tf.reduce_sum(tf.reshape(target, [self.L, self.n_aa])*tf.log(prediction)))
		# cross_entropy = -tf.reduce_sum(target*tf.log(prediction))
		# #BATCHES
		# cross_entropy = (1./self.B) * -tf.reduce_sum(tf.reshape(target, [self.L, self.B, self.n_aa])*tf.log(prediction))
		weight_decay = (self.rho * tf.reduce_sum(tf.square(self.w)))
		adj_cost =  self.lmbd * self.pull_adj()
		repel_cost = self.nu * self.repel_all()

		cost_function = cross_entropy + weight_decay + repel_cost + adj_cost

		train_op = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.mom).minimize(cost_function)
		# train_op = tf.train.RMSPropOptimizer.__init__(learning_rate=lr, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')
		
		# train_op = tf.train.GradientDescentOptimizer(self.lr)
		# grads_and_vars = train_op.compute_gradients(cost_function)
		# modified_grad_and_vars = [(gv[0], gv[1]) for gv in grads_and_vars]
		# apply_grad = train_op.apply_gradients(modified_grad_and_vars)

		# debug = self.sigma_gate
		# sqquu = tf.square(self.sigma_gate)
		

		previous_mean_valid_error = -1
		converged = 0

		sess = tf.Session()
		sess.run(tf.initialize_all_variables())

		for i in range(20000):

			#FOR BATCHES
			# print  'Batch ' + str(i)
			# batch = next_batch(X_train, self.B)
			# input1 = convert_batch_to_B_L_by_L(batch, self.n_aa) #(B, L, L, n_aa, 1)
			# target1 = convert_batch_to_one_hot(batch, self.n_aa) #(B, L, n_aa, 1)
			# if i %1 == 0:
			# 	print 'Before step'
			# 	batch = next_batch(X_valid,self.B)
			# 	input1 = convert_batch_to_B_L_by_L(batch, self.n_aa) #(B, L, L, n_aa, 1)
			# 	target1 = convert_batch_to_one_hot(batch, self.n_aa) #(B, L, n_aa, 1)
			# 	valid_error, ce, wd, pt, rc = sess.run([cost_function, cross_entropy, weight_decay, adj_cost, repel_cost], 
			# 						feed_dict={input_tensor: input1, target: target1, sigma_gate_holder: self.sigma_gate})
			# 	print 'cost: ' + str(valid_error) + ' CE: ' + str(ce) + ' WD: ' + str(wd) + ' AC: ' + str(pt) + ' RC: ' + str(rc)
			# _, cost, pred, gate, ce, wd, pt, rc = sess.run([train_op, cost_function, prediction, gating_network, cross_entropy, weight_decay, adj_cost, repel_cost], 
			# 	feed_dict={input_tensor: input1, target: target1, sigma_gate_holder: self.sigma_gate})
			# print 'cost: ' + str(cost) + 'CE: ' + str(ce) + ' WD: ' + str(wd) + ' AC: ' + str(pt) + ' RC: ' + str(rc)


			samp = i
			print '\nSamp ' + str(samp)

			# cost, pred, gate, ce, wd, pt, rc = sess.run([cost_function, prediction, gating_network, cross_entropy, weight_decay, adj_cost, repel_cost], 
			# 	feed_dict={input_tensor: convert_samp_to_L_by_L(X_train[samp], self.n_aa), target: convert_samp_to_one_hot(X_train[samp], self.n_aa), sigma_gate_holder: self.sigma_gate})
			# print str(i) + ' cost: ' + str(cost) + ' CE: ' + str(ce) + ' WD: ' + str(wd) + ' AC: ' + str(pt) + ' RC: ' + str(rc)

			_, cost, pred, gate, ce, wd, pt, rc = sess.run([train_op, cost_function, prediction, gating_network, cross_entropy, weight_decay, adj_cost, repel_cost], 
				feed_dict={input_tensor: convert_samp_to_L_by_L(X_train[samp], self.n_aa), 
							target: convert_samp_to_one_hot(X_train[samp], self.n_aa), 
							sigma_gate_holder: self.sigma_gate,
							help_predict_holder: self.help_predict})
			print str(i) + ' cost: ' + str(cost) + ' CE: ' + str(ce) + ' WD: ' + str(wd) + ' AC: ' + str(pt) + ' RC: ' + str(rc)

			# cost, pred, gate, ce, wd, pt, rc = sess.run([cost_function, prediction, gating_network, cross_entropy, weight_decay, adj_cost, repel_cost], 
			# 	feed_dict={input_tensor: convert_samp_to_L_by_L(X_train[samp], self.n_aa), target: convert_samp_to_one_hot(X_train[samp], self.n_aa), sigma_gate_holder: self.sigma_gate})
			# print str(i) + ' cost: ' + str(cost) + ' CE: ' + str(ce) + ' WD: ' + str(wd) + ' AC: ' + str(pt) + ' RC: ' + str(rc)

			# sess.run(gating_network, feed_dict={sigma_gate_holder: self.sigma_gate})
			gate = np.reshape(gate, [166,166])
			print gate.shape
			print gate[0]
			print '0_90: ' + str(gate[0][90])
			print 'mean: ' + str(np.mean(gate[0]))
			print sum(gate[0])
			print '3_18: ' + str(gate[3][18])
			print '12_60: ' + str(gate[12][60])
			print self.sigma_gate
			print self.help_predict



			if math.isnan(cost) or math.isnan(pt) or math.isinf(pt):
				print 'WTF IS GOING ON?!?!?!?!?'
				print 'why is it nan??! '
				print 'CE: ' + str(ce) + ' WD: ' + str(wd) + ' AC: ' + str(pt) + ' RC: ' + str(rc)
				print sess.run(self.coordinates).T
				print 
				break

			#Every 100 samples/updates, print the validation cost and check for convergence
			# if i%50 == 0:
			# 	print 'checking validation'
			# 	valid_costs = []
			# 	for samp2 in range(len(X_valid)):
			# 		valid_error, ce, wd, pt, rc = sess.run([cost_function, cross_entropy, weight_decay, adj_cost, repel_cost], 
			# 			feed_dict={input_tensor: convert_samp_to_L_by_L(X_valid[samp2], self.n_aa), 
			# 						target: convert_samp_to_one_hot(X_valid[samp2], self.n_aa), 
			# 						sigma_gate_holder: self.sigma_gate,
			# 						help_predict_holder: self.help_predict})
			# 		valid_costs.append(valid_error)
			# 	mean_valid_cost = np.mean(valid_costs)
			# 	print str(samp) + ' Valid cost ' + str(mean_valid_cost) + ':  CE: ' + str(ce) + ' WD: ' + str(wd) + ' AC: ' + str(pt) + ' RC: ' + str(rc)
					
			# 	gate = np.reshape(gate, [166,166])
			# 	print gate.shape
			# 	print gate[0]
			# 	print '90: ' + str(gate[0][90])
			# 	print 'mean: ' + str(np.mean(gate[0]))
			# 	print sum(gate[0])
			# 	print self.sigma_gate
			# 	print self.help_predict

				# if previous_mean_valid_error - mean_valid_cost < self.tol and previous_mean_valid_error != -1:
				# 	print 'Tol reached'
				# 	converged = 1
				# 	break
				# previous_mean_valid_error = mean_valid_cost	


			# if i %1 == 0:

			# 	print 'After step'
			# 	batch = next_batch(X_valid,self.B)
			# 	input1 = convert_batch_to_B_L_by_L(batch, self.n_aa) #(B, L, L, n_aa, 1)
			# 	target1 = convert_batch_to_one_hot(batch, self.n_aa) #(B, L, n_aa, 1)
			# 	valid_error, ce, wd, pt, rc = sess.run([cost_function, cross_entropy, weight_decay, adj_cost, repel_cost], 
			# 						feed_dict={input_tensor: input1, target: target1, sigma_gate_holder: self.sigma_gate})
			# 	print 'cost: ' + str(valid_error) + ' CE: ' + str(ce) + ' WD: ' + str(wd) + ' AC: ' + str(pt) + ' RC: ' + str(rc)
				
			# 	# print sess.run(gating_network, feed_dict={sigma_gate_holder: self.sigma_gate})[0]
			# 	# print sess.run(debug)
			# 	print self.sigma_gate
			# 	# print sess.run(sqquu)

			# 	# if previous_mean_valid_error - valid_error < self.tol and previous_mean_valid_error != -1:
			# 	# 	print 'Tol reached'
			# 	# 	break
			# 	# previous_mean_valid_error = valid_error	



			self.sigma_gate = (i+1.)/10.
			if self.sigma_gate > 100.:
				self.sigma_gate = 100.

			self.help_predict = self.help_predict - 1.
			if self.help_predict < 1.:
				self.help_predict = 1.

		#Save the parameters as numpy rather than tensorflow
		# self.w = sess.run(self.w)
		# self.b = sess.run(self.b)
		self.coordinates = sess.run(self.coordinates)
		





	def gate(self, sig):
		'''
		Compute the gating coefficients based on the coordinates of the columns
		Input: self.coordinates
		Output: LxLx1 matrix where each row sums to one. The larger the value, the closer you are together.
		'''
		#Step 1: coordinates dot ones transpose -> 3xLxL -> just replicates the 1D vector L times
		output = tf.batch_matmul(self.coordinates, tf.transpose(tf.ones_like(self.coordinates), perm=[0, 2, 1]))
		# output = tf.batch_matmul(self.bbb, tf.transpose(tf.ones_like(self.bbb), perm=[0, 2, 1]))
		
		#Step 2: minus its tranpose -> 3xLxL -> symmetric difference matrix
		output = output - tf.transpose(output, perm=[0, 2, 1]) 
		#Step 3: add 1000 to the diagonals, ie dist to itself is 1000 instead of 0 -> so that the weight column x has to predict itself is very low, preferably zero
		# output = output + tf.reshape(tf.concat(0,[tf.diag(tf.ones([self.L]))*99999, tf.diag(tf.ones([self.L]))*99999, tf.diag(tf.ones([self.L]))*99999]),[self.D,self.L,self.L])
		#could use the function tile for the stuff above, or just add to the diagonal of the next stuff
		#Step 4: square difference and sum over the dimensions -> LxL -> dist of coord to each coord, symmetric
		output = tf.reduce_sum(tf.square(output),0)
		#Step 4.1:
		output = -(sig*output)
		#Make the diagonal very negative, meaning they are far apart, so it get low gating weight to itself
		output = output + tf.diag(tf.ones([self.L]))*(-99999)
		#Step 5: softmax so rows sum to 1
		output = tf.nn.softmax(output)
		#Step 6: reshape so it works later -> LxLx1
		output = tf.reshape(output, [self.L, self.L, 1])
		return output


	def feed_forward(self, input1, gating_network):

		#Step 1: Output of each expert (net) -> LxLx21x21 * LxLx21x1 = LxLx21x1 = LxLx21
		output = tf.reshape(tf.batch_matmul(self.w, input1), [self.L, self.L, self.n_aa]) + self.b  

		# #Step 2: Since the softmax function only takes in a matrix (not tensor), have to do it myself
		exp_ = tf.exp(output) #numerator
		output = tf.reshape(tf.reduce_sum(exp_, 2), [self.L,self.L,1]) #sum the outputs for each of the nets
		sums = tf.tile(output, [1,1,self.n_aa]) #tile (replicate) the sum to each numerator
		output = exp_ / sums #softmax -> LxLx21

		# #Step 3: Multiply by the gating function -> Lx21xL * LxLx1 = Lx21x1 -> prediction for each column
		output = tf.batch_matmul(tf.transpose(output, perm=[0, 2, 1]), gating_network)

		# #Step 4: Reshape to fit other stuff -> Lx21
		output = tf.reshape(output, [self.L, self.n_aa])




		#Step 1 WITH BATCHES: Output of each expert (net) -> LxLxBx21 * LxLx21x21 = LxLxBx21
		# input1 = tf.transpose(input1, perm=[1, 2, 0, 3, 4])
		# input1 = tf.reshape(input1, [self.L, self.L, self.B, self.n_aa])
		# output = tf.batch_matmul(input1, self.w)
		# bbb = tf.reshape(self.b, [self.L, self.L, 1, self.n_aa])
		# bbb = tf.tile(bbb, [1,1, self.B, 1])
		# output = output + bbb

		#Step 2 WITH BATCHES: Since the softmax function only takes in a matrix (not tensor), have to do it myself
		# exp_ = tf.exp(output) #numerator, LxLxBx21
		# output = tf.reshape(tf.reduce_sum(exp_, 3), [self.L,self.L,self.B,1]) #sum the outputs for each of the nets, LxLxBx1
		# sums = tf.tile(output, [1,1,1,self.n_aa]) #tile (replicate) the sum to each numerator, LxLxBx21
		# output = exp_ / sums #softmax -> LxLxBx21

		#Step 3 WITH BATCHES: Multiply by the gating function -> LxBx21xL * LxBxLx1 = LxBx21x1 -> prediction for each column
		# gn = tf.reshape(gating_network, [self.L, 1, self.L, 1])
		# gn = tf.tile(gn, [1,self.B,1,1])
		# output = tf.batch_matmul(tf.transpose(output, perm=[0, 2, 3, 1]), gn)

		#Step 4 WITH BATCHES: Reshape to fit other stuff -> LxBx21
		# output = tf.reshape(output, [self.L, self.B, self.n_aa])





		#EMBEDDING
		# #First tile the embedding, need to reshape so that it has the right rank
		# tiled = tf.reshape(self.embedding, [1,1,self.n_aa,self.K]) #1x1x21x5
		# tiled = tf.tile(tiled, [self.L,self.L,1,1]) #LxLx21x5
		# tiled = tf.transpose(tiled, perm=[0,1,3,2]) #LxLx5x21
		# #Use embedding: embedding * input -> LxLx5x21 * LxLx21x1 = LxLx5x1
		# output = tf.batch_matmul(tiled, input1)
		# #Multiply by G: LxLx5x5 * LxLx5x1 = LxLx5x1
		# output = tf.batch_matmul(self.G, output)
		# #Decode
		# tiled = tf.reshape(self.decoding, [1,1,self.K,self.n_aa]) #1x1x5x21
		# tiled = tf.tile(tiled, [self.L,self.L,1,1]) #LxLx5x21
		# tiled = tf.transpose(tiled, perm=[0,1,3,2]) #LxLx21x5
		# output = tf.batch_matmul(tiled,output) #LxLx21x5 * LxLx5x1 = LxLx21x1
		# #Add bias
		# output = tf.reshape(output, [self.L, self.L, self.n_aa]) + self.b  






		return output



	def pull_adj(self):

		range1 = tf.range(1,self.L)
		range2 = tf.range(0,self.L-1)
		#Step 1: the first gather takes the dimension, the second gather excludes the first element, repeat for each dimension
		first_D0 = tf.gather(tf.reshape(tf.gather(tf.reshape(self.coordinates, [self.D, self.L]), [0]), [self.L]), range1)
		first_D1 = tf.gather(tf.reshape(tf.gather(tf.reshape(self.coordinates, [self.D, self.L]), [1]), [self.L]), range1)
		first_D2 = tf.gather(tf.reshape(tf.gather(tf.reshape(self.coordinates, [self.D, self.L]), [2]), [self.L]), range1)
		#Step 2: the first gather takes the dimension, the second gather excludes the last element, repeat for each dimension
		second_D0 = tf.gather(tf.reshape(tf.gather(tf.reshape(self.coordinates, [self.D, self.L]), [0]), [self.L]), range2)
		second_D1 = tf.gather(tf.reshape(tf.gather(tf.reshape(self.coordinates, [self.D, self.L]), [1]), [self.L]), range2)
		second_D2 = tf.gather(tf.reshape(tf.gather(tf.reshape(self.coordinates, [self.D, self.L]), [2]), [self.L]), range2)
		#Step 3: exp(dist betw adj) -> L-1 vector -> sum vector -> scalar -> so if adj are far apart, this will be large, minmizing will pull adj together
		output = tf.reduce_sum(tf.exp(self.sigma_adj*(tf.square(first_D0 - second_D0) + tf.square(first_D1 - second_D1) + tf.square(first_D2 - second_D2))))
		return output


	def repel_all(self):

		#Step 1: Coordinates dot ones -> 3xLx1 * 3x1xL = 3xLxL -> just replicates the 1D rows L times
		output = tf.batch_matmul(self.coordinates, tf.transpose(tf.ones_like(self.coordinates), perm=[0, 2, 1]))
		#Step 2: Minus its tranpose -> 3xLxL -> symmetric difference matrix, diagonal is zeros
		output = output - tf.transpose(output, perm=[0, 2, 1])
		#Step 3: negative square the difference, sum over the dimensions, exp the result -> LxL -> so if far apart, then this is small
		output = tf.exp(self.sigma_repel*tf.reduce_sum(-tf.square(output),0))
		#Step 4: sum all elements -> scalar -> want to min this by making all coordinates far apart (repel)
		output = tf.reduce_sum(output)
		return output


def convert_to_one_hot(aa, n_aa):
	vec = np.zeros((n_aa,1))
	vec[aa] = 1
	return vec

def convert_samp_to_one_hot(samp, n_aa):

	one_hot_samp = []
	for i in range(len(samp)):
		vec = np.zeros((n_aa,1))
		vec[samp[i]] = 1
		one_hot_samp.append(vec)
	return np.array(one_hot_samp)


def convert_samp_to_L_by_L(samp, n_aa):

	L_by_L = []
	for i in range(len(samp)):
		this_samp = []
		for j in range(len(samp)):
			if j == i:
				this_samp.append(convert_to_one_hot(0, n_aa))
			else:
				this_samp.append(convert_to_one_hot(samp[j], n_aa))
		L_by_L.append(this_samp)

	return np.array(L_by_L)


def next_batch(X, batch_size):
	'''
	For now, just return some random samples
	'''
	indexes = random.sample(range(len(X)), batch_size)
	x_batch = X[indexes]

	return x_batch

def convert_batch_to_B_L_by_L(batch, n_aa):

	B_L_by_L = []
	for b in range(len(batch)):
		L_by_L = []
		for i in range(len(batch[b])):
			this_samp = []
			for j in range(len(batch[b])):
				if j == i:
					this_samp.append(convert_to_one_hot(0, n_aa))
				else:
					this_samp.append(convert_to_one_hot(batch[b][j], n_aa))
			L_by_L.append(this_samp)

		L_by_L = np.array(L_by_L)
		B_L_by_L.append(L_by_L)

	return np.array(B_L_by_L)


def convert_batch_to_one_hot(batch, n_aa):

	B_one_hot_samp = []
	for b in range(len(batch)):
		one_hot_samp = []
		for i in range(len(batch[b])):
			vec = np.zeros((n_aa,1))
			vec[batch[b][i]] = 1
			one_hot_samp.append(vec)
		one_hot_samp = np.array(one_hot_samp)
		B_one_hot_samp.append(one_hot_samp)

	return np.array(B_one_hot_samp)






