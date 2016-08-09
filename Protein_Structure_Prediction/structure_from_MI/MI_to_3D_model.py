









import numpy as np
import tensorflow as tf
import math
import json
import random





class Structure_Learner():
	'''
	Learn the coordinates of each column
	'''

	def __init__(self, L=None, tol=2., B=20, lr=.01, mom=.9, nu=.1, rho=.01, sigma_repel=.01, sigma_adj=.1, coordinates=[]):

		self.L = L
		# self.n_aa = n_aa
		self.D = 3 #Dimensions of space
		self.tol =  tol
		self.B = B #batch size

		#Hyperparameters
		self.lr = lr #learning rate divided by batch size
		self.mom = mom #momentum
		self.nu = nu #weight of repel all cost
		self.rho = rho #weight decay
		self.sigma_repel = sigma_repel
		self.sigma_adj = sigma_adj
		self.sigma_gate = 1. #start at 10 then go to 1000

		#Parameters
		if coordinates == []:
			self.coordinates = tf.Variable(tf.random_normal([self.D, L, 1], stddev=0.01))
		else:
			coords = np.reshape(coordinates.T, [self.D,L,1])
			coords = np.float32(coords)
			self.coordinates = tf.Variable(coords)


		
	def fit(self, MI):
		'''
		Input should be ints
		'''

		target_ = np.identity(self.L)

		input_tensor = tf.placeholder(tf.float32, shape=[self.L, self.L])
		target = tf.placeholder(tf.float32, shape=[self.L, self.L])

		sigma_gate_holder = tf.placeholder(tf.float32, shape=[])
		# help_predict_holder = tf.placeholder(tf.float32, shape=[])

		gating_network = self.gate(sigma_gate_holder)
		prediction = self.feed_forward(input_tensor, gating_network)

		# cross_entropy = help_predict_holder*(-tf.reduce_sum(tf.reshape(target, [self.L, self.n_aa])*tf.log(prediction)))
		cross_entropy = -tf.reduce_sum(target*tf.log(prediction))
		adj_cost =  self.nu * self.pull_adj()
		repel_cost = self.rho * self.repel_all()

		cost_function = cross_entropy + repel_cost + adj_cost

		train_op = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.mom).minimize(cost_function)

		previous_mean_valid_error = -1
		converged = 0
		ind=0


		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())


			# for samp in range(len(X_train)):
			for iiii in range(999999):


				# gn = sess.run(gating_network, feed_dict={ sigma_gate_holder: self.sigma_gate})
				# print gn.shape

				# ff = sess.run(prediction,
				# 	feed_dict={input_tensor: MI, 
				# 				target: target_, 
				# 				sigma_gate_holder: self.sigma_gate})
				# print ff.shape

				# sdfsdfa



				_, cost, pred, gate, ce, pt, rc = sess.run([train_op, cost_function, prediction, gating_network, cross_entropy, adj_cost, repel_cost], 
					feed_dict={input_tensor: MI, 
								target: target_, 
								sigma_gate_holder: self.sigma_gate})
				print str(iiii) + ' cost: ' + str(cost) + ' CE: ' + str(ce) + ' AC: ' + str(pt) + ' RC: ' + str(rc) +'  ' + str(self.sigma_gate)

				# print pred

				if math.isnan(cost) or math.isnan(pt) or math.isinf(pt):
					print 'WTF IS GOING ON?!?!?!?!?'
					print 'why is it nan??! '
					print 'CE: ' + str(ce) + ' AC: ' + str(pt) + ' RC: ' + str(rc)
					print sess.run(self.coordinates).T
					print 
					break


				if previous_mean_valid_error - cost < self.tol and previous_mean_valid_error != -1 and iiii > 1000:
					print 'Tol reached'
					converged = 1
					break
				previous_mean_valid_error = cost	

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


				self.sigma_gate = (iiii+1.)/10.
				if self.sigma_gate > 100.:
					self.sigma_gate = 100.


			#Save the parameters as numpy rather than tensorflow
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


	def feed_forward(self, input1, gn_):

		# #Step 1: Output of each expert (net) -> LxLx21x21 * LxLx21x1 = LxLx21x1 = LxLx21
		# output = tf.reshape(tf.batch_matmul(experts, input1), [self.L, self.L, self.n_aa]) #+ self.b  

		# # #Step 2: Since the softmax function only takes in a matrix (not tensor), have to do it myself
		# exp_ = tf.exp(output) #numerator
		# output = tf.reshape(tf.reduce_sum(exp_, 2), [self.L,self.L,1]) #sum the outputs for each of the nets
		# sums = tf.tile(output, [1,1,self.n_aa]) #tile (replicate) the sum to each numerator
		# output = exp_ / sums #softmax -> LxLx21

		# #Step 3: Multiply by the gating function -> Lx21xL * LxLx1 = Lx21x1 -> prediction for each column


		#gating is LxL, input is LxL


		gn_ = tf.reshape(gn_, (self.L, self.L))

		output = tf.matmul(input1, gn_)
		# output = tf.matmul(gn_, input1)


		# #Step 4: Reshape to fit other stuff -> Lx21
		# output = tf.reshape(output, [self.L, self.n_aa])

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












