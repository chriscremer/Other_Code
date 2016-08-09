


#Versions before were not using repel all cost, so i added it to the cost


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math



##########################################################
#Make fake data for setting up the model

#100 protein samples with 10 amino acids each.
#Amino acids are 1-21 and the gap is 0
n_samps = 500
L = 20
n_aa = 21
X = np.random.randint(1,n_aa,size=(n_samps,L))
for i in range(len(X)):
	if X[i][9] > n_aa/2:
		X[i][0] = 2
	else:
		X[i][0] = 1
for i in range(len(X)):
	if X[i][3] > 10:
		X[i][18] = 5
	X[i][12] = X[i][6]
# for i in range(len(X)):
# 	if X[i][5] >5:
# 		X[i][3] = 5
# 	else:
# 		X[i][3] = 6
# for i in range(len(X)):
# 	if X[i][2] > 15:
# 		X[i][7] = 10
# 	else:
# 		X[i][7] = 20
# for i in range(len(X)):
# 	if i %2 ==0:
# 		X[i][6] == 10
# 		X[i][4] == 4
# 	else:
# 		X[i][6] == 5
# 		X[i][4] == 1


def convert_to_one_hot(aa):
	vec = np.zeros((n_aa,1))
	vec[aa] = 1
	# return vec.T[0]
	return vec


def convert_samp_to_one_hot(samp):
	one_hot_samp = []
	for i in range(len(samp)):
		one_hot_samp.append(convert_to_one_hot(samp[i]))
	return np.array(one_hot_samp)


def convert_samp_to_L_by_L(samp):
	#its really L x L x 21
	#the target position becomes 0
	L_by_L = []
	for i in range(len(samp)):
		this_samp = []
		for j in range(len(samp)):
			if j == i:
				this_samp.append(convert_to_one_hot(0))
			else:
				this_samp.append(convert_to_one_hot(samp[j]))
		L_by_L.append(this_samp)

	return np.array(L_by_L)


# print convert_samp_to_L_by_L([2,3,1]).T
# fsdfa

training_error = []
validation_error = []
best_error = -1
n_increasing_valid_errors = 0
previous_error = -1

##########################################################

D = 2 #Dimensions
nu = .1 #repel all
lmbd = nu * (L-1) #adj
rho = .1/(L*n_aa)  #.01 #weight decay

#Define placeholders 
input_tensor = tf.placeholder("float", shape=[L, L, n_aa, 1])
target = tf.placeholder("float", shape=[L, n_aa, 1])

#Define all variables
w = tf.Variable(tf.random_normal([L, L, n_aa, n_aa], stddev=0.01))
b = tf.Variable(tf.random_normal([L, L, n_aa], stddev=0.01))
# column_1D_positions = tf.Variable(tf.random_normal([L, 1], stddev=0.01))
# column_3D_positions = tf.Variable(tf.random_normal([3, L, 1], stddev=0.01))
column_2D_positions = tf.Variable(tf.random_normal([D, L, 1], stddev=0.01))




#Define model
def model(input1, gating_network):

	input_times_w_plus_b = tf.reshape(tf.batch_matmul(w, input1), [L, L, n_aa]) + b
	#the softmax takes in a matrix have to do it myself
	exp_ = tf.exp(input_times_w_plus_b)
	sums = tf.reshape(tf.reduce_sum(exp_, 2), [L,L,1])
	try1 = tf.tile(sums, [1,1,n_aa])
	activation_function = exp_ / try1
	# activation_function = tf.sigmoid(input_times_w_plus_b)
	# activation_function = tf.nn.relu(softmaxed)
	use_gate = tf.batch_matmul(tf.transpose(activation_function, perm=[0, 2, 1]), tf.transpose(gating_network, perm=[0,1,2])) #perm=[1,0,2]
	# output = tf.nn.softmax(tf.reshape(use_gate, [L, n_aa]))
	output = tf.reshape(use_gate, [L, n_aa])
	return output


def gate():

	L_by_L = tf.batch_matmul(column_2D_positions, tf.transpose(tf.ones_like(column_2D_positions), perm=[0, 2, 1]))
	#for the next line, if D changes need to chnage number of:   tf.diag(tf.ones([L]))*100, one for every dimension
	dif_matrix = L_by_L - tf.transpose(L_by_L, perm=[0, 2, 1]) + tf.reshape(tf.concat(0,[tf.diag(tf.ones([L]))*100, tf.diag(tf.ones([L]))*100]),[D,L,L])
	square = tf.square(dif_matrix)
	dist_matrix = tf.reduce_sum(square,0)
	negative = -dist_matrix
	# exp = tf.exp(negative)
	sm = tf.nn.softmax(negative)
	gating = tf.reshape(sm, [L, L, 1])
	return gating



def pull_adj():

	range1 = tf.range(1,L)
	range2 = tf.range(0,L-1)

	first_D0 = tf.gather(tf.reshape(tf.gather(tf.reshape(column_2D_positions, [D, L]), [0]), [L]), range1)
	first_D1 = tf.gather(tf.reshape(tf.gather(tf.reshape(column_2D_positions, [D, L]), [1]), [L]), range1)

	second_D0 = tf.gather(tf.reshape(tf.gather(tf.reshape(column_2D_positions, [D, L]), [0]), [L]), range2)
	second_D1 = tf.gather(tf.reshape(tf.gather(tf.reshape(column_2D_positions, [D, L]), [1]), [L]), range2)

	sum1 = tf.reduce_sum(tf.exp(tf.square(first_D0 - second_D0) + (tf.square(first_D1 - second_D1))))

	return sum1


def repel_all():

	L_by_L = tf.batch_matmul(column_2D_positions, tf.transpose(tf.ones_like(column_2D_positions), perm=[0, 2, 1]))
	dif_matrix = L_by_L - tf.transpose(L_by_L, perm=[0, 2, 1])
	square = tf.square(dif_matrix)
	dist_matrix = tf.exp(tf.reduce_sum(square,0))
	#the negative means ill be minimizing the negative distance, which is maximizing the distance betw points
	#Maybe negative doesnt work (nan) so ill try recipricol
	total = 1./tf.reduce_sum(dist_matrix)

	return total


gating_network = gate()
prediction = model(input_tensor, gating_network)

cross_entropy = -tf.reduce_sum(tf.reshape(target, [L, n_aa])*tf.log(prediction))
weight_decay = rho * tf.reduce_sum(tf.square(w))
adj_cost =  lmbd * pull_adj()
repel_cost = nu * repel_all()

cost_function = cross_entropy + weight_decay + adj_cost + repel_cost

train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost_function)
# train_op = tf.train.MomentumOptimizer(0.2, .01).minimize(cross_entropy)
# train_op = tf.train.AdagradOptimizer(0.4).minimize(cross_entropy)


#FOR PRINTING
range1 = tf.range(1,L)
range2 = tf.range(0,L-1)
first_D0 = tf.gather(tf.reshape(tf.gather(tf.reshape(column_2D_positions, [D, L]), [0]), [L]), range1)
first_D1 = tf.gather(tf.reshape(tf.gather(tf.reshape(column_2D_positions, [D, L]), [1]), [L]), range1)
second_D0 = tf.gather(tf.reshape(tf.gather(tf.reshape(column_2D_positions, [D, L]), [0]), [L]), range2)
second_D1 = tf.gather(tf.reshape(tf.gather(tf.reshape(column_2D_positions, [D, L]), [1]), [L]), range2)
adj_dist = tf.exp(tf.square(first_D0 - second_D0) + (tf.square(first_D1 - second_D1)))

#FOR DEBUGGING
#LxLx21
# input_times_w_plus_b = tf.exp(tf.reshape(tf.batch_matmul(w, input_tensor), [L, L, n_aa]) + b)
# #LxL
# # sums = tf.reduce_sum(input_times_w_plus_b, 2)
# sums = tf.reshape(tf.reduce_sum(input_times_w_plus_b, 2), [L,L,1])
# try1 = tf.tile(sums, [1,1,n_aa])
# yess = input_times_w_plus_b / try1

#do one over sums then make sums the same size as input then *
#making it the same size isnt easy
#use tile
#so ill do my own softmax
#so ill exp everythin, dividing will be the tricky part
#so i have a LxLx21, I want the sum of the Lx21s ill have L of them
# sums = tf.reduce_sum(input_times_w_plus_b, )
# activation_function = tf.nn.softmax(tf.reshape(input_times_w_plus_b), [])
# input_times_w_plus_b = tf.exp(tf.reshape(tf.batch_matmul(w, input_tensor), [L, L, n_aa]) + b)


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(50):
	print
	print  'Iter ' + str(i)

	this_iter_train_error = []
	this_iter_valid_error = []

	#For every sample
	for samp in range(len(X)):

		if samp < (n_samps*.7):
			# test = sess.run(input_times_w_plus_b, feed_dict={input_tensor: convert_samp_to_L_by_L(X[samp]), target: convert_samp_to_one_hot(X[samp])})

			# test2 = sess.run(sums, feed_dict={input_tensor: convert_samp_to_L_by_L(X[samp]), target: convert_samp_to_one_hot(X[samp])})
			# print test
			# print test.shape
			# print test2
			# print test2.shape

			# test3 = sess.run(try1, feed_dict={input_tensor: convert_samp_to_L_by_L(X[samp]), target: convert_samp_to_one_hot(X[samp])})
			# print test3
			# print test3.shape

			# test4 = sess.run(yess, feed_dict={input_tensor: convert_samp_to_L_by_L(X[samp]), target: convert_samp_to_one_hot(X[samp])})
			# print test4
			# print test4.shape
			# asdfald;aksjdfl;

			_, cost, pred, gate, ce, wd, pt, rc, ad = sess.run([train_op, cost_function, prediction, gating_network, cross_entropy, weight_decay, adj_cost, repel_cost, adj_dist], 
				feed_dict={input_tensor: convert_samp_to_L_by_L(X[samp]), target: convert_samp_to_one_hot(X[samp])})

			# if math.isnan(cost):
			# 	print 'got it '
			# 	afsfds


			this_iter_train_error.append(cost)
			if samp == 0:
				# print 'Input'
				# print X[samp]
				# print np.reshape(convert_samp_to_L_by_L(X[samp]), (L,L,n_aa)).shape
				# print np.reshape(convert_samp_to_L_by_L(X[samp]), (L,L,n_aa))[0]
				# adfsa

				# print 'Expected'
				# print np.reshape(convert_samp_to_one_hot(X[samp]), (L,n_aa)).T
				# print 'Predicted'
				# print pred.T
				# print 'Gating'
				# print gate
				print 'CE: ' + str(ce) + ' WD: ' + str(wd) + ' AC: ' + str(pt) + ' RC: ' + str(rc)
				print 'Adj Dist: ' + str(ad)


		else:
			valid_error = sess.run(cost_function, feed_dict={input_tensor: convert_samp_to_L_by_L(X[samp]), target: convert_samp_to_one_hot(X[samp])})

			this_iter_valid_error.append(valid_error)


	#Saving and Printing status
	avg_train_error = np.mean(this_iter_train_error)
	avg_valid_error = np.mean(this_iter_valid_error)

	training_error.append(avg_train_error)
	validation_error.append(avg_valid_error)
	print 'Training Error ' + str(avg_train_error) + '  Validation Error ' + str(avg_valid_error)

	if best_error <= avg_valid_error:
		n_increasing_valid_errors += 1
		print 'Consecutive increases = ' + str(n_increasing_valid_errors)
	else:
		n_increasing_valid_errors = 0

	if n_increasing_valid_errors >= 5:
		print 'Early Stopping'
		break

	if i%5==0 and (best_error > avg_valid_error or best_error == -1):
		best_error = avg_valid_error
		save_path = saver.save(sess, "/Users/Chris/Code/temp/model.ckpt")
		print("Model saved in file: %s" % save_path)

	if best_error > avg_valid_error or best_error == -1:
		best_error = avg_valid_error

	previous_error = avg_valid_error

# Restore variables from disk.
saver.restore(sess, "/Users/Chris/Code/temp/model.ckpt")
print("Model restored.")
final_positions = sess.run(column_2D_positions)
gating = sess.run(gating_network)		

sess.close()

print 'Final Positions '
for i in range(len(final_positions.T[0])):
	print str(i) + ' ' + str(final_positions.T[0][i])

print 'Gating Network'
for i in range(len(gating)):
	print str(i)
	for j in range(len(gating[i])):
		print str(j) + ' ' + str(gating[i][j])

print 'Data'
print X[:10]

#Plot
plt.scatter(final_positions[0], final_positions[1])
for i in range(len(final_positions[0])):
	plt.annotate(str(i), xy = (final_positions[0][i], final_positions[1][i]))
plt.show()








