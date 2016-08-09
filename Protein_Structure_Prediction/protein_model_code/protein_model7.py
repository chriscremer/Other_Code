

import tensorflow as tf
import numpy as np




##########################################################
#Make fake data for setting up the model

#100 protein samples with 10 amino acids each.
#Amino acids are 1-21 and the gap is 0
n_samps = 500
len_protein = 8
n_aa = 21
X = np.random.randint(1,n_aa,size=(n_samps,len_protein))
for i in range(len(X)):
	if X[i][1] > 10:
		X[i][0] = 2
	else:
		X[i][0] = 1
for i in range(len(X)):
	if X[i][5] >5:
		X[i][3] = 5
	else:
		X[i][3] = 6
for i in range(len(X)):
	if X[i][2] > 15:
		X[i][7] = 10
	else:
		X[i][7] = 20
# for i in range(len(X)):
# 	if X[i][6] > 10:
# 		X[i][7] = 2
# 	else:
# 		X[i][7] = 1
# for i in range(len(X)):
# 	if X[i][8] > 10:
# 		X[i][9] = 2
# 	else:
# 		X[i][9] = 1

# print X[:5]

# afdsaf

def convert_to_one_hot(aa):
	vec = np.zeros((n_aa,1))
	vec[aa] = 1
	# return vec.T[0]
	return vec

# def convert_samp_to_one_hot(samp, minus_this_aa):
# 	one_hot_samp = []
# 	for i in range(len(samp)):
# 		if i == minus_this_aa:
# 			continue
# 		one_hot_samp.append(convert_to_one_hot(samp[i]))
# 	# print np.array(one_hot_samp).shape
# 	# print np.array(one_hot_samp)
# 	return np.array(one_hot_samp)

def convert_samp_to_one_hot(samp):
	one_hot_samp = []
	for i in range(len(samp)):
		one_hot_samp.append(convert_to_one_hot(samp[i]))
	return np.array(one_hot_samp)

# def convert_samp_to_L_by_L_minus_1(samp):
# 	#Ill start by making it a long vector
# 	#len will be L * L-1 * 21
# 	long_vector = []
# 	for i in range(len(samp)):
# 		for j in range(len(samp)):
# 			if j == i:
# 				continue
# 			long_vector.extend(convert_to_one_hot(samp[i]))

# 	# aaa = np.array(long_vector)
# 	# print aaa.shape

# 	return np.array(long_vector)

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
				this_samp.append(convert_to_one_hot(samp[i]))
		L_by_L.append(this_samp)

	# aaa = np.array(long_vector)
	# print aaa.shape

	return np.array(L_by_L)

# convert_samp_to_L_by_L_minus_1([2,3,1,3])


training_error = []
validation_error = []

##########################################################

#Define placeholders 
# input_aa = tf.placeholder("float", shape=[len_protein-1, n_aa, 1])
input_tensor = tf.placeholder("float", shape=[len_protein, len_protein, n_aa, 1])

# expected_aa = tf.placeholder("float", shape=[n_aa,1])
target = tf.placeholder("float", shape=[len_protein, n_aa, 1])


#Define all variables
# w = tf.Variable(tf.random_normal([len_protein-1, n_aa, n_aa], stddev=0.01))
# b = tf.Variable(tf.random_normal([len_protein-1, n_aa], stddev=0.01))

w = tf.Variable(tf.random_normal([len_protein, len_protein, n_aa, n_aa], stddev=0.01))
b = tf.Variable(tf.random_normal([len_protein, len_protein, n_aa], stddev=0.01))

# this_aa_position = tf.Variable(tf.random_normal([1], stddev=0.01))
# other_aa_positions = tf.Variable(tf.random_normal([len_protein-1], stddev=0.01))
# column_1D_positions = tf.Variable(tf.random_normal([len_protein, 1], stddev=0.01))
column_3D_positions = tf.Variable(tf.random_normal([3, len_protein, 1], stddev=0.01))




#Define model
def model(input1, gating_network):

	# return tf.nn.softmax(tf.matmul(tf.transpose(gating_network), (tf.reshape(tf.batch_matmul(w, input_aa), [len_protein-1, n_aa]) + b)))


	input_times_w = tf.reshape(tf.batch_matmul(w, input1), [len_protein, len_protein, n_aa])
	input_times_w_plus_b = input_times_w + b
	activation_function = tf.nn.relu(input_times_w_plus_b)
	# activation_function = tf.sigmoid(input_times_w_plus_b)
	use_gate = tf.batch_matmul(tf.transpose(activation_function, perm=[0, 2, 1]), gating_network)
	softmax_output = tf.nn.softmax(tf.reshape(use_gate, [len_protein, n_aa]))
	return softmax_output



# gating_network = tf.reshape(tf.exp(-tf.square(other_aa_positions - this_aa_position)) / tf.reduce_sum(tf.exp(-tf.square(other_aa_positions - this_aa_position))), [len_protein-1,1])
def gate():
	L_by_L = tf.batch_matmul(column_3D_positions, tf.transpose(tf.ones_like(column_3D_positions), perm=[0, 2, 1]))
	dif_matrix = L_by_L - tf.transpose(L_by_L, perm=[0, 2, 1])
	square = tf.square(dif_matrix)
	dist_matrix = tf.reduce_sum(square,0)
	negative = -dist_matrix
	# exp = tf.exp(negative)
	sm = tf.nn.softmax(negative)
	gating = tf.reshape(sm, [len_protein, len_protein, 1])
	return gating

gating_network = gate()


prediction = model(input_tensor, gating_network)

#Sum over columns (L), then over amino acid predictions (21) for each column
#Prediction and target are L by 21 matrices
cross_entropy = -tf.reduce_sum(target*tf.transpose(tf.log(prediction)))

# train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)
# train_op = tf.train.MomentumOptimizer(0.001, .0001).minimize(cross_entropy)
train_op = tf.train.AdagradOptimizer(0.1).minimize(cross_entropy)


aaa = tf.transpose(tf.reshape(column_3D_positions, [3,len_protein]))
bbb = column_3D_positions
# aaa = tf.transpose(column_1D_positions)
# bbb = tf.shape(tf.reshape(tf.batch_matmul(w, input_tensor), [len_protein, len_protein, n_aa]))
# ccc = tf.shape(gating_network)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
	print
	print  'Iter ' + str(i)

	this_iter_train_error = []
	this_iter_valid_error = []

	#For every sample
	for samp in range(len(X)):

		# print sess.run(aaa,feed_dict={input_tensor: convert_samp_to_L_by_L(X[samp])})

		if samp < 400:
			_, cost, pred, gate = sess.run([train_op, cross_entropy, prediction, gating_network], 
				feed_dict={input_tensor: convert_samp_to_L_by_L(X[samp]), target: convert_samp_to_one_hot(X[samp])})

			this_iter_train_error.append(cost)

		else:
			valid_error = sess.run(cross_entropy, feed_dict={input_tensor: convert_samp_to_L_by_L(X[samp]), target: convert_samp_to_one_hot(X[samp])})

			this_iter_valid_error.append(valid_error)

	training_error.append(np.mean(this_iter_train_error))
	validation_error.append(np.mean(this_iter_valid_error))
	print 'Training Error ' + str(np.mean(this_iter_train_error)) + '  Validation Error ' + str(np.mean(this_iter_valid_error))


	# 	if samp == 0:

	# 		# print X[samp]
	# 		# print 'Expected ' + str(convert_samp_to_one_hot(X[samp]))
	# 		# print 'Predicted ' + str(pred)
	# 		# print gate
	# 		pos = sess.run(aaa)
			
	# 		print 'Dist 0 to 1: ' + str(sum((pos[0] - pos[1])**2))
	# 		print 'Dist 3 to 5: ' + str(sum((pos[3] - pos[5])**2))
	# 		print 'Dist 2 to 7: ' + str(sum((pos[2] - pos[7])**2))
	# 		print 'Dist 4 to 6: ' + str(sum((pos[4] - pos[6])**2))
	# 		print cost

	# if i % 50 == 0:
	# 	print sess.run(bbb)


			
			# print sess.run(bbb)
			# print sess.run(ccc)


sess.close()		









