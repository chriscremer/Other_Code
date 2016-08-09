




import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



##########################################################
#Make fake data for setting up the model

#100 protein samples with 10 amino acids each.
#Amino acids are 1-21 and the gap is 0
n_samps = 500
L = 4
n_aa = 10
X = np.random.randint(1,n_aa,size=(n_samps,L))
for i in range(len(X)):
	if X[i][1] > n_aa/2:
		X[i][0] = 2
	else:
		X[i][0] = 1
# for i in range(len(X)):
# 	if X[i][6] > 10:
# 		X[i][6] = 5
# 	X[i][7] = X[i][6] *2
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

#Define placeholders 
# input_aa = tf.placeholder("float", shape=[L-1, n_aa, 1])
input_tensor = tf.placeholder("float", shape=[L, L, n_aa, 1])

# expected_aa = tf.placeholder("float", shape=[n_aa,1])
target = tf.placeholder("float", shape=[L, n_aa, 1])


#Define all variables
# w = tf.Variable(tf.random_normal([L-1, n_aa, n_aa], stddev=0.01))
# b = tf.Variable(tf.random_normal([L-1, n_aa], stddev=0.01))

w = tf.Variable(tf.random_normal([L, L, n_aa, n_aa], stddev=0.01))
b = tf.Variable(tf.random_normal([L, L, n_aa], stddev=0.01))

D = 2

# this_aa_position = tf.Variable(tf.random_normal([1], stddev=0.01))
# other_aa_positions = tf.Variable(tf.random_normal([L-1], stddev=0.01))
# column_1D_positions = tf.Variable(tf.random_normal([L, 1], stddev=0.01))
# column_3D_positions = tf.Variable(tf.random_normal([3, L, 1], stddev=0.01))
column_2D_positions = tf.Variable(tf.random_normal([D, L, 1], stddev=0.01))




#Define model
def model(input1, gating_network):

	# return tf.nn.softmax(tf.matmul(tf.transpose(gating_network), (tf.reshape(tf.batch_matmul(w, input_aa), [L-1, n_aa]) + b)))
	input_times_w = tf.reshape(tf.batch_matmul(w, input1), [L, L, n_aa])
	input_times_w_plus_b = input_times_w + b
	activation_function = tf.nn.relu(input_times_w_plus_b)
	# activation_function = tf.sigmoid(input_times_w_plus_b)
	use_gate = tf.batch_matmul(tf.transpose(activation_function, perm=[0, 2, 1]), tf.transpose(gating_network, perm=[0,1,2])) #perm=[1,0,2]
	softmax_output = tf.nn.softmax(tf.reshape(use_gate, [L, n_aa]))
	return softmax_output



# gating_network = tf.reshape(tf.exp(-tf.square(other_aa_positions - this_aa_position)) / tf.reduce_sum(tf.exp(-tf.square(other_aa_positions - this_aa_position))), [L-1,1])
def gate():

	L_by_L = tf.batch_matmul(column_2D_positions, tf.transpose(tf.ones_like(column_2D_positions), perm=[0, 2, 1]))
	dif_matrix = L_by_L - tf.transpose(L_by_L, perm=[0, 2, 1]) + tf.reshape(tf.concat(0,[tf.diag(tf.ones([L]))*100, tf.diag(tf.ones([L]))*100]),[D,L,L])
	square = tf.square(dif_matrix)
	dist_matrix = tf.reduce_sum(square,0)
	negative = -dist_matrix
	# exp = tf.exp(negative)
	sm = tf.nn.softmax(negative)
	gating = tf.reshape(sm, [L, L, 1])
	return gating

gating_network = gate()

# aaa = tf.reshape(tf.concat(0,[tf.diag(tf.ones([L]))*10, tf.diag(tf.ones([L]))*10]),[D,L,L])

prediction = model(input_tensor, gating_network)

#Sum over columns (L), then over amino acid predictions (21) for each column
#Prediction and target are L by 21 matrices
cross_entropy = -tf.reduce_sum(tf.reshape(target, [L, n_aa])*tf.log(prediction))

train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
# train_op = tf.train.MomentumOptimizer(0.2, .01).minimize(cross_entropy)
# train_op = tf.train.AdagradOptimizer(0.4).minimize(cross_entropy)

aaa = tf.shape(target)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
	print
	print  'Iter ' + str(i)

	this_iter_train_error = []
	this_iter_valid_error = []

	#For every sample
	for samp in range(len(X)):

		if samp < 400:



			_, cost, pred, gate = sess.run([train_op, cross_entropy, prediction, gating_network], 
				feed_dict={input_tensor: convert_samp_to_L_by_L(X[samp]), target: convert_samp_to_one_hot(X[samp])})

			# print sess.run(aaa, feed_dict={input_tensor: convert_samp_to_L_by_L(X[samp]), target: convert_samp_to_one_hot(X[samp])})
			# dsfaf

			this_iter_train_error.append(cost)
			if samp == 0:
				# print 'Input'
				# print X[samp]
				# print np.reshape(convert_samp_to_L_by_L(X[samp]), (L,L,n_aa)).shape
				# print np.reshape(convert_samp_to_L_by_L(X[samp]), (L,L,n_aa))[0]
				# adfsa

				print 'Expected'
				print np.reshape(convert_samp_to_one_hot(X[samp]), (L,n_aa)).T
				print 'Predicted'
				print pred.T
				print 'Gating'
				print gate


		else:
			valid_error = sess.run(cross_entropy, feed_dict={input_tensor: convert_samp_to_L_by_L(X[samp]), target: convert_samp_to_one_hot(X[samp])})

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

		# Restore variables from disk.
		saver.restore(sess, "/Users/Chris/Code/temp/model.ckpt")
		print("Model restored.")
		final_positions = sess.run(column_2D_positions)
		gating = sess.run(gating_network)
		break

	if i%10==0 and (best_error > avg_valid_error or best_error == -1):
		best_error = avg_valid_error
		save_path = saver.save(sess, "/Users/Chris/Code/temp/model.ckpt")
		print("Model saved in file: %s" % save_path)

	if best_error > avg_valid_error or best_error == -1:
		best_error = avg_valid_error

	previous_error = avg_valid_error


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








