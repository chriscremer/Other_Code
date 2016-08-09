





import tensorflow as tf
import numpy as np




##########################################################
#Make fake data for setting up the model

#100 protein samples with 10 amino acids each.
#Amino acids are 1-21 and the gap is 0
n_samps = 100
len_protein = 10
n_aa = 21
X = np.random.randint(0,n_aa,size=(n_samps,len_protein))
for i in range(len(X)):
	if X[i][1] > 10:
		X[i][0] = 2
	else:
		X[i][0] = 1
# for i in range(len(X)):
# 	if X[i][5] == 1:
# 		X[i][3] = 5
# 	else:
# 		X[i][3] = 6
# for i in range(len(X)):
# 	if X[i][4] > 10:
# 		X[i][5] = 2
# 	else:
# 		X[i][5] = 1
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


def convert_to_one_hot(aa):
	vec = np.zeros((n_aa,1))
	vec[aa] = 1
	# return vec.T[0]
	return vec


def convert_samp_to_one_hot(samp, minus_this_aa):
	one_hot_samp = []
	for i in range(len(samp)):
		if i == minus_this_aa:
			continue
		one_hot_samp.append(convert_to_one_hot(samp[i]))
	# print np.array(one_hot_samp).shape
	# print np.array(one_hot_samp)
	return np.array(one_hot_samp)

##########################################################

for l in range(len_protein):


	#Define placeholders 
	input_aa = tf.placeholder("float", shape=[len_protein-1, n_aa, 1])
	expected_aa = tf.placeholder("float", shape=[n_aa,1])


	#Define all variables
	w = tf.Variable(tf.random_normal([len_protein-1, n_aa, n_aa], stddev=0.01))
	b = tf.Variable(tf.random_normal([len_protein-1, n_aa], stddev=0.01))

	this_aa_position = tf.Variable(tf.random_normal([1], stddev=0.01))
	other_aa_positions = tf.Variable(tf.random_normal([len_protein-1], stddev=0.01))



	#Define model
	def model(input_aa, w, gating_network):

		return tf.nn.softmax(tf.matmul(tf.transpose(gating_network), (tf.reshape(tf.batch_matmul(w, input_aa), [len_protein-1, n_aa]) + b)))


	aaa = this_aa_position
	bbb = other_aa_positions

	# gating_network = tf.reshape(tf.exp(-tf.square(other_aa_positions - this_aa_position)) / tf.reduce_sum(tf.exp(-tf.square(other_aa_positions - this_aa_position))), [len_protein-1,1])
	gating_network = tf.transpose(tf.nn.softmax(tf.transpose(tf.reshape(-tf.square(other_aa_positions - this_aa_position), [len_protein-1,1]))))

	prediction = model(input_aa, w, gating_network)

	cross_entropy = -tf.reduce_sum(expected_aa*tf.transpose(tf.log(prediction)))

	train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	for i in range(1000):
		print
		print 'Position ' + str(l) + ' Iter ' + str(i)

		#For every sample
		for samp in range(len(X)):

			_, cost, pred, gate = sess.run([train_op, cross_entropy, prediction, gating_network], feed_dict={input_aa: convert_samp_to_one_hot(X[samp], l), expected_aa: convert_to_one_hot(X[samp][l])})
	
			if samp == 0:

				print X[samp]
				print 'Expected ' + str(convert_to_one_hot(X[samp][l]).T)
				print 'Predicted ' + str(pred)
				print gate
				print cost
				print sess.run(aaa)
				print sess.run(bbb)


	sess.close()		






