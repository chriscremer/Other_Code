




import tensorflow as tf
import numpy as np




##########################################################
#Make fake data for setting up the model

#100 protein samples with 10 amino acids each.
#Amino acids are 1-21 and the gap is 0
n_samps = 100
len_protein = 10
X = np.random.randint(0,22,size=(n_samps,len_protein))
for i in range(len(X)):
	if X[i][0] > 10:
		X[i][1] = 2
	else:
		X[i][1] = 1
for i in range(len(X)):
	if X[i][1] == 1:
		X[i][3] = 5
	else:
		X[i][3] = 6
for i in range(len(X)):
	if X[i][4] > 10:
		X[i][5] = 2
	else:
		X[i][5] = 1
for i in range(len(X)):
	if X[i][6] > 10:
		X[i][7] = 2
	else:
		X[i][7] = 1
for i in range(len(X)):
	if X[i][8] > 10:
		X[i][9] = 2
	else:
		X[i][9] = 1


def convert_to_one_hot(aa):
	vec = np.zeros((22,1))
	vec[aa] = 1
	return vec.T[0]

def convert_samp_to_one_hot(samp, minus_this_aa):
	one_hot_samp = []
	for i in range(len(samp)):
		if i == minus_this_aa:
			continue
		one_hot_samp.append(convert_to_one_hot(samp[i]))
	return np.array(one_hot_samp)

# def convert_X_to_one_hot(X):
# 	newX = []
# 	for i in range(len(X)):
# 		samp = []
# 		for aa in X[i]:
# 			samp.append(convert_to_one_hot(aa))
# 		newX.append(samp)
# 	return np.array(newX)

# def make_matrix_replicate(samp):

# 	print samp.shape

# 	for i in range(len_protein-1):
# 		concatenate


# make_matrix_replicate(convert_samp_to_one_hot(X[0], 1))
# asfds

##########################################################

#will praobably need a loop here to make sessions for each aa column


#Define placeholders 
input_aa = tf.placeholder("float", shape=[len_protein-1, 22])
expected_aa = tf.placeholder("float", shape=[22])


#Define all variables
w = tf.Variable(tf.random_normal([len_protein-1, 22, 22], stddev=0.01))
b = tf.Variable(tf.random_normal([len_protein-1, 22], stddev=0.01))
gating_network = tf.Variable(tf.random_normal([len_protein-1], stddev=0.01))


# def init_weights(shape):
# 	return tf.Variable(tf.random_normal(shape, stddev=0.01))

# def init_biases(shape):
# 	return tf.Variable(tf.random_normal(shape, stddev=0.01))

# w = []
# b = []
# for i in range(len_protein):
# 	predicting_this_position_w = []
# 	predicting_this_position_b = []
# 	for j in range(len_protein-1):
# 		if j == 0:
# 			continue
# 		predicting_this_position_w.append(init_weights([22, 22]))
# 		predicting_this_position_b.append(init_biases([22]))
# 	w.append(predicting_this_position_w)
# 	b.append(predicting_this_position_b)

# w = init_weights([len_protein, 22, 22])
# b = init_biases([len_protein, 22])

#Gating network
# gating = tf.Variable([tf.softmax(tf.random_normal([len_protein]))])
# gating_list = [tf.Variable(.1) for i in range(len_protein)]
# gating_network = [tf.Variable(tf.random_normal([len_protein-1], stddev=0.01)) for i in range(len_protein)] 



# w = init_weights([22, 22])
# b = init_biases([22])


#Define model

def model(predicting, input_aa, w, gating_network):

	# aa = tf.matmul(gating, )


	# sum1 = tf.constant(tf.zeros([22]))
	# for i in range(len_protein):
	# 	if i == 0:
	# 		continue
	# return input_aa + gating_list[0]
	# return tf.nn.softmax((gating_list[0] * (tf.matmul(input_aa[0], w[0]) + b[0])) + (gating_list[1] * (tf.matmul(input_aa[1], w[1]) + b[1]))) 
	# return (gating_list[0] * (tf.matmul(input_aa[0], w[0]) + b[0])) + (gating_list[1] * (tf.matmul(input_aa[1], w[1]) + b[1]))
	# return (tf.matmul(input_aa[0], w[0]) + b[0]) + (tf.matmul(input_aa[1], w[1]) + b[1])
	# return (tf.matmul(input_aa[0], w[0])) + (tf.matmul(input_aa[1], w[1]))

	# return tf.nn.softmax(gating_list[0] * (tf.matmul(tf.slice(input_aa, [0,0], [1,22])  , w[0]))  +  gating_list[1] * (tf.matmul( tf.slice(input_aa, [1,0], [1,22])  , w[1])))

	l = predicting.eval()

	return tf.nn.softmax(tf.reduce_sum((tf.batch_matmul(input_aa, w[l]) + b[l]) * gating_network[l]), 1)





prediction = model(predicting, input_aa, w, gating_network)

print 'HELLLLO'

cross_entropy = -tf.reduce_sum(expected_aa*tf.log(prediction))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, expected_aa)) 
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)

# log_times = tf.log(prediction)
# print1 = tf.matmul(tf.transpose(input_aa), w)
# print2 = w[0]
# print3 = (gating[0] * (tf.matmul(input_aa[0], w[0]) + b[0])) + (gating[1] * (tf.matmul(input_aa[1], w[1]) + b[1]))

# print4 = gating_list[0]
# print5 = prediction
# print6 = gating_list[0]*prediction

print '@@@@@@@@@@@@'

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


#Predict aa1 using aa0

for i in range(100):
	print
	print i 

	#For every sample
	for samp in range(len(X)):
		#For every column, this the colum were tryin to predict
		for l in range(len(X[samp])):

			_, cost, pred = sess.run([train_op, cross_entropy, prediction], feed_dict={predicting: l, input_aa: convert_samp_to_one_hot(X[samp], l), expected_aa: convert_to_one_hot(X[samp][l])})
	
			if samp == 0 and l == 0:
				# print sess.run(print4, feed_dict={input_aa: convert_samp_to_one_hot(X[samp], l)})
				# print sess.run(print5, feed_dict={input_aa: convert_samp_to_one_hot(X[samp], l)})
				# print sess.run(print6, feed_dict={input_aa: convert_samp_to_one_hot(X[samp], l)})

				print 'Expected ' + str(convert_to_one_hot(X[samp][l]))
				print 'Predicted ' + str(pred)
				print cost


	# print convert_to_one_hot(samp[1])
	# print sess.run(prediction, feed_dict={input_aa: convert_to_one_hot(samp[0])})# - convert_to_one_hot(samp[1])
	# print sess.run(print4, feed_dict={input_aa: convert_samp_to_one_hot(samp, l)})


	# print sess.run(log_times,  feed_dict={input_aa: convert_to_one_hot(samp[0]), expected_aa: convert_to_one_hot(samp[1])})
	# print sess.run(cross_entropy, feed_dict={input_aa: convert_to_one_hot(samp[0]), expected_aa: convert_to_one_hot(samp[1])})
	






