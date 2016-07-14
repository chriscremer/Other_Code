


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from os.path import expanduser
home = expanduser("~")
from load_mnist import load_mnist


import cPickle, gzip, numpy

# Load the dataset
f = gzip.open(home + '/Documents/MNIST_data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


def class_counter(labels):
	'''
	Count the number of each class
	'''
	n_classes = [0]*10

	if len(labels.T) == 1:
		for i in range(len(labels)):
			n_classes[labels[i]] += 1
	else:
		for i in range(len(labels)):
			n_classes[np.argmax(labels[i])] += 1
	return n_classes



def convert_array_to_one_hot_matrix(array1):
	one_hot_matrix = []
	for i in range(len(array1)):
		vec = np.zeros((10,1))
		vec[array1[i]] = 1
		one_hot_matrix.append(vec)
	one_hot_matrix = np.array(one_hot_matrix)
	return np.reshape(one_hot_matrix, [one_hot_matrix.shape[0], one_hot_matrix.shape[1]]) 

def next_batch(X, y, batch_size,):
	'''
	For now, just return some random samples
	'''
	indexes = random.sample(range(len(y)), batch_size)
	x_batch = X[indexes]
	y_batch = y[indexes]

	return [x_batch, y_batch]


def next_batch2(X, y, batch_size, index):
	'''
	For now, just return some random samples
	'''

	count = 0
	x_batch = []
	y_batch = []
	while count < batch_size:

		x_batch.append(X[index])
		y_batch.append(y[index])

		index += 1
		count += 1 
		if index == len(y):
			index = 0

	# indexes = random.sample(range(len(y)), batch_size)
	# x_batch = X[indexes]
	# y_batch = y[indexes]

	x_batch = np.array(x_batch)
	y_batch = np.array(y_batch)

	return [x_batch, y_batch], index


def bias_sampling(X, y):
	temp_X = []
	temp_y = []
	#store the class 0 samples then add them to X
	for i in range(len(X)):
		if np.argmax(y[i]) == 0:
			temp_X.append(X[i])
			temp_y.append(y[i])
	temp_X = np.array(temp_X)
	temp_y = np.array(temp_y)
	for i in range(20):
		X = np.concatenate((X, temp_X), axis=0)
		y = np.concatenate((y, temp_y), axis=0)
	return X, y


def bias_sampling2(X, y):

	n_classes = [0]*10

	temp_X = []
	temp_y = []
	#store the class 0 samples then add them to X
	for i in range(len(X)):
		if n_classes[np.argmax(y[i])] < 250 or np.argmax(y[i]) in [0,1,2,3,4]:
			temp_X.append(X[i])
			temp_y.append(y[i])
			n_classes[np.argmax(y[i])] += 1
	temp_X = np.array(temp_X)
	temp_y = np.array(temp_y)
	# for i in range(20):
	# 	X = np.concatenate((X, temp_X), axis=0)
	# 	y = np.concatenate((y, temp_y), axis=0)
	return temp_X, temp_y


def calc_class_weights(labels):

	counts = class_counter(labels)
	mean = np.mean(counts)

	weights = [0]*10
	for i in range(len(weights)):
		weights[i] = mean / counts[i]

	return weights






train_x = train_set[0]
train_y = convert_array_to_one_hot_matrix(train_set[1])
valid_x = valid_set[0]
valid_y = convert_array_to_one_hot_matrix(valid_set[1])
test_x = test_set[0]
test_y = convert_array_to_one_hot_matrix(test_set[1])

print 'Training set ' +  str(train_x.shape)
print class_counter(train_y)
print
print 'Validation set ' +  str(valid_x.shape)
print class_counter(valid_y)
print
print 'Test set ' +  str(test_x.shape)
print class_counter(test_y) 
print

print 'Bias Sampling'
train_x, train_y = bias_sampling2(train_x, train_y)
print 'Training set ' +  str(train_x.shape)
print class_counter(train_y)
print

#Shuffle data
indexes = random.sample(range(len(train_y)), len(train_y))
train_x = train_x[indexes]
train_y = train_y[indexes]

# Parameters
n_batch = 1
lr = .0001
mom = .9
tol = .1

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 100 # 1st layer num features
n_hidden_2 = 100 # 2nd layer num features
n_classes = 10 # MNIST total classes (0-9 digits)


class_weights = calc_class_weights(train_y)
print class_weights

indexes = random.sample(range(len(valid_y)), 50)
background_samps = train_x[indexes]
background_target = train_y[indexes]

# # Create model
# def model(_X, _weights, _biases):
# 	#Hidden layer with RELU activation
# 	layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
# 	#Hidden layer with RELU activation
# 	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) 
# 	return tf.nn.softmax(tf.matmul(layer_2, weights['out']) + biases['out'])
	
# #Define variables
# weights = 	{
# 				'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
# 				'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
# 				'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
# 			}
# biases = 	{
# 				'b1': tf.Variable(tf.random_normal([n_hidden_1])),
# 				'b2': tf.Variable(tf.random_normal([n_hidden_2])),
# 				'out': tf.Variable(tf.random_normal([n_classes]))
# 			}

def background_model(_X, background_samps, background_target):

	#get distances to x
	#sum them weighted on distance
	dists = []

	for i in range(len(background_samps)):

		dists.append(sum(((_X - background_samps[i])**2)[0]))



	# print dists

	# dist_sum = sum(dists)

	# print (dists/dist_sum)

	# gdsffdsg

	# pred = np.zeros((10,1))

	# for i in range(len(background_samps)):
	# 	pred = pred + ((dists[i]/dist_sum)*background_target[i])

	# return pred

	return [background_target[np.argmin(dists)]]



# Create model2
def model(_X, _weights, _biases):
	#Hidden layer with RELU activation
	layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
	return tf.nn.softmax(tf.matmul(layer_1, weights['out']) + biases['out'])
	
#Define variables
weights = 	{
				'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
				'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
			}
biases = 	{
				'b1': tf.Variable(tf.random_normal([n_hidden_1])),
				'out': tf.Variable(tf.random_normal([n_classes]))
			}


x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
class_weight = tf.placeholder("float", shape=[])
B = tf.placeholder("float", shape=[None,10])

# W = tf.Variable(tf.random_normal([784,10], stddev=0.01))
# b = tf.Variable(tf.random_normal([10], stddev=0.01))
# y = tf.nn.softmax(tf.matmul(x,W) + b)

#background and foreground model
F = model(x,weights,biases)
P = (.5*F) + (.5*B)
cross_entropy_P = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(P,1e-10,1.0)))
cross_entropy_F = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(F,1e-10,1.0)))

#Regular model
# y = model(x,weights,biases)
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
# opt = tf.train.GradientDescentOptimizer(lr)
opt = tf.train.MomentumOptimizer(lr, mom)
grads_and_vars = opt.compute_gradients(cross_entropy_P)

# grads_and_vars = opt.compute_gradients(cross_entropy)
modified_grad_and_vars = [(gv[0], gv[1]) for gv in grads_and_vars]
# modified_grad_and_vars2 = [(class_weight*gv[0], gv[1]) for gv in grads_and_vars]
apply_grad = opt.apply_gradients(modified_grad_and_vars)

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
correct_prediction = tf.equal(tf.argmax(P,1), tf.argmax(y_,1))
correct_prediction_F = tf.equal(tf.argmax(F,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction_F, tf.float32))

early_stop_count = 0
prev_valid_ce = -1
best_valid_ce = -1

batch_index = 0
ind = 0

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print 'Begin Training'
for i in range(999999):


	# for j in range(len(train_y)):
	# print 'Iter ' + str(i)

	# batch, batch_index = next_batch2(train_x, train_y, n_batch, batch_index)

	if ind == len(train_y):
		ind = 0
	batch = [[train_x[ind]], [train_y[ind]]]

	#class of this samp
	# class1 = np.argmax(train_y[ind])


	ind +=1



	# if i % 50000 == 0:

		# print 'batch class distribution'
		# print class_counter(batch[1])
		
		# print 'W'
		# print sess.run([debug], feed_dict={x: batch[0], y_: batch1[0]})[0][402]
		# print sess.run([debug], feed_dict={x: batch[0], y_: batch1[0]})[0].shape

		# print class1
		# print class_weights[class1]

		# print 'gradient'
		# gs_vs = sess.run([grad for grad, _ in modified_grad_and_vars], feed_dict={x: batch[0], y_: batch[1], class_weight: class_weights[class1]})
		# print gs_vs[0].shape
		# print gs_vs[1].shape
		# print gs_vs[0][0]
		# print gs_vs[0][402]
		# # print gs_vs[1]

		# gs_vs = sess.run([grad for grad, _ in modified_grad_and_vars2], feed_dict={x: batch[0], y_: batch[1], class_weight: class_weights[class1]})
		# print gs_vs[0].shape
		# print gs_vs[1].shape
		# print gs_vs[0][0]
		# print gs_vs[0][402]

		# fasdad

		# print 'y'
		# print sess.run([y], feed_dict={x: train_x, y_: train_y})[0][0]
		# print 'y_'
		# print train_y[0]

		# print 'train ce'
		# print sess.run([cross_entropy], feed_dict={x: train_x, y_: train_y})
		# print 'valid ce'
		# print sess.run([cross_entropy], feed_dict={x: valid_x, y_: valid_y})[0]

		# print 'train acc'
		# print sess.run([accuracy], feed_dict={x: train_x, y_: train_y})
		# print class_counter(sess.run([y], feed_dict={x: train_x, y_: train_y})[0])
		# print 'valid acc'
		# print sess.run([accuracy], feed_dict={x: valid_x, y_: valid_y})
		# print class_counter(sess.run([y], feed_dict={x: valid_x, y_: valid_y})[0])


	#check for convergence
	if i % 10000 == 0:

		valid_ce = sess.run([cross_entropy_F], feed_dict={x: valid_x, y_: valid_y})[0]
		print i, valid_ce
		if best_valid_ce - valid_ce < tol and best_valid_ce != -1:
			early_stop_count += 1
			print 'early stop count ' + str(early_stop_count)
			if early_stop_count == 3:
				print 'Converged'
				print 'train acc'
				print sess.run([accuracy], feed_dict={x: train_x, y_: train_y})
				print class_counter(sess.run([F], feed_dict={x: train_x, y_: train_y})[0])
				print 'valid acc'
				print sess.run([accuracy], feed_dict={x: valid_x, y_: valid_y})
				print class_counter(sess.run([F], feed_dict={x: valid_x, y_: valid_y})[0])
				print 'test acc'
				print sess.run([accuracy], feed_dict={x: test_x, y_: test_y})
				print class_counter(sess.run([F], feed_dict={x: test_x, y_: test_y})[0])
				break
		else:
			early_stop_count = 0

		if best_valid_ce > valid_ce or best_valid_ce == -1:
			best_valid_ce = valid_ce

		# prev_valid_ce = valid_ce

		#make backgroudn predcitions for next 10000
		B_preds = []
		print 'beginning B preds'
		for j in range(10000):
			B_preds.append(background_model([train_x[j]], background_samps, background_target)) 

		#new sample
		indexes = random.sample(range(len(valid_y)), 50)
		background_samps = train_x[indexes]
		background_target = train_y[indexes]

	_ = sess.run([apply_grad], feed_dict={x: batch[0], y_: batch[1], B: B_preds[i%10000]})


print 'All done'


