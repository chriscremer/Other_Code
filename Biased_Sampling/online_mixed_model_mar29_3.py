

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

def next_batch(X, y, batch_size):
	'''
	For now, just return some random samples
	'''
	indexes = random.sample(range(len(y)), batch_size)
	x_batch = X[indexes]
	y_batch = y[indexes]

	return [x_batch, y_batch]





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
		if n_classes[np.argmax(y[i])] < 2500 or np.argmax(y[i]) == 0:
			temp_X.append(X[i])
			temp_y.append(y[i])
			n_classes[np.argmax(y[i])] += 1
	temp_X = np.array(temp_X)
	temp_y = np.array(temp_y)
	# for i in range(20):
	# 	X = np.concatenate((X, temp_X), axis=0)
	# 	y = np.concatenate((y, temp_y), axis=0)
	return temp_X, temp_y




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

# print 'Bias Sampling'
# train_x, train_y = bias_sampling2(train_x, train_y)
# print 'Training set ' +  str(train_x.shape)
# print class_counter(train_y)
# print



# Parameters
n_batch = 1000
lr = .1/n_batch
mom = .9
tol = 10.

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 100 # 1st layer num features
n_hidden_2 = 100 # 2nd layer num features
n_classes = 10 # MNIST total classes (0-9 digits)

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

# W = tf.Variable(tf.random_normal([784,10], stddev=0.01))
# b = tf.Variable(tf.random_normal([10], stddev=0.01))
# y = tf.nn.softmax(tf.matmul(x,W) + b)

y = model(x,weights,biases)
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

# opt = tf.train.GradientDescentOptimizer(lr)
opt = tf.train.MomentumOptimizer(lr, mom)
grads_and_vars = opt.compute_gradients(cross_entropy)
modified_grad_and_vars = [(gv[0], gv[1]) for gv in grads_and_vars]
apply_grad = opt.apply_gradients(modified_grad_and_vars)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

early_stop_count = 0
prev_valid_ce = -1

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print 'Begin Training'
for i in range(1501):

	print 'Iter ' + str(i)

	batch = next_batch(train_x, train_y, n_batch)

	if i % 100 == 0:

		print 'batch class distribution'
		print class_counter(batch[1])
		
		# print 'W'
		# print sess.run([debug], feed_dict={x: batch[0], y_: batch1[0]})[0][402]
		# print sess.run([debug], feed_dict={x: batch[0], y_: batch1[0]})[0].shape

		# print 'gradient'
		# gs_vs = sess.run([grad for grad, _ in modified_grad_and_vars], feed_dict={x: batch[0], y_: batch[1]})
		# print gs_vs[0].shape
		# print gs_vs[1].shape
		# print gs_vs[0][0]
		# print gs_vs[0][402]
		# print gs_vs[1]

		print 'y'
		print sess.run([y], feed_dict={x: train_x, y_: train_y})[0][0]
		print 'y_'
		print train_y[0]

		print 'train ce'
		print sess.run([cross_entropy], feed_dict={x: train_x, y_: train_y})
		print 'valid ce'
		print sess.run([cross_entropy], feed_dict={x: valid_x, y_: valid_y})[0]

		print 'train acc'
		print sess.run([accuracy], feed_dict={x: train_x, y_: train_y})
		print class_counter(sess.run([y], feed_dict={x: train_x, y_: train_y})[0])
		print 'valid acc'
		print sess.run([accuracy], feed_dict={x: valid_x, y_: valid_y})
		print class_counter(sess.run([y], feed_dict={x: valid_x, y_: valid_y})[0])


	#check for convergence
	if i % 10 == 0:
		valid_ce = sess.run([cross_entropy], feed_dict={x: valid_x, y_: valid_y})[0]
		if prev_valid_ce - valid_ce < tol and prev_valid_ce != -1:
			early_stop_count += 1
			print 'early stop count ' + str(early_stop_count)
			if early_stop_count == 3:
				print 'Converged'
				print 'train acc'
				print sess.run([accuracy], feed_dict={x: train_x, y_: train_y})
				print class_counter(sess.run([y], feed_dict={x: train_x, y_: train_y})[0])
				print 'valid acc'
				print sess.run([accuracy], feed_dict={x: valid_x, y_: valid_y})
				print class_counter(sess.run([y], feed_dict={x: valid_x, y_: valid_y})[0])
				print 'test acc'
				print sess.run([accuracy], feed_dict={x: test_x, y_: test_y})
				print class_counter(sess.run([y], feed_dict={x: test_x, y_: test_y})[0])
				break
		else:
			early_stop_count = 0

		prev_valid_ce = valid_ce
		


	_ = sess.run([apply_grad], feed_dict={x: batch[0], y_: batch[1]})


print 'All done'


