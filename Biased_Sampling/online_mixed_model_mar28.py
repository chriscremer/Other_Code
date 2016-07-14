

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
	for i in range(len(labels)):
		n_classes[labels[i]] += 1
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
	For now, just return some samples
	'''

	x_batch = X[random.sample(range(len(y)), batch_size)]
	y_batch = y[random.sample(range(len(y)), batch_size)]

	return [x_batch, y_batch]



print train_set[1].shape
print train_set[1][0]



images = train_set[0]
labels = convert_array_to_one_hot_matrix(train_set[1])

print images.shape
print labels.shape



# print 'Loading data'
# path = home + '/Documents/MNIST_data/'
# images, labels = load_mnist(path=path)
# print labels.shape
# print images.shape
# images = np.reshape(images, [images.shape[0],images.shape[1]*images.shape[2]])
# class_counts = class_counter(labels)
# labels = convert_array_to_one_hot_matrix(labels)

# print 'X shape: ' + str(images.shape)
# print 'Y shape: ' + str(labels.shape)
# print 'Class counts: ' + str(class_counts)




print 'Begin Training'
N = len(labels)
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
W = tf.Variable(tf.random_normal([784,10], stddev=0.01))
b = tf.Variable(tf.random_normal([10], stddev=0.01))

y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

debug = W
debug2 = tf.matmul(x,W)
debug3 = tf.matmul(x,W) + b

sess = tf.Session()
# sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


for i in range(1000):
	print
	print 'Iter ' + str(i)

	# print 'W'
	# print sess.run([debug], feed_dict={x: images, y_: labels})[0][402]
	# print sess.run([debug], feed_dict={x: images, y_: labels})[0].shape

	# print 'xW'
	# print sess.run([debug2], feed_dict={x: images, y_: labels})[0][402]
	# print sess.run([debug2], feed_dict={x: images, y_: labels})[0].shape

	# print 'xW+b'
	# print sess.run([debug3], feed_dict={x: images, y_: labels})[0][402]
	# print sess.run([debug3], feed_dict={x: images, y_: labels})[0].shape

	print 'acc'
	print sess.run([accuracy], feed_dict={x: images, y_: labels})
	
	# print 'ce'
	# print sess.run([cross_entropy], feed_dict={x: images, y_: labels})
	

	batch = next_batch(images, labels, 500)
	# print batch[0].shape
	# print batch[1].shape
	
	# print batch[0].shape
	# print batch[1].shape
	# print batch[1]
	_ = sess.run([train_step], feed_dict={x: batch[0], y_: batch[1]})



	# batch = next_batch(images, labels, 1)
	# print batch[0].shape
	# print batch[1].shape
	# dict_batch = {x: batch[0], y_: batch[1]}
	# dict_all = {x: images, y_: labels}

	# W =  sess.run([debug])

	# print np.array(W).shape
	# print W[0]

	# _, W = sess.run([train_step, debug], feed_dict=dict_batch)
	# # train_step.run(feed_dict={x: batch[0], y_: batch[1]})
	# print np.array(W).shape
	# print W[0]

	# acc = sess.run([accuracy], feed_dict=dict_all)
	# # print(accuracy.eval(feed_dict={x: images, y_: labels}))
	# print acc




fdsdsfsa


batches = []
for i in range(100):
	batch1 = []
	for j in range(N):
		ind = random.randint(0, len(X)-1)
		batch1.append([X[ind],Y[ind]])
	batches.append(batch1)

X = batches
X = np.array(X)
print X.shape


batch = tf.placeholder("float", shape=[N, D])
means = tf.Variable(tf.random_normal([K,D,1], stddev=0.01))
mixture_weights = tf.Variable(tf.random_normal([K,1], stddev=0.01))



def log_likelihood(batch):

	#batch is NxD matrix, where N is length of batch, D is dimension of samples
	#P(D|w) = prod( sum( pi*N(samp|k))
	#exp(-square(mean-samp))

	#multiplying by ones replicates the matrix, becomes (N,D,K)
	tmp1 = tf.batch_matmul(tf.reshape(batch, [N,D,1]), tf.ones([N,1,K]))
	#same but with the means matrix
	tmp2 = tf.batch_matmul(means, tf.ones([K,1,N]))
	tmp2 = tf.transpose(tmp2, [2,1,0])
	# (x - mu)
	tmp3 = tmp1 - tmp2
	tmp4 = tmp1 - tmp2
	# (x - mu).T(x - mu)
	tmp3 = tf.batch_matmul(tf.transpose(tmp3, [0,2,1]), tmp3)
	tmp3 = tf.reduce_sum(tmp3,2)
	# -(x - mu).T(x - mu)
	tmp3 = -tmp3
	# exp(-(x - mu).T(x - mu))
	tmp3 = tf.exp(tmp3)
	#multiply by mixture weights
	tmp3 = tf.matmul(tmp3, mixture_weights)
	#log
	tmp3 = tf.log(tmp3)
	#sum over all samples of the batch
	tmp3 = tf.reduce_sum(tmp3,0)

	return tmp3
	
return_means = means
return_mixtures = mixture_weights
cost_function = tf.neg(log_likelihood(batch))
# train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost_function)
# opt = tf.train.GradientDescentOptimizer(0.001)
grad = tf.train.GradientDescentOptimizer(0.001).compute_gradients(cost_function)
# apply_grad = opt.apply_gradients(grad)


sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(3):

	for batch_i in range(len(X)):

			g, cost, m, mx = sess.run([grad, cost_function, return_means, return_mixtures], feed_dict={batch: X[batch_i]})
			# out = sess.run([cost_function], feed_dict={batch: X[batch_i]})

			print cost[0]
			print g
			print m.T
			print mx.T
			# print out[0].shape
			# afsd










