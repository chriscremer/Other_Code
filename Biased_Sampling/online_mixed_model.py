

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


#Make data
#3 clusters, 2 features
#equal variance


k1_mean = [1.,2.]
k2_mean = [1.5,1.]
k3_mean = [2.,1.7]

cov = [[.01,.0],[0.,.01]]

X = []
Y = []

x, y = np.random.multivariate_normal(k1_mean, cov, 100).T
plt.plot(x, y, 'x')
X.extend(x)
Y.extend(y)

x, y = np.random.multivariate_normal(k2_mean, cov, 100).T
plt.plot(x, y, 'x')
X.extend(x)
Y.extend(y)

x, y = np.random.multivariate_normal(k3_mean, cov, 100).T
plt.plot(x, y, 'x')
X.extend(x)
Y.extend(y)
# plt.show()

#test batches
# X = [[[1.,2.],[3.,2.],[1.,5.],[4.,2.]],[[3.,3.],[1.,7.],[2.,4.],[3.,7.]]]
X = np.array(X)
Y = np.array(Y)

K=3 #number of clusters
D=2 #number of dimensions
N=4 #number of samps per batch


batches = []
for i in range(1000):
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










