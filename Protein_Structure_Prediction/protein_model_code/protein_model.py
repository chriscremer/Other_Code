

####################################
#MULTIPLY

# import tensorflow as tf

# a = tf.placeholder("float") # Create a symbolic variable 'a'
# b = tf.placeholder("float") # Create a symbolic variable 'b'

# y = tf.mul(a, b) # multiply the symbolic variables

# sess = tf.Session() # create a session to evaluate the symbolic expressions

# print "%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2}) # eval expressions with parameters for a and b
# print "%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3})


####################################
#LINEAR REGRESSION

# import tensorflow as tf
# import numpy as np

# trX = np.linspace(-1, 1, 101)
# print trX

# trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise
# print trY

# X = tf.placeholder("float") # create symbolic variables
# Y = tf.placeholder("float")


# def model(X, w):
# 	return tf.mul(X, w) # lr is just X*w so this model line is pretty simple


# w = tf.Variable(0.0, name="weights") # create a shared variable (like theano.shared) for the weight matrix
# y_model = model(X, w)

# cost = (tf.pow(Y-y_model, 2)) # use sqr error for cost function

# train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

# sess = tf.Session()
# init = tf.initialize_all_variables() # you need to initialize variables (in this case just variable W)
# sess.run(init)

# for i in range(100):
# 	# print i
# 	for (x, y) in zip(trX, trY): 
# 		sess.run(train_op, feed_dict={X: x, Y: y})

# print(sess.run(w))  # something around 2




####################################
#LOGISTIC REGRESSION

# import tensorflow as tf
# import numpy as np
# # import input_data

# from tensorflow.examples.tutorials.mnist import input_data


# def init_weights(shape):
#     return tf.Variable(tf.random_normal(shape, stddev=0.01))


# def model(X, w):
#     return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# X = tf.placeholder("float", [None, 784]) # create symbolic variables
# Y = tf.placeholder("float", [None, 10])

# w = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression

# py_x = model(X, w)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute mean cross entropy (softmax is applied internally)
# train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
# predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

# sess = tf.Session()
# init = tf.initialize_all_variables()
# sess.run(init)

# for i in range(100):
#     for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
#         sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
#     print i, np.mean(np.argmax(teY, axis=1) ==
#                      sess.run(predict_op, feed_dict={X: teX, Y: teY}))



# sdsafa



# Protein Model


import tensorflow as tf
import numpy as np

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
	if X[i][2] > 10:
		X[i][3] = 2
	else:
		X[i][3] = 1
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
	return vec.T

input_aa = tf.placeholder("float", shape=[None, 22])
expected_aa = tf.placeholder("float", shape=[None, 22])

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

# weights_to_predict_0 = []
# for i in range(len_protein):
# 	if i == 0:
# 		continue
# 	layers_to_predict_0.append(init_weights([22, 22]))


w = init_weights([22, 22])
b = init_biases([22])

def model(input_aa, w):
	return tf.nn.softmax(tf.matmul(input_aa, w) + b) 

prediction = model(input_aa, w)

cross_entropy = -tf.reduce_sum(expected_aa*tf.log(prediction))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, expected_aa)) 
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)

log_times = tf.log(prediction)
print1 = tf.matmul(tf.transpose(input_aa), w)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


#Predict aa1 using aa0

for i in range(100):
	print
	print i 
	for samp in X:
		sess.run(train_op, feed_dict={input_aa: convert_to_one_hot(samp[0]), expected_aa: convert_to_one_hot(samp[1])})

	print convert_to_one_hot(samp[1])
	print sess.run(prediction, feed_dict={input_aa: convert_to_one_hot(samp[0])})# - convert_to_one_hot(samp[1])

	# print sess.run(log_times,  feed_dict={input_aa: convert_to_one_hot(samp[0]), expected_aa: convert_to_one_hot(samp[1])})
	print sess.run(cross_entropy, feed_dict={input_aa: convert_to_one_hot(samp[0]), expected_aa: convert_to_one_hot(samp[1])})
	
	# print sess.run(print1, feed_dict={input_aa: convert_to_one_hot(samp[0]), expected_aa: convert_to_one_hot(samp[1])})
	
	# sdfas
	# if i == 1:
	# 	last_w = sess.run(w)
	# if i == 2:
	# 	print last_w - sess.run(w)
	# 	fdsasa


# 	print i, np.mean(np.argmax(teY, axis=1) ==
# 					 sess.run(predict_op, feed_dict={X: teX, Y: teY}))


# y = sess.run(model, feed_dict={x: X})


tf.shape(x)







