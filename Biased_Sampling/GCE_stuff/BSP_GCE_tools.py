

import numpy as np
import random



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


def next_batch_and_return_indexes(X, y, batch_size):
	'''
	For now, just return some random samples
	'''
	indexes = random.sample(range(len(y)), batch_size)
	x_batch = X[indexes]
	y_batch = y[indexes]

	return [x_batch, y_batch], indexes


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


def bias_sampling2(X, y, bias):


	mean_class = np.mean(np.array(class_counter(y)))

	if bias == 0:
		return X, y
	else:
		keep = mean_class / float(bias)
	# elif bias == 5:
	# 	keep = 1000
	# elif bias == 10:
	# 	keep = 500
	# elif bias == 20:
	# 	keep = 250


	n_classes = [0]*10

	temp_X = []
	temp_y = []
	#store the class 0 samples then add them to X
	for i in range(len(X)):
		if n_classes[np.argmax(y[i])] < keep or np.argmax(y[i]) in [0,1,2,3,4]:
			temp_X.append(X[i])
			temp_y.append(y[i])
			n_classes[np.argmax(y[i])] += 1
	temp_X = np.array(temp_X)
	temp_y = np.array(temp_y)
	# for i in range(20):
	# 	X = np.concatenate((X, temp_X), axis=0)
	# 	y = np.concatenate((y, temp_y), axis=0)
	return temp_X, temp_y


def bias_sampling3(X, y, major_bias, total_n_samples):

	major_per_class = major_bias*total_n_samples / 5
	minor_per_class = (1-major_bias)*total_n_samples / 5

	n_classes = [0]*10
	temp_X = []
	temp_y = []
	for i in range(len(X)):
		if np.argmax(y[i]) in [0,1,2,3,4]:
			if n_classes[np.argmax(y[i])] < major_per_class:
				temp_X.append(X[i])
				temp_y.append(y[i])
				n_classes[np.argmax(y[i])] += 1
		elif np.argmax(y[i]) in [5,6,7,8,9]:
			if n_classes[np.argmax(y[i])] < minor_per_class:
				temp_X.append(X[i])
				temp_y.append(y[i])
				n_classes[np.argmax(y[i])] += 1

	temp_X = np.array(temp_X)
	temp_y = np.array(temp_y)

	return temp_X, temp_y





def calc_class_weights(labels):

	counts = class_counter(labels)
	mean = np.mean(counts)

	weights = [0]*10
	for i in range(len(weights)):
		weights[i] = mean / counts[i]

	return weights



def weight_array(labels, weights):

	weight_array_ = []
	for i in range(len(labels)):
		weight_array_.append(weights[np.argmax(labels[i])])

	# return np.reshape(np.array(weight_array), (len(weight_array),1))
	weight_array_ = np.array(weight_array_)
	return weight_array_







