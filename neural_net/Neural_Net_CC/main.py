

import numpy as np
import csv
import random

from NN_cc import Network
from costs import *
from activations import *

from sklearn import preprocessing


if __name__ == "__main__":


	####################################
	#Load data
	####################################
	MY_DATASET = '/data1/morrislab/ccremer/simulated_data/simulated_classification_data_100_samps_1000_feats_3_distinct.csv'

	X = []
	y = []
	header = True
	with open(MY_DATASET, 'r') as f:
		csvreader = csv.reader(f, delimiter=',', skipinitialspace=True)
		for row in csvreader:
			if header:
				header = False
				continue
			X.append(map(float,row[1:-1]))
			if str(row[-1]) == '0.0':
				y.append([1.0,0.0])
			else:
				y.append([0.0,1.0])

	X = np.array(X)
	y = np.array(y)

	#preprocess
	preprocessor = preprocessing.StandardScaler()
	preprocessor.fit(X)
	X = preprocessor.transform(X)
	#X_test = preprocessor.transform(X_test)

	training_data= []
	for i in range(0,70):
		training_data.append((np.array(X[i], ndmin=2).T, np.array(y[i], ndmin=2).T))

	evaluation_data= []
	for i in range(70,100):
		evaluation_data.append((np.array(X[i], ndmin=2).T, np.array(y[i], ndmin=2).T))


	print 'Numb of Samples: ' + str(len(training_data))
	print 'X shape: ' + str(training_data[0][0].shape)
	print 'y shape: ' + str(training_data[0][1].shape)



	####################################
	#Train Model
	####################################

	#dimension of input, hidden layer, dimension of output
	net = Network(layers=[len(X[0]), 3, len(y[0])], 
					activations=[Sigmoid_Activation, Linear_Activation],
					cost=QuadraticCost,
					regularization='l1')
	evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(training_data=training_data, 
																						epochs=1000,
																						mini_batch_size=2,
																						learn_rate=0.01,
																						lmbda=0.1,
																						monitor_training_cost=True,
																						monitor_training_accuracy=True,
																						evaluation_data=evaluation_data,
																						monitor_evaluation_cost=True,
																						monitor_evaluation_accuracy=True
																					)

