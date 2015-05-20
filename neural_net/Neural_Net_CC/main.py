

import numpy as np
import csv

from NN_cc import Network

if __name__ == "__main__":


	####################################
	#Load data
	####################################
	MY_DATASET = '/data1/morrislab/ccremer/simulated_data/simulated_classification_data_10_samps_5_feats_3_distinct.csv'

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

	#X = np.array(X)
	#y = np.array(y)

	training_data= []
	for i in range(len(y)):
		training_data.append((np.array(X[i], ndmin=2).T, np.array(y[i], ndmin=2).T))

	print 'Numb of Samples: ' + str(len(training_data))
	print 'X shape: ' + str(training_data[0][0].shape)
	print 'y shape: ' + str(training_data[0][1].shape)



	####################################
	#Train Model
	####################################

	#dimension of input, hidden layer, dimension of output
	net = Network([len(X[0]), 5, len(y[0])])
	#SGD(self, training_data, epochs, mini_batch_size, learn_rate, lmbda = 0.0, 
	#evaluation_data=None, monitor_evaluation_cost=False,monitor_evaluation_accuracy=False,monitor_training_cost=False, monitor_training_accuracy=False):
	evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(training_data=training_data, 
																						epochs=1000,
																						mini_batch_size=2,
																						learn_rate=0.01,
																						lmbda=0.0,
																						monitor_training_cost=True,
																						monitor_training_accuracy=False,
																						evaluation_data=None,
																						monitor_evaluation_cost=False,
																						monitor_evaluation_accuracy=False
																					)


	# #to get dimensions of input
	# print 'Getting dimensions of input...'
	# numb_of_genes = 0
	# with open(data_file, 'rU') as f:
	#         reader = csv.reader(f, delimiter=' ')
	#         for row in reader:
	#             numb_of_genes += 1
	#             numb_of_samples = len(row)
	# print 'numb_of_genes ' + str(numb_of_genes)
	# print 'numb_of_samples ' + str(numb_of_samples)
	# f.close()
	# print 'Done.'


	# print 'Putting data in a list from the file...'
	# data_list = []
	# for i in range(0, numb_of_samples):
	#     data_list.append([])
	# #gather data into lists
	# with open(data_file, 'rU') as f:
	#         reader = csv.reader(f, delimiter=' ')
	#         for row in reader:
	#             column = 1
	#             for sample_exp in row[1:-1]:
	#                 #add the expr to the sample list
	#                 data_list[column-1].append(float(sample_exp))
	#                 column += 1
	# f.close()
	# print "Done."
	# #print len(data_list[0])


	# print 'Rescaling data...'
	# #get max gene expression for each gene, for rescaling
	# gene_maxs = []
	# with open(data_file, 'rU') as f:
	#         reader = csv.reader(f, delimiter=' ')
	#         for row in reader:
	#             temp_list = []
	#             for sample_exp in row[1:-1]:
	#                 temp_list.append(float(sample_exp))
	#             gene_max = max(temp_list)
	#             gene_maxs.append(gene_max)
	# f.close()
	# #print "len gene_maxs" + str(len(gene_maxs))
	# #rescale by divinding each expression by the max expression of that gene
	# for sample in data_list:
	#     for column in range(len(sample)):
	#         if gene_maxs[column] < 0.0001:
	#             continue
	#         sample[column] = sample[column] / gene_maxs[column]
	# #confirm all gene expressions are between 0 and 1
	# for sample in data_list:
	#     for expr in sample:
	#         if expr > 1 or expr < 0:
	#             print 'PROBLEM HERE2'
	#             print expr
	#             error
	#             break
	# print 'Done.'





	# #for autoencoder, need to make the input the same as the output
	# #need to reshape the arrays to make them vectors ie have the 1 in the second position
	# '''
	# training_data = []
	# for sample in data_list:
	# 	training_data.append(
	# 							((np.reshape(np.array(sample), (numb_of_genes,1) )), (np.reshape(np.array(sample), (numb_of_genes,1) )))
	# 						)
	# '''
	# training_data = []
	# for sample in data_list:
	# 	training_data.append(
	# 							((np.transpose(np.array(sample, ndmin=2))), np.transpose((np.array(sample, ndmin=2))))
	# 						)

	# for sample in training_data:
	# 	print training_data[0][0].shape



	# with open('workfile.txt', 'w') as f2:
	# 	f2.write('This is a test\n')

	# training_data = training_data[:10]

	# net = Network([len(data_list[0]), 10000, len(data_list[0])])
	# evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(training_data, 100, 2, 0.01, monitor_training_cost=True)