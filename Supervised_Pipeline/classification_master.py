

#for many things
import numpy as np

#for going through files
import csv

#add parent directory to path to get my packages there
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

#print sys.path
sys.path.insert(0,parentdir) 
sys.path.insert(0,currentdir+'/methods')
#print sys.path


def classification(X_train, y_train, X_test=None, sample_names=None, feature_names=None):
	'''
	Input:
		X_train: numpy array, shape (n_samples, n_features)
		y_train : numpy array, shape (n_samples,), use floats for classes (ie 0.0 and 1.0)
		OPTIONAL:
		X_test: numpy array, shape (n_samples, n_features)
		sample_names: list of the names of the samples
		feature_names: list of the names of the features

	Do:
		Run different methods (iterations? parallel? cross-validation?)
		Print the ordred performance of the different methods
		Plot some graphs

	Output:
		If X_test != None:
			Return the predictions for the test data
		else:
			return None

	'''


	print 'Training Data: ' + str(X_train.shape)
	print 'Labels: ' + str(y_train.shape)
	

	#for every file in methods directory, load method object, 
	#use cross validation to select hypers and report performance of methods
	#if test data, run best method on it

	method_list = []
	for file in os.listdir("methods"):
		if file.endswith(".py"):
			#print(file)
			file_ = file[:-3]
			#print file_
			new_module = __import__(file_)
			clf = new_module.method()
			method_list.append(clf)

	for method in method_list:
		method.select_hypers(X_train, y_train)
		method.average_two_fold_performace(X_train, y_train)

	for method in method_list:
		print method.name + ' ' + str(method.performance)




	return None




if __name__ == "__main__":



	#test

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
				y.append(0.0)
			else:
				y.append(1.0)

	X = np.array(X)
	y = np.array(y)


	output = classification(X, y)







