
import csv
import numpy as np
import random
from sklearn.cross_validation import StratifiedKFold
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import preprocessing


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
		X.append(map(float, (row[1:-1])))
		y.append(float(row[-1]))
		# if str(row[-1]) == '0.0':
		# 	y.append([1, 0])
		# else:
		# 	y.append([0, 1])
X = np.asarray(X)
y = np.asarray(y) 

# #################
# # this is wrong, should be fitted to training data then used to transform all data sets
# ###########
# X = scaler.fit_transform(X)






# model = linear_model.LogisticRegression(penalty='l1')
# model.fi
# clf = linear_model.Lasso(alpha = 0.3)

all_scores = []

for iter in range(1):

	#shuffle it up
	random_shuffle = random.sample(range(len(y)), len(y))
	#print random_samp
	#print X.shape
	X = X[random_shuffle]
	#print y.shape
	y = y[random_shuffle]

	#stratified 2 fold cross validation
	cv_outer = StratifiedKFold(y, n_folds=2)
	for i, (train_index, test_index) in enumerate(cv_outer):
		#print 'Outer Fold ' + str(i)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#preprocess
		preprocessor = preprocessing.StandardScaler()
		preprocessor.fit(X_train)
		X_train = preprocessor.transform(X_train)
		X_test = preprocessor.transform(X_test)


		clf = linear_model.Lasso(alpha=0.001)
		#clf = linear_model.LogisticRegression(penalty='l1')
		#clf = naive_bayes.GaussianNB()
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)

		#print clf.coef_ 

		score = []
		for j in range(len(predictions)):

			#print predictions[j]
			print 'Predict ' + str(predictions[j] ) + ' Label ' + str(y_test[j])
			if predictions[j] >= 0.5 and y_test[j] >= 0.5:
				score.append(1.0)
			elif predictions[j] < 0.5 and y_test[j] < 0.5:
				score.append(1.0)
			else:
				score.append(0.0)

		#print np.mean(score)
		#print np.mean(score)
		all_scores.append(np.mean(score))


print np.mean(all_scores)

#try lasso vs L1 log reg... interesting..