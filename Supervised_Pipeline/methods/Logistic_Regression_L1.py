

from sklearn import linear_model
from sklearn import cross_validation

import numpy as np
import random


class method():

	def __init__(self):
		self.name='Logistic Regression L1'
		self.hyper= 0.0
		self.performance = 0.0
		#model = linear_model.LogisticRegression(penalty='l1')

	@staticmethod
	def accuracy(target, output):
		'''Model accuracy'''

		results = []
		for i in range(len(output)):
			if output[i] > 0.5 and target[i] > 0.5:
				results.append(1.0)
			elif output[i] < 0.5 and target[i] < 0.5:
				results.append(1.0)
			else:
				results.append(0.0)

		return np.mean(results)

	def select_hypers(self, X_train, y_train):
		'''
		Hyperparameter selection
		'''

		best_hyper = 0
		best_hyper_score = 0
		for hyper1 in [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0]:
			#accuracy sum over all crosses
			acc_sum = 0
			#2-fold cross validation for hyperparameter selection
			cv_inner = cross_validation.StratifiedKFold(y_train, n_folds=2)
			#leave one out, takes longer but get better hyperparameters
			#cv_inner = cross_validation.LeaveOneOut(n=len(X_train))
			for j, (train_index2, valid_index) in enumerate(cv_inner):

				X_train2, X_valid = X_train[train_index2], X_train[valid_index]
				y_train2, y_valid = y_train[train_index2], y_train[valid_index]

				clf = linear_model.LogisticRegression(penalty='l1', C=hyper1)
				clf.fit(X_train2, y_train2)
				output = clf.fit(X_train2, y_train2).predict(X_valid)
				acc = self.accuracy(y_valid, output)

				acc_sum += acc

			#if this hyper is best then save it
			if acc_sum > best_hyper_score:
				best_hyper_score = acc_sum
				best_hyper = hyper1
		#print 'best hyper ' + str(best_hyper)


		self.hyper = best_hyper


	def average_two_fold_performace(self, X_train, y_train):
		'''
		Return the average accuracy and AUC
		'''
		accuracies_list = []
		for iteration in range(20):

			#shuffle it up
			random_shuffle = random.sample(range(len(y_train)), len(y_train))
			X_train = X_train[random_shuffle]
			y_train = y_train[random_shuffle]

			cv_inner = cross_validation.StratifiedKFold(y_train, n_folds=2)
			for j, (train_index2, valid_index) in enumerate(cv_inner):

				X_train2, X_valid = X_train[train_index2], X_train[valid_index]
				y_train2, y_valid = y_train[train_index2], y_train[valid_index]

				clf = linear_model.LogisticRegression(penalty='l1', C=self.hyper)
				clf.fit(X_train2, y_train2)
				output = clf.fit(X_train2, y_train2).predict(X_valid)
				acc = self.accuracy(y_valid, output)

				accuracies_list.append(acc)

		self.performance = np.mean(accuracies_list)

		






