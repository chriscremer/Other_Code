


#Kaggle - Winton Challenge


import csv
import numpy as np

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn import ensemble
from sklearn import gaussian_process



def read_data():

	# ids = []
	# X = []
	# t = []
	# header = []

	print 'Reading train data...'

	train_X = np.genfromtxt('../Downloads/Kaggle_Winton/train.csv', skip_header=1, usecols=range(1,147), delimiter=',')
	train_Y = np.genfromtxt('../Downloads/Kaggle_Winton/train.csv', skip_header=1, usecols=range(147,209), delimiter=',')
	train_weights = np.genfromtxt('../Downloads/Kaggle_Winton/train.csv', skip_header=1, usecols=range(209,211), delimiter=',')


	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	imp.fit(train_X)
	train_X = imp.transform(train_X)



	# print train_X[0]
	# print train_X.shape

	# if np.isnan(train_X[0][0]):
	# 	print 'yes'
	# else:
	# 	print 'no'
	# afsdfa

	# with open('../Downloads/Kaggle_Winton/train.csv', 'rb') as f:
	# 	reader = csv.reader(f, delimiter=',')
	# 	count =0
	# 	for row in reader:
	# 		if count == 0:
	# 			header = row
	# 			count += 1
	# 			continue

	# 		ids.append(row[0])
	# 		X.append(row[1:147])
	# 		t.append(row[147:209])

	# 		count +=1

	#index 147-208 is where targets are
	#indexes 1-146 are the inputs
	# for i in range(len(header)):
	# 	print str(i) + '  ' +str(header[i])

	#Fill missing data with zeros for now. Will need to improve this later
	# for i in range(len(X)):
	# 	for j in range(len(X[i])):
	# 		if X[i][j] == '':
	# 			X[i][j] = '0.0'


	# for i in range(len(X)):
	# 	for j in range(len(X[i])):
	# 		X[i][j] = float(X[i][j])
	# 	for j in range(len(t[i])):
	# 		t[i][j] = float(t[i][j])




	print train_X.shape
	print train_Y.shape

	print 'Reading train data... Complete'

	#Read test data
	print 'Reading test data...'

	test_X = np.genfromtxt('../Downloads/Kaggle_Winton/test_2.csv', skip_header=1, usecols=range(1,147), delimiter=',')
	test_X = imp.transform(test_X)
	print test_X.shape
	# test_X = None

	print 'Reading test data... Complete'


	return train_X, train_Y, test_X, train_weights

def predict_multiple_outputs(model, X, y, X_test):

	print y.T[0].shape

	fasdf

	predictions =[]
	for i in range(len(y[0])):
		model.fit(X, y.T[i])
		pred = model.predict(X_test)
		predictions.append(pred)

	predictions = np.array(predictions).T
	return predictions



def predict(train_X, train_Y, test_X, train_weights):
	#Model
	print 'Making Predictions'

	'''

	scaler = preprocessing.StandardScaler()
	scaler.fit(train_X)
	train_X = scaler.transform(train_X)
	train_X = train_X[:10000]
	test_X = scaler.transform(test_X)

	model = linear_model.LinearRegression()
	# model.fit(train_X,train_Y)
	# predictions = model.predict(test_X)

	# model2 = svm.SVR()

	model2 = ensemble.RandomForestRegressor()

	# model2 = gaussian_process.GaussianProcess()


	kf = KFold(len(train_X), n_folds=2)
	count =0
	for train_index, test_index in kf:
		print 'Fold ' + str(count)
		
		fold_X_train, fold_X_test = train_X[train_index], train_X[test_index]
		fold_y_train, fold_y_test = train_Y[train_index], train_Y[test_index]

		print fold_X_train.shape


		model.fit(fold_X_train, fold_y_train)
		pred = model.predict(fold_X_test)
		err = metrics.mean_absolute_error(fold_y_test, pred)
		print 'Lin Reg Error ' + str(err)

		# pred = predict_multiple_outputs(model2, fold_X_train, fold_y_train, fold_X_test)
		model2.fit(fold_X_train, fold_y_train)
		pred = model2.predict(fold_X_test)
		err = metrics.mean_absolute_error(fold_y_test, pred)
		print 'Model2 Error ' + str(err)

		count += 1


	afsdfs
	# weights = []
	# for i in range(len(train_preds)):
	# 	this_samp = [train_weights[i][0]]*60
	# 	this_samp.append(train_weights[i][1])
	# 	this_samp.append(train_weights[i][1])
	# 	weights.append(this_samp)

	# print weights[0]


	'''


	#Predict the last known price
	#return is change in price, not price, so not good
	# predictions = []
	# for i in range(len(test_X)):
	# 	predictions.append([test_X[i][-1]]*62)
	# predictions = np.array(predictions)
	

	#instead predict the median 1-min return and the mean 1-day return
	# predictions = []
	# for i in range(len(test_X)):
	# 	predictions.append([np.median(test_X[i][28:])]*62)
	# predictions = np.array(predictions)


	#now predict 0 for intraday because its almost impossible
	#but for last two days, try to use the 25 features, maybe its related to news
	# model = linear_model.LinearRegression()
	# model.fit(train_X.T[:25].T, train_Y.T[-2:].T)
	# preds = model.predict(test_X.T[:25].T)
	# print preds.shape
	# predictions = []
	# for i in range(len(test_X)):
	# 	pred = [0.0]*60
	# 	pred.append(preds[i][0])
	# 	pred.append(preds[i][1])
	# 	predictions.append(pred)
	# predictions = np.array(predictions)

	#those features didnt help, ill use the median for the last two
	predictions = []
	for i in range(len(test_X)):
		pred = [0.0]*60
		med = np.median(test_X[i][28:])
		pred.append(med)
		pred.append(med)
		predictions.append(pred)
	predictions = np.array(predictions)



	print predictions.shape

	return predictions



def write_predictions(predictions):

	#Write predictions
	print 'Writing predictions'


	f = open('../Downloads/Kaggle_Winton/my_submission_median.csv', 'w')
	f.write('Id,Predicted\n')
	for i in range(len(predictions)):

		for j in range(1,63):

			f.write(str(i+1) + '_' + str(j) + ',' + str(predictions[i][j-1]) + '\n')

	print 'Saved'





if __name__ == "__main__":

	train_X, train_Y, test_X, train_weights = read_data()

	predictions = predict(train_X, train_Y, test_X, train_weights)

	write_predictions(predictions)





