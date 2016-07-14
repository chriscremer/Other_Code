
import math
import numpy as np
import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats.stats import pearsonr  



#GAUSSIAN LINEAR REGRESSION

data_x = [[1,2], [2,7], [3,4]]
data_y = [[1], [2], [3]]

#data_x = [[1,9,7],[2,8,1],[3,2,8],[4,5,3],[5,4,9],[6,3,1],[7,2,9],[8,1,1]]
#data_y = [[1],[2],[3],[4],[5],[6],[7],[8]]
parameters_w = []




def average(data_x):
	#list of mean for each dimension
	list_of_means = []
	#list for taking mean of each dimension
	temp = []
	#for each dimention/variable
	for i in range(len(data_x[0])):
		for sample in data_x:
			temp.append(sample[i])
		list_of_means.append(np.mean(temp))
	return list_of_means

def std_dev(data_x):
	#list of std_dev for each dimension
	list_of_std = []
	#list for taking std_dev of each dimension
	temp = []
	#for each dimention/variable
	for i in range(len(data_x[0])):
		for sample in data_x:
			temp.append(sample[i])
		list_of_std.append(np.std(temp))
	return list_of_std


#sigmoidal basis function
# im using it as feature scaling
def basis_function(sample, dimension, mean, std_dev):
	#bias
	if dimension == 0:
		return 1
	logistic_sigmoid_function = 1.0/(1 + math.e**-(((sample[dimension-1]-mean)/std_dev)))
	return logistic_sigmoid_function


def make_design_matrix(data_x):
	design_matrix = []
	for sample in data_x:
		sample_row = []
		for dimension in range(len(data_x) + 1):
			sample_row.append(basis_function(sample, dimension, list_of_means[dimension-1], list_of_means[dimension-1]))
		design_matrix.append(sample_row)

	return design_matrix


def normal_equations(design_matrix, data_y):
	phi = np.matrix(design_matrix)
	y = np.matrix(data_y)
	w_ML = (phi.T * phi).I * phi.T*y
	return w_ML



def y(one_sample_x, parameters_w):
	sum = 0
	for i in range(len(one_sample_x)):
		#features of x times their respective parameter
		sum += one_sample_x[i] * parameters_w[i]
	return sum


def squared_error(parameters, data_x, data_y):
	sum = 0
	for i in range(len(data_y)):
		sum += (data_y[i][0] - y(data_x[i], parameters))**2
	return sum

def precision(w_ML, design_matrix, targets):
	N = float(len(targets))
	squared_error_value = squared_error(w_ML, design_matrix, targets)
	devided_by_N = squared_error_value / N
	precision = 1.0 / devided_by_N
	return precision

def loglikelihood(data_x, data_y, parameters, precision_value):
	N = float(len(data_y))
	half_error = (1/2.0) * squared_error(parameters, data_x, data_y)
	loglike = (N/2)*math.log(precision_value) - (N/2)*math.log(2*math.pi) - precision * half_error
	return loglike



#main
'''
list_of_means = average(data_x)
lis_of_stds = std_dev(data_x)

#add the bias
for sample in data_x:
	sample.insert(0, 1)

parameters = normal_equations(data_x, data_y)
print 'parameters:'
print parameters

precision = precision(parameters, data_x, data_y)
#print precision

loglike = loglikelihood(data_x, data_y, parameters, precision)
#print loglike

likelihood = math.e**float(loglike)
#print likelihood
'''

'''
x = np.linspace(1,10,10)
y = np.array(parameters[0]+parameters[1]*x)

plt.figure(1)
plt.plot(x,y.T,color='r')

for i in range(len(data_x)):
	plt.scatter(data_x[i][1], data_y[i])

plt.savefig('learn_reg.pdf')
'''

y_pos = 0.9
colours = ['red', 'blue', 'green', 'black', 'purple']

#plot for each single feature their x vs y-value
plt.figure()

for feature_i in range(len(data_x[0])):

	new_Xs = []
	for sample_i in range(len(data_x)):
		new_Xs.append([data_x[sample_i][feature_i]])
		plt.scatter(data_x[sample_i][feature_i], data_y[sample_i], color=colours[feature_i])
		print feature_i
	#add the bias
	for sample in new_Xs:
		sample.insert(0, 1)
	#regression
	parameters = normal_equations(new_Xs, data_y)

	x = np.linspace(1,10,10)
	y = np.array(parameters[0]+parameters[1]*x)

	plt.plot(x,y.T,color=colours[feature_i])

	feature_y_predictions = []
	for sample in data_x:
		feature_y_predictions.append((parameters[0]+float(parameters[1]*sample[feature_i])).tolist()[0])

	#feature_y_predictions.ravel().tolist()

	#print parameters
	#print feature_y_predictions
	#print ''
	#print data_y
	PCC = pearsonr(feature_y_predictions, data_y)
	print PCC
	plt.annotate('Feature ' + str(feature_i) + ':' + str(PCC[0]), xy=(0.4,y_pos), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color=colours[feature_i])
	y_pos -= 0.05

	#plt.annotate('b='+str(parameters[0]), xy=(0.8,0.9), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='green')
	#plt.annotate('m='+str(parameters[1]), xy=(0.8,0.8), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='green')


'''
#for all features
for sample in data_x:
	sample.insert(0, 1)
parameters = normal_equations(data_x, data_y)
feature_y_predictions = []
for sample in data_x:
	feature_y_predictions.append((parameters[0]+parameters[1]*sample[1]+parameters[2]*sample[2]+parameters[3]*sample[3]).tolist()[0])
PCC = pearsonr(feature_y_predictions, data_y)
print PCC
plt.annotate('Using all features:' + str(PCC[0]), xy=(0.4,y_pos), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color=colours[3])
y_pos -= 0.05

#for features 1 and 2
for sample in data_x:
	sample.pop(3)
print data_x

parameters = normal_equations(data_x, data_y)
feature_y_predictions = []
for sample in data_x:
	feature_y_predictions.append((parameters[0]+parameters[1]*sample[1]+parameters[2]*sample[2]).tolist()[0])
PCC = pearsonr(feature_y_predictions, data_y)
print PCC
plt.annotate('Using features 1 + 2:' + str(PCC[0]), xy=(0.4,y_pos), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color=colours[4])
y_pos -= 0.05
plt.annotate('parameters: ' + str(parameters[0]) + ' ' + str(parameters[1]) + ' ' + str(parameters[2]), xy=(0.4,y_pos), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color=colours[4])


plt.savefig('features_best_fit3.pdf')

'''
#for all features of samples with only 2 dimensions
for sample in data_x:
	sample.insert(0, 1)
parameters = normal_equations(data_x, data_y)
feature_y_predictions = []
for sample in data_x:
	feature_y_predictions.append((parameters[0]+parameters[1]*sample[1]+parameters[2]*sample[2]).tolist()[0])
PCC = pearsonr(feature_y_predictions, data_y)
print PCC
plt.annotate('Using all features:' + str(PCC[0]), xy=(0.4,y_pos), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color=colours[3])
y_pos -= 0.05

plt.savefig('features_best_fit3.pdf')