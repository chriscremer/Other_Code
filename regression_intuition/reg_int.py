


import math


#for now data its just x,y
data = [[1,5], [2,6], [3,9], [4,17]]

#initial parameter weights
parameters = [1.0, 1.0]

#alpha
alpha = 1/5.0

#number of samples, same as len(data)
n = 4.0

#hypothesis = para1 + para2*x
def hypothesis(parameters, sample):

	return parameters[0] + parameters[1]*sample[0]


#objective function = alpha*1/2n* sum((h-y)^2)
#					= alpha*1/2n* sum(((para1 + para2*x) -y)^2)
#partial derivatives of obj func =
#	
#	derivative of para1 = alpha*1/n* sum((para1 + para2*x) -y)
#	derivative of para2 = alpha*1/n* sum(((para1 + para2*x) -y) *x )

def obj(parameters, data):
	sum = 0
	for samp in data:
		sum += math.pow((parameters[0] + parameters[1]*samp[0] - samp[1]), 2)

	return alpha*(1/(2*n))*sum

def deriv_para1(parameters, data):

	sum = 0
	for samp in data:
		sum += (parameters[0] + parameters[1]*samp[0] - samp[1])

	return alpha*(1/n)*sum

def deriv_para2(parameters, data):

	sum = 0
	for samp in data:
		sum += (parameters[0] + parameters[1]*samp[0] - samp[1]) * samp[0]

	return alpha*(1/n)*sum



#gradient descent
#number of iterations
for i in range(0,20):

	print "true y values: 5, 6, 9, 17"
	print "current parameters = " + str(parameters)
	print "current hypo values: " + str(hypothesis(parameters, data[0])) + ' ' \
									+ str(hypothesis(parameters, data[1])) + ' ' \
									+ str(hypothesis(parameters, data[2])) + ' ' \
									+ str(hypothesis(parameters, data[3]))
	print ' '

	temp_para1 = parameters[0] - deriv_para1(parameters, data)
	temp_para2 = parameters[1] - deriv_para2(parameters, data)


	parameters = [temp_para1, temp_para2]


print obj(parameters, data)
