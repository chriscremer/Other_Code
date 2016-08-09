


import numpy as np

import plotly
import plotly.graph_objs as go

import pickle
from os.path import expanduser
home = expanduser("~")

import BSP_GCE_tools as tools

# import cPickle, gzip, numpy
import gzip

# Load the dataset
f = gzip.open(home + '/Documents/MNIST_data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x = train_set[0]
train_y = tools.convert_array_to_one_hot_matrix(train_set[1])
valid_x = valid_set[0]
valid_y = tools.convert_array_to_one_hot_matrix(valid_set[1])
test_x = test_set[0]
test_y = tools.convert_array_to_one_hot_matrix(test_set[1])

print 'Training set ' +  str(train_x.shape)
print tools.class_counter(train_y)
print
print 'Validation set ' +  str(valid_x.shape)
print tools.class_counter(valid_y)
print
print 'Test set ' +  str(test_x.shape)
print tools.class_counter(test_y) 
print

fasfasdfs
#run pca on data.


data = [
	go.Scatter(
		x = bias_values,
		y = means[0],
		mode = 'lines',
		name = 'Training',
		error_y=dict(
			type='data',
			array=stds[0],
			visible=True)
	),

	# go.Scatter(
	# 	x = bias_values,
	# 	y = means[1],
	# 	mode = 'lines',
	# 	name = 'Validation',
	# 	error_y=dict(
	# 		type='data',
	# 		array=stds[1],
	# 		visible=True)
	# ),

	go.Scatter(
		x = bias_values,
		y = means[2],
		mode = 'lines',
		name = 'Test',
		error_y=dict(
			type='data',
			array=stds[2],
			visible=True)
	)
]

# data = [trace1,trace2,trace3]

layout = go.Layout(
	xaxis=dict(
		title='Bias'
		# range=[0,l],
		# type='linear',
		# dtick=20,
	),
	yaxis=dict(
		title='Accuracy'
		# range=[0,l],
		# type='linear',
		# dtick=20,
		# autorange=False,
		# showgrid=True,
		# zeroline=False,
		# showline=False,
		# autotick=True,
		# ticks='',
		# showticklabels=False
	),

	# width=950,
	# height=800
)


fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename=home + '/Documents/Bias_Sampling/accuracies_squared_10.html')

print 'saved plot'


