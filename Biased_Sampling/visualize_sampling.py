


import numpy as np

import plotly
import plotly.graph_objs as go

import random

import pickle
from os.path import expanduser
home = expanduser("~")

import cPickle, gzip

import BSP_GCE_tools as tools

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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


valid_x_alt, valid_y_alt = tools.bias_sampling3(valid_x, valid_y, .9, 5000)
print 'Validation set ' +  str(valid_x_alt.shape)
print tools.class_counter(valid_y_alt)

# valid_x_alt = valid_x
# valid_y_alt = valid_y


#run pca on data.
pca = PCA(n_components=5)
valid_pca = pca.fit_transform(valid_x_alt)

# model = TSNE(n_components=2)
# valid_tsne = model.fit_transform(valid_pca)

# print valid_tsne.shape

data = valid_pca

#shuffle
indexes = random.sample(range(len(valid_y_alt)), len(valid_y_alt))
data = data[indexes]
valid_y_alt = valid_y_alt[indexes]

x1 = []
y1 = []
x2 = []
y2 = []
# for i in range(len(valid_pca)):
for i in range(3000):

	if np.argmax(valid_y_alt[i]) > 4:

		x2.append(float(data[i][0]))
		y2.append(float(data[i][1]))
	else:
		x1.append(float(data[i][0]))
		y1.append(float(data[i][1]))

# x = list(x)
# print len(x)

# x= valid_pca.T[0]
# y= valid_pca.T[1]

# print len(x)
# fasf
# for class_ in range(1):

# 	print class_

# 	x = []
# 	y = []

# 	for i in range(len(valid_y_alt)):

# 		if np.argmax(valid_y_alt[i]) == class_:

# 			x.append(valid_pca.T[0])
# 			y.append(valid_pca.T[1])

# 	print len(x)

	# a = go.Scatter(x=x,y=y,mode='markers')


data = [go.Scatter(x=x1,y=y1,mode='markers',name='Over-sampled'),go.Scatter(x=x2,y=y2,mode='markers',name='Under-sampled')]

# data = [
# 	go.Scatter(
# 		x = valid_pca.T[0],
# 		y = valid_pca.T[1],
# 		mode = 'markers'
# 		# mode = 'lines',
# 		# name = 'Training',
# 		# error_y=dict(
# 		# 	type='data',
# 		# 	array=stds[0],
# 		# 	visible=True)
# 	),

# 	# go.Scatter(
# 	# 	x = bias_values,
# 	# 	y = means[1],
# 	# 	mode = 'lines',
# 	# 	name = 'Validation',
# 	# 	error_y=dict(
# 	# 		type='data',
# 	# 		array=stds[1],
# 	# 		visible=True)
# 	# ),

# 	# go.Scatter(
# 	# 	x = bias_values,
# 	# 	y = means[2],
# 	# 	mode = 'lines',
# 	# 	name = 'Test',
# 	# 	error_y=dict(
# 	# 		type='data',
# 	# 		array=stds[2],
# 	# 		visible=True)
# 	# )
# ]

# # data = [trace1,trace2,trace3]

layout = go.Layout(
	xaxis=dict(
		title='PC1'
		# range=[0,l],
		# type='linear',
		# dtick=20,
	),
	yaxis=dict(
		title='PC2'
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

print 'plotting'

fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename=home + '/Documents/Bias_Sampling/valid_pca.html')

print 'saved plot'


