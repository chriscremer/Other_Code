

import theano
from pylearn2.models import mlp
from pylearn2.train_extensions import best_params
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.utils import serial
from pylearn2.termination_criteria import MonitorBased
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from sklearn.preprocessing import StandardScaler
import numpy as np
from random import randint
import itertools
import os
import csv
import random
import pylearn2 as PL
#os.system('rm /tmp/best.pkl')

 
MY_DATASET = '/data1/morrislab/ccremer/simulated_data/simulated_classification_data_10_samps_5_feats_3_distinct.csv'
 
scaler = StandardScaler()
 
class the_data(DenseDesignMatrix):
    def __init__(self, X=None, y=None):
        X = X
        y = y
        if X is None:
			X = []
			y = []
			header = True
			with open(MY_DATASET, 'r') as f:
				csvreader = csv.reader(f, delimiter=',', skipinitialspace=True)
				for row in csvreader:
					if header:
						header = False
						continue
					#X.append(row[1:-1])
                    X.append(map(float,row[1:-1]))
					if str(row[-1]) == '0.0':
						y.append([1, 0])
					else:
						y.append([0, 1])

			X = np.asarray(X)
			#################
			# this is wrong, should be fitted to training data then used to transform all data sets
			###########
			X = scaler.fit_transform(X)
			y = np.asarray(y)

			#shuffle it up
			random_shuffle = random.sample(range(len(y)), len(y))
			#print random_samp
			print X.shape
			X = X[random_shuffle]
			print y.shape
			y = y[random_shuffle]

			#print X
			#print y
        super(the_data, self).__init__(X=X, y=y)
 
    @property
    def nr_inputs(self):
        return len(self.X[0])
 
    def split(self, prop=.8):
        cutoff = int(len(self.y) * prop)
        X1, X2 = self.X[:cutoff], self.X[cutoff:]
        y1, y2 = self.y[:cutoff], self.y[cutoff:]
        return the_data(X1, y1), the_data(X2, y2)
 
    def __len__(self):
        return self.X.shape[0]
 
    def __iter__(self):
        return itertools.izip_longest(self.X, self.y)
    

# create datasets
ds_train = the_data()
ds_train, ds_valid = ds_train.split(0.7)
ds_valid, ds_test = ds_valid.split(0.7)


#####################################
#Define Model
#####################################
 
# create sigmoid hidden layer with 20 nodes, init weights in range -0.05 to 0.05 and add
# a bias with value 1
hidden_layer = mlp.Sigmoid(layer_name='h0', dim=1, irange=.05, init_bias=1.)
# softmax output layer
output_layer = mlp.Softmax(2, 'softmax', irange=.05)
layers = [hidden_layer, output_layer]

# create neural net
ann = mlp.MLP(layers, nvis=ds_train.nr_inputs)

#####################################
#Define Training
#####################################

#L1 Weight Decay
L1_cost = PL.costs.cost.SumOfCosts([PL.costs.cost.MethodCost(method='cost_from_X'), PL.costs.mlp.L1WeightDecay(coeffs=[0.1, 0.01])])

# momentum
initial_momentum = .5
final_momentum = .99
start = 1
saturate = 20
momentum_adjustor = learning_rule.MomentumAdjustor(final_momentum, start, saturate)
momentum_rule = learning_rule.Momentum(initial_momentum)
 
# learning rate
start = .1
saturate = 20
decay_factor = .00001
learning_rate_adjustor = sgd.LinearDecayOverEpoch(start, saturate, decay_factor)

# termination criterion that stops after 50 epochs without
# any increase in misclassification on the validation set
termination_criterion = MonitorBased(channel_name='objective', N=20, prop_decrease=0.0)
 
# create Stochastic Gradient Descent trainer 
trainer = sgd.SGD(learning_rate=.001,
                    batch_size=10,
                    monitoring_dataset=ds_valid, 
                    termination_criterion=termination_criterion, 
                    cost=L1_cost)
#learning_rule=momentum_rule,
trainer.setup(ann, ds_train) 

# add monitor for saving the model with best score
monitor_save_best = best_params.MonitorBasedSaveBest('objective','./tmp/best.pkl')
 

#####################################
#Train model
####################################

# train neural net until the termination criterion is true
while True:
    trainer.train(dataset=ds_train)
    ann.monitor.report_epoch()
    ann.monitor()
    monitor_save_best.on_monitor(ann, ds_valid, trainer)
    if not trainer.continue_learning(ann):
        break

    #try to print params
    for param in range(len(ann.get_params())):
        print ann.get_params()[param]
        print ann.get_param_values()[param]

    #momentum_adjustor.on_monitor(ann, ds_valid, trainer)
    #learning_rate_adjustor.on_monitor(ann, ds_valid, trainer)
 

#####################################
#Now examine model
#####################################

# load the best model
ann = serial.load('./tmp/best.pkl')

#try to print params
for param in range(len(ann.get_params())):
	print ann.get_params()[param]
	print ann.get_param_values()[param]

 
# function for classifying a input vector
def classify(inp):
    inp = np.asarray(inp)
    inp.shape = (1, ds_train.nr_inputs)
    print ann.fprop(theano.shared(inp, name='inputs')).eval()
    return np.argmax(ann.fprop(theano.shared(inp, name='inputs')).eval())
 
# function for calculating and printing the models accuracy on a given dataset
def score(dataset):
    nr_correct = 0
    for features, label in dataset:
    	print 'Predict ' + str(classify(features)) + ' Label ' + str(np.argmax(label))
        if classify(features) == np.argmax(label):
            nr_correct += 1
    print '%s/%s correct' % (nr_correct, len(dataset))
 
print
print 'Accuracy of train set:'
score(ds_train)
print 'Accuracy of validation set:'
score(ds_valid)
print 'Accuracy of test set:'
score(ds_test)