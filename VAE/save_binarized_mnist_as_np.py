
import pickle
from os.path import expanduser
home = expanduser("~")


import numpy as np

def lines_to_np_array(lines):
    return np.array([[int(i) for i in line.split()] for line in lines])


with open('binarized_mnist_train.amat') as f:
    lines = f.readlines()
train_data = lines_to_np_array(lines).astype('float32')

print train_data.shape

with open('binarized_mnist_valid.amat') as f:
    lines = f.readlines()
validation_data = lines_to_np_array(lines).astype('float32')
print validation_data.shape
with open('binarized_mnist_test.amat') as f:
    lines = f.readlines()
test_data = lines_to_np_array(lines).astype('float32')
print test_data.shape


with open(home+'/Documents/MNIST_data/binarized_mnist.pkl', 'wb') as f:
	pickle.dump( [train_data, validation_data, test_data], f)

















