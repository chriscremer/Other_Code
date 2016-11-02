


import pickle
import numpy as np

from os.path import expanduser
home = expanduser("~")

# import gzip
# with gzip.open('mnist.pkl.gz', 'rb') as f:
with open(home+ '/Documents/MNIST_data/mnist.pkl', 'rb') as f:
	train_set, valid_set, test_set = pickle.load(f)


train_x, train_y = train_set

print train_x.shape
print train_y.shape