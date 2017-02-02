
import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

def load_binarized_mnist():

    location = home+'/data/binarized_mnist.pkl'

    with open(location, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    train_x = train_set
    valid_x = valid_set
    test_x = test_set

    return train_x #, valid_x, test_x




def get_sequence(vector_length):
    #since mnist is 50000, do the same
    data = []
    for i in range(50000):
        position = np.random.randint(0, vector_length)
        state = np.zeros([vector_length])
        state[position] = 1.
        data.append(state)
        # state = np.array([.5]*vector_length) + (np.random.randn(vector_length) *.1)

    return np.array(data)



def load_my_data():
    #784 so its the same as MNIST
    #since mnist is 50000, do the same
    data = []
    block_size = 784/4 #196
    for i in range(50000):
        position = np.random.randint(0, 4)
        state = np.zeros([784])
        start = block_size * position
        end = start + block_size
        state[start:end] = 1.
        data.append(state)
        # state = np.array([.5]*vector_length) + (np.random.randn(vector_length) *.1)

    return np.array(data)






# print load_binarized_mnist(home+'/data/binarized_mnist.pkl').shape
# print get_sequence(10).shape








