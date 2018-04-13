

import numpy as np
import pickle

from os.path import expanduser
home = expanduser("~")

import torch




def load_mnist():

    print ('Loading MNIST')

    #Regular MNIST
    with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
        mnist_data = pickle.load(f, encoding='latin1')
    train_x = mnist_data[0][0]
    train_y = mnist_data[0][1]
    valid_x = mnist_data[1][0]
    valid_y = mnist_data[1][1]
    test_x = mnist_data[2][0]
    test_y = mnist_data[2][1]

    assert np.max(train_x) <= 1.
    assert np.min(train_x) >= 0.

    return train_x, train_y, valid_x, valid_y




def load_cifar10():

    print ('Loading CIFAR10')


    #CIFAR10
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    file_ = home+'/Documents/cifar-10-batches-py/data_batch_'
    for i in range(1,6):
        file__ = file_ + str(i)
        b1 = unpickle(file__)
        if i ==1:
            train_x = b1['data']
            train_y = b1['labels']
        else:
            train_x = np.concatenate([train_x, b1['data']], axis=0)
            train_y = np.concatenate([train_y, b1['labels']], axis=0)
    file__ = home+'/Documents/cifar-10-batches-py/test_batch'
    b1 = unpickle(file__)
    test_x = b1['data']
    test_y = np.array(b1['labels'])
    valid_x = test_x
    valid_y = test_y

    train_x = train_x / 255.
    valid_x = valid_x / 255.

    assert np.max(train_x) <= 1.
    assert np.min(train_x) >= 0.

    return train_x, train_y, valid_x, valid_y








# class MyDataset(torch.utils.Dataset):
#     def __init__(self, data_dir):
#         self.data_files = os.listdir(data_dir)
#         sort(self.data_files)

#     def __getindex__(self, idx):
#         return load_file(self.data_files[idx])

#     def __len__(self):
#         return len(self.data_files)


# dset = MyDataset()
# loader = torch.utils.DataLoader(dset, num_workers=8)






# class MyDataset2(torch.utils.data.Dataset):
#     def __init__(self, data_files):
#         self.data_files = data_files
#         # sort(self.data_files)

#     def __getindex__(self, idx):
#         fadfa
#         return load_file(self.data_files[idx])

#     def __len__(self):
#         return len(self.data_files)















