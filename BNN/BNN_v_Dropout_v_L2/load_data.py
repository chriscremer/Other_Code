



import cPickle
import pickle
import numpy as np


from os.path import expanduser
home = expanduser("~")




def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict





def convert_to_onehot(data_y):

    train_y_one_hot = []
    for i in range(len(data_y)):
        one_hot=np.zeros(10)
        one_hot[data_y[i]]=1.
        train_y_one_hot.append(one_hot)
    data_y = np.array(train_y_one_hot)

    return data_y





def load_cifar10():

    print 'Loading data'
    file_ = home+'/Documents/cifar-10-batches-py/data_batch_'
    for i in range(1,6):
        file__ = file_ + str(i)
        b1 = unpickle(file__)
        if i ==1:
            data_x = b1['data']
            data_y = b1['labels']
        else:
            data_x = np.concatenate([data_x, b1['data']], axis=0)
            data_y = np.concatenate([data_y, b1['labels']], axis=0)

    file__ = home+'/Documents/cifar-10-batches-py/test_batch'
    b1 = unpickle(file__)
    test_x = b1['data']
    test_y = b1['labels']

    data_y = convert_to_onehot(data_y)
    test_y = convert_to_onehot(test_y)


    return data_x, data_y, test_x, test_y










def load_mnist():


    print 'Loading data'
    with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
        mnist_data = pickle.load(f)

    train_x = mnist_data[0][0]
    train_y = mnist_data[0][1]
    valid_x = mnist_data[1][0]
    valid_y = mnist_data[1][1]
    test_x = mnist_data[2][0]
    test_y = mnist_data[2][1]

    # print train_x.shape
    # print train_y.shape

    train_y = convert_to_onehot(train_y)
    valid_y = convert_to_onehot(valid_y)
    test_y = convert_to_onehot(test_y)



    return train_x, valid_x, test_x, train_y, valid_y, test_y





def load_binarized_mnist():

    with open(home+'/Documents/MNIST_data/binarized_mnist.pkl', 'rb') as f:
        train_x, valid_x, test_x = pickle.load(f)


    return train_x, valid_x, test_x








def get_equal_each_class(n_classes, n_datapoints_each_class, data_x, data_y):
    
    count = [0]*n_classes

    for i in range(len(data_x)):

        if count[np.argmax(data_y[i])] < n_datapoints_each_class:

            if i ==0:
                new_x = np.reshape(data_x[i], [1,-1])
                new_y = np.reshape(data_y[i], [1,n_classes])

            else:
                new_x = np.concatenate([new_x, np.reshape(data_x[i], [1,-1])], axis=0)
                new_y = np.concatenate([new_y, np.reshape(data_y[i], [1,n_classes])], axis=0)

            count[np.argmax(data_y[i])] +=1


    return new_x, new_y















