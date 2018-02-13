





import numpy as np

import os
from os.path import expanduser
home = expanduser("~")
import time
import pickle


# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt




import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F




# from bottleneck_NN import bottleneck_NN

# from bottleneck_BNN import bottleneck_BNN

from bottleneck_BNN_stochastic_firsthalf import bottleneck_BNN








def train(model, train_x, train_y, batch_size, display_epoch, path_to_save_variables):

    # train_y = torch.from_numpy(np.zeros(len(train_x)))
    train_x = torch.from_numpy(train_x).float().type(model.dtype)
    train_y = torch.from_numpy(train_y).type(model.dtype)

    train_ = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=.0001)

    for epoch in range(1, epochs + 1):

        for batch_idx, (data, target) in enumerate(train_loader):

            batch_x = Variable(data)#.type(model.dtype)
            batch_y = Variable(target).type(torch.LongTensor)#.type(model.dtype)

            acc = model.accuracy(batch_x, batch_y)
            # print (acc.data[0])

            optimizer.zero_grad()

            loss = model.forward(batch_x, batch_y)

            loss.backward()
            optimizer.step()


        if epoch%display_epoch==0:
            acc = model.accuracy(batch_x, batch_y)
            print ('Train Epoch: {}/{}'.format(epoch, epochs),
                'LL:{:.5f}'.format(loss.data[0]),
                'acc:{:.3f}'.format(acc.data[0]))


        # if total_epochs >= start_at and (total_epochs-start_at)%save_freq==0:

        #     # save params
        #     save_file = path_to_save_variables+'_encoder_'+str(total_epochs)+'.pt'
        #     torch.save(model.q_dist.state_dict(), save_file)
        #     print ('saved variables ' + save_file)
        #     # save_file = path_to_save_variables+'_generator_'+str(total_epochs)+'.pt'
        #     # torch.save(model.generator.state_dict(), save_file)
            # print ('saved variables ' + save_file)



    # save params
    save_params(model, path_to_save_variables)
    # save_file = path_to_save_variables+'_encoder_'+str(total_epochs)+'.pt'
    # torch.save(model.q_dist.state_dict(), save_file)
    # print ('saved variables ' + save_file)
    # save_file = path_to_save_variables+'_generator_'+str(total_epochs)+'.pt'
    # torch.save(model.generator.state_dict(), save_file)
    # print ('saved variables ' + save_file)


    print ('done training')








def load_params(model, path_to_load_variables=''):
    # model.load_state_dict(torch.load(path_to_load_variables))
    if path_to_load_variables != '':
        model.load_state_dict(torch.load(path_to_load_variables, map_location=lambda storage, loc: storage))
        print ('loaded variables ' + path_to_load_variables)


def save_params(model, path_to_save_variables=''):
    if path_to_save_variables != '':
        torch.save(model.state_dict(), path_to_save_variables)
        print ('saved variables ' + path_to_save_variables)










if __name__ == '__main__':


    # Which gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    train_ = 0
    viz_ = 1


    n_all_classes = 10
    n_limited_classes = 5


    print ('Loading data')
    with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
        mnist_data = pickle.load(f, encoding='latin1')

    train_x = mnist_data[0][0]
    train_y = mnist_data[0][1]
    valid_x = mnist_data[1][0]
    valid_y = mnist_data[1][1]
    test_x = mnist_data[2][0]
    test_y = mnist_data[2][1]

    print (train_x.shape)
    print (train_y.shape)


    x_size = len(train_x[0])

    #make limited dataset
    new_x = []
    new_y = []
    for i in range(len(train_y)):
        if train_y[i] < n_limited_classes:
            new_x.append(train_x[i])
            new_y.append(train_y[i])
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    print ('Limited Dataset')
    print (new_x.shape)
    print (new_y.shape)
    original_train_x = train_x
    original_train_y = train_y
    train_x = new_x
    train_y = new_y

    # #same on test set
    # new_x_test = []
    # new_y_test = []
    # for i in range(len(test_y)):

    #     if test_y[i] < n_limited_classes:
    #         new_x_test.append(test_x[i])
    #         new_y_test.append(test_y[i])
    # new_x_test = np.array(new_x_test)
    # new_y_test = np.array(new_y_test)

    print()


    hyper_config = { 
                    'x_size': x_size,
                    'output_size': n_limited_classes,
                    'act_func': F.relu,# F.tanh,# 
                    'first_half': [[x_size,200],[200,200],[200,2]],
                    'second_half': [[2,200],[200,200],[200,n_limited_classes]],
                    'use_gpu': False,
                    'which_gpu': '0'
                }

    print ('Init model')

    # model = bottleneck_NN(hyper_config)
    model = bottleneck_BNN(hyper_config)



    if torch.cuda.is_available():
        model.cuda()
    print (model)



    path_to_load_variables=''
    # path_to_save_variables=home+'/Documents/tmp/bottleneck_NN_5_classes.ckpt'
    # path_to_save_variables=home+'/Documents/tmp/bottleneck_BNN_firsthalfsotchastic_activation_reg.ckpt'
    path_to_save_variables=home+'/Documents/tmp/bottleneck_BNN_firsthalfsotchastic_5_classes.ckpt'



    if train_:

        batch_size = 50
        epochs = 20

        print ('\nTraining')
        train(model, train_x=train_x, train_y=train_y, batch_size=batch_size, 
                    display_epoch=1, path_to_save_variables=path_to_save_variables)


    else:
        load_params(model, path_to_load_variables=path_to_save_variables)

        # for_acc_x = Variable(torch.from_numpy(train_x).float().type(model.dtype))
        # for_acc_y = Variable(torch.from_numpy(train_y).type(model.dtype)).type(torch.LongTensor)#.type(model.dtype)
        # print ('Train Accuracy:', model.accuracy(for_acc_x, for_acc_y).data.numpy())








    if viz_:

        argmax_ = 1
        max_ = 1
        entropy_ = 1
        view_encodings_ = 1



        if view_encodings_:

            #choose samples
            n = 1
            # samples_from_each = [[] for x in range(n_all_classes)]
            samples_from_each = [[] for x in range(5)]
            for i in range(len(original_train_y)):
                if original_train_y[i] < 5:
                    if len(samples_from_each[original_train_y[i]]) < n:
                        samples_from_each[original_train_y[i]].append(original_train_x[i])   
                        # if np.array([len(x) for x in samples_from_each]).all():
                        #     break


        fig = plt.figure(figsize=(10,6), facecolor='white')

        rows = 1
        columns = 1

        #BOUNDARIES
        numticks = 200
        limit_value= 7

        x_min_max = [-limit_value,limit_value]
        y_min_max = [-limit_value,limit_value]

        x = np.linspace(*x_min_max, num=numticks)
        y = np.linspace(*y_min_max, num=numticks)
        X, Y = np.meshgrid(x, y)

        flat = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
        flat_torch_variable = Variable(torch.from_numpy(flat).float().type(model.dtype)) #[4000, 2]

        predictions_torch_variable = model.predict_from_encoding(flat_torch_variable) #, using_mean=True)  # [N,C]
        predictions = predictions_torch_variable.data.numpy()

        predictions_argmax = np.argmax(predictions, axis=1).reshape(X.shape) #[N] ->[numticks, numticks]
        predictions_max = np.max(predictions, axis=1).reshape(X.shape) #[N] -> [numticks, numticks]
        predictions_entropy = -np.sum(predictions*np.log(np.maximum(predictions, np.exp(-30))), axis=1).reshape(X.shape) #[N]

        # levels = [-1] + list(range(0,n_all_classes))
        # ax = plt.subplot2grid((rows,columns), (0,0), frameon=False)#, colspan=3)
        # cs = ax.contourf(X, Y, predictions_argmax, levels=levels, cmap='jet')
        # ax.annotate('Arg Max', xytext=(.4, 1.05), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Black')
        # plt.gca().set_aspect('equal', adjustable='box')
        # ax.tick_params(labelsize=6)

        # ax = plt.subplot2grid((rows,columns), (0,1), frameon=False)#, colspan=3)
        # cs = ax.contourf(X, Y, predictions_max, cmap='jet')
        # ax.annotate('Max', xytext=(.4, 1.05), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Black')
        # plt.gca().set_aspect('equal', adjustable='box')
        # ax.tick_params(labelsize=6)


        ax = plt.subplot2grid((rows,columns), (0,0), frameon=False)#, colspan=3)
        cs = ax.contourf(X, Y, predictions_entropy, cmap='jet')
        ax.annotate('Entropy', xytext=(.4, 1.05), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Black')
        plt.gca().set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=6)


        for j in range(12):
            #convert to encodings
            samples_from_each_encoded = []
            for i in range(5):
                pytorch_input = Variable(torch.from_numpy(np.array(samples_from_each[i])))
                # encodings_ = model.encode(pytorch_input).data.numpy() #[n,2]
                encodings_ = model.encode(pytorch_input).data.numpy() #[n,2]
                samples_from_each_encoded.append(encodings_)

            #See scatter
            for c in range(5):
                    aaa = ax.scatter(samples_from_each_encoded[c].T[0], samples_from_each_encoded[c].T[1], alpha=.5, label=str(c), c='black', marker='x', s=1)
                    for ii in range(len(samples_from_each_encoded[c])):
                        if c < n_limited_classes:
                            plt.text(samples_from_each_encoded[c][ii][0], samples_from_each_encoded[c][ii][1], str(c), color="black", fontsize=6)
                        else:
                            # plt.text(samples_from_each_encoded[c][ii][0], samples_from_each_encoded[c][ii][1], str(c), color="red", fontsize=6)
                            continue







        plt.show()





    print('\nDone\n')
















