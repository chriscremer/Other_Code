




import numpy as np
from os.path import expanduser
home = expanduser("~")
import time
import os


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from collections import deque


from load_data import load_mnist, load_cifar10

from fully_connected_net import Net as FCN
from conv_net import MNIST_ConvNet, CIFAR_ConvNet
from resnet import ResNet18
from PreActResNet import PreActResNet18, PreActResNet10




def train(train_x, train_y, valid_x, valid_y):

    epochs = 1
    batch_size = 100

    train_x = torch.from_numpy(train_x).float().type(torch.FloatTensor).cuda()
    train_y = torch.from_numpy(train_y)

    valid_x = torch.from_numpy(valid_x).float().type(torch.FloatTensor).cuda()
    valid_y = torch.from_numpy(valid_y)

    train_ = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True)

    valid_ = torch.utils.data.TensorDataset(valid_x, valid_y)
    valid_loader = torch.utils.data.DataLoader(valid_, batch_size=batch_size, shuffle=True)

    prev_accs_train = deque(maxlen=10)
    prev_accs_valid = deque(maxlen=10)

    start = time.time()

    model.train()#for BN
    


    for epoch in range(epochs):

        for batch_idx, (data, target) in enumerate(train_loader):

            batch = Variable(data)#.type(model.dtype)
            target = Variable(target).cuda()#.type(model.dtype)

            optimizer.zero_grad()

            output = model.forward(batch)

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if batch_idx % 200 == 0:

                pred = torch.max(output, dim=1)[1]
                train_batch_acc = target.eq(pred).float().mean()
                prev_accs_train.append(train_batch_acc.data.cpu().numpy()[0])

                model.eval()
                for batch_idx_valid, (data, target) in enumerate(valid_loader):
                    batch = Variable(data)#.type(model.dtype)
                    target = Variable(target).cuda()#.type(model.dtype)
                    output = model.forward(batch)
                    loss_valid = criterion(output, target)
                    pred = torch.max(output, dim=1)[1]
                    valid_batch_acc = target.eq(pred).float().mean()
                    prev_accs_valid.append(valid_batch_acc.data.cpu().numpy()[0])
                    break
                model.train()#for BN

                batch_time = time.time() - start 

                print ('Epoch:{:3d}/{:3d}'.format(epoch, epochs),
                    'batch:{:4d}'.format(batch_idx),
                    'time:{:.3f}'.format(batch_time),
                    '   Train loss:{:.3f}'.format(loss.data.cpu().numpy()[0]),
                    'acc:{:.3f}'.format(train_batch_acc.data.cpu().numpy()[0]),
                    'avgacc:{:.3f}'.format(np.mean(prev_accs_train)),
                    '  Valid loss:{:.3f}'.format(loss_valid.data.cpu().numpy()[0]),
                    'acc:{:.3f}'.format(valid_batch_acc.data.cpu().numpy()[0]),
                    'avgacc:{:.3f}'.format(np.mean(prev_accs_valid))
                    )
                start = time.time()





if __name__ == "__main__":

    load_ = 1
    save_ = 1
    save_file = home+'/Documents/tmp/model.pt'



    #Load data

    # train_x, train_y, valid_x, valid_y = load_mnist()

    train_x, train_y, valid_x, valid_y = load_cifar10()
    train_x = np.reshape(train_x, [train_x.shape[0], 3, 32, 32])
    valid_x = np.reshape(valid_x, [valid_x.shape[0], 3, 32, 32])

    print (train_x.shape)
    print (train_y.shape)
    print (valid_x.shape)
    print (valid_y.shape)
    print()


    #Init model
    print ('Loading model')
    use_cuda = True# torch.cuda.is_available()
    n_gpus = 1#2 #torch.cuda.device_count()
    if n_gpus < 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1' #which gpu

    if load_:
        loaded_state = torch.load(save_file)
        model = loaded_state['model']
        # model.load_params(path_to_load_variables=save_file)
        print ('loaded model ' + save_file)

    else:
        # model = CIFAR_ConvNet()
        # model = ResNet18()
        # model = PreActResNet18()
        model = PreActResNet10()

    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(n_gpus))

    print (model)
    print ()

    #Train model
    optimizer = optim.SGD(model.parameters(), lr=.005, momentum=.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    train(train_x, train_y, valid_x, valid_y)


    if save_:
        state = {
            'model': model.module if use_cuda else model,
            'epoch': 1,
        }
        # torch.save(model.state_dict(), save_file)
        torch.save(state, save_file)
        print ('saved model ' + save_file)















