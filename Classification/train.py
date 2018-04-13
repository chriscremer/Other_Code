




import numpy as np
from os.path import expanduser
home = expanduser("~")
import time
import pickle


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from collections import deque



#Load data
print ('Loading data')

#Regular MNIST
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
print (valid_x.shape)
print (valid_y.shape)




#Init model

#Fully Connected
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         # self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(784, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         # x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         # x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         # x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


# Conv Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)






model = Net()
# if args.cuda:
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=.005, momentum=.9)
criterion = nn.CrossEntropyLoss()





def train(train_x, train_y, valid_x, valid_y):

    epochs = 10

    train_x = torch.from_numpy(train_x).float().type(torch.FloatTensor).cuda()
    train_y = torch.from_numpy(train_y)

    valid_x = torch.from_numpy(valid_x).float().type(torch.FloatTensor).cuda()
    valid_y = torch.from_numpy(valid_y)

    train_ = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_, batch_size=20, shuffle=True)

    valid_ = torch.utils.data.TensorDataset(valid_x, valid_y)
    valid_loader = torch.utils.data.DataLoader(valid_, batch_size=20, shuffle=True)

    prev_accs_train = deque(maxlen=10)
    prev_accs_valid = deque(maxlen=10)


    for epoch in range(10):

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


                for batch_idx_valid, (data, target) in enumerate(valid_loader):
                    batch = Variable(data)#.type(model.dtype)
                    target = Variable(target).cuda()#.type(model.dtype)
                    output = model.forward(batch)
                    loss_valid = criterion(output, target)
                    pred = torch.max(output, dim=1)[1]
                    valid_batch_acc = target.eq(pred).float().mean()
                    prev_accs_valid.append(valid_batch_acc.data.cpu().numpy()[0])


                print ('Epoch: {:3d}/{:3d}'.format(epoch, epochs),
                    'batch: {:4d}'.format(batch_idx),
                    '  Train loss:{:.3f}'.format(loss.data.cpu().numpy()[0]),
                    'acc:{:.3f}'.format(train_batch_acc.data.cpu().numpy()[0]),
                    'avgacc:{:.3f}'.format(np.mean(prev_accs_train)),
                    '  Valid loss:{:.3f}'.format(loss_valid.data.cpu().numpy()[0]),
                    'acc:{:.3f}'.format(valid_batch_acc.data.cpu().numpy()[0]),
                    'avgacc:{:.3f}'.format(np.mean(prev_accs_valid))
                    )


        


train(train_x, train_y, valid_x, valid_y)















