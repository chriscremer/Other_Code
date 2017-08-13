

import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")




import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



def train(model, train_x, train_y, valid_x=[], valid_y=[], 
            path_to_load_variables='', path_to_save_variables='', 
            epochs=10, batch_size=20, display_epoch=2):
    

    if path_to_load_variables != '':
        model.load_state_dict(torch.load(path_to_load_variables))
        print 'loaded variables ' + path_to_load_variables

    train = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=.001)

    for epoch in range(1, epochs + 1):

        for batch_idx, (data, target) in enumerate(train_loader):

            if data.is_cuda:
                data, target = Variable(data), Variable(target).type(torch.cuda.LongTensor)
            else:
                data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = model.loss(output, target)
            loss.backward()
            optimizer.step()

            if epoch%display_epoch==0 and batch_idx == 0:
                acc = model.accuracy(output, target)
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.4f}, acc {:.3f}'.format(
                    epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0], acc.data[0]))


    if path_to_save_variables != '':
        torch.save(model.state_dict(), path_to_save_variables)
        print 'Saved variables to ' + path_to_save_variables




class My_Model(nn.Module):
    def __init__(self):
        super(My_Model, self).__init__()
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 10)
        self.model_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        # return F.log_softmax(x)

    def loss(self, prediction, true):

        return self.model_loss(prediction, true)

    def accuracy(self, prediction, true):

        values, indices = torch.max(prediction,1)
        correct_prediction = torch.eq(indices, true).float()
        return torch.mean(correct_prediction)


print 'Loading data'
with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
    mnist_data = pickle.load(f)

train_x = mnist_data[0][0]
train_y = mnist_data[0][1]
valid_x = mnist_data[1][0]
valid_y = mnist_data[1][1]
test_x = mnist_data[2][0]
test_y = mnist_data[2][1]

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)

print train_x.shape
print train_y.shape



model = My_Model()

print 'GPU available:', torch.cuda.is_available()
if torch.cuda.is_available():
    print 'loading cuda'
    model.cuda()
    train_x = train_x.cuda()


path_to_load_variables=''
# path_to_load_variables=home+'/Documents/tmp/pytorch_first.pt'
# path_to_save_variables=home+'/Documents/tmp/pytorch_first.pt'
path_to_save_variables=''



train(model=model, train_x=train_x, train_y=train_y, valid_x=[], valid_y=[], 
            path_to_load_variables=path_to_load_variables, 
            path_to_save_variables=path_to_save_variables, 
            epochs=100, batch_size=200, display_epoch=1)



#Quesitons:
# biases in NN?  done automatically
# accuracy. done
# logits of cross entropy?? im just outputing linear now.
# look at validation
# maybe early stopping
# get it to work on gpus. done















# faddfs
# def train(self, train_x, train_y, valid_x=[], valid_y=[], display_step=5, 
#                     path_to_load_variables='', path_to_save_variables='', 
#                     epochs=10, batch_size=20):
#         '''
#         Train.
#         '''
#         random_seed=1
#         rs=np.random.RandomState(random_seed)
#         n_datapoints = len(train_y)
#         one_over_N = 1./float(n_datapoints)
#         arr = np.arange(n_datapoints)

#         if path_to_load_variables == '':
#             self.sess.run(self.init_vars)

#         else:
#             #Load variables
#             self.saver.restore(self.sess, path_to_load_variables)
#             print 'loaded variables ' + path_to_load_variables

#         #start = time.time()
#         for epoch in range(1,epochs+1):

#             #shuffle the data
#             rs.shuffle(arr)
#             train_x = train_x[arr]
#             train_y = train_y[arr]

#             data_index = 0
#             for step in range(n_datapoints/batch_size):

#                 #Make batch
#                 batch = []
#                 batch_y = []
#                 while len(batch) != batch_size:
#                     batch.append(train_x[data_index]) 
#                     one_hot=np.zeros(self.output_size)
#                     one_hot[train_y[data_index]]=1.
#                     batch_y.append(one_hot)
#                     data_index +=1

#                 # Fit training using batch data
#                 _ = self.sess.run((self.optimizer), feed_dict={self.x: batch, self.y: batch_y, 
#                                                         self.one_over_N: one_over_N})

#                 # Display logs per epoch step
#                 if step % display_step == 0:

#                     # cost,logpy,logpW,logqW,pred = self.sess.run((self.cost,self.logpy, self.logpW, self.logqW,self.prediction), feed_dict={self.x: batch, self.y: batch_y, 
#                     #                                     self.batch_fraction_of_dataset: 1./float(n_datapoints)})

#                     cost = self.sess.run((self.cost), feed_dict={self.x: batch, self.y: batch_y, 
#                                                         self.one_over_N: one_over_N})



#                     # cost = -cost #because I want to see the NLL
#                     print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "cost=", "{:.4f}".format(float(cost))#,logpy,logpW,logqW #, 'time', time.time() - start



#         if path_to_save_variables != '':
#             self.saver.save(self.sess, path_to_save_variables)
#             print 'Saved variables to ' + path_to_save_variables









# # Training settings
# parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for training (default: 64)')
# parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                     help='input batch size for testing (default: 1000)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                     help='learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                     help='SGD momentum (default: 0.5)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)


# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x)

# model = Net()
# if args.cuda:
#     model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         # if args.cuda:
#         #     data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.data[0]))

# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))


# for epoch in range(1, args.epochs + 1):
#     train(epoch)
    # test()










