
import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F







#todo
# relu
# resnet






        # fig = plt.figure(figsize=(10,6), facecolor='white')









# n = 500
# d = 20

# data = np.random.uniform(-10,10,n)
# train_x = np.reshape(data, [n/d,d])
# data = np.random.uniform(-10,10,n)
# train_y = np.reshape(data, [n/d,d]) #[100,5]

# train_x = Variable(torch.FloatTensor(train_x), requires_grad = True)
# train_y = Variable(torch.FloatTensor(train_y), requires_grad = True)

# weights = []
# for l in range(20):
#     weights.append(Variable(torch.randn(d,d), requires_grad = True))
#     # weights2 = Variable(torch.randn(5,5), requires_grad = True)

# last = Variable(torch.randn(d,d), requires_grad = True)


# acts = [0]*len(weights)

# learning_rate = .0001
# for i in range(10000):

#     for l in range(len(weights)):

#         if l == 0:
#             a = torch.matmul(train_x, weights[l])
#             a = F.sigmoid(a)
#             acts[l] = a
#             a.retain_grad()
#         else:
#             a = torch.matmul(a, weights[l])
#             a = F.sigmoid(a)
#             acts[l] = a
#             a.retain_grad()

#     out = torch.matmul(a, last)
#     out.retain_grad()

#     error = torch.mean((out-train_y)**2, 0)
#     error = torch.sum(error)

#     print i, error.data[0]
#     error.backward()


#     if i==10:
#         acts_grads = []
#         for l in range(len(weights)):
#             acts_grads.append(acts[l].grad[0].data.numpy())
#         acts_grads.append(out.grad[0].data.numpy())
#         acts_grads = np.array(acts_grads)
#         print acts_grads.shape
#         plt.imshow(np.abs(acts_grads.T), cmap='Blues')
#         plt.colorbar()
#         plt.show()
#         fdsf


#     for l in range(len(weights)):

#         weights[l].data -= learning_rate * weights[l].grad.data
#         weights[l].grad.data.zero_()

#     last.data -= learning_rate * last.grad.data
#     last.grad.data.zero_()


# fdsa


















# #this is the resnet one



# n = 500
# d = 20

# data = np.random.uniform(-10,10,n)
# train_x = np.reshape(data, [n/d,d])
# data = np.random.uniform(-10,10,n)
# train_y = np.reshape(data, [n/d,d]) #[100,5]

# train_x = Variable(torch.FloatTensor(train_x), requires_grad = True)
# train_y = Variable(torch.FloatTensor(train_y), requires_grad = True)

# weights = []
# for l in range(20):
#     weights.append(Variable(torch.randn(d,d), requires_grad = True))
#     # weights2 = Variable(torch.randn(5,5), requires_grad = True)

# last = Variable(torch.randn(d,d), requires_grad = True)


# acts = [0]*len(weights)

# learning_rate = .0001
# for i in range(10000):

#     for l in range(len(weights)):

#         if l == 0:
#             a = torch.matmul(train_x, weights[l])
#             # a = F.sigmoid(a)
#             a = F.relu(a)

#             acts[l] = a
#             a.retain_grad()
#         else:
#             a2 = torch.matmul(a, weights[l])
#             # a2 = F.sigmoid(a2)
#             a2 = F.relu(a2)

#             a = a+a2
#             acts[l] = a
#             a.retain_grad()

#     out = torch.matmul(a, last)
#     out.retain_grad()

#     error = torch.mean((out-train_y)**2, 0)
#     error = torch.sum(error)

#     print i, error.data[0]
#     error.backward()


#     if i%500==0 and i!= 0:


#         plt.cla()

#         fig = plt.figure(frameon=False)
#         ax = fig.add_axes([0, 0, 1, 1])
#         ax.axis('off')

#         # ax = plt.subplot2grid((1,1), (0,0), frameon=False)


#         acts_grads = []
#         for l in range(len(weights)):
#             # acts_grads.append(acts[l].grad[0].data.numpy())
#             acts_grads.append(np.mean(np.abs(acts[l].grad.data.numpy()),0))
#         # acts_grads.append(out.grad[0].data.numpy())


#         acts_grads = np.array(acts_grads)



#         # print acts_grads.shape
#         plt.imshow(acts_grads.T, cmap='Blues', vmax=2)#, aspect='auto')
#         # plt.colorbar()
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # plt.xticks([])
#         # plt.yticks([])

#         # plt.show()
#         # plt.savefig('aaa'+str(i)+'.png')
#         # plt.gcf()
#         plt.savefig(home+'/Documents/tmp/aaa' +str(i) + '.png', bbox_inches='tight')
#         print 'saved' + home+'/Documents/tmp/aaa' +str(i) + '.png'
#         # fdsf


#     for l in range(len(weights)):

#         weights[l].data -= learning_rate * weights[l].grad.data
#         weights[l].grad.data.zero_()

#     last.data -= learning_rate * last.grad.data
#     last.grad.data.zero_()






# # regular neural net v1



# n = 500
# d = 20
# depth = 20

# data = np.random.uniform(-10,10,n)
# train_x = np.reshape(data, [n/d,d])
# data = np.random.uniform(-10,10,n)
# train_y = np.reshape(data, [n/d,d]) #[100,5]

# train_x = Variable(torch.FloatTensor(train_x), requires_grad = True)
# train_y = Variable(torch.FloatTensor(train_y), requires_grad = True)

# weights = []
# for l in range(depth):
#     weights.append(Variable(torch.randn(d,d), requires_grad = True))
#     # weights2 = Variable(torch.randn(5,5), requires_grad = True)

# last = Variable(torch.randn(d,d), requires_grad = True)


# acts = [0]*len(weights)

# learning_rate = .0001
# for i in range(10000):

#     for l in range(len(weights)):

#         if l == 0:
#             a = torch.matmul(train_x, weights[l])
#             a = F.sigmoid(a)
#             # a = F.relu(a)

#             acts[l] = a
#             a.retain_grad()
#         else:
#             a = torch.matmul(a, weights[l])
#             a = F.sigmoid(a)
#             # a = F.relu(a)

#             # a = a+a2
#             acts[l] = a
#             a.retain_grad()

#     out = torch.matmul(a, last)
#     out.retain_grad()

#     error = torch.mean((out-train_y)**2, 0)
#     # error = torch.sum(error)
#     error = torch.mean(error)


#     print i, error.data[0]
#     error.backward()


#     if i%100==0 and i!= 0:


#         plt.cla()

#         fig = plt.figure(frameon=False)
#         ax = fig.add_axes([0, 0, 1, 1])
#         ax.axis('off')

#         # ax = plt.subplot2grid((1,1), (0,0), frameon=False)


#         acts_grads = []
#         for l in range(len(weights)):
#             # acts_grads.append(acts[l].grad[0].data.numpy())
#             acts_grads.append(np.mean(np.abs(acts[l].grad.data.numpy()),0))
#         # acts_grads.append(out.grad[0].data.numpy())


#         acts_grads = np.array(acts_grads)



#         # print acts_grads.shape
#         plt.imshow(acts_grads.T, cmap='Blues', vmax=2)#, aspect='auto')
#         # plt.colorbar()
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # plt.xticks([])
#         # plt.yticks([])

#         # plt.show()
#         # plt.savefig('aaa'+str(i)+'.png')
#         # plt.gcf()
#         plt.savefig(home+'/Documents/tmp/aaa' +str(i) + '.png', bbox_inches='tight')
#         print 'saved' + home+'/Documents/tmp/aaa' +str(i) + '.png'
#         # fdsf


#     for l in range(len(weights)):

#         weights[l].data -= learning_rate * weights[l].grad.data
#         weights[l].grad.data.zero_()

#     last.data -= learning_rate * last.grad.data
#     last.grad.data.zero_()





















# regular neural net v2



n = 500
d = 5
depth = 5

data = np.random.uniform(-10,10,n)
train_x = np.reshape(data, [n/d,d])
data = np.random.uniform(-10,10,n)
train_y = np.reshape(data, [n/d,d]) #[100,5]

train_x = Variable(torch.FloatTensor(train_x), requires_grad = True)
train_y = Variable(torch.FloatTensor(train_y), requires_grad = True)

weights = []
for l in range(depth):
    weights.append(Variable(torch.randn(d,d), requires_grad = True))
    # weights2 = Variable(torch.randn(5,5), requires_grad = True)

last = Variable(torch.randn(d,d), requires_grad = True)


acts = [0]*len(weights)

learning_rate = .001
for i in range(10000):

    for l in range(len(weights)):

        if l == 0:
            a = torch.matmul(train_x, weights[l])
            # a = F.sigmoid(a)
            a = F.relu(a)

            acts[l] = a
            a.retain_grad()
        else:
            a = torch.matmul(a, weights[l])
            # a = F.sigmoid(a)
            a = F.relu(a)

            # a = a+a2
            acts[l] = a
            a.retain_grad()

    out = torch.matmul(a, last)
    out.retain_grad()

    error = torch.mean((out-train_y)**2, 0)
    # error = torch.sum(error)
    error = torch.mean(error)


    print i, error.data[0]
    error.backward()


    if i%100==0 and i!= 0:


        plt.cla()

        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        # ax = plt.subplot2grid((1,1), (0,0), frameon=False)


        acts_grads = []
        for l in range(len(weights)):
            # acts_grads.append(acts[l].grad[0].data.numpy())
            acts_grads.append(np.mean(np.abs(acts[l].grad.data.numpy()),0))
        # acts_grads.append(out.grad[0].data.numpy())


        acts_grads = np.array(acts_grads)



        # print acts_grads.shape
        plt.imshow(acts_grads.T, cmap='Blues', vmax=2)#, aspect='auto')
        # plt.colorbar()
        ax.set_xticks([])
        ax.set_yticks([])
        # plt.xticks([])
        # plt.yticks([])

        # plt.show()
        # plt.savefig('aaa'+str(i)+'.png')
        # plt.gcf()
        plt.savefig(home+'/Documents/tmp/aaa' +str(i) + '.png', bbox_inches='tight')
        print 'saved' + home+'/Documents/tmp/aaa' +str(i) + '.png'
        # fdsf


    for l in range(len(weights)):

        weights[l].data -= learning_rate * weights[l].grad.data
        weights[l].grad.data.zero_()

    last.data -= learning_rate * last.grad.data
    last.grad.data.zero_()





















# #this is the long resnet one



# n = 500
# d = 20

# data = np.random.uniform(-10,10,n)
# train_x = np.reshape(data, [n/d,d])
# data = np.random.uniform(-10,10,n)
# train_y = np.reshape(data, [n/d,d]) #[100,5]

# train_x = Variable(torch.FloatTensor(train_x), requires_grad = True)
# train_y = Variable(torch.FloatTensor(train_y), requires_grad = True)

# weights = []
# for l in range(100):
#     weights.append(Variable(torch.randn(d,d), requires_grad = True))
#     # weights2 = Variable(torch.randn(5,5), requires_grad = True)

# last = Variable(torch.randn(d,d), requires_grad = True)


# acts = [0]*len(weights)

# learning_rate = .00001
# for i in range(10000):

#     for l in range(len(weights)):

#         if l == 0:
#             a = torch.matmul(train_x, weights[l])
#             a = F.sigmoid(a)
#             acts[l] = a
#             a.retain_grad()
#         else:
#             a2 = torch.matmul(a, weights[l])
#             a2 = F.sigmoid(a2)
#             a = a+a2
#             acts[l] = a
#             a.retain_grad()

#     out = torch.matmul(a, last)
#     out.retain_grad()

#     error = torch.mean((out-train_y)**2, 0)
#     error = torch.sum(error)

#     print i, error.data[0]
#     error.backward()


#     if i%500==0 and i!= 0:


#         plt.cla()

#         fig = plt.figure(frameon=False)
#         ax = fig.add_axes([0, 0, 1, 1])
#         ax.axis('off')

#         # ax = plt.subplot2grid((1,1), (0,0), frameon=False)


#         acts_grads = []
#         for l in range(len(weights)):
#             # acts_grads.append(acts[l].grad[0].data.numpy())
#             acts_grads.append(np.mean(np.abs(acts[l].grad.data.numpy()),0))
#         # acts_grads.append(out.grad[0].data.numpy())


#         acts_grads = np.array(acts_grads)



#         # print acts_grads.shape
#         plt.imshow(acts_grads.T, cmap='Blues', vmax=2)#, aspect='auto')
#         # plt.colorbar()
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # plt.xticks([])
#         # plt.yticks([])

#         # plt.show()
#         # plt.savefig('aaa'+str(i)+'.png')
#         # plt.gcf()
#         plt.savefig(home+'/Documents/tmp/aaa' +str(i) + '.png', bbox_inches='tight')
#         print 'saved' + home+'/Documents/tmp/aaa' +str(i) + '.png'
#         # fdsf


#     for l in range(len(weights)):

#         weights[l].data -= learning_rate * weights[l].grad.data
#         weights[l].grad.data.zero_()

#     last.data -= learning_rate * last.grad.data
#     last.grad.data.zero_()



