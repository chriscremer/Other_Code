



# with relu, cant get past the first iteration because it all blows up, the error is massive and so are the grads
# so thats why Im thinking of doing layer normaliztion



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


fig = plt.figure(frameon=False)

learning_rate = .001
cmap = 'seismic'
vmin = -3
vmax = 3

n = 100  #number data points
d = 10   #data and layer width
depth = 10   #net depth
activation_func = F.relu # F.sigmoid#  # 

data = np.random.uniform(0,1,n*d)
train_x = np.reshape(data, [n,d])
data = np.random.uniform(0,1,n*d)
train_y = np.reshape(data, [n,d]) 
train_x = Variable(torch.FloatTensor(train_x), requires_grad = True)
train_y = Variable(torch.FloatTensor(train_y), requires_grad = True)

weights = []
for l in range(depth-1):
    weights.append(Variable(torch.randn(d,d), requires_grad = True))
    # print weights[l]
last = Variable(torch.randn(d,d), requires_grad = True)
# print last

acts = []


for i in range(10000):

    for l in range(len(weights)):

        if l == 0:
            a = torch.matmul(train_x, weights[l])
            a = activation_func(a)  # [B,D]
            acts.append(a)
            a.retain_grad()
        else:
            a2 = torch.matmul(a, weights[l])
            a2 = activation_func(a2)
            a = a + a2
            acts.append(a)
            a.retain_grad()

    out = torch.matmul(a, last)
    out.retain_grad()

    error = torch.mean((out-train_y)**2, 0) #avg over batch and dimensions
    error = torch.sum(error)
    print i, error.data[0]


    #Backpropagate
    error.backward()


    #PLOT
    if i%1==0 and i!= 0:
    # if 1:
        # print weights
        print last

        frame = np.zeros((d,depth))
        frame2= np.zeros((d,depth))


        #FORWARD PROP
        for l in range(len(weights)):

            ax = plt.subplot2grid((2,1), (0,0))#, frameon=False)
            frame.T[l] = acts[l].data.numpy()[0] #first datapoint
            print frame.T[l]
            plt.imshow(frame, cmap=cmap, vmax=vmax, vmin=vmin)#, aspect='auto')
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()

            ax = plt.subplot2grid((2,1), (1,0))#, frameon=False)
            plt.imshow(frame2, cmap=cmap, vmax=vmax, vmin=vmin)#, aspect='auto')
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()

            plt.savefig(home+'/Documents/tmp/aaa' +str(i) + str(l) +'.png', bbox_inches='tight')
            print 'saved' + home+'/Documents/tmp/aaa' +str(i) + str(l) + '.png'

        print l+1
        print out.data.numpy()[0]
        fsdf
        ax = plt.subplot2grid((2,1), (0,0))#, frameon=False)
        frame.T[l+1] = out.data.numpy()[0] #first datapoint
        plt.imshow(frame, cmap=cmap, vmax=vmax, vmin=vmin)#, vmax=2)#, aspect='auto')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()

        ax = plt.subplot2grid((2,1), (1,0))#, frameon=False)
        plt.imshow(frame2, cmap=cmap, vmax=vmax, vmin=vmin)#, aspect='auto')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()

        plt.savefig(home+'/Documents/tmp/aaa' +str(i) + str(l+1) + '.png', bbox_inches='tight')
        print 'saved' + home+'/Documents/tmp/aaa' +str(i) + str(l+1) + '.png'

        


        #BACKPROP
        acts_grads = []
        backward_layers= range(len(weights))[::-1]
        for l in backward_layers:

            ax = plt.subplot2grid((2,1), (0,0))#, frameon=False)
            plt.imshow(frame, cmap=cmap, vmax=vmax, vmin=vmin)#, aspect='auto')
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()

            frame2.T[l+1] = acts[l].grad.data.numpy()[0] #first datapoint
            # print frame2.T[l] 
            ax = plt.subplot2grid((2,1), (1,0))#, frameon=False)
            plt.imshow(frame2, cmap=cmap, vmax=vmax, vmin=vmin)#, aspect='auto')
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()

            plt.savefig(home+'/Documents/tmp/aaa' +str(i) + str(l) +'back' +'.png', bbox_inches='tight')
            print 'saved' + home+'/Documents/tmp/aaa' +str(i) + str(l)+'back' + '.png'


        ax = plt.subplot2grid((2,1), (0,0))#, frameon=False)
        plt.imshow(frame, cmap=cmap, vmax=vmax, vmin=vmin)#, aspect='auto')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()

        frame2.T[l] = out.grad.data.numpy()[0] #first datapoint
        ax = plt.subplot2grid((2,1), (1,0))#, frameon=False)
        plt.imshow(frame2, cmap=cmap, vmax=vmax, vmin=-3)#, aspect='auto')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()

        plt.savefig(home+'/Documents/tmp/aaa' +str(i) + str(l-1) +'back' +'.png', bbox_inches='tight')
        print 'saved' + home+'/Documents/tmp/aaa' +str(i) + str(l-1)+'back' + '.png'

        


        # fsafa

        # # print acts_grads.shape
        # plt.imshow(acts_grads.T, cmap='Blues', vmax=2)#, aspect='auto')
        # # plt.colorbar()
        # ax.set_xticks([])
        # ax.set_yticks([])
        # # plt.xticks([])
        # # plt.yticks([])

        # # plt.show()
        # # plt.savefig('aaa'+str(i)+'.png')
        # # plt.gcf()
        # plt.savefig(home+'/Documents/tmp/aaa' +str(i) + '.png', bbox_inches='tight')
        # print 'saved' + home+'/Documents/tmp/aaa' +str(i) + '.png'
        # # fdsf


    #Gradient descent
    for l in range(len(weights)):

        weights[l].data -= learning_rate * weights[l].grad.data
        weights[l].grad.data.zero_()

    last.data -= learning_rate * last.grad.data
    last.grad.data.zero_()




















