



# with relu, cant get past the first iteration because it all blows up, the error is massive and so are the grads
# so thats why Im thinking of doing layer normaliztion

#this one will have layer norm

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

import time


fig = plt.figure(frameon=False)

learning_rate = .001
cmap = 'seismic'
vmin = -3
vmax = 3

n = 100  #number data points
d = 10   #data and layer width
depth = 3   #net depth
activation_func = F.tanh #F.sigmoid#  F.relu #  # 

data = np.random.uniform(-10,10,n*d)
train_x = np.reshape(data, [n,d])
data = np.random.uniform(-3,3,n*d)
train_y = np.reshape(data, [n,d]) 
train_x = Variable(torch.FloatTensor(train_x), requires_grad = True)
train_y = Variable(torch.FloatTensor(train_y), requires_grad = True)

weights = []
gains = []
biases = []
for l in range(depth-1):
    weights.append(Variable(torch.randn(d,d), requires_grad = True))
    gains.append(Variable(torch.randn(d), requires_grad = True))
    biases.append(Variable(torch.randn(d), requires_grad = True))

last = Variable(torch.randn(d,d), requires_grad = True)


acts = []


for i in range(10000):

    for l in range(len(weights)):

        if l == 0:
            z = torch.matmul(train_x, weights[l])  #[B,D]

            #layer norm
            layer_mean = torch.mean(z,1,keepdim=True)
            layer_var = torch.var(z,1,keepdim=True)
            z = (z - layer_mean) * (gains[l]/layer_var) + biases[l]

            a = activation_func(z)  # [B,D]
            acts.append(a)
            a.retain_grad()
        else:
            z = torch.matmul(a, weights[l])  #[B,D]

            #layer norm
            layer_mean = torch.mean(z,1,keepdim=True)
            layer_var = torch.var(z,1,keepdim=True)
            z = (z - layer_mean) * (gains[l]/layer_var) + biases[l]

            #resnet 
            a2 = activation_func(z)  # [B,D]

            # a = a +a2
            a = a2
            # a = activation_func(z)  # [B,D]

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
    print_count = 0
    if i%100000==0 and i!= 0:
    # if 1:
        # print weights
        # print last

        frame = np.zeros((d,depth))
        frame2= np.zeros((d,depth))


        #FORWARD PROP
        for l in range(len(weights)):

            ax = plt.subplot2grid((2,1), (0,0))#, frameon=False)
            frame.T[l] = acts[l].data.numpy()[0] #first datapoint
            # print frame.T[l]
            plt.imshow(frame, cmap=cmap, vmax=vmax, vmin=vmin)#, aspect='auto')
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()

            ax = plt.subplot2grid((2,1), (1,0))#, frameon=False)
            plt.imshow(frame2, cmap=cmap, vmax=vmax, vmin=vmin)#, aspect='auto')
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()

            plt.savefig(home+'/Documents/tmp/a' +str(print_count) + 'aa' +str(i) + str(l) +'.png', bbox_inches='tight')
            print 'saved' + home+'/Documents/tmp/' +str(print_count) + 'aaa' +str(i) + str(l) + '.png'
            print_count +=1

        # print l+1
        # print out.data.numpy()[0]
        # fsdf
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

        plt.savefig(home+'/Documents/tmp/a' +str(print_count) + 'aa' +str(i) + str(l+1) + '.png', bbox_inches='tight')
        print 'saved' + home+'/Documents/tmp/' +str(print_count) + 'aaa' +str(i) + str(l+1) + '.png'
        print_count +=1

        


        #BACKPROP
        ax = plt.subplot2grid((2,1), (0,0))#, frameon=False)
        plt.imshow(frame, cmap=cmap, vmax=vmax, vmin=vmin)#, aspect='auto')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()

        frame2.T[l+1] = out.grad.data.numpy()[0] #first datapoint
        ax = plt.subplot2grid((2,1), (1,0))#, frameon=False)
        plt.imshow(frame2, cmap=cmap, vmax=vmax, vmin=-3)#, aspect='auto')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()

        plt.savefig(home+'/Documents/tmp/a' +str(print_count) + 'aa' +str(i) + str(l+1) +'back' +'.png', bbox_inches='tight')
        print 'saved' + home+'/Documents/tmp/' +str(print_count) + 'aaa' +str(i) + str(l+1)+'back' + '.png'
        print_count +=1


        backward_layers= range(len(weights))[::-1]
        for l in backward_layers:

            ax = plt.subplot2grid((2,1), (0,0))#, frameon=False)
            plt.imshow(frame, cmap=cmap, vmax=vmax, vmin=vmin)#, aspect='auto')
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()

            frame2.T[l] = acts[l].grad.data.numpy()[0] #first datapoint
            # print frame2.T[l] 
            ax = plt.subplot2grid((2,1), (1,0))#, frameon=False)
            plt.imshow(frame2, cmap=cmap, vmax=vmax, vmin=vmin)#, aspect='auto')
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()

            plt.savefig(home+'/Documents/tmp/a' +str(print_count) + 'aa' +str(i) + str(l) +'back' +'.png', bbox_inches='tight')
            print 'saved' + home+'/Documents/tmp/' +str(print_count) + 'aaa' +str(i) + str(l)+'back' + '.png'
            print_count +=1



        fsafa


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




















