
import os
from os.path import expanduser
home = expanduser("~")

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import math

np.random.seed(0)


def samp_func():

    des = 2* (np.random.rand()  -.5)
    freq = 2*np.random.rand() 
    amp = 10* (np.random.rand()  -.5)

    # f = lambda x: (-x*des+np.sin(freq*x) + np.random.rand()*.5).flatten()
    f = lambda x: (-x*des+amp*np.sin(freq*x)).flatten()
    return f




# class Norm(nn.Module):
#     def __init__(self, d_model, eps = 1e-6):
#         super().__init__()
    
#         self.size = d_model
#         # create two learnable parameters to calibrate normalisation
#         self.alpha = nn.Parameter(torch.ones(self.size))
#         self.bias = nn.Parameter(torch.zeros(self.size))
#         self.eps = eps
#     def forward(self, x):
#         norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
#         / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
#         return norm






class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.k = 9
        k=self.k

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=20, kernel_size=k, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=k, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv3 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=k, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv4 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=k, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv5 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=k, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv6 = nn.Conv1d(in_channels=20, out_channels=2, kernel_size=k, stride=1, padding=0, dilation=1, groups=1, bias=True)


        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(20)


        # self.norm_1 = Norm(20)
        # self.norm_2 = Norm(20)
        # self.norm_3 = Norm(20)


        self.act_func = F.leaky_relu

    def conv_masked(self, x, conv, kernel_size):

        #[B,C,L]

        channels = x.shape[1]

        # print (x.shape)

        #shift right 
        x = torch.cat([torch.zeros(B,channels,kernel_size).cuda(),x],dim=2)
        x = x[:,:,:-1]
        # batch_shifted = torch.unsqueeze(batch_shifted, dim=2)
        # print (batch.shape)
        # fsdaf

        #pad for the filter, k-(k/2)
        # batch_shifted = torch.cat([torch.zeros(B,2,1).cuda(),batch_shifted],dim=1)

        # 

        x = conv(x)

        return x



    def forward(self, x):

        # print (x.shape) #B,L,C

        x = x.permute(0,2,1)  #B,C,L


        # x = self.conv_masked(x, self.conv1)
        # out = self.conv_masked(x, self.conv2)

        k = self.k

        # x = self.conv_masked(x, self.conv1, k)
        # # x2 = self.norm_1(x)
        # x = x +  self.conv_masked(self.act_func(self.conv_masked(x, self.conv2, k)), self.conv3, k)
        # # x2 = self.norm_2(x)
        # x = x +  self.conv_masked(self.act_func(self.conv_masked(x, self.conv4, k)), self.conv5, k)
        # out = self.conv_masked(x, self.conv6, k)



        x = self.conv_masked(x, self.conv1, k)
        # x2 = self.norm_1(x)
        x = x +  self.conv_masked(self.act_func(self.conv_masked(self.bn1(x), self.conv2, k)), self.conv3, k)
        # x2 = self.norm_2(x)
        x = x +  self.conv_masked(self.act_func(self.conv_masked(self.bn2(x), self.conv4,k )), self.conv5, k)
        # out = self.conv_masked(self.norm_3(x), self.conv6)
        out = self.conv_masked(x, self.conv6, k)



        # x = self.norm_3(x)
        # x = self.predict_output(x)


        # out = self.act_func(self.m1(z))
        # out = self.m2(out)
        mean = out[:,0]
        logvar = out[:,1]
        logvar = torch.clamp(logvar, min=-10., max=10.)
        mean = mean.view(B,1,n)
        logvar = logvar.view(B,1,n)
        out = torch.cat([mean,logvar], dim=1)


        out = out.permute(0,2,1) #B,L,C

        # print (out.shape)

        return out










save_to_dir = home + '/Documents/SL3/'
exp_name = 'test_cnn'

exp_dir = save_to_dir + exp_name + '/'
params_dir = exp_dir + 'params/'
images_dir = exp_dir + 'images/'
# code_dir = exp_dir + 'code/'

if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    print ('Made dir', exp_dir) 

if not os.path.exists(params_dir):
    os.makedirs(params_dir)
    print ('Made dir', params_dir) 

if not os.path.exists(images_dir):
    os.makedirs(images_dir)
    print ('Made dir', images_dir) 

n = 30 # 40     
# x_lim = [-12.5,12.5]
x_lim = [0, 25]
X_linspace = np.linspace(x_lim[0], x_lim[1], n).reshape(-1,1)
B = 32
state_size = 20
noise = .3



# model = RNN().cuda()
# model = Att().cuda()
model = CNN().cuda()



# model = nn.GRUCell(1, 20).cuda()

# model.concat_position(3)

# fsdfa


lr = .0004
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=.0000001)


train_ = 1


# name = '_'
# step = 50000
# save_to=os.path.join(params_dir, "model_" +name + str(step)+".pt")
# state_dict = torch.load(save_to)
# model.load_state_dict(state_dict)
# print ('loaded params', save_to)

total_steps = 30000

if train_:

    for step in range(total_steps):

        #Make batch
        batch = []
        # batch.append(np.zeros())
        for i in range(B):

            f = samp_func()
            seq = f(X_linspace) + np.random.randn(n)*noise #+ np.random.rand(n) *.1  + np.random.rand(n)
            # seq = np.insert(seq, 0, 0)
            batch.append(seq)

        batch = torch.from_numpy(np.array(batch)).float().cuda()

        batch = torch.unsqueeze(batch, dim=2)  #B,L,C

        # print (batch.shape)
        # fsd




        # #shift right 
        # batch_shifted = torch.cat([torch.zeros(B,1).cuda(),batch],dim=1)
        # batch_shifted = batch_shifted[:,:-1]
        # batch_shifted = torch.unsqueeze(batch_shifted, dim=2)
        # # print (batch.shape)
        # # fsdaf

        # #pad for the filter, k-(k/2)
        # batch_shifted = torch.cat([torch.zeros(B,2,1).cuda(),batch_shifted],dim=1)

        # batch_shifted = batch_shifted.permute(0,2,1)



        # batch_withposition = concat_position(batch_shifted)

        # print (batch.shape)
        # fsafs

        # print (batch_shifted.shape)
        preds = model.forward(batch)  #[B,L,2]

        


        # print (preds.shape)
        # fdsfa
        # fdsfsa

        # print (preds.shape)
        # print (batch.shape)

        mean = torch.unsqueeze(preds[:,:,0], dim=2)
        logvar = torch.unsqueeze(preds[:,:,1], dim=2)

        # batch = torch.unsqueeze(batch, dim=2)
        # print (batch.shape)
        # print (mean.shape)
        # fadsdf


        L = torch.sum((batch - mean)**2 / torch.exp(logvar) + logvar, 1)
        L = torch.mean(L)
        # print (L.shape)


        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        if step %100 == 0:
            print (step, total_steps, L.data.item())



    name = '_'
    save_to=os.path.join(params_dir, "model_"+name + str(step+1)+".pt")
    torch.save(model.state_dict(), save_to)
    print ('saved params', save_to)


else:
    name = '_'
    step = 100000
    save_to=os.path.join(params_dir, "model_" +name + str(step)+".pt")
    state_dict = torch.load(save_to)
    model.load_state_dict(state_dict)
    print ('loaded params', save_to)

















# PLOTS:
ylim = [-20,20]

rows = 2
cols = 2

with torch.no_grad():

    for plot_i in range (3):

        fig = plt.figure(figsize=(6+cols,2+rows), facecolor='white', dpi=150) 


        ax = plt.subplot2grid((rows,cols), (0,0), frameon=False, colspan=1, rowspan=1)
        for ii in range(30):  

            f = samp_func()
            seq = f(X_linspace) + np.random.randn(n)*noise
            # seq2 = f(X_linspace)


            
            ax.plot(X_linspace, seq, 'b-', label='True', linewidth=1., alpha=.3)
            # ax.plot(X_linspace, seq2, 'g-', label='No Noise', linewidth=1.)

        ax.tick_params(labelsize=6)
        ax.set_title('Sequence Distribution',fontsize=6,family='serif')
        ax.set_ylim(ylim[0], ylim[1]) 
        ax.grid(alpha=.2)









        ax = plt.subplot2grid((rows,cols), (0,1), frameon=False, colspan=1, rowspan=1)

        B = 1

        for ii in range(30):

            outputs = torch.zeros(n)
            outputs = outputs.view(B,n,1).cuda()

            # if ii ==29:
            #     means = []
            #     logvars = []

            for t in range(n):
                # print (outputs.shape)

                # batch_shifted = torch.cat([torch.zeros(B,1,1).cuda(),outputs],dim=1)
                # batch_shifted = batch_shifted[:,:-1]
                # # batch_shifted = torch.unsqueeze(batch_shifted, dim=2)

                # # batch_withposition = concat_position(batch_shifted)

                # #pad for the filter, k-(k/2)
                # batch_shifted = torch.cat([torch.zeros(B,2,1).cuda(),batch_shifted],dim=1)

                # batch_shifted = outputs.permute(0,2,1)


                # outputs = concat_position(outputs)

                preds = model.forward(outputs)  #[B,L,2]

                mean = torch.unsqueeze(preds[:,:,0], dim=2).data.cpu().numpy()[0][t]
                logvar = torch.unsqueeze(preds[:,:,1], dim=2).data.cpu().numpy()[0][t]

                x_cur =  np.random.randn()* np.exp(.5*logvar) + mean
                # print (outputs[0,t,0].shape)
                # print (x_cur.shape)
                outputs[0,t,0] = torch.from_numpy(x_cur)
                outputs = outputs[:,:,0].view(1,n,1)


                # if ii ==29:
                #     means.append(mean)
                #     logvars.append(logvar)

            outputs = outputs.view(n).data.cpu().numpy()

            ax.plot(X_linspace, outputs, 'b-', label='Samp '+str(ii), linewidth=1., alpha=.3)

            # if ii ==29:

            #     # print (len(me))
            #     mean = np.reshape(np.array(means), [n])
            #     std = np.exp( np.array(logvars) *.5 )
            #     std = np.reshape(std, [n])

            #     ax.plot(X_linspace, mean, 'g-', label='Pred', linewidth=1.)
            #     plt.gca().fill_between(X_linspace.flat, mean-std, mean+std, color="#dddddd")


        ax.tick_params(labelsize=6)
        ax.set_title('Sequence Samples from Model',fontsize=6,family='serif')
        ax.set_ylim(ylim[0], ylim[1]) 
        # ax.legend(fontsize=6)
        ax.grid(alpha=.2)














        ax = plt.subplot2grid((rows,cols), (1,0), frameon=False, colspan=1, rowspan=1)

        # state = torch.zeros(B,state_size).cuda()
        # c = torch.zeros(B,state_size).cuda()
        # out_mean = torch.unsqueeze(torch.from_numpy(np.array([0])),dim=1).float().cuda()

        # seq2 = seq.data.cpu().numpy()#.item()
        # seq1 = np.insert(seq2, 0, 0)
        B =1
        batch = torch.unsqueeze(torch.from_numpy(seq),dim=0).float().cuda()
        # seq2 = torch.unsqueeze(seq2, dim=2)
        # print (seq2.shape)


        #shift right 
        # batch_shifted = torch.cat([torch.zeros(B,1).cuda(),batch],dim=1)
        # batch_shifted = batch_shifted[:,:-1]
        batch_shifted = torch.unsqueeze(batch, dim=2)
        # batch_withposition = concat_position(batch_shifted)
        #pad for the filter, k-(k/2)
        # batch_shifted = torch.cat([torch.zeros(B,2,1).cuda(),batch_shifted],dim=1)

        # batch_shifted = batch_shifted.permute(0,2,1)

        # seq2 = concat_position(seq2)
        # print (seq2.shape)

        # if plot_i ==2:
        #     print (seq2)
        
        preds = model.forward(batch_shifted)  #[B,L,2]

        mean = torch.unsqueeze(preds[:,:,0], dim=2).data.cpu().numpy()
        logvar = torch.unsqueeze(preds[:,:,1], dim=2).data.cpu().numpy()

        # means = np.array(means)
        mean = np.reshape(mean, [n])
        std = np.exp( np.array(logvar) *.5 )
        std = np.reshape(std, [n])

        # print (mean.shape, std.shape)
        ax.plot(X_linspace, seq, 'b-', label='True', linewidth=1.)
        ax.plot(X_linspace, mean, 'g-', label='Pred', linewidth=1.)
        plt.gca().fill_between(X_linspace.flat, mean-std, mean+std, color="#dddddd")
        ax.tick_params(labelsize=6)
        ax.set_title('Filtering',fontsize=6,family='serif')
        ax.set_ylim(ylim[0], ylim[1]) 
        ax.legend(fontsize=6)
        ax.grid(alpha=.2)
















        ax = plt.subplot2grid((rows,cols), (1,1), frameon=False, colspan=1, rowspan=1)


        B = 1

        seq = torch.unsqueeze(torch.from_numpy(np.array(seq)),dim=0).float().cuda()

        for ii in range(10):

            outputs = torch.zeros(n)
            # print (seq.shape)
            outputs[:n//2] = seq[0][:n//2]
            outputs = outputs.view(B,n,1).cuda()

            for t in range(n//2,n):
                # print (outputs.shape)

                # batch_shifted = torch.cat([torch.zeros(B,1,1).cuda(),outputs],dim=1)
                # batch_shifted = batch_shifted[:,:-1]
                # # batch_shifted = torch.unsqueeze(batch_shifted, dim=2)
                # # batch_withposition = concat_position(batch_shifted)
                # # outputs = concat_position(outputs)

                # #pad for the filter, k-(k/2)
                # batch_shifted = torch.cat([torch.zeros(B,2,1).cuda(),batch_shifted],dim=1)

                # batch_shifted = outputs.permute(0,2,1)

                preds = model.forward(outputs)  #[B,L,2]

                mean = torch.unsqueeze(preds[:,:,0], dim=2).data.cpu().numpy()[0][t]
                logvar = torch.unsqueeze(preds[:,:,1], dim=2).data.cpu().numpy()[0][t]


                # if plot_i ==2 and ii==0:
                #     print (outputs)
                #     print (mean)
                #     fads
                #     print (t)
                #     print (seq[0])
                #     print (seq[0][t])
                #     print (mean)
                #     print (np.exp(.5*logvar))
                #     print (np.random.rand()* np.exp(.5*logvar))

                #     print (seq[0].cpu().numpy()[:n//2])
                #     print (seq[0].cpu().numpy()[n//2:])
                #     print (X_linspace[n//2:])

                

                x_cur =  np.random.randn()* np.exp(.5*logvar) + mean

                # if plot_i ==2:
                #     print (x_cur)

                    # fasfdsa
                
                # print (outputs[0,t,0].shape)
                # print (x_cur.shape)
                outputs[0,t,0] = torch.from_numpy(x_cur)
                outputs = outputs[:,:,0].view(1,n,1)

                # print (outputs)



            outputs = outputs.view(n).data.cpu().numpy()
            # print(outputs)
            # print(outputs.shape)
            # print (outputs[n//2:])

            # print (X_linspace[n//2:])
            

            ax.plot(X_linspace[n//2:], outputs[n//2:], 'g-', label='Samp '+str(ii), linewidth=1., alpha=.3)
            # ax.plot(X_linspace, outputs, 'g-', label='Samp '+str(ii), linewidth=1., alpha=.3)


        first_half = seq[0].cpu().numpy()[:n//2]
        second_half = seq[0].cpu().numpy()[n//2:]

        ax.plot(X_linspace[:n//2], first_half, 'b-', label='True', linewidth=1.)
        ax.plot(X_linspace[n//2:], second_half, 'b-', label='True', linewidth=1., alpha=.3)

        # print (second_half)

        ax.tick_params(labelsize=6)
        ax.set_title('Sequence Samples Given First Half',fontsize=6,family='serif')
        ax.set_ylim(ylim[0], ylim[1]) 
        ax.grid(alpha=.2)









        plt.tight_layout()
        # pl.gca().set_aspect('equal')

        # pl.show()
        file_ =images_dir + 'test' + str(plot_i)+'.png'
        plt.savefig(file_) #, bbox_inches='tight')
        print ('saved image', file_)



























