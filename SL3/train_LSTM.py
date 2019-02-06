
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


np.random.seed(0)


def samp_func():

    des = 2*np.random.rand() -.5
    freq = 2*np.random.rand() 

    # f = lambda x: (-x*des+np.sin(freq*x) + np.random.rand()*.5).flatten()
    f = lambda x: (-x*des+np.sin(freq*x)).flatten()
    return f


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()


        # kwargs['act_func'] = F.leaky_relu

        # self.__dict__.update(kwargs)

        self.act_func = F.leaky_relu

        self.L = 50
        # state_size = 10

        # self.rnn_cell = nn.GRUCell(input_size=1, hidden_size=state_size)
        self.rnn_cell = nn.LSTMCell(input_size=state_size, hidden_size=state_size)
        # self.rnn_cell2 = nn.LSTMCell(input_size=state_size, hidden_size=state_size)
        
        self.m3 = nn.Linear(1, state_size)
        self.m1 = nn.Linear(state_size, self.L)
        self.m2 = nn.Linear(self.L, 2)


        # self.m3 = nn.Linear(2, self.L)
        # self.m4 = nn.Linear(self.L, self.L)
        # self.m5 = nn.Linear(self.L, state_size)
        # self.m6 = nn.Linear(state_size, 1)

        # self.m7 = nn.Linear(1, self.L)
        # self.m8 = nn.Linear(self.L, self.L)
        # self.m9 = nn.Linear(self.L, 1)



    def predict_output(self, z):

        out = self.act_func(self.m1(z))
        out = self.m2(out)
        mean = out[:,:1]
        logvar = out[:,1:]
        logvar = torch.clamp(logvar, min=-15., max=10.)
        return mean, logvar


    def update_state(self, x, h, c):

        x = self.act_func(self.m3(x))
        h, c = self.rnn_cell(x, (h,c))

        # h, c = self.rnn_cell2(h, (h,c))



        # h = self.rnn_cell(x, prev_z)

        # # print (x.shape)
        # # print (prev_z.shape)
        # # fsd
        # prev_z = self.m6(prev_z)

        # x = torch.cat([x, prev_z], 1)
        # upt = self.act_func(self.m3(x))
        # # upt = self.act_func(self.m4(upt))
        # z = self.m5(upt)

        # # z = prev_z + upt

        return h,c


    # def NN(self, x):

    #     out = self.act_func(self.m7(x))
    #     out = self.act_func(self.m8(out))
    #     out = self.m9(out)

    #     return out





save_to_dir = home + '/Documents/SL3/'
exp_name = 'test_LSTM'

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

n = 40     
# x_lim = [-12.5,12.5]
x_lim = [0, 25]
X_linspace = np.linspace(x_lim[0], x_lim[1], n).reshape(-1,1)
B = 32
state_size = 20
noise = .5

model = RNN().cuda()
# model = nn.GRUCell(1, 20).cuda()


lr = .0004
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=.0000001)


train_ = 0


# name = '_'
# step = 9999
# save_to=os.path.join(params_dir, "model_" +name + str(step)+".pt")
# state_dict = torch.load(save_to)
# model.load_state_dict(state_dict)
# print ('loaded params', save_to)


if train_:

    for step in range(10000):

        #Make batch
        batch = []
        # batch.append(np.zeros())
        for i in range(B):

            f = samp_func()
            seq = f(X_linspace) + np.random.rand(n)*noise #+ np.random.rand(n) *.1  + np.random.rand(n)
            seq = np.insert(seq, 0, 0)
            batch.append(seq)

        batch = torch.from_numpy(np.array(batch)).float().cuda()

        state = torch.zeros(B,state_size).cuda()
        c = torch.zeros(B,state_size).cuda()

        L = 0
        for t in range(1,n):

            state, c = model.update_state(torch.unsqueeze(batch[:,t],dim=1), state, c)
            out_mean, out_logvar = model.predict_output(state)

            # out_mean = model.NN(torch.unsqueeze(batch[:,t],dim=1))

            # out_mean = torch.unsqueeze(batch[:,t],dim=1) + .00001*out_mean

            target = torch.unsqueeze(batch[:,t+1],dim=1)

            # logprob = -.5*  torch.mean((target - out_mean)**2 / torch.exp(out_logvar) 

            L += torch.mean(((target - out_mean)**2  / torch.exp(out_logvar)) + out_logvar)
            # L += torch.mean( torch.abs(target - out_mean) ) #/ torch.exp(out_logvar) + out_logvar)
            


        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        if step %50 == 0:
            print (step, L.data.item())



    name = '_'
    save_to=os.path.join(params_dir, "model_"+name + str(step)+".pt")
    torch.save(model.state_dict(), save_to)
    print ('saved params', save_to)


else:
    name = '_'
    step = 9999
    save_to=os.path.join(params_dir, "model_" +name + str(step)+".pt")
    state_dict = torch.load(save_to)
    model.load_state_dict(state_dict)
    print ('loaded params', save_to)








# PLOTS:
ylim = [-15,15]

rows = 2
cols = 2


for plot_i in range (3):

    fig = plt.figure(figsize=(6+cols,2+rows), facecolor='white', dpi=150)   

    f = samp_func()
    seq = f(X_linspace) + np.random.rand(n)*noise
    seq2 = f(X_linspace)


    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False, colspan=1, rowspan=1)
    ax.plot(X_linspace, seq, 'b-', label='True', linewidth=1.)
    ax.plot(X_linspace, seq2, 'g-', label='No Noise', linewidth=1.)
    ax.tick_params(labelsize=6)
    ax.set_title('Example Sequence',fontsize=6,family='serif')
    ax.set_ylim(ylim[0], ylim[1]) 










    ax = plt.subplot2grid((rows,cols), (0,1), frameon=False, colspan=1, rowspan=1)

    B = 1
    state = torch.zeros(B,state_size).cuda()
    c = torch.zeros(B,state_size).cuda()
    out_mean = torch.unsqueeze(torch.from_numpy(np.array([0])),dim=1).float().cuda()
    seq = torch.unsqueeze(torch.from_numpy(np.array(seq)),dim=0).float().cuda()
    means = []
    for t in range(n//2):
        # print (seq.shape)
        # print (state.shape)
        # fds
        state, c = model.update_state(torch.unsqueeze(seq[:,t],dim=1), state, c)
        # state = model.update_state(torch.unsqueeze(batch[:,t],dim=1), state)
        # out_mean, out_logvar = model.predict_output(state)
        # means.append(out_mean.data.item())
        means.append(seq[:,t].data.item())

    ax.plot(X_linspace[:len(means)], means, 'b-', label='True', linewidth=1.)
    half_seq = seq[0].cpu().numpy()[len(means):]
    ax.plot(X_linspace[len(means):], half_seq, 'b-', label='True', linewidth=1., alpha=.3)


    # out_mean, out_logvar = model.predict_output(state)

    init_state = state.clone()
    init_c = c.clone()

    for ii in range(5):
        means = []
        # logvars = []
        for t in range(n//2):
            if t == 0:
                out_mean, out_logvar = model.predict_output(init_state)
                state, c = model.update_state(out_mean, init_state, init_c)
            else:
                out_mean, out_logvar = model.predict_output(state)
                state, c = model.update_state(out_mean, state, c)

            
            
            # means.append(out_mean.data.item())
            x_cur = out_mean + np.random.rand()* torch.exp(.5*out_logvar)
            means.append(x_cur.data.item())
            # logvars.append(out_logvar.data.item())

        means = np.array(means)
        # std = np.exp( np.array(logvars) *.5 )

        ax.plot(X_linspace[n//2:], means, 'g-', label='True', linewidth=1., alpha=.3)
        # plt.gca().fill_between(X_linspace[n//2:].flat, means-std, means+std, color="#dddddd")

    ax.tick_params(labelsize=6)
    ax.set_title('Sample given first half',fontsize=6,family='serif')
    ax.set_ylim(ylim[0], ylim[1]) 










    ax = plt.subplot2grid((rows,cols), (1,0), frameon=False, colspan=1, rowspan=1)

    state = torch.zeros(B,state_size).cuda()
    c = torch.zeros(B,state_size).cuda()
    # out_mean = torch.unsqueeze(torch.from_numpy(np.array([0])),dim=1).float().cuda()

    seq2 = seq.data.cpu().numpy()#.item()
    seq1 = np.insert(seq2, 0, 0)
    seq = torch.unsqueeze(torch.from_numpy(np.array(seq1)),dim=0).float().cuda()
    means = []
    logvars = []
    for t in range(n):
        # print (seq.shape)
        # print (state.shape)
        # fds


        state, c = model.update_state(torch.unsqueeze(seq[:,t],dim=1), state, c)
        out_mean, out_logvar = model.predict_output(state)


        # out_mean = model.NN(torch.unsqueeze(seq[:,t],dim=1))
        # out_mean = torch.unsqueeze(seq[:,t],dim=1) + .00001*out_mean

        # means.append(out_mean.data.item())
        means.append(out_mean.data.item())
        logvars.append(out_logvar.data.item())

    means = np.array(means)
    std = np.exp( np.array(logvars) *.5 )
    ax.plot(X_linspace, seq2[0], 'b-', label='True', linewidth=1.)
    ax.plot(X_linspace, means, 'g-', label='Pred', linewidth=1.)
    plt.gca().fill_between(X_linspace.flat, means-std, means+std, color="#dddddd")
    ax.tick_params(labelsize=6)
    ax.set_title('Filtering',fontsize=6,family='serif')
    ax.set_ylim(ylim[0], ylim[1]) 
    ax.legend(fontsize=6)












    ax = plt.subplot2grid((rows,cols), (1,1), frameon=False, colspan=1, rowspan=1)

    B = 1
    state = torch.zeros(B,state_size).cuda()
    c = torch.zeros(B,state_size).cuda()
    out_mean = torch.unsqueeze(torch.from_numpy(np.array([0])),dim=1).float().cuda()
    means = []
    for t in range(n):
        state, c = model.update_state(out_mean, state, c)
        out_mean, out_logvar = model.predict_output(state)
        means.append(out_mean.data.item())

    ax.plot(X_linspace, means, 'r-', label='Means', linewidth=1.)



    for ii in range(10):
        B = 1
        state = torch.zeros(B,state_size).cuda()
        c = torch.zeros(B,state_size).cuda()
        x_cur = torch.unsqueeze(torch.from_numpy(np.array([0])),dim=1).float().cuda()
        means = []
        for t in range(n):

            state, c = model.update_state(x_cur, state, c)
            out_mean, out_logvar = model.predict_output(state)
            x_cur =  np.random.rand()* torch.exp(.5*out_logvar) + out_mean
            means.append(x_cur.data.item())

        ax.plot(X_linspace, means, 'b-', label='Samp '+str(ii), linewidth=1., alpha=.3)





    ax.tick_params(labelsize=6)
    ax.set_title('Samples, no info',fontsize=6,family='serif')
    ax.set_ylim(ylim[0], ylim[1]) 
    # ax.legend(fontsize=6)












    plt.tight_layout()
    # pl.gca().set_aspect('equal')

    # pl.show()
    file_ =images_dir + 'test' + str(plot_i)+'.png'
    plt.savefig(file_) #, bbox_inches='tight')
    print ('saved image', file_)




















