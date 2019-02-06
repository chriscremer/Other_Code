
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


def samp_func():

    des = 2*np.random.rand() -.5
    freq = 2*np.random.rand() 
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
        
        # self.m1 = nn.Linear(state_size, self.L)
        self.m2 = nn.Linear(state_size, 2)

        # self.m3 = nn.Linear(2, self.L)
        # self.m4 = nn.Linear(self.L, self.L)
        self.m5 = nn.Linear(1, state_size)
        self.m6 = nn.Linear(state_size, state_size)


        # self.m7 = nn.Linear(1, self.L)
        # self.m8 = nn.Linear(self.L, self.L)
        # self.m9 = nn.Linear(self.L, 1)



    def predict_output(self, z):

        # out = self.act_func(self.m1(z))
        out = self.m2(z)
        mean = out[:,:1]
        logvar = out[:,1:]
        logvar = torch.clamp(logvar, min=-15., max=10.)
        return mean, logvar


    def update_state(self, x, prev_z):

        # print (x.shape)
        # print (prev_z.shape)
        # fsd
        # prev_z = self.m6(prev_z)

        # x = torch.cat([x, prev_z], 1)
        # upt = self.act_func(self.m3(x))
        # upt = self.act_func(self.m4(upt))
        # z = self.m5(x/10.)
        z = self.m5(x)

        z2 = self.m6(prev_z)
        z = z+z2

        # z = prev_z + upt

        return z


    def NN(self, x):

        out = self.act_func(self.m7(x))
        out = self.act_func(self.m8(out))
        out = self.m9(out)

        return out





save_to_dir = home + '/Documents/SL3/'
exp_name = 'test_RNN'

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
x_lim = [0,25.]
# x_lim = [0,10.]
X_linspace = np.linspace(x_lim[0], x_lim[1], n).reshape(-1,1)
B = 32 # 32
state_size = 20


model = RNN().cuda()

lr = .0004
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=.0000001)


train_ = 1

if train_:

    for step in range(1000):

        #Make batch
        batch = []
        # batch.append(np.zeros())
        for i in range(B):

            f = samp_func()
            seq = f(X_linspace)
            seq = np.insert(seq, 0, 0)
            batch.append(seq)

        batch = torch.from_numpy(np.array(batch)).float().cuda()

        # print (batch.shape)
        # fas

        state = torch.zeros(B,state_size).cuda()

        L = 0
        for t in range(0,n):

            state = model.update_state(torch.unsqueeze(batch[:,t],dim=1), state)
            out_mean, out_logvar = model.predict_output(state)

            # out_mean = model.NN(torch.unsqueeze(batch[:,t],dim=1))

            # out_mean = torch.unsqueeze(batch[:,t],dim=1) + .00001*out_mean

            # a= torch.zeros(2,1)
            # b = torch.zeros(1,2)
            # a[0] = 1
            # b[0] = 2
            # print (a.shape)
            # print (b.shape)
            # print (a)
            # print (b)
            # c = a - b
            # print (c.shape)
            # print (c)
            # fsddf


            # print (batch[:,t].shape)
            # print (out_mean.shape)
            target = torch.unsqueeze(batch[:,t+1],dim=1)
            # fds
            # L += torch.mean((batch[:,t] - out_mean)**2 ) #/ torch.exp(out_logvar) + out_logvar)
            L += torch.mean(torch.abs(target - out_mean)) #/ torch.exp(out_logvar) + out_logvar)
            
            # if step %100 == 0 and t ==5 :
            #     print (batch[:,t][0], batch[:,t+1][0], out_mean[0])

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
    step = 999
    save_to=os.path.join(params_dir, "model_" +name + str(step)+".pt")
    state_dict = torch.load(save_to)
    model.load_state_dict(state_dict)
    print ('loaded params', save_to)








# PLOTS:
ylim = [-15,15]

rows = 2
cols = 2
fig = plt.figure(figsize=(6+cols,2+rows), facecolor='white', dpi=150)   


# for i in range (rows):
#     for j in range (cols):

# des = 2*np.random.rand() -.5
# freq = 2*np.random.rand() 
# f = lambda x: (-x*des+np.sin(freq*x)).flatten()
f = samp_func()
seq = f(X_linspace)
ax = plt.subplot2grid((rows,cols), (0,0), frameon=False, colspan=1, rowspan=1)
ax.plot(X_linspace, seq, 'b-', label='True', linewidth=1.)
ax.tick_params(labelsize=6)
ax.set_title('Example Sequence',fontsize=6,family='serif')
ax.set_ylim(ylim[0], ylim[1]) 


ax = plt.subplot2grid((rows,cols), (0,1), frameon=False, colspan=1, rowspan=1)

B = 1
state = torch.zeros(B,state_size).cuda()
out_mean = torch.unsqueeze(torch.from_numpy(np.array([0])),dim=1).float().cuda()
seq = torch.unsqueeze(torch.from_numpy(np.array(seq)),dim=0).float().cuda()
means = []
for t in range(n//2):
    # print (seq.shape)
    # print (state.shape)
    # fds
    state = model.update_state(torch.unsqueeze(seq[:,t],dim=1), state)
    # state = model.update_state(torch.unsqueeze(batch[:,t],dim=1), state)
    # out_mean, out_logvar = model.predict_output(state)
    # means.append(out_mean.data.item())
    means.append(seq[:,t].data.item())

out_mean, out_logvar = model.predict_output(state)
for t in range(n//2):
    state = model.update_state(out_mean, state)
    out_mean, out_logvar = model.predict_output(state)
    # means.append(out_mean.data.item())
    means.append(out_mean)

ax.plot(X_linspace, means, 'b-', label='True', linewidth=1.)
ax.tick_params(labelsize=6)
ax.set_title('Sample Seq 1',fontsize=6,family='serif')
ax.set_ylim(ylim[0], ylim[1]) 


ax = plt.subplot2grid((rows,cols), (1,1), frameon=False, colspan=1, rowspan=1)

B = 1
state = torch.zeros(B,state_size).cuda()
out_mean = torch.unsqueeze(torch.from_numpy(np.array([0])),dim=1).float().cuda()
means = []
for t in range(n):
    state = model.update_state(out_mean, state)
    out_mean, out_logvar = model.predict_output(state)
    means.append(out_mean.data.item())

ax.plot(X_linspace, means, 'b-', label='True', linewidth=1.)
ax.tick_params(labelsize=6)
ax.set_title('Sample Seq 2',fontsize=6,family='serif')
ax.set_ylim(ylim[0], ylim[1]) 





ax = plt.subplot2grid((rows,cols), (1,0), frameon=False, colspan=1, rowspan=1)

state = torch.zeros(B,state_size).cuda()
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


    state = model.update_state(torch.unsqueeze(seq[:,t],dim=1), state)
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
ax.set_title('Vars',fontsize=6,family='serif')
ax.set_ylim(ylim[0], ylim[1]) 
ax.legend(fontsize=6)




plt.tight_layout()
# pl.gca().set_aspect('equal')

# pl.show()
plt.savefig(images_dir + 'test1.png') #, bbox_inches='tight')
print ('saved image', images_dir + 'test.png')




















