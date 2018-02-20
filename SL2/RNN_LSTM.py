





import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F






class RNN(nn.Module):
    def __init__(self, specs):
        super(RNN, self).__init__()

        torch.manual_seed(1)


        self.input_size = specs['input_size']
        self.output_size = specs['output_size']
        self.z_size = specs['z_size']

        self.lstm = nn.LSTM(self.z_size, self.z_size)  # Input dim to lstm, hidden dim of lstm

        # self.update_net = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_size+self.z_size, specs['update_net'][0]),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(specs['update_net'][0], specs['update_net'][1]),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(specs['update_net'][1], self.z_size),
        # )

        # print (specs['input_net'][0])
        # print (specs['input_net'][1])

        # fsd

        # self.input_net = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_size, specs['input_net'][0]),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(specs['input_net'][0], specs['input_net'][1]),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(specs['input_net'][1], self.z_size),
        # )



        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, self.z_size),
        )


        # self.output_net = torch.nn.Sequential(
        #     torch.nn.Linear(self.z_size, specs['output_net'][0]),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(specs['output_net'][0], specs['output_net'][1]),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(specs['output_net'][1], self.output_size),
        # )


        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.z_size, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, self.output_size),
        )



    def init_state(self, batch_size):

        # self.state = Variable(torch.zeros(batch_size, self.z_size))
        # return Variable(torch.zeros(batch_size, self.z_size))

        # initialize the hidden state and cell state
        # hidden = (Variable(torch.randn(1, batch_size, self.z_size)),Variable(torch.randn((1, batch_size, self.z_size)))) 
        hidden = (Variable(torch.zeros(1, batch_size, self.z_size)),Variable(torch.zeros((1, batch_size, self.z_size)))) 
        return hidden


    def input_encoding(self, x):

        # print (x)

        return self.input_net(x) 

    def output_prediction(self, state):
        '''
        z: [B,Z]
        out: [B,O]
        '''
        return self.output_net(state) 


    def update_state(self, x_encoding, hidden):

        # print (x.size()) #[B,X]
        # print (self.state.size())  #[B,Z]

        # x_z = torch.cat((x,state), dim=1) #[B,x+z]
        # new_state = self.update_net(x_z)

        x_encoding = torch.unsqueeze(x_encoding, dim=0) #[1,B,Z]

        # print (x_encoding)


        _, hidden = self.lstm(x_encoding, hidden)  #  ([1,B,Z],[1,B,Z])

        # return new_state
        # self.state = new_state #does this erase the grad??
        return hidden


    def train(self, data_x, data_y, batch_size):

        np.random.seed(0)

        n_data = data_x.shape[0]

        timesteps = 10

        # convert to pytorch
        data_x = Variable(torch.FloatTensor(data_x))
        data_y = Variable(torch.LongTensor(data_y))


        loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=.001)

        # for p in self.parameters():
        #     print (p.size())
        # fds

        for step_ in range(10000):


            # make a batch of sequences
            batch_x = []
            batch_y = []
            for i in range(batch_size):
                seq = []
                for t in range(timesteps):

                    data_index = np.random.randint(n_data)
                    seq.append(data_x[data_index])

                    if t == 0:
                        label = data_y[data_index]

                batch_x.append(torch.stack(seq, dim=0))
                batch_y.append(label)

            batch_x = torch.stack(batch_x, dim=1)  #[T,B,X]
            # batch_y = torch.unsqueeze(torch.stack(batch_y, dim=0), dim=0)  #[1,B,1]
            batch_y = torch.squeeze(torch.stack(batch_y, dim=0))  #[B,1]
            # print (batch_x.size())
            # print (batch_y.size())




            # Run model
            state = self.init_state(batch_size)
            for t in range(timesteps):
                x_encoding = self.input_encoding(batch_x[t])
                state = self.update_state(x_encoding, state)

            # print (state)

            pred = self.output_prediction(state[0][0]) #[B,Y]
            # print (pred.size())

            loss = loss_func(pred, batch_y) #[1]

            # print (loss.size())
            # fsad

            if step_ %100 == 0:
                print (step_, loss.data.numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            




# inputs = [Variable(torch.randn((1, 3))) for _ in range(5)]  # make a sequence of length 5


# for i in inputs:
#     # Step through the sequence one element at a time.
#     # after each step, hidden contains the hidden state.
#     #out contains the hidden state
#     #hidden contains (hidden, cell)  states
#     #so this is pretty confusing, no reason to have out there
#     # out, hidden = lstm(i.view(1, 1, -1), hidden)  #[1,B,X]
#     print ('out')
#     print (out)
#     print ('hidden')
#     print (hidden)
#     # print(out.data.numpy())
#     # print(hidden.data.numpy())


















    # def step(self):


        #update state

        #output value

    # def reset_state():

    #     self.state = Variable(pytorch.zeros())

    # def encode(self, x, a, prev_z):
    #     '''
    #     x: [B,X]
    #     a: [B,A]
    #     prev_z: [P,B,Z]
    #     '''

    #     out = torch.cat((x,a), 1) #[B,XA]
    #     out = torch.unsqueeze(out, 0) #[1,B,XA]
    #     out = out.repeat(self.k, 1, 1)#.float() #[P,B,XA]
    #     out = torch.cat((out,prev_z), 2) #[P,B,XAZ]
    #     out = out.view(self.k*self.B, self.input_size+self.action_size+self.z_size) #[P*B,XAZ]
    #     out = self.encoder_net(out) #[P*B,Z*2]
    #     out = out.view(self.k, self.B, 2*self.z_size) #[P,B,XAZ]
    #     mean = out[:,:,:self.z_size]
    #     logvar = out[:,:,self.z_size:]
    #     return mean, logvar


    # def sample(self, mu, logvar):
    #     '''
    #     mu, logvar: [P,B,Z]
    #     '''
    #     eps = Variable(torch.FloatTensor(self.k, self.B, self.z_size).normal_()) #[P,B,Z]
    #     z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
    #     # logpz = lognormal(z, Variable(torch.zeros(self.k, self.B, self.z_size)), 
    #     #                     Variable(torch.zeros(self.k, self.B, self.z_size)))  #[P,B]
    #     logqz = lognormal(z, mu, logvar)
    #     return z, logqz




    # def transition_prior(self, prev_z, a):
    #     '''
    #     prev_z: [P,B,Z]
    #     a: [B,A]
    #     '''
    #     a = torch.unsqueeze(a, 0) #[1,B,A]
    #     out = a.repeat(self.k, 1, 1) #[P,B,A]
    #     out = torch.cat((out,prev_z), 2) #[P,B,AZ]
    #     out = out.view(-1, self.action_size+self.z_size) #[P*B,AZ]
    #     out = self.transition_net(out) #[P*B,Z*2]
    #     out = out.view(self.k, self.B, 2*self.z_size) #[P,B,2Z]
    #     mean = out[:,:,:self.z_size] #[P,B,Z]
    #     logvar = out[:,:,self.z_size:]
    #     return mean, logvar


    # def forward(self, x, a, k=1, current_state=None):
    #     '''
    #     x: [B,T,X]
    #     a: [B,T,A]
    #     output: elbo scalar
    #     '''
        
    #     self.B = x.size()[0]
    #     self.T = x.size()[1]
    #     self.k = k

    #     a = a.float()
    #     x = x.float()

    #     # log_probs = [[] for i in range(k)]
    #     # log_probs = []
    #     logpxs = []
    #     logpzs = []
    #     logqzs = []


    #     weights = Variable(torch.ones(k, self.B)/k)
    #     # if current_state==None:
    #     prev_z = Variable(torch.zeros(k, self.B, self.z_size))
    #     # else:
    #     #     prev_z = current_state
    #     for t in range(self.T):
    #         current_x = x[:,t] #[B,X]
    #         current_a = a[:,t] #[B,A]

    #         #Encode
    #         mu, logvar = self.encode(current_x, current_a, prev_z) #[P,B,Z]
    #         #Sample
    #         z, logqz = self.sample(mu, logvar) #[P,B,Z], [P,B]
    #         #Decode
    #         x_hat = self.decode(z)  #[P,B,X]
    #         logpx = log_bernoulli(x_hat, current_x)  #[P,B]
    #         #Transition/Prior prob
    #         prior_mean, prior_log_var = self.transition_prior(prev_z, current_a) #[P,B,Z]
    #         logpz = lognormal(z, prior_mean, prior_log_var) #[P,B]






    #         log_alpha_t = logpx + logpz - logqz #[P,B]
    #         log_weights_tmp = torch.log(weights * torch.exp(log_alpha_t))

    #         max_ = torch.max(log_weights_tmp, 0)[0] #[B]
    #         log_p_hat = torch.log(torch.sum(torch.exp(log_weights_tmp - max_), 0)) + max_ #[B]

    #         # p_hat = torch.sum(alpha_t,0)  #[B]
    #         normalized_alpha_t = log_weights_tmp - log_p_hat  #[P,B]

    #         weights = torch.exp(normalized_alpha_t) #[P,B]

    #         #if resample
    #         if t%2==0:
    #             # print weights
    #             #[B,P] indices of the particles for each bactch
    #             sampled_indices = torch.multinomial(torch.t(weights), k, replacement=True).detach()
    #             new_z = []
    #             for b in range(self.B):
    #                 tmp = z[:,b] #[P,Z]
    #                 z_b = tmp[sampled_indices[b]] #[P,Z]
    #                 new_z.append(z_b)
    #             new_z = torch.stack(new_z, 1) #[P,B,Z]
    #             weights = Variable(torch.ones(k, self.B)/k)
    #             z = new_z

    #         logpxs.append(logpx)
    #         logpzs.append(logpz)
    #         logqzs.append(logqz)
    #         # log_probs.append(logpx + logpz - logqz)
    #         prev_z = z



    #     logpxs = torch.stack(logpxs) 
    #     logpzs = torch.stack(logpzs)
    #     logqzs = torch.stack(logqzs) #[T,P,B]

    #     logws = logpxs + logpzs - logqzs  #[T,P,B]
    #     logws = torch.mean(logws, 0)  #[P,B]

    #     # elbo = logpx + logpz - logqz  #[P,B]

    #     if k>1:
    #         max_ = torch.max(logws, 0)[0] #[B]
    #         elbo = torch.log(torch.mean(torch.exp(logws - max_), 0)) + max_ #[B]
    #         elbo = torch.mean(elbo) #over batch
    #     else:
    #         elbo = torch.mean(logws)

    #     # print log_probs[0]


    #     # #for printing
    #     logpx = torch.mean(logpxs)
    #     logpz = torch.mean(logpzs)
    #     logqz = torch.mean(logqzs)
    #     # self.x_hat_sigmoid = F.sigmoid(x_hat)

    #     # elbo = torch.mean(torch.stack(log_probs)) #[1]
    #     # elbo = logpx + logpz - logqz

    #     return elbo, logpx, logpz, logqz



