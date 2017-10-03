




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

from utils import lognormal2 as lognormal
from utils import log_bernoulli







def train(model, train_x, train_y, valid_x=[], valid_y=[], 
            path_to_load_variables='', path_to_save_variables='', 
            epochs=10, batch_size=20, display_epoch=2, k=1):
    

    if path_to_load_variables != '':
        model.load_state_dict(torch.load(path_to_load_variables))
        print 'loaded variables ' + path_to_load_variables

    train = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=.0001)

    for epoch in range(1, epochs + 1):

        for batch_idx, (data, target) in enumerate(train_loader):

            if data.is_cuda:
                data, target = Variable(data), Variable(target).type(torch.cuda.LongTensor)
            else:
                data, target = Variable(data), Variable(target)

            optimizer.zero_grad()

            elbo, logpx, logpz, logqz = model.forward(data, k=k)
            loss = -(elbo)

            loss.backward()
            optimizer.step()

            if epoch%display_epoch==0 and batch_idx == 0:
                print 'Train Epoch: {}/{} [{}/{} ({:.0f}%)]'.format(epoch, epochs, 
                        batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader)), \
                    'Loss:{:.4f}'.format(loss.data[0]), \
                    'logpx:{:.4f}'.format(logpx.data[0]), \
                    'logpz:{:.4f}'.format(logpz.data[0]), \
                    'logqz:{:.4f}'.format(logqz.data[0]) 


    if path_to_save_variables != '':
        torch.save(model.state_dict(), path_to_save_variables)
        print 'Saved variables to ' + path_to_save_variables




def test(model, data_x, path_to_load_variables='', batch_size=20, display_epoch=4, k=10):
    

    if path_to_load_variables != '':
        model.load_state_dict(torch.load(path_to_load_variables))
        print 'loaded variables ' + path_to_load_variables

    elbos = []
    data_index= 0
    for i in range(len(data_x)/ batch_size):

        batch = data_x[data_index:data_index+batch_size]
        data_index += batch_size

        elbo, logpx, logpz, logqz = model(Variable(batch), k=k)
        elbos.append(elbo.data[0])

        if i%display_epoch==0:
            print i,len(data_x)/ batch_size, elbo.data[0]

    return np.mean(elbos)







def test_ais(model, data_x, path_to_load_variables='', batch_size=20, display_epoch=4, k=10):

    n_intermediate_dists = 10
    n_HMC_steps = 10


    if path_to_load_variables != '':
        model.load_state_dict(torch.load(path_to_load_variables))
        print 'loaded variables ' + path_to_load_variables


    logws = []
    data_index= 0
    for i in range(len(data_x)/ batch_size):

        # print i, len(data_x)/ batch_size

        batch = data_x[data_index:data_index+batch_size]
        data_index += batch_size


        #AIS

        schedule = np.linspace(0.,1.,n_intermediate_dists)
        model.B = batch_size

        logw = Variable(torch.zeros(k, model.B))
        # logw = torch.zeros(k, model.B)
        



        #Encode x
        mean, logvar = model.encode(Variable(batch)) #[B,Z]
        # print mean.data.numpy().shape
        # fasdf

        #Init z
        z, logpz, logqz = model.sample(mean, logvar, k=k)  #[P,B,Z], [P,B], [P,B]
        # print logpz.data.numpy().shape
        # fasdf


        for (t0, t1) in zip(schedule[:-1], schedule[1:]): 

            #Compute intermediate distribution log prob
            # (1-t)*logp1(z) + (t)*logpT(z)
            logp1 = lognormal(z, mean, logvar)  #[P,B]
            log_prior = lognormal(z, Variable(torch.zeros(model.B, model.z_size)), Variable(torch.zeros(model.B, model.z_size)))  #[P,B]
            log_likelihood = log_bernoulli(model.decode(z), Variable(batch))
            logpT = log_prior + log_likelihood

            #log pt-1(zt-1)
            log_intermediate_1 = (1-float(t0))*logp1 + float(t0)*logpT
            #log pt(zt-1)
            log_intermediate_2 = (1-float(t1))*logp1 + float(t1)*logpT

            logw += log_intermediate_2 - log_intermediate_1



            #HMC

            step_size = .1
            v = Variable(torch.FloatTensor(z.size()).normal_()) 
            v0 = v
            z0 = z

            gradients = torch.autograd.grad(outputs=log_intermediate_2, inputs=z,
                              grad_outputs=torch.ones(log_intermediate_2.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

            v = v + .5 *step_size*gradients
            z = z + step_size*v

            for LF_step in range(n_HMC_steps):
                # print LF_step

                logp1 = lognormal(z, mean, logvar)  #[P,B]
                log_prior = lognormal(z, Variable(torch.zeros(model.B, model.z_size)), Variable(torch.zeros(model.B, model.z_size)))  #[P,B]
                log_likelihood = log_bernoulli(model.decode(z), Variable(batch))
                logpT = log_prior + log_likelihood
                log_intermediate_2 = (1-float(t1))*logp1 + float(t1)*logpT
                gradients = torch.autograd.grad(outputs=log_intermediate_2, inputs=z,
                                  grad_outputs=torch.ones(log_intermediate_2.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

                v = v + step_size*gradients
                z = z + step_size*v



            logp1 = lognormal(z, mean, logvar)  #[P,B]
            log_prior = lognormal(z, Variable(torch.zeros(model.B, model.z_size)), Variable(torch.zeros(model.B, model.z_size)))  #[P,B]
            log_likelihood = log_bernoulli(model.decode(z), Variable(batch))
            logpT = log_prior + log_likelihood
            log_intermediate_2 = (1-float(t1))*logp1 + float(t1)*logpT
            gradients = torch.autograd.grad(outputs=log_intermediate_2, inputs=z,
                              grad_outputs=torch.ones(log_intermediate_2.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

            v = v + .5 *step_size*gradients


            #MH step
            logp1 = lognormal(z0, mean, logvar)  #[P,B]
            log_prior = lognormal(z0, Variable(torch.zeros(model.B, model.z_size)), Variable(torch.zeros(model.B, model.z_size)))  #[P,B]
            log_likelihood = log_bernoulli(model.decode(z0), Variable(batch))
            logpT = log_prior + log_likelihood
            log_intermediate_2 = (1-float(t1))*logp1 + float(t1)*logpT
            logpv0 = lognormal(v0, Variable(torch.zeros(model.B, model.z_size)), Variable(torch.zeros(model.B, model.z_size))) #[P,B]
            hamil_0 =  log_intermediate_2 + logpv0

            logp1 = lognormal(z, mean, logvar)  #[P,B]
            log_prior = lognormal(z, Variable(torch.zeros(model.B, model.z_size)), Variable(torch.zeros(model.B, model.z_size)))  #[P,B]
            log_likelihood = log_bernoulli(model.decode(z), Variable(batch))
            logpT = log_prior + log_likelihood
            log_intermediate_2 = (1-float(t1))*logp1 + float(t1)*logpT
            logpvT = lognormal(v, Variable(torch.zeros(model.B, model.z_size)), Variable(torch.zeros(model.B, model.z_size))) #[P,B]

            hamil_T = log_intermediate_2 + logpvT
            # print hamil_T.data.numpy().shape

            accept_prob = torch.exp(hamil_T - hamil_0)
            accept = accept_prob > Variable(torch.FloatTensor(accept_prob.size()).uniform_())
            accept = accept.type(torch.FloatTensor)
            accept = accept.view(k, model.B, 1)
            # print accept.data.numpy().shape

            # print torch.mean(accept)


            z = (accept * z) + ((1-accept) * z0)

            avg_acceptance_rate = torch.mean(accept)
            # print avg_acceptance_rate.data.numpy()

            if avg_acceptance_rate.data.numpy() > .7:
                step_size = 1.02 * step_size
            else:
                step_size = .98 * step_size

            if step_size < 0.0001:
                step_size = 0.0001
            if step_size > 0.5:
                step_size = 0.5



        #lgo sum exp
        max_ = torch.max(logw,0)[0] #[B]
        logw = torch.log(torch.mean(torch.exp(logw - max_), 0)) + max_ #[B]

        logws.append(torch.mean(logw).data.numpy())


        if i%display_epoch==0:
            print i,len(data_x)/ batch_size, np.mean(logws)

    return np.mean(logws)


















class IWAE(nn.Module):
    def __init__(self):
        super(IWAE, self).__init__()

        self.z_size = 50

        self.fc1 = nn.Linear(784, 200)
        self.fc1_2 = nn.Linear(200, 200)
        self.fc2 = nn.Linear(200, self.z_size*2)
        self.fc3 = nn.Linear(self.z_size, 200)
        self.fc3_2 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 784)


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc1_2(h1))

        h2 = self.fc2(h1)
        mean = h2[:,:self.z_size]
        logvar = h2[:,self.z_size:]
        return mean, logvar

    def sample(self, mu, logvar, k):
        eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_()) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
        logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size)), 
                            Variable(torch.zeros(self.B, self.z_size)))  #[P,B]
        logqz = lognormal(z, mu, logvar)
        return z, logpz, logqz

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc3_2(h3))

        return self.fc4(h3)


    def forward(self, x, k=1):
        
        self.B = x.size()[0]
        mu, logvar = self.encode(x)
        z, logpz, logqz = self.sample(mu, logvar, k=k)
        x_hat = self.decode(z)
        logpx = log_bernoulli(x_hat, x)  #[P,B]


        elbo = logpx + logpz - logqz  #[P,B]

        if k>1:
            max_ = torch.max(elbo, 0)[0] #[B]
            elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

        elbo = torch.mean(elbo) #[1]

        #for printing
        logpx = torch.mean(logpx)
        logpz = torch.mean(logpz)
        logqz = torch.mean(logqz)
        self.x_hat_sigmoid = F.sigmoid(x_hat)

        return elbo, logpx, logpz, logqz





















if __name__ == '__main__':

    train_ = 1
    eval_iw = 1
    eval_ais = 1

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
    test_x = torch.from_numpy(test_x)
    train_y = torch.from_numpy(train_y)

    print train_x.shape
    print test_x.shape
    print train_y.shape
    print


    model = IWAE()

    if torch.cuda.is_available():
        print 'GPU available, loading cuda'#, torch.cuda.is_available()
        model.cuda()
        train_x = train_x.cuda()


    path_to_load_variables=''
    # path_to_load_variables=home+'/Documents/tmp/pytorch_first.pt'
    path_to_save_variables=home+'/Documents/tmp/pytorch_first.pt'
    # path_to_save_variables=''


    if train_:
        print 'Training'
        train(model=model, train_x=train_x, train_y=train_y, valid_x=[], valid_y=[], 
                    path_to_load_variables=path_to_load_variables, 
                    path_to_save_variables=path_to_save_variables, 
                    epochs=200, batch_size=100, display_epoch=1, k=1)


    if eval_iw:
        print 'Evaluating with IW'
        print test(model=model, data_x=test_x, path_to_load_variables=path_to_save_variables, 
                    batch_size=20, display_epoch=100, k=1000)

    if eval_ais:
        print 'Evaluating with AIS'
        print test_ais(model=model, data_x=test_x, path_to_load_variables=path_to_save_variables, 
                    batch_size=50, display_epoch=1, k=10)




    print 'Done.'


















