



import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import lognormal2 as lognormal
from utils import lognormal5 as lognormal_decoder

from utils import log_bernoulli



#plot it
fig = plt.figure(figsize=(10,6), facecolor='white')

plt.ion()
plt.show(block=False)






def plot_it(data):


    plt.cla()
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.scatter(data,means)
    # plt.scatter(data,means-std)
    # plt.scatter(data,means+std)


    #for each x, get its q

    # means, logvars = model.encode(Variable(data_pytorch))

    # print data.shape
    # print means.shape

    # print data


    means, logvars = model.encode(data)  #q(z|x)
    # print means

    x_mean, x_logvar = model.decode(torch.unsqueeze(means,1)) #p(x|z_mean)
    # print np.arange(-3,3,20)
    zs = np.expand_dims(np.expand_dims(np.array(np.arange(-3,3,.2)),1),0)
    x_mean_other, x_logvar_other = model.decode(Variable(torch.Tensor(zs))) #p(x|z_mean)


    xs = np.expand_dims(np.array(np.arange(-15,15,1.)),1)
    xs_torch = Variable(torch.Tensor(xs))
    means_other, logvars_other = model.encode(xs_torch) #p(x|z_mean)

    elbos = model.forward_for_plot(xs_torch, k=50)
    elbos = elbos.data.numpy()
    # print elbos
    # fsad



    s=.7
    xmin = -3.1
    xmax = 3.1
    ymin = -15.1
    ymax = 15.1

    rows = 2
    cols = 3

    # print data
    data = np.reshape(data.data.numpy(), [-1]) 
    # print np.reshape(data.numpy(), [-1])
    # fads



    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

    binwidth = 1.
    bins = np.arange(ymin, ymax, binwidth)
    ax.hist(data, bins=bins, orientation='horizontal', normed=True)

    # ax.set_xlim([0,])
    ax.annotate('p(x_train)', xytext=(.1, 1.), xy=(0, 1), textcoords='axes fraction')








    ax = plt.subplot2grid((rows,cols), (1,0), frameon=False)

    binwidth = 1.
    xs111 = np.reshape(xs, [-1])
    bins = np.arange(ymin, ymax, binwidth)
    ax.hist(xs111, weights=np.exp(elbos), bins=bins, orientation='horizontal', normed=True)

    # ax.set_xlim([0,])
    ax.annotate('p(x_model)', xytext=(.1, 1.), xy=(0, 1), textcoords='axes fraction')









    ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)

    means = means.data.numpy()
    means = np.reshape(means, [-1])
    logvars = logvars.data.numpy()
    logvars = np.reshape(logvars, [-1])
    variances = np.exp(logvars)
    std = np.sqrt(variances)
    # plt.title('q(z|x_data)')
    plt.xlabel('z')
    plt.ylabel('x')
    # plt.scatter([0.]*n,data)
    # print means
    # print data
    plt.scatter(means,data,s=s)
    plt.scatter(means-std,data,s=s)
    plt.scatter(means+std,data,s=s)
    for i in range(len(means)):
        plt.plot([means[i]-std[i],means[i]+std[i]], [data[i],data[i]])
    ax.annotate('q(z|x_train)', xytext=(.1, 1.), xy=(0, 1), textcoords='axes fraction')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])






    # ax = plt.subplot2grid((1,2), (0,1), frameon=False)



    # means1 = x_mean.data.numpy()
    # means1 = np.reshape(means1, [-1])
    # logvars1 = x_logvar.data.numpy()
    # logvars1 = np.reshape(logvars1, [-1])
    # variances1 = np.exp(logvars1)
    # std1 = np.sqrt(variances1)
    # plt.title('p(x|z_mean)')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # # plt.scatter([0.]*n,data)
    # plt.scatter(means1,means,s=s)
    # plt.scatter(means1-std1,means,s=s)
    # plt.scatter(means1+std1,means,s=s)
    # for i in range(len(means1)):
    #     plt.plot([means1[i]-std1[i],means1[i]+std1[i]], [means[i],means[i]])


    ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)

    means1 = x_mean.data.numpy()
    means1 = np.reshape(means1, [-1])
    logvars1 = x_logvar.data.numpy()
    logvars1 = np.reshape(logvars1, [-1])
    variances1 = np.exp(logvars1)
    std1 = np.sqrt(variances1)
    # plt.title('p(x|z_mean)')
    plt.xlabel('z')
    plt.ylabel('x')
    # plt.scatter([0.]*n,data)
    plt.scatter(means,means1,s=s)


    plt.scatter(means,means1-std1,s=s)
    plt.scatter(means,means1+std1,s=s)
    for i in range(len(means1)):
        plt.plot([means[i],means[i]],[means1[i]-std1[i],means1[i]+std1[i]])

    ax.annotate('p(x|z_mean)', xytext=(.1, 1.), xy=(0, 1), textcoords='axes fraction')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])

    plt.scatter(means,data,s=4.)#, marker='_')





    ax = plt.subplot2grid((rows,cols), (1,2), frameon=False)

    zs = np.reshape(zs, [-1])
    means1 = x_mean_other.data.numpy()
    means1 = np.reshape(means1, [-1])
    logvars1 = x_logvar_other.data.numpy()
    logvars1 = np.reshape(logvars1, [-1])
    variances1 = np.exp(logvars1)
    std1 = np.sqrt(variances1)
    # plt.title('p(x|z)')
    plt.xlabel('z')
    plt.ylabel('x')
    # plt.scatter([0.]*n,data)
    plt.scatter(zs,means1,s=s)
    plt.scatter(zs,means1-std1,s=s)
    plt.scatter(zs,means1+std1,s=s)
    for i in range(len(means1)):
        # print zs.shape
        # print means1.shape
        plt.plot([zs[i],zs[i]],[means1[i]-std1[i],means1[i]+std1[i]])
    ax.annotate('p(x|z_range)', xytext=(.1, 1.), xy=(0, 1), textcoords='axes fraction')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])






    ax = plt.subplot2grid((rows,cols), (1,1), frameon=False)

    xs = np.reshape(xs, [-1])
    means = means_other.data.numpy()
    means = np.reshape(means, [-1])
    logvars = logvars_other.data.numpy()
    logvars = np.reshape(logvars, [-1])
    variances = np.exp(logvars)
    std = np.sqrt(variances)
    # plt.title('q(z|x_data)')
    plt.xlabel('z')
    plt.ylabel('x')
    # plt.scatter([0.]*n,data)
    plt.scatter(means,xs,s=s)
    plt.scatter(means-std,xs,s=s)
    plt.scatter(means+std,xs,s=s)
    for i in range(len(means)):
        plt.plot([means[i]-std[i],means[i]+std[i]], [xs[i],xs[i]])
    ax.annotate('q(z|x_range)', xytext=(.1, 1.), xy=(0, 1), textcoords='axes fraction')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])





    # plt.show()
    plt.draw()
    plt.pause(1.0/30.0)







def train(model, train_x, train_y, valid_x=[], valid_y=[], 
            path_to_load_variables='', path_to_save_variables='', 
            epochs=10, batch_size=20, display_epoch=2, k=1):
    

    if path_to_load_variables != '':
        model.load_state_dict(torch.load(path_to_load_variables))
        print 'loaded variables ' + path_to_load_variables

    train = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=.005)


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



                plot_it(data)


                plt.savefig(home+'/Documents/tmp/thing'+str(epoch)+'.png')
                print 'Saved fig'




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






class IWAE(nn.Module):
    def __init__(self):
        super(IWAE, self).__init__()

        torch.manual_seed(1000)

        self.z_size = 1

        la = 10

        self.fc1 = nn.Linear(1, la)
        self.fc2 = nn.Linear(la, self.z_size*2)
        self.fc3 = nn.Linear(self.z_size, la)
        self.fc4 = nn.Linear(la, 1*2)


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = self.fc2(h1)
        mean = h2[:,:self.z_size]
        logvar = h2[:,self.z_size:]
        return mean, logvar

    def sample(self, mu, logvar, k):
        eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_()) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
        # z = eps.mul(torch.exp(.5*logvar)) + mu.detach()  #[P,B,Z]

        logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size)), 
                            Variable(torch.zeros(self.B, self.z_size)))  #[P,B]
        # logqz = lognormal(z, mu, logvar)
        logqz = lognormal(z, mu.detach(), logvar.detach())

        return z, logpz, logqz

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        out = self.fc4(h3)

        mean = out[:,:,:1]
        logvar = out[:,:,1:] #[P,B,1]
        return mean, logvar


    def forward(self, x, k=1):
        
        self.B = x.size()[0]
        mu, logvar = self.encode(x)
        z, logpz, logqz = self.sample(mu, logvar, k=k)
        # x_hat = self.decode(z)
        x_mean, x_logvar = self.decode(z)  #[P,B,1]

        # logpx = log_bernoulli(x_hat, x)  #[P,B]
        logpx = lognormal_decoder(x, x_mean, x_logvar)  #[P,B]



        # elbo = logpx + .00000001*logpz - logqz  #[P,B]
        elbo = logpx + logpz - logqz  #[P,B]


        if k>1:
            max_ = torch.max(elbo, 0)[0] #[B]
            elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

        elbo = torch.mean(elbo) #[1]

        #for printing
        logpx = torch.mean(logpx)
        logpz = torch.mean(logpz)
        logqz = torch.mean(logqz)
        # self.x_hat_sigmoid = F.sigmoid(x_hat)

        return elbo, logpx, logpz, logqz



    def forward_for_plot(self, x, k=1):
        
        self.B = x.size()[0]
        mu, logvar = self.encode(x)
        z, logpz, logqz = self.sample(mu, logvar, k=k)
        # x_hat = self.decode(z)
        x_mean, x_logvar = self.decode(z)  #[P,B,1]
        # logpx = log_bernoulli(x_hat, x)  #[P,B]
        logpx = lognormal_decoder(x, x_mean, x_logvar)  #[P,B]
        # elbo = logpx + .00000001*logpz - logqz  #[P,B]
        elbo = logpx + logpz - logqz  #[P,B]
        if k>1:
            max_ = torch.max(elbo, 0)[0] #[B]
            elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

        return elbo





# print 'Loading data'
# with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
#     mnist_data = pickle.load(f)

# train_x = mnist_data[0][0]
# train_y = mnist_data[0][1]
# valid_x = mnist_data[1][0]
# valid_y = mnist_data[1][1]
# test_x = mnist_data[2][0]
# test_y = mnist_data[2][1]

# train_x = torch.from_numpy(train_x)
# test_x = torch.from_numpy(test_x)
# train_y = torch.from_numpy(train_y)

# print train_x.shape
# print test_x.shape
# print train_y.shape

n = 20
np.random.seed(3)

data = np.random.uniform(-10,10,n)

data1 = np.reshape(data, [n,1])

# print data.shape
# data = torch.FloatTensor(torch.from_numpy(data))
data_pytorch = torch.FloatTensor(data1)



model = IWAE()

# if torch.cuda.is_available():
#     print 'GPU available, loading cuda'#, torch.cuda.is_available()
#     model.cuda()
#     train_x = train_x.cuda()


path_to_load_variables=''
# path_to_load_variables=home+'/Documents/tmp/pytorch_first.pt'
# path_to_save_variables=home+'/Documents/tmp/pytorch_first.pt'
path_to_save_variables=''



train(model=model, train_x=data_pytorch, train_y=data_pytorch, valid_x=[], valid_y=[], 
            path_to_load_variables=path_to_load_variables, 
            path_to_save_variables=path_to_save_variables, 
            epochs=160000, batch_size=n, display_epoch=1000, k=1)



# print test(model=model, data_x=test_x, path_to_load_variables='', 
#             batch_size=20, display_epoch=100, k=1000)


# update = lambda i: train(i, model=model, train_x=data_pytorch, train_y=data_pytorch, valid_x=[], valid_y=[], 
#             path_to_load_variables=path_to_load_variables, 
#             path_to_save_variables=path_to_save_variables, 
#             epochs=1, batch_size=10, display_epoch=1, k=1)

# anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
# anim.save(home+'/Documents/tmp/1d.gif', dpi=80, writer='imagemagick')


plt.savefig(home+'/Documents/tmp/thing.png')
print 'Saved fig'


print 'Done.'


















