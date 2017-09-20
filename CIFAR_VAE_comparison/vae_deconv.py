





# add conv layer
# deconv layer


import numpy as np
import pickle, cPickle
from os.path import expanduser
home = expanduser("~")

# import matplotlib.pyplot as plt


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import lognormal2 as lognormal
from utils import log_bernoulli

from scipy.misc import toimage


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict



print 'Loading data'
file_ = home+'/Documents/cifar-10-batches-py/data_batch_'

for i in range(1,6):
    file__ = file_ + str(i)
    b1 = unpickle(file__)
    if i ==1:
        train_x = b1['data']
        train_y = b1['labels']
    else:
        train_x = np.concatenate([train_x, b1['data']], axis=0)
        train_y = np.concatenate([train_y, b1['labels']], axis=0)

file__ = home+'/Documents/cifar-10-batches-py/test_batch'
b1 = unpickle(file__)
test_x = b1['data']
test_y = b1['labels']



# #View image
# for i in range(50):
#     img = np.reshape(test_x[i], [3,32,32])
#     img = img.transpose(1,2,0).astype("uint8")
#     print i, img.shape
#     plt.imshow(img,interpolation='nearest')
#     # plt.imshow(toimage(img))
#     # plt.show()
#     plt.savefig(home+'/Documents/tmp/img'+str(i)+'.png')
# fasdf


# #THis one is better
# print 'ereere'
# test_x = test_x.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
# print 'fffff'
# #Visualizing CIFAR 10
# fig, axes1 = plt.subplots(5,5,figsize=(3,3))
# for j in range(5):
#     for k in range(5):
#         i = np.random.choice(range(len(test_x)))
#         axes1[j][k].set_axis_off()
#         axes1[j][k].imshow(test_x[i:i+1][0])
# plt.show()
# fasdfa







train_x = train_x / 255.
test_x = test_x / 255.


train_x = torch.from_numpy(train_x).float()
test_x = torch.from_numpy(test_x)
train_y = torch.from_numpy(train_y)

print train_x.shape
print test_x.shape
print train_y.shape



batch_size = 50
epochs = 200
display_epoch = 10















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

            # if data.is_cuda:
            if torch.cuda.is_available():
                data = Variable(data).type(torch.cuda.FloatTensor)# , Variable(target).type(torch.cuda.LongTensor) 
            else:
                data = Variable(data)#, Variable(target)

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

        # if data.is_cuda:
        if torch.cuda.is_available():
            data = Variable(batch).type(torch.cuda.FloatTensor)
        else:
            data = Variable(batch)#, Variable(target)

        elbo, logpx, logpz, logqz = model.forward(data, k=k)
        # print elbo, logpx, logpz, logqz
        # fasdfa

        # elbo, logpx, logpz, logqz = model(Variable(batch), k=k)
        elbos.append(elbo.data[0])

        if i%display_epoch==0:
            print i,len(data_x)/ batch_size, np.mean(elbos)

    return np.mean(elbos)




def load_params(model, path_to_load_variables=''):

    if path_to_load_variables != '':
        # model.load_state_dict(torch.load(path_to_load_variables))
        model.load_state_dict(torch.load(path_to_load_variables, lambda storage, loc: storage)) 
        print 'loaded variables ' + path_to_load_variables

























#With conv layer
# print 'conv encoder, fc decoder'

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.z_size = 20

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=2, padding=0, dilation=1, bias=True)

        self.fc1 = nn.Linear(1960, 200)
        self.fc2 = nn.Linear(200, self.z_size*2)
        self.fc3 = nn.Linear(self.z_size, 200)
        self.fc4 = nn.Linear(200, 3072)


    def encode(self, x):

        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.conv1(x))

        x = x.view(-1, 1960)

        h1 = F.relu(self.fc1(x))
        h2 = self.fc2(h1)
        mean = h2[:,:self.z_size]
        logvar = h2[:,self.z_size:]
        return mean, logvar

    def sample(self, mu, logvar, k):
        if torch.cuda.is_available():
            eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_()).cuda() #[P,B,Z]
            z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
            logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size).cuda()), 
                                Variable(torch.zeros(self.B, self.z_size)).cuda())  #[P,B]
            logqz = lognormal(z, mu, logvar)
        else:
            eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_())#[P,B,Z]
            z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
            logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size)), 
                                Variable(torch.zeros(self.B, self.z_size)))  #[P,B]
            logqz = lognormal(z, mu, logvar) 
        return z, logpz, logqz

    def decode(self, z):
        # print z.size()
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)


    def forward(self, x, k=1):
        
        self.B = x.size()[0]
        mu, logvar = self.encode(x)
        z, logpz, logqz = self.sample(mu, logvar, k=k)  #[P,B,Z]
        x_hat = self.decode(z)  #[PB,X]
        x_hat = x_hat.view(k, self.B, -1)
        # print x_hat.size()
        # print x_hat.size()
        # print x.size()
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












#With conv layer
# with deconv layer
# print 'conv encoder, deconv decoder'

class VAE_deconv1(VAE):
    def __init__(self):
        super(VAE, self).__init__()

        self.x_size = 3072
        self.z_size = 100

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=2, padding=0, dilation=1, bias=True)

        self.fc1 = nn.Linear(1960, 200)
        self.fc2 = nn.Linear(200, self.z_size*2)
        self.fc3 = nn.Linear(self.z_size, 200)
        self.fc4 = nn.Linear(200, 1960)

        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size=5, stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)



    def decode(self, z):

        z = F.relu(self.fc3(z)) 
        z = F.relu(self.fc4(z))  #[B,1960]

        z = z.view(-1, 10, 14, 14)
        z = self.deconv1(z)
        z = z.view(-1, self.x_size)
        
        return z








#With conv layer
# with deconv layer
# print 'conv encoder,  deconv then conv decoder'

class VAE_deconv2(VAE):
    def __init__(self):
        super(VAE, self).__init__()

        self.x_size = 3072
        self.z_size = 100

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=2, padding=0, dilation=1, bias=True)

        self.fc1 = nn.Linear(1960, 200)
        self.fc2 = nn.Linear(200, self.z_size*2)
        self.fc3 = nn.Linear(self.z_size, 200)
        self.fc4 = nn.Linear(200, 1960)

        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size=5, stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)

        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2, dilation=1, bias=True)




    def decode(self, z):

        z = F.relu(self.fc3(z)) 
        z = F.relu(self.fc4(z))  #[B,1960]

        z = z.view(-1, 10, 14, 14)
        z = F.relu(self.deconv1(z))
        z = self.conv2(z)
        z = z.view(-1, self.x_size)
        
        return z


















print 'conv encoder, deconv decoder'
model = VAE_deconv1()

if torch.cuda.is_available():
    print 'GPU available, loading cuda'#, torch.cuda.is_available()
    model.cuda()
    # train_x = train_x.cuda()

path_to_load_variables=''
# path_to_load_variables=home+'/Documents/tmp/pytorch_first.pt'
path_to_save_variables=home+'/Documents/tmp/pytorch_vae_cifar_deconv.pt'
# path_to_save_variables=''


train(model=model, train_x=train_x, train_y=train_y, valid_x=[], valid_y=[], 
            path_to_load_variables=path_to_load_variables, 
            path_to_save_variables=path_to_save_variables, 
            epochs=epochs, batch_size=batch_size, display_epoch=display_epoch, k=1)



print test(model=model, data_x=test_x, path_to_load_variables='', 
            batch_size=5, display_epoch=100, k=1000)

print 'Done.'
fsadfa































# print 'conv encoder, fc decoder'
# model = VAE()
# path_to_save_variables=home+'/Documents/tmp/pytorch_fc.pt'


# print 'conv encoder, deconv decoder'
# model = VAE_deconv1()
# path_to_save_variables=home+'/Documents/tmp/pytorch_deconv.pt'


# print 'conv encoder,  deconv then conv decoder'
# model = VAE_deconv2()
# path_to_save_variables=home+'/Documents/tmp/pytorch_deconv_conv.pt'






# if torch.cuda.is_available():
#     print 'GPU available, loading cuda'#, torch.cuda.is_available()
#     model.cuda()
#     # train_x = train_x.cuda()


# path_to_load_variables=''
# # path_to_load_variables=home+'/Documents/tmp/pytorch_first.pt'
# path_to_save_variables=home+'/Documents/tmp/pytorch_fc.pt'
# # path_to_save_variables=''


model1 = VAE()
load_params(model1, home+'/Documents/tmp/pytorch_fc.pt')

model2 = VAE_deconv1()
load_params(model2, home+'/Documents/tmp/pytorch_deconv.pt')

model3 = VAE_deconv2()
load_params(model3, home+'/Documents/tmp/pytorch_deconv_conv.pt')


#Reconstruct images
n_imgs = 10
fig, axes1 = plt.subplots(n_imgs,4,figsize=(3,6))


for j in range(n_imgs):
    # for i in range(n_imgs):
    i = j
    # i = np.random.choice(range(50))
    axes1[j][0].set_axis_off()
    axes1[j][1].set_axis_off()
    axes1[j][2].set_axis_off()
    axes1[j][3].set_axis_off()


    real_img = train_x[i].numpy()* 255
    real_img = np.reshape(real_img, [3,32,32])
    real_img = real_img.transpose(1,2,0).astype("uint8")
    axes1[j][0].imshow(real_img)


    x = Variable(train_x[i].view(1,3072))


    model1.forward(x)
    recon = model1.x_hat_sigmoid
    recon = recon.data.numpy()* 255
    recon = np.reshape(recon, [3,32,32])
    recon = recon.transpose(1,2,0).astype("uint8")
    axes1[j][1].imshow(recon)


    model2.forward(x)
    recon = model2.x_hat_sigmoid
    recon = recon.data.numpy()* 255
    recon = np.reshape(recon, [3,32,32])
    recon = recon.transpose(1,2,0).astype("uint8")
    axes1[j][2].imshow(recon)


    model3.forward(x)
    recon = model3.x_hat_sigmoid
    recon = recon.data.numpy()* 255
    recon = np.reshape(recon, [3,32,32])
    recon = recon.transpose(1,2,0).astype("uint8")
    axes1[j][3].imshow(recon)

plt.annotate('Real', xy=(0, 0), xytext=(.18,.9), textcoords='figure fraction', size=6)
plt.annotate('FC', xy=(0, 0), xytext=(.4,.9), textcoords='figure fraction', size=6)
plt.annotate('Deconv', xy=(0, 0), xytext=(.56,.9), textcoords='figure fraction', size=6)
plt.annotate('Deconv+Conv', xy=(0, 0), xytext=(.74,.9), textcoords='figure fraction', size=6)

# ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5))

plt.show()
fasdfa




# # print train_x.size
# real_img = train_x[0]
# x = Variable(real_img)#.type(torch.cuda.FloatTensor)
# print x.size()
# x = x.view(1,3072)

# recon = model.x_hat_sigmoid
# print recon.size()

# recon = recon.data.numpy()

# recon = np.reshape(recon, [3,32,32]) 
# recon = recon.transpose(1,2,0).astype("uint8")
# print i, recon.shape
# plt.imshow(recon,interpolation='nearest')
# # plt.imshow(toimage(img))
# plt.show()




# fdsa


















# train(model=model, train_x=train_x, train_y=train_y, valid_x=[], valid_y=[], 
#             path_to_load_variables=path_to_load_variables, 
#             path_to_save_variables=path_to_save_variables, 
#             epochs=epochs, batch_size=batch_size, display_epoch=display_epoch, k=1)



# test_elbo =  test(model=model, data_x=test_x, path_to_load_variables='', 
#             batch_size=5, display_epoch=100, k=10)


# print test_elbo

# print 'Done.'







print 'conv encoder, deconv decoder'

model = VAE_deconv1()

if torch.cuda.is_available():
    print 'GPU available, loading cuda'#, torch.cuda.is_available()
    model.cuda()
    # train_x = train_x.cuda()


path_to_load_variables=''
# path_to_load_variables=home+'/Documents/tmp/pytorch_first.pt'
path_to_save_variables=home+'/Documents/tmp/pytorch_vae_cifar_deconv.pt'
# path_to_save_variables=''



train(model=model, train_x=train_x, train_y=train_y, valid_x=[], valid_y=[], 
            path_to_load_variables=path_to_load_variables, 
            path_to_save_variables=path_to_save_variables, 
            epochs=epochs, batch_size=batch_size, display_epoch=display_epoch, k=1)



print test(model=model, data_x=test_x, path_to_load_variables='', 
            batch_size=5, display_epoch=100, k=1000)

print 'Done.'













print 'conv encoder,  deconv then conv decoder'

model = VAE_deconv2()

if torch.cuda.is_available():
    print 'GPU available, loading cuda'#, torch.cuda.is_available()
    model.cuda()
    # train_x = train_x.cuda()


path_to_load_variables=''
# path_to_load_variables=home+'/Documents/tmp/pytorch_first.pt'
path_to_save_variables=home+'/Documents/tmp/pytorch_deconv_conv.pt'
# path_to_save_variables=''



train(model=model, train_x=train_x, train_y=train_y, valid_x=[], valid_y=[], 
            path_to_load_variables=path_to_load_variables, 
            path_to_save_variables=path_to_save_variables, 
            epochs=epochs, batch_size=batch_size, display_epoch=display_epoch, k=1)



print test(model=model, data_x=test_x, path_to_load_variables='', 
            batch_size=5, display_epoch=100, k=1000)













print 'Done.'
















