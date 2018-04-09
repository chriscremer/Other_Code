




import numpy as np
import pickle#, cPickle
from os.path import expanduser
home = expanduser("~")

# import matplotlib.pyplot as plt


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



from vae_utils import lognormal2 as lognormal
from vae_utils import log_bernoulli

# from scipy.misc import toimage
from vae import VAE
# from vae import train
import time



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


train_ = 0





print ('load data')


data = pickle.load( open( home + '/Documents/tmp/montezum_frames.pkl', "rb" ) )

data = np.array(data)
data = data  / 255.0
print (data.shape)
train_x = np.float32(data)




print ('init vae')
model = VAE()
model.cuda()


path_to_load_variables = home + '/Documents/tmp/montezum_vae.ckpt'

if path_to_load_variables != '':
    model.load_state_dict(torch.load(path_to_load_variables))
    print ('loaded variables ' + path_to_load_variables)


learning_rate= .0001# .0004 ##
epochs = 100
batch_size = 100
display_epoch = 2
k  =1
# dtype = 

if train_:

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    time_ = time.time()
    start_time = time.time()

    n_data = len(train_x)
    arr = np.array(range(n_data))

    for epoch in range(1, epochs + 1):

        # print (epoch)

        #shuffle
        np.random.shuffle(arr)
        train_x = train_x[arr]

        data_index= 0
        for i in range(int(n_data/batch_size)):
            batch = train_x[data_index:data_index+batch_size]
            data_index += batch_size

            # batch = Variable(torch.from_numpy(batch)).type(model.dtype)
            batch = Variable(torch.from_numpy(batch)).cuda()
            optimizer.zero_grad()

            elbo, logpx, logpz, logqz = model.forward(batch, k=k)

            loss = -(elbo)
            loss.backward()
            optimizer.step()


        if epoch%display_epoch==0:
            print ('Train Epoch: {}/{}'.format(epoch, epochs),
                'LL:{:.3f}'.format(-loss.data[0]),
                'logpx:{:.3f}'.format(logpx.data[0]),
                'logpz:{:.3f}'.format(logpz.data[0]),
                'logqz:{:.3f}'.format(logqz.data[0]),
                'T:{:.2f}'.format(time.time()-time_),
                )
            time_ = time.time()



    path_to_save_variables = home + '/Documents/tmp/montezum_vae.ckpt'

    if path_to_save_variables != '':
        torch.save(model.state_dict(), path_to_save_variables)
        print ('Saved variables to ' + path_to_save_variables)


# train = torch.utils.data.TensorDataset(train_x, train_y)
# train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
# optimizer = optim.Adam(model.parameters(), lr=.0001)

# for epoch in range(1, epochs + 1):

#     for batch_idx, (data, target) in enumerate(train_loader):

#         # if data.is_cuda:
#         if torch.cuda.is_available():
#             data = Variable(data).type(torch.cuda.FloatTensor)# , Variable(target).type(torch.cuda.LongTensor) 
#         else:
#             data = Variable(data)#, Variable(target)

#         optimizer.zero_grad()

#         elbo, logpx, logpz, logqz = model.forward(data, k=k)
#         loss = -(elbo)

#         loss.backward()
#         optimizer.step()

#         if epoch%display_epoch==0 and batch_idx == 0:
#             print 'Train Epoch: {}/{} [{}/{} ({:.0f}%)]'.format(epoch, epochs, 
#                     batch_idx * len(data), len(train_loader.dataset),
#                     100. * batch_idx / len(train_loader)), \
#                 'Loss:{:.4f}'.format(loss.data[0]), \
#                 'logpx:{:.4f}'.format(logpx.data[0]), \
#                 'logpz:{:.4f}'.format(logpz.data[0]), \
#                 'logqz:{:.4f}'.format(logqz.data[0]) 



# #plot just real frames
# fig = plt.figure(figsize=(12,8), facecolor='white')
# rows = 5
# cols = 10
# for i in range(cols*rows):

#     frame = train_x[i*32]

#     batch = Variable(torch.from_numpy(frame)).cuda()
#     elbo, logpx, logpz, logqz, recon = model.forward3(batch, k=1)
#     # probs.append(elbo.data.cpu().numpy())

#     recon = recon[0].view(84,84).data.cpu().numpy()
#     # recons.append(recon)



#     # plot frame
#     ax = plt.subplot2grid((rows,cols), (int(i/cols),i%cols), frameon=False)#, colspan=3)
#     # ax = plt.subplot2grid((rows,cols), (0,i%cols), frameon=False)#, colspan=3)
#     # state1 = np.squeeze(state[0])
#     state1 = np.squeeze(frame)
#     ax.imshow(state1, cmap='gray')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     # ax.set_title('State '+str(step),family='serif')


#     # # plot recon
#     # ax = plt.subplot2grid((rows,cols), (1,i), frameon=False)#, colspan=3)
#     # # state1 = np.squeeze(state[0])
#     # # state1 = train_x[step]
#     # ax.imshow(recon, cmap='gray')
#     # ax.set_xticks([])
#     # ax.set_yticks([])
#     # # ax.set_title('State '+str(step),family='serif')




# #plot fig
# plt.tight_layout(pad=1.5, w_pad=.4, h_pad=1.)
# plt_path = home + '/Documents/tmp/plots_recons.png'
# plt.savefig(plt_path)
# print ('saved',plt_path)
# plt.close(fig)










#plot frames and recons
fig = plt.figure(figsize=(14,8), facecolor='white')
rows = 2
cols = 10
for i in range(cols):

    frame = train_x[i*32*15]

    batch = Variable(torch.from_numpy(frame)).cuda()
    elbo, logpx, logpz, logqz, recon = model.forward3(batch, k=100)
    # probs.append(elbo.data.cpu().numpy())
    print (i*32*15, elbo.data.cpu().numpy())

    recon = recon[0].view(84,84).data.cpu().numpy()
    # recons.append(recon)



    # plot frame
    ax = plt.subplot2grid((rows,cols), (0,i), frameon=False)#, colspan=3)
    # state1 = np.squeeze(state[0])
    state1 = np.squeeze(frame)
    ax.imshow(state1, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title('State '+str(step),family='serif')


    # plot recon
    ax = plt.subplot2grid((rows,cols), (1,i), frameon=False)#, colspan=3)
    # state1 = np.squeeze(state[0])
    # state1 = train_x[step]
    ax.imshow(recon, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title('State '+str(step),family='serif')




#plot fig
plt.tight_layout(pad=1.5, w_pad=.4, h_pad=1.)
plt_path = home + '/Documents/tmp/plots_recons2.png'
plt.savefig(plt_path)
print ('saved',plt_path)
plt.close(fig)








