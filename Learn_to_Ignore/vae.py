
#Train vae on cifar

import os
from os.path import expanduser
home = expanduser("~")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torch.autograd import Variable

import time
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import imageio
import math
import pickle



def lognormal(x, mean, logvar):
    '''
    x: [P,B,Z]
    mean,logvar: [B,Z]
    output: [P,B]
    '''

    return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(2))


    
def log_bernoulli(pred_no_sig, target):
    '''
    pred_no_sig is [P, B, X] 
    t is [B, X]
    output is [P, B]
    '''

    return -(torch.clamp(pred_no_sig, min=0)
                        - pred_no_sig * target
                        + torch.log(1. + torch.exp(-torch.abs(pred_no_sig)))).sum(2) #sum over dimensions



def load_cifar10():

    print ('Loading CIFAR10')


    #CIFAR10
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

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
    test_y = np.array(b1['labels'])
    valid_x = test_x
    valid_y = test_y

    train_x = train_x / 255.
    valid_x = valid_x / 255.

    assert np.max(train_x) <= 1.
    assert np.min(train_x) >= 0.

    return train_x, train_y, valid_x, valid_y






class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()


        self.act_func = F.leaky_relu# F.relu # # F.tanh ##  F.elu F.softplus

        self.x_size = 3072
        self.z_size = 100

        self.intermediate_size = 1152

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=0, dilation=1, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=0, dilation=1, bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=0, dilation=1, bias=True)

        self.fc1 = nn.Linear(self.intermediate_size, 200)
        self.fc2 = nn.Linear(200, self.z_size*2)
        self.fc3 = nn.Linear(self.z_size, 200)
        self.fc4 = nn.Linear(200, self.intermediate_size)

        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        self.deconv3 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

        self.optimizer = optim.Adam(self.parameters(), lr=.0001, weight_decay=.0000001)


    def encode(self, x):

        # (w-k)/s + 1
        # x = x.view(-1, 3, 32, 32)
        # print (x.shape)

        x = self.act_func(self.conv1(x))
        # print (x.shape)

        x = self.act_func(self.conv2(x))
        # print (x.shape)

        x = self.act_func(self.conv3(x))

        # print (x.shape)
        # fdsf

        x = x.view(-1, self.intermediate_size)

        h1 = self.act_func(self.fc1(x))
        h2 = self.fc2(h1)
        mean = h2[:,:self.z_size]
        logvar = h2[:,self.z_size:]
        return mean, logvar

    def sample(self, mu, logvar, k):
        # if torch.cuda.is_available():
        # eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_()).cuda() #[P,B,Z]
        eps = torch.FloatTensor(k, self.B, self.z_size).normal_().cuda() #[P,B,Z]


        z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
        # logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size).cuda()), 
        #                     Variable(torch.zeros(self.B, self.z_size)).cuda())  #[P,B]
        logpz = lognormal(z, torch.zeros(self.B, self.z_size).cuda(), 
                            torch.zeros(self.B, self.z_size).cuda()) #[P,B]
        logqz = lognormal(z, mu, logvar)


        # else:
        #     eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_())#[P,B,Z]
        #     z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
        #     logpz = lognormal(z, Variable(torch.zeros(self.B, self.z_size)), 
        #                         Variable(torch.zeros(self.B, self.z_size)))  #[P,B]
        #     logqz = lognormal(z, mu, logvar) 
        return z, logpz, logqz



    def decode(self, z):

        z = self.act_func(self.fc3(z)) 
        z = self.act_func(self.fc4(z))  #[B,1960]

        z = z.view(-1, 32, 6, 6)
        # print (z.shape)
        z = self.act_func(self.deconv1(z))
        z = self.act_func(self.deconv2(z))
        z = self.deconv3(z)
        # z = z.view(-1, self.x_size)
        
        return z





    def forward(self, x, k=1):
        
        self.B = x.size()[0]
        mu, logvar = self.encode(x)
        z, logpz, logqz = self.sample(mu, logvar, k=k)  #[P,B,Z]
        x_hat = self.decode(z)  #[PB,X]
        # x_hat = x_hat.view(k, self.B, -1)
        # print x_hat.size()
        # print x_hat.size()
        # print x.size()


        logpx = -F.binary_cross_entropy_with_logits(input=x_hat, target=x) * self.x_size


        # print (logpx.shape)
        # print (logpz.shape)
        # print (logqz.shape)
        # fsda
        # logpx = log_bernoulli(x_hat, x)  #[P,B]

        logpz = torch.mean(logpz)
        logqz = torch.mean(logqz)

        elbo = logpx + logpz - logqz  #[P,B]

        # if k>1:
        #     max_ = torch.max(elbo, 0)[0] #[B]
        #     elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]

        # elbo = torch.mean(elbo) #[1]

        # #for printing
        # logpx = torch.mean(logpx)
        # logpz = torch.mean(logpz)
        # logqz = torch.mean(logqz)
        # self.x_hat_sigmoid = F.sigmoid(x_hat)

        return elbo, logpx, logpz, logqz







def train(model, epochs, train_x, valid_x, save_dir, params_dir):

    # epochs = 30
    batch_size = 32# 100

    display_step = 300
    save_epoch = 10
    plot_epoch = 2


    train_x = torch.from_numpy(train_x).float().type(torch.FloatTensor).cuda()
    # train_y = torch.from_numpy(train_y)

    valid_x = torch.from_numpy(valid_x).float().type(torch.FloatTensor).cuda()
    # valid_y = torch.from_numpy(valid_y)

    train_ = torch.utils.data.TensorDataset(train_x) #, train_y)
    train_loader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True)

    valid_ = torch.utils.data.TensorDataset(valid_x) #, valid_y)
    valid_loader = torch.utils.data.DataLoader(valid_, batch_size=batch_size, shuffle=True)

    recent_train_elbos = deque(maxlen=10)
    recent_valid_elbos = deque(maxlen=10)

    all_train_elbos = []
    all_valid_elbos = []

    start_time = time.time()

    model.train()#for BN
    
    start_epoch = 0

    for epoch in range(epochs):

        for batch_idx, (data) in enumerate(train_loader):

            batch = data[0] #Variable(data)#.type(model.dtype)

            model.optimizer.zero_grad()

            elbo, logpx, logpz, logqz = model.forward(batch)

            loss = -elbo

            loss.backward()
            model.optimizer.step()

            if batch_idx % display_step == 0:

                for i_batch, data in enumerate(valid_loader):
                    batch = data[0]
                    valid_elbo, valid_logpx, valid_logpz, valid_logqz = model.forward(batch)
                    break 

                recent_train_elbos.append(elbo.data.item())
                recent_valid_elbos.append(valid_elbo.data.item())

                T = time.time() - start_time
                start_time = time.time()

                print ('Train Epoch: {}/{}'.format(epoch+start_epoch, epochs+start_epoch),
                    # 'N_Steps:{:5d}'.format(total_steps),
                    'T:{:.2f}'.format(T),
                    'Loss:{:.4f}'.format(loss.data.item()),
                    'logpx:{:.4f}'.format(logpx.data.item()),
                    'logpz:{:.4f}'.format(logpz.data.item()),
                    'logqz:{:.4f}'.format(logqz.data.item()),
                    'TrainAvg:{:.4f}'.format(np.mean(recent_train_elbos)),
                    'ValidAvg:{:.4f}'.format(np.mean(recent_valid_elbos)),
                    # 'ValidLoss:{:.4f}'.format(valid_loss.data.item()),
                    # 'tran:{:.4f}'.format(tran_loss.data.item()), 
                    # 'term:{:.4f}'.format(terminal_loss.data.item()),
                    )

                # if total_steps!=0:
                all_train_elbos.append(np.mean(recent_train_elbos))
                all_valid_elbos.append(np.mean(recent_valid_elbos))


        if epoch % save_epoch==0 and epoch > 0:
            #Save params
            save_params_v3(save_dir=params_dir, model=model, epochs=epoch+start_epoch)

        if epoch % plot_epoch==0 and epoch > 0 and len(all_train_elbos) > 2:
            #plot the training curve
            plt.plot(all_train_elbos[1:], label='Train')
            plt.plot(all_valid_elbos[1:], label='Valid')
            # save_dir = home+'/Documents/tmp/Doom/'
            plt_path = save_dir+'training_plot.png'
            plt.legend()
            plt.savefig(plt_path)
            print ('saved training plot',plt_path)
            plt.close()









def load_params_v3(save_dir, model, epochs):
    save_to=os.path.join(save_dir, "model_params" + str(epochs)+".pt")
    model.load_state_dict(torch.load(save_to))
    print ('loaded', save_to)





def save_params_v3(save_dir, model, epochs):
    save_to=os.path.join(save_dir, "model_params" + str(epochs)+".pt")
    torch.save(model.state_dict(), save_to)
    print ('saved', save_to)











if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] =  '0' #'1' #

    load_ = 0
    save_ = 0

    epochs = 30
    start_epoch = 0

    save_dir = home+'/Documents/tmp/cifar_vae/'
    params_dir = save_dir + 'params/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print ('Made dir', save_dir) 

    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
        print ('Made dir', params_dir) 


    #Load data
    train_x, train_y, valid_x, valid_y = load_cifar10()
    train_x = np.reshape(train_x, [train_x.shape[0], 3, 32, 32])
    valid_x = np.reshape(valid_x, [valid_x.shape[0], 3, 32, 32])

    print (train_x.shape)
    print (train_y.shape)
    print (valid_x.shape)
    print (valid_y.shape)
    print()


    print ('Init VAE')
    vae = VAE()
    vae.cuda()
    print ('VAE Initilized\n')

    print ('Train')
    train(vae, epochs=epochs, train_x=train_x, valid_x=valid_x, save_dir=save_dir, params_dir=params_dir)
    save_params_v3(save_dir=params_dir, model=vae, epochs=epochs+start_epoch)

    print ('Done.')
    fdsa

















































































    fsad


    #Init model
    print ('Loading model')
    use_cuda = True# torch.cuda.is_available()
    n_gpus = 1#2 #torch.cuda.device_count()
    if n_gpus < 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'# '1' #which gpu

    if load_:
        loaded_state = torch.load(save_file)
        model = loaded_state['model']
        # model.load_params(path_to_load_variables=save_file)
        print ('loaded model ' + save_file)

    else:
        # model = CIFAR_ConvNet()
        # model = ResNet18()
        # model = PreActResNet18()
        model = PreActResNet10()

    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(n_gpus))

    print (model)
    print ()

    #Train model
    # optimizer = optim.SGD(model.parameters(), lr=.005, momentum=.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=.0005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    train(train_x, train_y, valid_x, valid_y)


    if save_:
        state = {
            'model': model.module if use_cuda else model,
            'epoch': 1,
        }
        # torch.save(model.state_dict(), save_file)
        torch.save(state, save_file)
        print ('saved model ' + save_file)









fsdfa



















def preprocess_action(action):
    #onehot
    n_actions = 4
    B = action.size()[0]
    y_onehot = torch.FloatTensor(B, n_actions).zero_().cuda()
    y_onehot.scatter_(1, action.view(B,1).long(), 1)
    return y_onehot
 

def preprocess(img):
    # numpy to pytoch, float, scale, cuda, downsample
    # img = torch.from_numpy(img).cuda().float() / 255.
    # img = (img.float() / 255.).cuda()
    img = img.cuda().float() / 255.
    # img = F.avg_pool2d(img, kernel_size=2, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
    # print (img.shape) #[3,240,320]
    # fsdaf
    return img 






class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        image_channels = 3
        self.z_size = 100

        self.act_func = F.leaky_relu# F.relu # # F.tanh ##  F.elu F.softplus
        self.intermediate_size = 1120 # 1728 #1536

        #ENCODER
        self.conv1_gp = nn.Conv2d(image_channels, 32, kernel_size=10, stride=5)
        self.conv2_gp = nn.Conv2d(32, 64, kernel_size=7, stride=4)
        self.conv3_gp = nn.Conv2d(64, 32, kernel_size=3, stride=2)
        self.fc1_gp = nn.Linear(self.intermediate_size, 200)
        self.fc2_gp = nn.Linear(200, self.z_size)


        #DECODER
        self.fc3_gp = nn.Linear(self.z_size, 200)
        self.fc4_gp = nn.Linear(200, self.intermediate_size)
        self.deconv1_gp = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)#, output_padding=(0,1))
        self.deconv2_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=7, stride=4)#, output_padding=(3,3))
        self.deconv3_gp = torch.nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=10, stride=5)#, output_padding=(4,4))


        params_gp = [list(self.conv1_gp.parameters()) + list(self.conv2_gp.parameters()) + list(self.conv3_gp.parameters()) +
                    list(self.fc1_gp.parameters()) + list(self.fc2_gp.parameters()) + 
                    list(self.fc3_gp.parameters()) + list(self.fc4_gp.parameters()) +
                    list(self.deconv1_gp.parameters()) + list(self.deconv2_gp.parameters()) + list(self.deconv3_gp.parameters())]
         

                    
        self.optimizer = optim.Adam(params_gp[0], lr=.0001, weight_decay=.0000001)


    def encode(self, x):
        B = x.shape[0]
        # print (x.size())  #[B,32,480,640]
        x = self.act_func(self.conv1_gp(x))  #[B,32,59,79]
        # print (x.size())
        x = self.act_func(self.conv2_gp(x)) #[B,64,13,18]
        # print (x.size())
        x = self.act_func(self.conv3_gp(x)) #[B,32,5,7]
        # print (x.size())
        # fds
        x = x.view(B, self.intermediate_size)
        x = self.act_func(self.fc1_gp(x))
        z = self.fc2_gp(x)

        return z



    def decode(self, z):
        B = z.size()[0]
        z = self.act_func(self.fc3_gp(z)) 
        z_pre = self.act_func(self.fc4_gp(z))  #[B,11264]
        # print (z_pre.shape)
        z = z_pre.view(B, 32, 5, 7)
        z = self.act_func(self.deconv1_gp(z)) #[B,64,13,18]
        # print (z.size())
        z = self.act_func(self.deconv2_gp(z)) #[B,64,59,79]
        # print (z.size())
        x = self.deconv3_gp(z) # [B,1,452,580]
        # print (x.size())
        # fdf


        # z = self.act_func(self.fc_isterm1(z_pre)) 
        # isTerm = F.sigmoid(self.fc_isterm2(z))


        # return F.sigmoid(z)
        return x#, isTerm



    def forward(self, x):
        # x: [B,2,84,84]
        B = x.size()[0]

        z = self.encode(x) 
        # za = torch.cat((z1.detach(),a),1) #[B,z+a]
        # z2_prior = self.transition(za)
        # z2 = self.encode(s2)
        recon = self.decode(z) 

        # z_loss = torch.mean(z2**2) * .001

        recon_loss = F.binary_cross_entropy_with_logits(input=recon, target=x) #, reduce=False)

        loss = recon_loss #+ tran_loss + terminal_loss + z_loss

        return loss, recon_loss#, tran_loss, terminal_loss, z_loss











    def train2(self, epochs, trainingset, validationset, save_dir, start_epoch):

        batch_size = 32

        display_step = int(len(trainingset) / 32)  # 200 
        
        loss_list = []
        valid_loss_list = []
        total_steps = 0
        epoch_time = 0.0


        training_dataloader = DataLoader(trainingset, batch_size=batch_size, shuffle=True, num_workers=2)
        validation_dataloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=2)

        for epoch in range(epochs):

            start_time = time.time()
            for i_batch, batch in enumerate(training_dataloader):

                batch = preprocess(batch) #.cuda()

                self.optimizer.zero_grad()
                loss, recon_loss = self.forward(batch) #, DQN)
                loss.backward()
                self.optimizer.step()

                if total_steps%display_step==0: # and batch_idx == 0:

                    for i_batch, batch in enumerate(validation_dataloader):
                        batch = preprocess(batch) 
                        valid_loss, valid_recon_loss = self.forward(batch)
                        break 


                    print ('Train Epoch: {}/{}'.format(epoch+start_epoch, epochs+start_epoch),
                        'N_Steps:{:5d}'.format(total_steps),
                        'T:{:.2f}'.format(epoch_time),
                        'Loss:{:.4f}'.format(loss.data.item()),
                        'Recon:{:.4f}'.format(recon_loss.data.item()),
                        'ValidLoss:{:.4f}'.format(valid_loss.data.item()),
                        # 'tran:{:.4f}'.format(tran_loss.data.item()), 
                        # 'term:{:.4f}'.format(terminal_loss.data.item()),
                        )
                    if total_steps!=0:
                        loss_list.append(loss.data.item())
                        valid_loss_list.append(valid_loss.data.item())

                total_steps+=1

            epoch_time = time.time() - start_time

            if epoch % 10==0 and epoch > 2:
                #Save params
                save_params_v3(save_dir=save_dir, model=self, epochs=epoch+start_epoch)

                if len(loss_list) > 7:
                    #plot the training curve
                    plt.plot(loss_list[2:], label='Train')
                    plt.plot(valid_loss_list[2:], label='Valid')
                    # save_dir = home+'/Documents/tmp/Doom/'
                    plt_path = save_dir+'training_plot.png'
                    plt.legend()
                    plt.savefig(plt_path)
                    print ('saved training plot',plt_path)
                    plt.close()









def load_params_v3(save_dir, model, epochs):
    save_to=os.path.join(save_dir, "MP_params" + str(epochs)+".pt")
    model.load_state_dict(torch.load(save_to))
    print ('loaded', save_to)





def save_params_v3(save_dir, model, epochs):
    save_to=os.path.join(save_dir, "MP_params" + str(epochs)+".pt")
    torch.save(model.state_dict(), save_to)
    print ('saved', save_to)



































if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1' # '0' #,1'

    resolution = (240,320) # (480, 640)

    epochs = 500

    save_dir =  home+'/Documents/tmp/Doom2/' 
    load_dataset_from = save_dir+'doom_dataset_10000.pkl'

    exp_path = save_dir+'vae_with_valid/'

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        print ('Made dir', exp_path) 


    run_what = 'train'
    # run_what = 'viz'

    if run_what == 'train':
        train_dkf_ = 1
        viz_ = 0
    else:
        train_dkf_ = 0
        viz_ = 1  

    start_epoch = 0 #500





    print ("Load dataset")
    dataset = pickle.load( open( load_dataset_from, "rb" ) )
    print ('numb trajectories', len(dataset))
    # print ([len(x) for x in dataset])
    print ('Avg len', np.mean([len(x) for x in dataset]), np.mean([len(x) for x in dataset])*12)
    #Put all trajectories into one list
    dataset_states = []
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset_states.append(dataset[i][j][0])
    
    training_set = dataset_states[:9000]
    validation_set = dataset_states[9000:]


    class DoomDataset(Dataset):

        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    training_set = DoomDataset(data=training_set)
    validation_set = DoomDataset(data=validation_set)
    print ('training set:', len(training_set))
    print ('validation set:', len(validation_set))
    print()
    





    print("Initializing VAE...")
    vae = VAE()
    if start_epoch>0:
        load_params_v3(save_dir=exp_path, model=vae, epochs=start_epoch)
        # load_params_v3(save_dir=exp_path + 'blur_v2_moretrainin500/', model=vae, epochs=start_epoch)
    vae.cuda()
    # n_gpus = 2
    # print (torch.cuda.device_count())
    # fad
    # MP = torch.nn.DataParallel(MP, device_ids=range(n_gpus)).cuda()
    # MP = torch.nn.DataParallel(MP, device_ids=[1])
    # print (MP)
    print("VAE initialized\n")
    









    if train_dkf_:

        print("Training")
        vae.train2(epochs=epochs, trainingset=training_set, validationset=validation_set,
                     save_dir=exp_path, start_epoch=start_epoch)

        save_params_v3(save_dir=exp_path, model=vae, epochs=epochs+start_epoch)

















    if viz_:

        # load_from = save_dir+'doom_dataset_10000.pkl'


        # print ("Load dataset")
        # # dir_ = 'RoadRunner_colour_4'
        # # dataset = pickle.load( open( home+'/Documents/tmp/breakout_2frames/breakout_trajectories_10000.pkl', "rb" ) )
        # dataset = pickle.load( open( load_from, "rb" ) )

        # print ('numb trajectories', len(dataset))
        # print ([len(x) for x in dataset])
        # print ('Avg len', np.mean([len(x) for x in dataset]), np.mean([len(x) for x in dataset])*12)
        # print()



        rows = 1
        cols = 1

        traj_numb = 100
        print ('traj', str(traj_numb), 'len', str(len(dataset[traj_numb])))

        gif_dir = exp_path + 'view_traj_recon'+str(traj_numb)+'/'


        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
            print ('Made dir', gif_dir) 
        else:
            print (gif_dir, 'exists')
            # fasdf 

        max_count = 100000 #1000
        # count = 0    

        cols = 2
        rows = 1

        traj = dataset[traj_numb]
        for i in range(len(traj)):

            s = traj[i][0]
            a = traj[i][1]
            t = traj[i][2]

            plt_path = gif_dir+'frame'+str(i)+'.png'
            # fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=640*rows)
            fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=150)


            frame = np.rollaxis(s, 1, 0)
            frame = np.rollaxis(frame, 2, 1)

            s1 = preprocess(torch.from_numpy(np.array([s])))
            recon = F.sigmoid(vae.decode(vae.encode( s1)))
            recon = recon.data.cpu().numpy()[0]

            recon = np.rollaxis(recon, 1, 0)
            recon = np.rollaxis(recon, 2, 1)






            #Plot Frame
            ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.1, 1.04, 'Real Frame '+str(i)+'  a:'+str(a)+'  t:'+str(t), transform=ax.transAxes, family='serif', size=6)

            #Plot recon
            ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)
            ax.imshow(recon)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.1, 1.04, 'Recon', transform=ax.transAxes, family='serif', size=6)


            # plt.tight_layout()
            plt.savefig(plt_path)
            print ('saved viz',plt_path)
            plt.close(fig)

            # frame_count+=1

            # count+=1



        # print ('game over', game.is_episode_finished() )
        # print (count)
        # print (len(frames))


        print('Making Gif')
        # frames_path = save_dir+'gif/'
        images = []
        for i in range(len(traj)):
            images.append(imageio.imread(gif_dir+'frame'+str(i)+'.png'))

        gif_path_this = gif_dir+ 'gif_'+str(traj_numb)+'.gif'
        # imageio.mimsave(gif_path_this, images)
        imageio.mimsave(gif_path_this, images, duration=.1)
        print ('made gif', gif_path_this)






    print ('Done.')
































































