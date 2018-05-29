
#Train rnn to predict next z and termination

from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import time
from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import imageio
import os
import math
import pickle


from train_vae import VAE





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






class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.h_size = 50
        self.z_size = 100
        self.action_size = 2

        self.act_func = F.leaky_relu # F.relu # # F.tanh ##  F.elu F.softplus
        # self.intermediate_size = 1120 # 1728 #1536
        inter_size = 100

        # AZ Encoder
        self.fc1_gp = nn.Linear(self.z_size+self.action_size, 100)
        self.fc2_gp = nn.Linear(100, inter_size)


        # H Updator
        self.fc3_gp = nn.Linear(inter_size, 100)
        self.fc4_gp = nn.Linear(100, self.h_size)

        # Output Predictor : z and t
        self.fc5_gp = nn.Linear(inter_size+self.h_size, 100)
        self.predict_z = nn.Linear(100, self.z_size)
        self.predict_term = nn.Linear(100, 1)


        params_gp = [list(self.fc1_gp.parameters()) + list(self.fc2_gp.parameters()) + 
                    list(self.fc3_gp.parameters()) + list(self.fc4_gp.parameters()) + 
                    list(self.fc5_gp.parameters()) + 
                    list(self.predict_z.parameters()) + list(self.predict_term.parameters()) ] 

                    
        self.optimizer = optim.Adam(params_gp[0], lr=.0001, weight_decay=.0000001)


    def encode_az(self, a, z):
        az = torch.cat((a, z), 1)
        intermediate = self.act_func(self.fc1_gp(az))
        intermediate = self.act_func(self.fc2_gp(intermediate))
        return intermediate



    def update_h(self, h, intermediate):
        intermediate = self.act_func(self.fc3_gp(intermediate))
        intermediate = self.fc4_gp(intermediate)
        h = h + intermediate
        return h



    def predict_output(self, h, intermediate):
        input_ = torch.cat((h, intermediate), 1)
        intermediate = self.act_func(self.fc5_gp(input_))
        z = self.predict_z(intermediate)
        term = self.predict_term(intermediate)
        return z, term



    def forward(self, z_seq, a_seq, term_seq):
        # x: [B,2,84,84]
        # T = x.size()[0]

        h = torch.zeros(1,self.h_size).cuda()
        z_losses = []
        term_losses = []
        for t in range(len(term_seq)-1):

            inter = self.encode_az(a_seq[t], z_seq[t])
            h = self.update_h(h, inter)
            z_pred, term_pred = self.predict_output(h, inter)

            z_loss = torch.mean((z_seq[t+1] - z_pred)**2)
            term_loss = F.binary_cross_entropy_with_logits(input=term_pred, target=term_seq[t+1])

            z_losses.append(z_loss)
            term_losses.append(term_loss)

        z_loss = torch.mean(torch.stack(z_losses))
        term_loss = torch.mean(torch.stack(term_losses)) 

        loss = z_loss + term_loss 

        return loss, z_loss, term_loss











    def train2(self, epochs, trainingset, validationset, save_dir, start_epoch):

        batch_size = 1

        display_step = 20 # int(len(trainingset) / batch_size)
        
        loss_list = []
        valid_loss_list = []
        total_steps = 0
        epoch_time = 0.0


        training_dataloader = DataLoader(trainingset, batch_size=batch_size, shuffle=True, num_workers=1)
        validation_dataloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=1)

        avg_train_loss = 500.
        avg_valid_loss = 500.
        for epoch in range(epochs):

            start_time = time.time()
            for seq_i, seq in enumerate(training_dataloader):

                # print (seq[0][0].shape)
                seq_len = len(seq)

                z_seq = torch.stack([vae.encode(preprocess(x[0])) for x in seq], 0) #[T,1,Z]

                a_seq = torch.stack([torch.from_numpy(np.array(x[1])) for x in seq], 0).float().cuda() #[T,A]
                a_seq = a_seq.view(seq_len, 1, 2)

                term_seq = torch.stack([x[2] for x in seq], 0).float().cuda() #[T,1]
                term_seq = term_seq.view(seq_len, 1, 1)


                self.optimizer.zero_grad()
                loss, z_loss, term_loss = self.forward(z_seq, a_seq, term_seq) #, DQN)
                loss.backward()
                self.optimizer.step()

                avg_train_loss = avg_train_loss + .1*(loss - avg_train_loss)

                if total_steps%display_step==0: # and batch_idx == 0:

                    # for seq_i, seq in enumerate(validation_dataloader):

                    #     seq_len = len(seq)
                    #     z_seq = torch.stack([vae.encode(preprocess(x[0])) for x in seq], 0) #[T,1,Z]
                    #     a_seq = torch.stack([torch.from_numpy(np.array(x[1])) for x in seq], 0).view(seq_len, 1, 2).float().cuda() #[T,A]
                    #     term_seq = torch.stack([x[2] for x in seq], 0).view(seq_len, 1, 1).float().cuda() #[T,1]
                    #     valid_loss, valid_z_loss, valid_term_loss = self.forward(z_seq, a_seq, term_seq) #, DQN)

                    #     avg_valid_loss = avg_valid_loss + .1*(valid_loss - avg_valid_loss)
    
                    #     break 

                    # valid_loss=0.


                    print ('Epoch: {}/{}'.format(epoch+start_epoch, epochs+start_epoch),
                        'N_Steps:{:5d}'.format(total_steps),
                        'T:{:.2f}'.format(epoch_time),
                        'Loss:{:.4f}'.format(loss.data.item()),
                        'z_loss:{:.4f}'.format(z_loss.data.item()),
                        'term_loss:{:.4f}'.format(term_loss.data.item()),
                        # 'ValidLoss:{:.4f}'.format(valid_loss.data.item()),
                        'AvgTra:{:.4f}'.format(avg_train_loss.data.item()),
                        # 'AvgVal:{:.4f}'.format(avg_valid_loss.data.item()),
                        )
                    if total_steps!=0:
                        loss_list.append(avg_train_loss.data.item())
                        # valid_loss_list.append(avg_valid_loss.data.item())

                total_steps+=1

            epoch_time = time.time() - start_time

            if epoch % 10==0 and epoch > 2:
                #Save params
                save_params_v3(save_dir=save_dir, model=self, epochs=epoch+start_epoch)

            if epoch % 3 ==0 and epoch > 2 and len(loss_list) > 7:
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

    exp_path = save_dir+'rnn_train/'

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        print ('Made dir', exp_path) 


    run_what = 'train'
    # run_what = 'viz'

    if run_what == 'train':
        train_rnn_ = 1
        viz_ = 0
    else:
        train_rnn_ = 0
        viz_ = 1  

    start_epoch = 0 #500





    print ("Load dataset")
    dataset = pickle.load( open( load_dataset_from, "rb" ) )
    print ('numb trajectories', len(dataset))
    # print ([len(x) for x in dataset])
    print ('Avg len', np.mean([len(x) for x in dataset])) #, np.mean([len(x) for x in dataset])*12)
    print ('Max len', np.max([len(x) for x in dataset])) 
    print ('Min len', np.min([len(x) for x in dataset])) 

    dsffad



    # Make every into nupy arrays
    states_dataset = []
    actions_dataset = []
    termnination_dataset = []
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset_states.append(dataset[i][j][0])

    
    training_set = dataset[:300]
    validation_set = dataset[300:]


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


    fdsa
    





    print("Initializing VAE...")
    vae = VAE()
    # if start_epoch>0:
        # load_params_v3(save_dir=exp_path, model=vae, epochs=start_epoch)
    load_params_v3(save_dir=save_dir+'vae_with_valid/', model=vae, epochs=500)
    vae.cuda()
    print("VAE initialized\n")
    




    print("Initializing RNN...")
    rnn = RNN()
    if start_epoch>0:
        load_params_v3(save_dir=exp_path, model=rnn, epochs=start_epoch)
        # load_params_v3(save_dir=exp_path + 'blur_v2_moretrainin500/', model=vae, epochs=start_epoch)
    rnn.cuda()
    print("RNN initialized\n")
    










    if train_rnn_:

        print("Training")
        rnn.train2(epochs=epochs, trainingset=training_set, validationset=validation_set,
                     save_dir=exp_path, start_epoch=start_epoch)

        save_params_v3(save_dir=exp_path, model=rnn, epochs=epochs+start_epoch)

















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
































































