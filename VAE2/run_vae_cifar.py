

import numpy as np
import _pickle as cPickle
from os.path import expanduser
home = expanduser("~")

import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import torch
# from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F

# from utils import lognormal2 as lognormal
# from utils import log_bernoulli

# from scipy.misc import toimage

from vae import VAE
from inference_net import Inference_Net


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def smooth_list(x, window_len=5, window='flat'):
    if len(x) < window_len:
        return x
    w = np.ones(window_len,'d') 
    y = np.convolve(w/ w.sum(), x, mode='same')
    y[-1] = x[-1]
    y[-2] = x[-2]
    return y


def logmeanexp(elbo):
    # [P,B]
    max_ = torch.max(elbo, 0)[0] #[B]
    elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
    return elbo






def get_batch(data, batch_size):

    N = len(data)
    idx = np.random.randint(N, size=batch_size)
    return data[idx]


def LL_to_BPD(LL):

    dim = 32*32*3
    nats = -LL + (np.log(256.)*dim)
    NPD = nats / dim
    BPD = NPD / np.log(2.)

    return BPD





def train(model, train_x, train_y, valid_x, valid_y, 
                save_dir, params_dir, images_dir,
                batch_size, 
                max_steps, display_step, save_steps, viz_steps, trainingplot_steps, load_step):

    # random.seed( 0 )
    
    all_dict = {}

    list_recorded_values = [ 
            'all_steps', 'all_warmup',
            'all_train_elbos', 'all_valid_elbos', 'all_svhn_elbos', 
            'all_train_logpx', 'all_valid_logpx', 'all_svhn_logpx', 
            'all_train_logpz', 'all_valid_logpz', 'all_svhn_logpz', 
            'all_train_logqz', 'all_valid_logqz', 'all_svhn_logqz', 
            'all_train_bpd', 'all_valid_bpd', 'all_svhn_bpd', 
    ]

    for label in list_recorded_values:
        all_dict[label] = []


    train = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    # optimizer = optim.Adam(model.parameters(), lr=.0001)

    # load_step = 0
    max_beta = 1.
    warmup_steps = 20000. 
    
    limit_numb = 1000


    step = 0
    # for epoch in range(1, epochs + 1):
    while step < max_steps:

        for batch_idx, (batch, target) in enumerate(train_loader):

            warmup = min((step+load_step) / float(warmup_steps), max_beta)

            # print ((batch).shape)
            # print ((batch + torch.FloatTensor(batch.shape).uniform_(0., 1./256.)).shape)
            # print (torch.min(batch + torch.FloatTensor(batch.shape).uniform_(0., 1./256.)))
            # print (torch.max(batch + torch.FloatTensor(batch.shape).uniform_(0., 1./256.)))
            # fasdf

            model.optimizer_x.zero_grad()
            outputs = model.forward(x=batch.cuda(), warmup=warmup) 
            loss = -outputs['welbo']
            loss.backward()
            model.optimizer_x.step()
            model.scheduler_x.step()

            if step % 2 ==0:

                infnet_valid.optimizer_x.zero_grad()
                valid_outputs = model.forward(x=get_batch(valid_x, batch_size).cuda(), warmup=1., inf_net=infnet_valid) 
                loss = -valid_outputs['welbo']
                loss.backward()
                infnet_valid.optimizer_x.step()
                infnet_valid.scheduler_x.step()

                infnet_svhn.optimizer_x.zero_grad()
                svhn_outputs = model.forward(x=get_batch(svhn, batch_size).cuda(), warmup=1., inf_net=infnet_svhn) 
                loss = -svhn_outputs['welbo']
                loss.backward()
                infnet_svhn.optimizer_x.step()
                infnet_svhn.scheduler_x.step()



            step +=1

            if step%display_step==0:
                print (
                    'S:{:5d}'.format(step),
                    # 'T:{:.2f}'.format(T),
                    'BPD:{:.4f}'.format(LL_to_BPD(outputs['elbo'].data.item())),
                    'welbo:{:.4f}'.format(outputs['welbo'].data.item()),
                    'elbo:{:.4f}'.format(outputs['elbo'].data.item()),
                    'lpx:{:.4f}'.format(outputs['logpx'].data.item()),
                    'lpz:{:.4f}'.format(outputs['logpz'].data.item()),
                    'lqz:{:.4f}'.format(outputs['logqz'].data.item()),
                    'lpx_v:{:.4f}'.format(valid_outputs['logpx'].data.item()),
                    'lpz_v:{:.4f}'.format(valid_outputs['logpz'].data.item()),
                    'lqz_v:{:.4f}'.format(valid_outputs['logqz'].data.item()),
                    'warmup:{:.4f}'.format(warmup),
                    )

                # model.eval()
                # with torch.no_grad():
                #     valid_outputs = model.forward(x=valid_x[:50].cuda(), warmup=1., inf_net=infnet_valid)
                #     svhn_outputs = model.forward(x=svhn[:50].cuda(), warmup=1., inf_net=infnet_svhn)
                # model.train()


                if step > limit_numb:

                    all_dict['all_steps'].append(step+load_step)
                    all_dict['all_warmup'].append(warmup)
                    all_dict['all_train_elbos'].append(outputs['elbo'].data.item())
                    all_dict['all_valid_elbos'].append(valid_outputs['elbo'].data.item())
                    all_dict['all_svhn_elbos'].append(svhn_outputs['elbo'].data.item())

                    all_dict['all_train_logpx'].append(outputs['logpx'].data.item())
                    all_dict['all_valid_logpx'].append(valid_outputs['logpx'].data.item())
                    all_dict['all_svhn_logpx'].append(svhn_outputs['logpx'].data.item())
        
                    all_dict['all_train_logpz'].append(outputs['logpz'].data.item())
                    all_dict['all_valid_logpz'].append(valid_outputs['logpz'].data.item())
                    all_dict['all_svhn_logpz'].append(svhn_outputs['logpz'].data.item())

                    all_dict['all_train_logqz'].append(outputs['logqz'].data.item())
                    all_dict['all_valid_logqz'].append(valid_outputs['logqz'].data.item())
                    all_dict['all_svhn_logqz'].append(svhn_outputs['logqz'].data.item())

                    all_dict['all_train_bpd'].append(LL_to_BPD(outputs['elbo'].data.item()))
                    all_dict['all_valid_bpd'].append(LL_to_BPD(valid_outputs['elbo'].data.item()))
                    all_dict['all_svhn_bpd'].append(LL_to_BPD(svhn_outputs['elbo'].data.item()))

            if step % trainingplot_steps==0 and step > 0 and len(all_dict['all_train_elbos']) > 2:
                plot_curves(model, save_dir, all_dict)

            if step % save_steps==0 and step > 0:

                model.save_params_v3(save_dir=params_dir, step=step+load_step)
                infnet_valid.save_params_v3(save_dir=params_dir, step=step+load_step, name='valid')
                infnet_svhn.save_params_v3(save_dir=params_dir, step=step+load_step, name='svhn')

            if step % viz_steps==0 and step > 0: 

                model.eval()
                with torch.no_grad():
                    train_recon = model.forward(x=train_x[:10].cuda(), warmup=1.)['x_recon']
                    valid_recon = model.forward(x=valid_x[:10].cuda(), warmup=1., inf_net=infnet_valid)['x_recon']
                    svhn_recon = model.forward(x=svhn[:10].cuda(), warmup=1., inf_net=infnet_svhn)['x_recon']
                    sample_prior = model.sample_prior(z=z_prior)
                model.train()

                vizualize(images_dir, step, train_real=train_x[:10], train_recon=train_recon,
                                            valid_real=valid_x[:10], valid_recon=valid_recon,
                                            svhn_real=svhn[:10], svhn_recon=svhn_recon,
                                            prior_samps=sample_prior)






def eval_model(model, train_x, train_y, valid_x, valid_y, 
                save_dir, params_dir, images_dir,
                batch_size, 
                max_steps, display_step, save_steps, viz_steps, trainingplot_steps, load_step):

    model.eval()

    k = 10
    n_data = 100
    print ('k=',k,'\n')
    LLs_train = compute_LL(model, train_x[:n_data], k=k)
    print ('Train', np.mean(LLs_train), np.std(LLs_train))
    print ()
    LLs_valid = compute_LL(model, valid_x[:n_data], k=k)
    print ('Valid', np.mean(LLs_valid), np.std(LLs_valid))
    print()
    LLs_svhn = compute_LL(model, svhn[:n_data], k=k, inf_net=infnet_svhn)
    print ('SVHN', np.mean(LLs_svhn), np.std(LLs_svhn))
    print()

    plt.hist([LLs_train, LLs_valid, LLs_svhn])
    plt_path = save_dir+'LL_hist.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()
    

    with torch.no_grad():
        train_recon = model.forward(x=train_x[:10].cuda(), warmup=1.)['x_recon']
        valid_recon = model.forward(x=valid_x[:10].cuda(), warmup=1.)['x_recon']
        svhn_recon = model.forward(x=svhn[:10].cuda(), warmup=1., inf_net=infnet_svhn)['x_recon']
        sample_prior = model.sample_prior(z=z_prior)

    # svhn_outputs = model.forward(x=svhn[:10].cuda(), warmup=1.)
    # print (svhn_outputs['logws'])
    # fsafd

    vizualize(images_dir, load_step, train_real=train_x[:10], train_recon=train_recon,
                                valid_real=valid_x[:10], valid_recon=valid_recon,
                                svhn_real=svhn[:10], svhn_recon=svhn_recon,
                                prior_samps=sample_prior)





def compute_LL(model, data, k=10, inf_net=None): 

    batch_size = 50
    # LLs = []

    model.eval()
    with torch.no_grad():

        # for i in range(n_batches):
        i  = 0
        while i < len(data):


            # img_batch, question_batch = make_batch(image_dataset, question_dataset, batch_size, indexes=indexes)#, 
            # shuffled_image = img_batch[torch.randperm(img_batch.shape[0])]
            if i + batch_size >= len(data):
                batchsize_ = len(data) - i 
            else:
                batchsize_ = batch_size

            batch = data[i:i+batchsize_]

            logws = []
            # logws_S = []
            for j in range(k):

                if inf_net ==None:
                    outputs = model.forward(x=batch.cuda())
                else:
                    outputs = model.forward(x=batch.cuda(), inf_net=inf_net)
                     
                logws.append(outputs['logws'])

            logws = torch.stack(logws, 0) #[k,B]
            # print (logws.shape)
            # fdfa
            # if k ==1:
            #     LLs.append( torch.mean(logws).data.item())
            # else:

            # print (logmeanexp(logws).data.cpu().numpy().shape)

            if i ==0:
                LLs = logmeanexp(logws).data.cpu().numpy()
            else:
                LLs = np.concatenate([LLs,logmeanexp(logws).data.cpu().numpy() ], 0)

            # LLs.append( torch.mean(logmeanexp(logws)).data.item())
            # LLs += logmeanexp(logws).data.cpu().numpy()

            # print (len(LLs))
            # fsd

            i += batchsize_

            if i %100 == 0:
                print (i,len(LLs), np.mean(LLs), np.std(LLs))


    # LL_mean = np.mean(LLs)
    # LL_std = np.std(LLs)

    return LLs









# def train_inf_net(inf_net, vae, train_x, train_y, valid_x, valid_y, 
#                 save_dir, params_dir, images_dir,
#                 batch_size, 
#                 max_steps, display_step, save_steps, viz_steps, trainingplot_steps):

#     # random.seed( 0 )
    
#     all_dict = {}

#     list_recorded_values = [ 
#             'all_steps', 'all_warmup',
#             'all_train_elbos', 'all_valid_elbos', 'all_svhn_elbos', 
#     ]

#     for label in list_recorded_values:
#         all_dict[label] = []


#     train = torch.utils.data.TensorDataset(train_x, train_y)
#     train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
#     # optimizer = optim.Adam(model.parameters(), lr=.0001)

#     load_step = 0
#     max_beta = 1.
#     warmup_steps = 20000. 
    
#     limit_numb = 10


#     step = 0
#     # for epoch in range(1, epochs + 1):
#     while step < max_steps:

#         for batch_idx, (batch, target) in enumerate(train_loader):

#             # warmup = min((step+load_step) / float(warmup_steps), max_beta)
#             warmup = 1.

#             inf_net.optimizer_x.zero_grad()
#             outputs = vae.forward(x=batch.cuda(), warmup=1., inf_net=inf_net) 
#             loss = -outputs['welbo']
#             loss.backward()
#             inf_net.optimizer_x.step()
#             inf_net.scheduler_x.step()


#             step +=1

#             if step%display_step==0:
#                 print (
#                     'S:{:5d}'.format(step),
#                     'welbo:{:.4f}'.format(outputs['welbo'].data.item()),
#                     'elbo:{:.4f}'.format(outputs['elbo'].data.item()),
#                     'lpx:{:.4f}'.format(outputs['logpx'].data.item()),
#                     'lpz:{:.4f}'.format(outputs['logpz'].data.item()),
#                     'lqz:{:.4f}'.format(outputs['logqz'].data.item()),
#                     'warmup:{:.4f}'.format(warmup),
#                     )

#                 vae.eval()
#                 with torch.no_grad():
#                     valid_outputs = vae.forward(x=valid_x[:50].cuda(), warmup=1., inf_net=inf_net)
#                     svhn_outputs = vae.forward(x=svhn[:50].cuda(), warmup=1., inf_net=inf_net)
#                 vae.train()


#                 if step > limit_numb:

#                     all_dict['all_steps'].append(step+load_step)
#                     all_dict['all_warmup'].append(warmup)
#                     all_dict['all_train_elbos'].append(outputs['elbo'].data.item())
#                     all_dict['all_valid_elbos'].append(valid_outputs['elbo'].data.item())
#                     all_dict['all_svhn_elbos'].append(svhn_outputs['elbo'].data.item())

        

#             if step % trainingplot_steps==0 and step > 0 and len(all_dict['all_train_elbos']) > 2:
#                 plot_curves(vae, save_dir, all_dict)

#             if step % save_steps==0 and step > 0:
#                 inf_net.save_params_v3(save_dir=params_dir, step=step+load_step)

#             if step % viz_steps==0 and step > 0: 

#                 vae.eval()
#                 with torch.no_grad():
#                     train_recon = vae.forward(x=train_x[:10].cuda(), warmup=1., inf_net=inf_net)['x_recon']
#                     valid_recon = vae.forward(x=valid_x[:10].cuda(), warmup=1., inf_net=inf_net)['x_recon']
#                     svhn_recon = vae.forward(x=svhn[:10].cuda(), warmup=1., inf_net=inf_net)['x_recon']
#                     sample_prior = vae.sample_prior(z=z_prior)
#                 vae.train()

#                 vizualize(images_dir, step, train_real=train_x[:10], train_recon=train_recon,
#                                             valid_real=valid_x[:10], valid_recon=valid_recon,
#                                             svhn_real=svhn[:10], svhn_recon=svhn_recon,
#                                             prior_samps=sample_prior)










def plot_curves(self, save_dir, all_dict):


    def make_curve_subplot(self, rows, cols, row, col, steps, values_list, label_list, ylabel, show_ticks, colspan, rowspan, set_xlabel=False, color_list=None, linestyle_list=None, title=0):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=colspan, rowspan=rowspan)

        for i in range(len(values_list)):
            if color_list is not None:
                # ax.plot(steps, smooth_list(values_list[i]), label=label_list[i], c=color_list[i], linestyle=linestyle_list[i])
                ax.plot(steps, values_list[i], label=label_list[i], c=color_list[i], linestyle=linestyle_list[i])
            else:
                # ax.plot(steps, smooth_list(values_list[i]), label=label_list[i])
                ax.plot(steps, values_list[i], label=label_list[i])

        if len(label_list) > 1:
            ax.legend(prop={'size':5}, loc=2) #upper left
        
        if not show_ticks:
            ax.tick_params(axis='x', colors='white')

        ax.set_ylabel(ylabel, size=6, family='serif')
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=.3)

        if set_xlabel:
            ax.set_xlabel('Steps', size=6, family='serif')

        if title:
            ax.set_title('Exp:'+self.exp_name   +  r'     $D_z$:'+str(self.z_size), 
                            size=6, family='serif')


    rows = 6
    cols = 2
    text_col_width = cols
    fig = plt.figure(figsize=(4+cols,4+rows), facecolor='white', dpi=150)
    
    col =0
    row = 0
    steps = all_dict['all_steps']
    

    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_train_bpd'], all_dict['all_valid_bpd'], all_dict['all_svhn_bpd']], 
        label_list=['Train: {:.2f}'.format( all_dict['all_train_bpd'][-1]), 
                    'Valid: {:.2f}'.format( all_dict['all_valid_bpd'][-1]),
                    'SVHN: {:.2f}'.format( all_dict['all_svhn_bpd'][-1]),
                    ], 
        ylabel='BPD', show_ticks=True, colspan=text_col_width, rowspan=1, title=1)
    row+=1


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_train_elbos'], all_dict['all_valid_elbos'], all_dict['all_svhn_elbos']], 
        label_list=['Train: {:.2f}'.format( all_dict['all_train_elbos'][-1]), 
                    'Valid: {:.2f}'.format( all_dict['all_valid_elbos'][-1]),
                    'SVHN: {:.2f}'.format( all_dict['all_svhn_elbos'][-1]),
                    ], 
        ylabel='ELBO', show_ticks=True, colspan=text_col_width, rowspan=1)
    row+=1


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_train_logpx'], all_dict['all_valid_logpx'], all_dict['all_svhn_logpx']], 
        label_list=['Train: {:.2f}'.format( all_dict['all_train_logpx'][-1]),
                    'Valid: {:.2f}'.format( all_dict['all_valid_logpx'][-1]),
                    'SVHN: {:.2f}'.format( all_dict['all_svhn_logpx'][-1]),], 
        ylabel='logp(x|z)', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1

    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_train_logpz'], all_dict['all_valid_logpz'], all_dict['all_svhn_logpz']], 
        label_list=['Train: {:.2f}'.format( all_dict['all_train_logpz'][-1]),
                    'Valid: {:.2f}'.format( all_dict['all_valid_logpz'][-1]),
                    'SVHN: {:.2f}'.format( all_dict['all_svhn_logpz'][-1]),], 
        ylabel='logp(z)', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_train_logqz'], all_dict['all_valid_logqz'], all_dict['all_svhn_logqz']], 
        label_list=['Train: {:.2f}'.format( all_dict['all_train_logqz'][-1]),
                    'Valid: {:.2f}'.format( all_dict['all_valid_logqz'][-1]),
                    'SVHN: {:.2f}'.format( all_dict['all_svhn_logqz'][-1]),], 
        ylabel='logq(z|x)', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_warmup']],# all_dict['all_marginf']], 
        label_list=[r'$Warmup$'], ylabel='Hyperparameters', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1


    plt_path = save_dir+'curves_plot.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()


















def vizualize(images_dir, step, train_real, train_recon,
                        valid_real, valid_recon,
                        svhn_real, svhn_recon,
                        prior_samps):



    def make_image_subplot(rows, cols, row, col, image, text):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False)
       
        image = image.data.cpu().numpy() * 256.
        image = np.rollaxis(image, 1, 0)
        image = np.rollaxis(image, 2, 1)# [112,112,3]
        image = np.uint8(image)
        ax.imshow(image) #, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)

    rows = 6
    cols= 7
    text_col_width = 1

    fig = plt.figure(figsize=(7+cols,2+rows), facecolor='white', dpi=150)

    fig.text(.2, .9, 'Training Set', family='serif', size=16)
    fig.text(.4, .9, 'Validation Set', family='serif', size=16)
    fig.text(.59, .9, 'Prior Samples', family='serif', size=16)
    fig.text(.78, .9, 'SVHN', family='serif', size=16)

    for i in range(rows):
        make_image_subplot(rows, cols, row=i, col=0, image=train_real[i], text='')
        make_image_subplot(rows, cols, row=i, col=1, image=train_recon[i], text='')

        make_image_subplot(rows, cols, row=i, col=2, image=valid_real[i], text='')
        make_image_subplot(rows, cols, row=i, col=3, image=valid_recon[i], text='')

        make_image_subplot(rows, cols, row=i, col=4, image=prior_samps[i], text='')

        make_image_subplot(rows, cols, row=i, col=5, image=svhn_real[i], text='')
        make_image_subplot(rows, cols, row=i, col=6, image=svhn_recon[i], text='')

    # plt.tight_layout()
    plt_path = images_dir + 'img'+str(step) +'.png'
    plt.savefig(plt_path)
    print ('saved viz',plt_path)
    plt.close(fig)

























if __name__ == "__main__":

    parser = argparse.ArgumentParser()



    parser.add_argument('--data_dir', type=str, default=home+'/Documents/')

    
    parser.add_argument('--save_to_dir', type=str, default=home+'/Documents/')
    parser.add_argument('--exp_name', default='test', type=str)
    # parser.add_argument('--data_dir', type=str, required=True)
    


    # parser.add_argument('--feature_dim', default='3,112,112')
    # parser.add_argument('--num_train_samples', default=None, type=int)
    # parser.add_argument('--num_val_samples', default=None, type=int)

    parser.add_argument('--which_gpu', default='0', type=str)

    parser.add_argument('--input_size', default=32, type=int)
    parser.add_argument('--x_enc_size', default=500, type=int)
    # parser.add_argument('--y_enc_size', default=200, type=int)
    parser.add_argument('--z_size', default=50, type=int)

    # parser.add_argument('--w_logpx', default=.05, type=float)
    # parser.add_argument('--w_logpy', default=1000., type=float)
    # parser.add_argument('--w_logqy', default=1., type=float)
    # parser.add_argument('--max_beta', default=1., type=float)

    parser.add_argument('--batch_size', default=20, type=int)
    # parser.add_argument('--embed_size', default=100, type=int)

    # parser.add_argument('--train_which_model', default='pxy', choices=['pxy', 'px', 'py'])

    # parser.add_argument('--single', default=0, type=int)
    # parser.add_argument('--singlev2', default=0, type=int)
    # parser.add_argument('--multi', default=0, type=int)

    # parser.add_argument('--flow_int', default=0, type=int)
    # parser.add_argument('--joint_inf', default=0, type=int)


    # parser.add_argument('--just_classifier', default=0, type=int)
    # parser.add_argument('--train_classifier', default=0, type=int)
    # parser.add_argument('--classifier_load_params_dir', default='', type=str)
    # parser.add_argument('--classifier_load_step', default=0, type=int)

    parser.add_argument('--params_load_dir', default='')
    parser.add_argument('--model_load_step', default=0, type=int)

    # parser.add_argument('--quick_check', default=0, type=int)
    # parser.add_argument('--quick_check_data', default='', type=str)


    parser.add_argument('--display_step', default=500, type=int)
    parser.add_argument('--trainingplot_steps', default=5000, type=int)
    parser.add_argument('--viz_steps', default=5000, type=int)
    parser.add_argument('--start_storing_data_step', default=2001, type=int)
    parser.add_argument('--save_params_step', default=50000, type=int)
    parser.add_argument('--max_steps', default=400000, type=int)



    args = parser.parse_args()
    args_dict = vars(args) #convert to dict

    os.environ['CUDA_VISIBLE_DEVICES'] = args.which_gpu #  '0' #'1' #


    print (args.exp_name, '\n')



    exp_dir = args.save_to_dir + args.exp_name + '/'
    params_dir = exp_dir + 'params/'
    images_dir = exp_dir + 'images/'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print ('Made dir', exp_dir) 
    # else:
    #     print (save_dir, 'already exists')
    #     sdfdafs

    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
        print ('Made dir', params_dir) 

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print ('Made dir', images_dir) 

















    






    print ('Loading CIFAR')
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


    # View cifar images
    # test_x = test_x.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    # # print 'fffff'
    # #Visualizing CIFAR 10
    # fig, axes1 = plt.subplots(5,5,figsize=(3,3))
    # for j in range(5):
    #     for k in range(5):
    #         i = np.random.choice(range(len(test_x)))
    #         axes1[j][k].set_axis_off()
    #         axes1[j][k].imshow(test_x[i:i+1][0])
    # # plt.show()
    # plt.savefig(home+'/Documents/tmp/img'+str(i)+'.png')
    # fasdfa


    train_x = train_x / 256.
    test_x = test_x / 256.

    train_x = torch.from_numpy(train_x).float()
    test_x = torch.from_numpy(test_x).float()
    train_y = torch.from_numpy(train_y)
    test_y = torch.from_numpy(np.array(test_y))

    train_x = train_x.view(-1, 3, 32, 32)
    test_x = test_x.view(-1, 3, 32, 32)

    train_x = torch.clamp(train_x, min=1e-5, max=1-1e-5)
    test_x = torch.clamp(test_x, min=1e-5, max=1-1e-5)


    # print (torch.min(train_x))
    # print (torch.max(train_x))

    # fasdf

    print (train_x.shape)
    print (test_x.shape)
    print ()

    # fafdsa






    print ('SVHN')
    import scipy.io
    mat = scipy.io.loadmat(home + '/Documents/SVHN/test_32x32.mat')

    # print (mat['X'].shape)
    # print (mat['y'].shape)

    svhn = mat['X'].transpose(3,2,0,1)
    svhn = torch.from_numpy(svhn).float()
    svhn = svhn / 256.
    svhn = torch.clamp(svhn, min=1e-5, max=1-1e-5)
    print (svhn.shape)
    # print (torch.min(svhn))
    # print (torch.max(svhn))
    # fsdfa





    print ('\nInit VAE')
    vae = VAE(args_dict)
    vae.cuda()
    if args.model_load_step>0:
        vae.load_params_v3(save_dir=args.params_load_dir, step=args.model_load_step)
    # print(vae)
    # fdsa
    print ('VAE Initilized\n')



    print ('\nInit inf net svhn')
    infnet_svhn = Inference_Net(args_dict)
    infnet_svhn.cuda()
    if args.model_load_step>0:
        infnet_svhn.load_params_v3(save_dir=args.params_load_dir, step=args.model_load_step, name='svhn')
    # print(vae)
    # fdsa
    print ('inf net Initilized\n')


    print ('\nInit inf net validaiton')
    infnet_valid = Inference_Net(args_dict)
    infnet_valid.cuda()
    if args.model_load_step>0:
        infnet_valid.load_params_v3(save_dir=args.params_load_dir, step=args.model_load_step, name='valid')
    # print(vae)
    # fdsa
    print ('inf net Initilized\n')



    # if train_:
    
    # train2(self=model, max_steps=max_steps, load_step=args.model_load_step,
    #         image_dir=image_dir, train_indexes=list(range(1,180001)), val_indexes=list(range(180001, n_images)), attr_dict=attr_dict, word_to_idx=word_to_idx,
    #         save_dir=exp_dir, params_dir=params_dir, images_dir=images_dir, batch_size=args.batch_size,
    #         display_step=display_step, save_steps=save_steps, trainingplot_steps=trainingplot_steps, viz_steps=viz_steps,
    #         classifier=classifier, start_storing_data_step=args.start_storing_data_step) #, k=args.k)

    # path_to_load_variables = ''
    # path_to_save_variables = ''
    # epochs = 200

    z_prior = torch.FloatTensor(10, args.z_size).normal_().cuda() 

    batch_size = 50
    display_step = 500
    save_steps = 50000
    max_steps = 300000
    viz_steps = 5000
    trainingplot_steps = 5000

    load_step = args.model_load_step

    train_ = 1
    train_svhn_inf_net = 0
    eval_ = 0

    if train_:
        print ('Training')
        train(model=vae, train_x=train_x, train_y=train_y, valid_x=test_x, valid_y=test_y, 
                    save_dir=exp_dir, params_dir=params_dir, images_dir=images_dir,
                    batch_size=batch_size, 
                    max_steps=max_steps, display_step=display_step, save_steps=save_steps, viz_steps=viz_steps,
                    trainingplot_steps=trainingplot_steps, load_step=load_step)


    # elif train_svhn_inf_net:

    #     print ('\nInit inf net')
    #     inf_net_svhn = Inference_Net(args_dict)
    #     inf_net_svhn.cuda()
    #     # if args.model_load_step>0:
    #     #     vae.load_params_v3(save_dir=args.params_load_dir, step=args.model_load_step)
    #     # print(vae)
    #     # fdsa
    #     print ('inf net Initilized\n')


    #     print ('Training svhn inf net')
    #     train_inf_net(inf_net=inf_net_svhn,  vae=vae, train_x=svhn, train_y=torch.ones(len(svhn)), valid_x=svhn, valid_y=torch.ones(len(svhn)), 
    #                 save_dir=exp_dir, params_dir=params_dir, images_dir=images_dir,
    #                 batch_size=batch_size, 
    #                 max_steps=max_steps, display_step=display_step, save_steps=save_steps, viz_steps=viz_steps,
    #                 trainingplot_steps=trainingplot_steps)

    elif eval_:
        print ('Eval')
        eval_model(model=vae, train_x=train_x, train_y=train_y, valid_x=test_x, valid_y=test_y, 
                    save_dir=exp_dir, params_dir=params_dir, images_dir=images_dir,
                    batch_size=batch_size, 
                    max_steps=max_steps, display_step=display_step, save_steps=save_steps, viz_steps=viz_steps,
                    trainingplot_steps=trainingplot_steps, load_step=args.model_load_step)

    # model.save_params_v3(save_dir=params_dir, step=max_steps+args.model_load_step)

    print ('Done.')









# Example:

#python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 0 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --flow 1  --exp_name flow_qy --quick_check 1 --model_load_step 100000

# python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --z_size 20 --exp_name gaus_qy

# python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --z_size 20 --quick_check 1 

# python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --quick_check 1 --flow 1 --z_size 20

# python3 train_jointVAE_v21.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 20 --w_logpx .01 --joint_inf 0 --quick_check 1
















