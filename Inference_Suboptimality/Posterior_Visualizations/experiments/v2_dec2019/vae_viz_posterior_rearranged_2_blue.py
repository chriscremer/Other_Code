


# include optimal ffg





import sys, os
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '../../models')
sys.path.insert(0, '../../models/utils')

import time
import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# from ais2 import test_ais

from pytorch_vae_v5 import VAE

from approx_posteriors_v5 import standard
from approx_posteriors_v5 import flow1
from approx_posteriors_v5 import aux_nf
from approx_posteriors_v5 import hnf

import argparse
from plotting_functions import plot_isocontours
from plotting_functions import plot_isocontours2
from plotting_functions import plot_isocontours2_exp
from plotting_functions import plot_isocontoursNoExp

from plotting_functions import plot_isocontours2_exp_norm
from plotting_functions import plot_scatter

from plotting_functions import plot_kde

from optimize_local import optimize_local_expressive_only_sample
from optimize_local import optimize_local_expressive_only_sample_2

# from optimize_local import optimize_local_gaussian_mean_logvar


from optimize_local2 import optimize_local_gaussian_mean_logvar
from optimize_local2 import optimize_local_gaussian_mean_logvar2
from optimize_local2 import optimize_local_flow1




from utils import lognormal4 


from plotting_functions2 import plot_isocontours_new




if __name__ == "__main__":

    print ('Loading data')
    with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
        mnist_data = pickle.load(f, encoding='latin1')
    train_x = mnist_data[0][0]
    valid_x = mnist_data[1][0]
    test_x = mnist_data[2][0]
    train_x = np.concatenate([train_x, valid_x], axis=0)
    # print (train_x.shape)

    # #For debug purposes
    # train_x = train_x[:1000]
    # test_x = test_x[:500]

    print (train_x.shape)
    print (test_x.shape)





    x_size = 784
    z_size = 2

    l_size = 100
    f_size = 30

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m",'--model', default="standard")
    # parser.add_argument("-e",'--epoch')
    # parser.add_argument("-s",'--set')
    # args = parser.parse_args()


    directory = home+'/Documents/tmp/2D_models'

    # if args.model == 'standard':


    which_model = 'standard'


    if which_model == 'standard':
        print(which_model)
        this_dir = directory+'/standard'    
        hyper_config = { 
                        'x_size': x_size,
                        'z_size': z_size,
                        'act_func': F.tanh,# F.relu,
                        'encoder_arch': [[x_size,l_size],[l_size,l_size],[l_size,z_size*2]],
                        'decoder_arch': [[z_size,l_size],[l_size,l_size],[l_size,x_size]],
                        'q_dist': standard,#hnf,#aux_nf,#flow1,#,
                    }

        model = VAE(hyper_config)

        path_to_save_variables=this_dir+'/params_standard' +'_'






    # elif args.model == 'flow1':

    #     this_dir = directory+'/flow1'
    #     # if not os.path.exists(this_dir):
    #     #     os.makedirs(this_dir)
    #     #     print ('Made directory:'+this_dir)

    #     # experiment_log = this_dir+'/log.txt'

    #     # with open(experiment_log, "a") as myfile:
    #     #     myfile.write("Flow1" +'\n')

    #     print('Init '+args.model)
    #     hyper_config = { 
    #                     'x_size': x_size,
    #                     'z_size': z_size,
    #                     'act_func': F.tanh,# F.relu,
    #                     'encoder_arch': [[x_size,l_size],[l_size,l_size],[l_size,z_size*2]],
    #                     'decoder_arch': [[z_size,l_size],[l_size,l_size],[l_size,x_size]],
    #                     'q_dist': flow1,#hnf,#aux_nf,#,#standard,#, #
    #                     'n_flows': 2,
    #                     'flow_hidden_size': f_size
    #                 }

    #     model = VAE(hyper_config)



    # elif args.model == 'aux_nf':




    # elif which_model == 'aux':
    #     print(which_model)
    #     this_dir = directory+'/aux_nf'

    #     hyper_config = { 
    #                     'x_size': x_size,
    #                     'z_size': z_size,
    #                     'act_func': F.tanh,# F.relu,
    #                     'encoder_arch': [[x_size,l_size],[l_size,l_size],[l_size,z_size*2]],
    #                     'decoder_arch': [[z_size,l_size],[l_size,l_size],[l_size,x_size]],
    #                     'q_dist': aux_nf,#aux_nf,#flow1,#standard,#, #, #, #,#, #,# ,
    #                     'n_flows': 2,
    #                     'qv_arch': [[x_size,l_size],[l_size,l_size],[l_size,z_size*2]],
    #                     'qz_arch': [[x_size+z_size,l_size],[l_size,l_size],[l_size,z_size*2]],
    #                     'rv_arch': [[x_size+z_size,l_size],[l_size,l_size],[l_size,z_size*2]],
    #                     'flow_hidden_size': f_size
    #                 }

    #     model = VAE(hyper_config)
    #     path_to_save_variables=this_dir+'/params_aux_nf_'





    # elif args.model == 'hnf':

    #     this_dir = directory+'/hnf'
    #     # if not os.path.exists(this_dir):
    #     #     os.makedirs(this_dir)
    #     #     print ('Made directory:'+this_dir)

    #     # experiment_log = this_dir+'/log.txt'

    #     # with open(experiment_log, "a") as myfile:
    #     #     myfile.write("hnf" +'\n')

    #     print('Init '+args.model)
    #     hyper_config = { 
    #                     'x_size': x_size,
    #                     'z_size': z_size,
    #                     'act_func': F.tanh,# F.relu,
    #                     'encoder_arch': [[x_size,l_size],[l_size,l_size],[l_size,z_size*2]],
    #                     'decoder_arch': [[z_size,l_size],[l_size,l_size],[l_size,x_size]],
    #                     'q_dist': hnf,#aux_nf,#flow1,#standard,#, #, #, #,#, #,# ,
    #                     'n_flows': 2,
    #                     'qv_arch': [[x_size,l_size],[l_size,l_size],[l_size,z_size*2]],
    #                     'qz_arch': [[x_size+z_size,l_size],[l_size,l_size],[l_size,z_size*2]],
    #                     'rv_arch': [[x_size+z_size,l_size],[l_size,l_size],[l_size,z_size*2]],
    #                     'flow_hidden_size': f_size
    #                 }

    #     model = VAE(hyper_config)

    # else:
    #     print ('What')
    #     fadas



    # path_to_load_variables=''
    # path_to_save_variables=home+'/Documents/tmp/pytorch_bvae'+str(i)+'.pt'
    # path_to_load_variables=home+'/Documents/tmp/pytorch_bvae'+str(i)+'.pt'
    # path_to_save_variables=''

    # model.load_state_dict(torch.load(path_to_save_variables, lambda storage, loc: storage)) 
    # print 'loaded variables ' + path_to_save_variables




    # path_to_save_variables=this_dir+'/params_'+args.model +'_'



    if torch.cuda.is_available():
        model.cuda()

    epoch_to_load = 101 #21 # 3 #3000

    this_ckt_file = path_to_save_variables + str(epoch_to_load) + '.pt'
    # this_ckt_file = path_to_save_variables + str(args.epoch) + '.pt'

    model.load_params(path_to_load_variables=this_ckt_file)

    print ('model loaded')
    

    

    # ffg_samps = [0,1,2,3]
    # ffg_samps = [5,6,7,8]


    # # used for standard
    # ffg_samps = [6,3,2,5]
    # ffg_samps = [0,1,2,3,4,5,6,7,8]
    # ffg_samps = list(range(9,14))
    # ffg_samps = list(range(15,24))
    # ffg_samps = list(range(40,50))
        # 33, 26
    # ffg_samps = [26,33]
    # ffg_samps = [5]
    # ffg_samps = [6]


    # ffg_samps = list(range(20,40))
    # ffg_samps = list(range(70,80))
    # ffg_samps = list(range(60,70))

    # ffg_samps = list(range(20,30))

    # ffg_samps = [70]
    # ffg_samps = [63]

    # ffg_samps = [24]  # this one was decent but coulnt get flow working on it 

    # ffg_samps = list(range(175,183))

    ffg_samps = [179] 



    rows = 4
    cols = len(ffg_samps) +1 #for annotation

    legend= False

    fig = plt.figure(figsize=(2+cols,rows), facecolor='white')

    # lim_val = .24
    # xlimits=[-lim_val, lim_val]
    # ylimits=[-lim_val, lim_val]

    x_text = .05
    y_text = .4

    #annotate
    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
    # ax.annotate('True Posterior', xytext=(.1, .5), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Blue', size='large')
    ax.annotate('   True\nPosterior', xytext=(x_text, y_text), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Black')#, size='large')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')


    ax = plt.subplot2grid((rows,cols), (1,0), frameon=False)
    # ax.annotate('Amortized\nFFG', xytext=(.1, .5), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Green', size='large')
    ax.annotate('        Amortized\nFactorized Gaussian', xytext=(x_text, y_text), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Black')#, size='large')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')


    ax = plt.subplot2grid((rows,cols), (2,0), frameon=False)
    # ax.annotate('Optimal\nFFG', xytext=(.1, .5), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Purple', size='large')
    ax.annotate('         Optimized\nFactorized Gaussian', xytext=(x_text, y_text), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Black')#, size='large')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')


    ax = plt.subplot2grid((rows,cols), (3,0), frameon=False)
    # ax.annotate('Optimal\nFlow', xytext=(.1, .5), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Red', size='large')
    ax.annotate('Optimized Flow', xytext=(x_text, y_text), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Black')#, size='large')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')



    for samp_i in range(len(ffg_samps)):



        #Get a sample
        # samp = train_x[ffg_samps[samp_i]]
        samp = test_x[ffg_samps[samp_i]]
        # samp = test_x[samp_i]

        # if args.set == 'train':
            # samp = train_x[np.random.randint(len(train_x))]
        # elif args.set == 'test':
            # samp = test_x[np.random.randint(len(test_x))]

        # print samp.shape
        # col = 0
        row = 0

        



        #Plot sample
        # ax = plt.subplot2grid((rows,cols), (row,samp_i), frameon=False)
        # ax.imshow(samp.reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        # ax.set_yticks([])
        # ax.set_xticks([])
        # if samp_i==0:  ax.annotate('Sample', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

        samp_torch = Variable(torch.from_numpy(np.array([samp]))).type(model.dtype)
        n_samps = 1000
        z = model.sample_q(x=samp_torch, k=n_samps)# [P,B,Z]
        z = z.view(-1,z_size)
        z = z.data.cpu().numpy()

        # print (z)

        center_val_x = np.mean(z, axis=0)[0] #z[0][0]
        center_val_y = np.mean(z, axis=0)[1] #z[0][0]
        # center_val_x = z[0][0]
        # center_val_y = z[0][1]


        # if samp_i == 0:
        #     lim_val = .13
        #     xlimits=[center_val_x-lim_val, center_val_x+lim_val]
        #     ylimits=[center_val_y-lim_val, center_val_y+lim_val]
        # elif samp_i == 1:
        #     lim_val = .24
        #     xlimits=[center_val_x-lim_val, center_val_x+lim_val]
        #     ylimits=[center_val_y-lim_val, center_val_y+lim_val]
        # elif samp_i == 2:
        #     xlimits=[center_val_x-.24, center_val_x+.06]
        #     ylimits=[center_val_y-.15, center_val_y+.15]
        # elif samp_i == 3:
        #     xlimits=[center_val_x-.25, center_val_x+.05]
        #     ylimits=[center_val_y-.15, center_val_y+.15]

        # xlimits = [-1.7,1.7]
        # ylimits = [-1.7,1.7]

        # xlimits = [-2,2]
        # ylimits = [-2,2]



        # zzzz = .25 #55
        # xlimits=[center_val_x-zzzz, center_val_x+zzzz]
        # ylimits=[center_val_y-zzzz, center_val_y+zzzz]

        if samp_i == 0:
            zzzz = .15
            xlimits=[center_val_x-zzzz, center_val_x+zzzz]
            ylimits=[center_val_y-.1, center_val_y+.33]

        # if samp_i == 1:
        #     xxxx = .2
        #     # yyyy = .55
        #     xlimits=[center_val_x-xxxx, center_val_x+xxxx]
        #     ylimits=[center_val_y-.9, center_val_y]



        # #Plot prior
        # col +=1
        # ax = plt.subplot2grid((rows,cols), (samp_i,col), frameon=False)
        # func = lambda zs: lognormal4(torch.Tensor(zs), torch.zeros(2), torch.zeros(2))
        # plot_isocontours(ax, func, cmap='Blues')
        # if samp_i==0:  ax.annotate('Prior p(z)', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')



        #Scatter plot of q
        # col +=1

        ax = plt.subplot2grid((rows,cols), (row,samp_i+1), frameon=False)

        # Ws, logpW, logqW = model.sample_W()  #_ , [1], [1]   
        # func = lambda zs: lognormal4(torch.Tensor(zs), torch.squeeze(mean.data), torch.squeeze(logvar.data))
        # plot_isocontours(ax, func, cmap='Reds',xlimits=xlimits,ylimits=ylimits)
        func = lambda zs: model.logposterior_func(samp_torch,zs)

        # levels = np.linspace(10., 120., 10)
        # print (levels)
        # levels = []
        # levels = np.linspace(0., 170., 10)
        levels = np.linspace(-20., 170., 10)
        # levels[0]
        # print (levels)
        # fasd
        legend = False

        # plot_isocontours2_exp_norm(ax, func, cmap='Greens', legend=legend,xlimits=xlimits,ylimits=ylimits)
        _ = plot_isocontours_new(ax, func, cmap='Greens', legend=legend, xlimits=xlimits,ylimits=ylimits, levels=levels)












        plot_amortized=1
        if plot_amortized:



            #Plot prob
            row +=1
            ax = plt.subplot2grid((rows,cols), (row, samp_i+1), frameon=False)

            func = lambda zs: model.logposterior_func(samp_torch,zs)

            # plot_isocontours2_exp_norm(ax, func, cmap='Greens', legend=legend,xlimits=xlimits,ylimits=ylimits,alpha=.2)
            _ = plot_isocontours_new(ax, func, cmap='Greens', legend=False,xlimits=xlimits,ylimits=ylimits,alpha=1., levels=levels)

            # plot_isocontours2_exp_norm(ax, func, cmap='Blues', legend=legend,xlimits=xlimits,ylimits=ylimits,alpha=1.)

            # plot_scatter(ax, samps=z ,xlimits=xlimits,ylimits=ylimits)
            # plot_kde(ax,samps=z,xlimits=xlimits,ylimits=ylimits,cmap='Blues')
            # plot_kde(ax,samps=z,xlimits=xlimits,ylimits=ylimits,cmap='Greens') 'Greys'

            mean, logvar = model.q_dist.get_mean_logvar(samp_torch)
            func = lambda zs: lognormal4(torch.Tensor(zs), torch.squeeze(mean.data.cpu()), torch.squeeze(logvar.data.cpu()))
            # plot_isocontours(ax, func, cmap='Greens',xlimits=xlimits,ylimits=ylimits)


            # plot_isocontours(ax, func, cmap='Blues',xlimits=xlimits,ylimits=ylimits)

            # plot_isocontours2_exp_norm(ax, func, cmap='Blues',xlimits=xlimits,ylimits=ylimits)
            # levels = []
            _ = plot_isocontours_new(ax, func, cmap='Blues',xlimits=xlimits,ylimits=ylimits, legend=False, levels=levels)













        plot_optimal= 1
        if plot_optimal: 


            #plot local optima guassian

            row +=1
            ax = plt.subplot2grid((rows,cols), (row, samp_i+1), frameon=False)
            func = lambda zs: model.logposterior_func(samp_torch,zs)
            plot_isocontours_new(ax, func, cmap='Greens', legend=False,xlimits=xlimits,ylimits=ylimits,alpha=1., levels=levels)
            # plot_isocontours2_exp_norm(ax, func, cmap='Blues', legend=legend,xlimits=xlimits,ylimits=ylimits,alpha=1.)


            # x = train_x[i]
            x = samp
            x = Variable(torch.from_numpy(x)).type(model.dtype)
            x = x.view(1,784)

            # save_to = this_dir+'/local_params'+str(samp_i)+'.pt'
            # load_from = save_to

            logposterior = lambda aa: model.logposterior_func2(x=x,z=aa)
            print ('optimiznig local ffg', samp_i)


            mean, logvar, z = optimize_local_gaussian_mean_logvar(logposterior, model, x)
            # mean, logvar, z = optimize_local_gaussian_mean_logvar2(logposterior, model, x)

            

            func = lambda zs: lognormal4(torch.Tensor(zs), torch.squeeze(mean.data.cpu()), torch.squeeze(logvar.data.cpu()))
            # plot_isocontours(ax, func, cmap='Purples',xlimits=xlimits,ylimits=ylimits)


            # plot_isocontours(ax, func, cmap='Blues',xlimits=xlimits,ylimits=ylimits)
            _ = plot_isocontours_new(ax, func, cmap='Blues',xlimits=xlimits,ylimits=ylimits, legend=False, levels=levels)



        
            # z = z.view(-1,z_size)
            # z = z.data.cpu().numpy()
            # plt.scatter(z[:,0],z[:,1], alpha=.2)


            # plot_kde(ax,samps=z,xlimits=xlimits,ylimits=ylimits,cmap='Reds')







        plot_optimal_2= 1
        if plot_optimal_2: 



            #plot local optima

            row +=1
            ax = plt.subplot2grid((rows,cols), (row, samp_i+1), frameon=False)
            func = lambda zs: model.logposterior_func(samp_torch,zs)
            # plot_isocontours2_exp_norm(ax, func, cmap='Greens', legend=legend,xlimits=xlimits,ylimits=ylimits,alpha=1.)
            # plot_isocontours2_exp_norm(ax, func, cmap='Blues', legend=legend,xlimits=xlimits,ylimits=ylimits,alpha=1.)

            plot_isocontours_new(ax, func, cmap='Greens', legend=False,xlimits=xlimits,ylimits=ylimits,alpha=1., levels=levels)


            # x = train_x[i]
            x = samp
            x = Variable(torch.from_numpy(x)).type(model.dtype)
            x = x.view(1,784)

            save_to = this_dir+'/local_params'+str(samp_i)+'.pt'
            load_from = save_to

            logposterior = lambda aa: model.logposterior_func2(x=x,z=aa)
            print ('optimiznig local flow', samp_i)
            # z = optimize_local_expressive_only_sample(logposterior, model, x, save_to=save_to, load_from=load_from)

            # z = optimize_local_expressive_only_sample_2(logposterior, model, x)

            z = optimize_local_expressive_only_sample(logposterior, model, x)

            z = z.view(-1,z_size)
            z = z.data.cpu().numpy()
            # plot_kde(ax,samps=z,xlimits=xlimits,ylimits=ylimits,cmap='Reds')
            plot_kde(ax,samps=z,xlimits=xlimits,ylimits=ylimits,cmap='Blues')



            # flow = optimize_local_flow1(logposterior, model, x)
            # func = lambda zs: flow.logprob(torch.Tensor(zs).cuda())
            # _ = plot_isocontours_new(ax, func, cmap='Blues',xlimits=xlimits,ylimits=ylimits, legend=False, levels=[])









        # #Plot prob
        # col +=1
        # ax = plt.subplot2grid((rows,cols), (samp_i,col), frameon=False)
        # Ws, logpW, logqW = model.sample_W()  #_ , [1], [1]   
        # func = lambda zs: log_bernoulli(model.decode(Ws, Variable(torch.unsqueeze(zs,1))), Variable(torch.unsqueeze(samp,0)))+ Variable(torch.unsqueeze(lognormal4(torch.Tensor(zs), torch.zeros(2), torch.zeros(2)), 1))
        # plot_isocontours2_exp(ax, func, cmap='Greens', legend=legend)
        # if samp_i==0:  ax.annotate('p(z,x|W2)', xytext=(.1, 1.1), xy=(0, 1), textcoords='axes fraction')
        # func = lambda zs: lognormal4(torch.Tensor(zs), torch.zeros(2), torch.zeros(2))
        # plot_isocontours(ax, func, cmap='Blues', alpha=.3)
        # func = lambda zs: lognormal4(torch.Tensor(zs), torch.squeeze(mean.data), torch.squeeze(logvar.data))
        # plot_isocontours(ax, func, cmap='Reds')

        # #Plot prob
        # col +=1
        # ax = plt.subplot2grid((rows,cols), (samp_i,col), frameon=False)
        # Ws, logpW, logqW = model.sample_W()  #_ , [1], [1]   
        # func = lambda zs: log_bernoulli(model.decode(Ws, Variable(torch.unsqueeze(zs,1))), Variable(torch.unsqueeze(samp,0)))+ Variable(torch.unsqueeze(lognormal4(torch.Tensor(zs), torch.zeros(2), torch.zeros(2)), 1))
        # plot_isocontours2_exp(ax, func, cmap='Greens', legend=legend)
        # if samp_i==0:  ax.annotate('p(z,x|W3)', xytext=(.1, 1.1), xy=(0, 1), textcoords='axes fraction')
        # func = lambda zs: lognormal4(torch.Tensor(zs), torch.zeros(2), torch.zeros(2))
        # plot_isocontours(ax, func, cmap='Blues', alpha=.3)
        # func = lambda zs: lognormal4(torch.Tensor(zs), torch.squeeze(mean.data), torch.squeeze(logvar.data))
        # plot_isocontours(ax, func, cmap='Reds')

        # #Plot reconstruction
        # col +=1
        # ax = plt.subplot2grid((rows,cols), (samp_i,col), frameon=False)
        # x_hat = model.reconstruct(Variable(torch.unsqueeze(samp,0))).data[0]
        # ax.imshow(x_hat.numpy().reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        # ax.set_yticks([])
        # ax.set_xticks([])
        # if samp_i==0:  ax.annotate('Reconstruction', xytext=(.1, 1.1), xy=(0, 1), textcoords='axes fraction')


    # plt.show()
    # plt.savefig(home+'/Documents/tmp/2D_models/standard/first.png')
    # name_file = home+'/Documents/tmp/2D_models/'+ args.model +'/'+args.model+'_'+args.epoch+'_'+args.set+ '.png'

    name_file = home+'/Documents/tmp/2D_models/rearranged_real_2.png'
    plt.savefig(name_file)
    print ('Saved fig', name_file)
    
    # name_file = home+'/Documents/tmp/2D_models/rearranged_real.eps'
    # plt.savefig(name_file)
    # print ('Saved fig', name_file)
    
    name_file = home+'/Documents/tmp/2D_models/rearranged_real_2.pdf'
    plt.savefig(name_file)
    print ('Saved fig', name_file)




 # assert not torch.is_tensor(other)


# alpha=.2
# rows = len(posteriors)
# columns = len(models) +1 #+1 for posteriors

# fig = plt.figure(figsize=(6+columns,4+rows), facecolor='white')

# for p_i in range(len(posteriors)):

#     print '\nPosterior', p_i, posterior_names[p_i]

#     posterior = ttp.posterior_class(posteriors[p_i])
#     ax = plt.subplot2grid((rows,columns), (p_i,0), frameon=False)#, colspan=3)
#     plot_isocontours(ax, posterior.run_log_post, cmap='Blues')
#     if p_i == 0: ax.annotate('Posterior', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

#     for q_i in range(len(models)):

#         print model_names[q_i]
#         ax = plt.subplot2grid((rows,columns), (p_i,q_i+1), frameon=False)#, colspan=3)
#         model = models[q_i](posteriors[p_i])
#         # model.train(10000, save_to=home+'/Documents/tmp/vars.ckpt')
#         model.train(10000, save_to='')
#         samps = model.sample(1000)
#         plot_kde(ax, samps, cmap='Reds')
#         plot_isocontours(ax, posterior.run_log_post, cmap='Blues', alpha=alpha)
#         if p_i == 0: ax.annotate(model_names[q_i], xytext=(.38, 1.1), xy=(0, 1), textcoords='axes fraction')

# # plt.show()
# plt.savefig(home+'/Documents/tmp/plots.png')
# print 'saved'









    print ('Done.')



    #Reulls scores 
    #[-148.14850021362304, -146.89197433471679, -152.84082611083986]
    # qW_weights = [.000000001, .000001, .0001]


    # training elbo: 
    # 1: 116/300 Loss:153.175 logpx:-146.690 logpz:-1.800 logqz:4.666 logpW:-18463.871 logqW:616290.312
    # 2: 104/300 Loss:148.872 logpx:-142.097 logpz:-1.209 logqz:5.088 logpW:-26428.045 logqW:452337.312
    # 3: 94/300 Loss:134.635 logpx:-149.966 logpz:-0.951 logqz:4.851 logpW:-4931132.500 logqW:-260631.391


    # next: 2 figues, 1) the true posteriors 2) the image uncertainty. 









