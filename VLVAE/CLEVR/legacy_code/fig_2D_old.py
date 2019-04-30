





import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../VLVAE'))

from os.path import expanduser
home = expanduser("~")

import numpy as np
import math
import pickle
import random
import subprocess
import json
import random
import shutil
import time
import argparse
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical, Beta#, Normal
from torch.optim import lr_scheduler
torch.backends.cudnn.enabled = True

try:
    import h5py
except Warning:
    pass 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# from vr.data import ClevrDataset, ClevrDataLoader
from Clevr_data_loader import ClevrDataset, ClevrDataLoader
# from cyclegangenerator_bottleneck import *
from classifier import attribute_classifier, attribute_classifier_with_relations, classifier_of_prior
# from textRNN import TextRNN
# from priorRNN import priorRNN
# from distributions import Normal, Flow1#, IAF_flow
from preprocess_statements import *
# from qRNN import qRNN



import scipy.stats as st


from VLVAE_model import VLVAE

from distributions import lognormal



# def lognormal(x, mean, logvar):
#     # x: [P,B,Z]
#     # mean, logvar: [B,Z]
#     # return: [P,B]
#     # assert x.shape[1] == mean.shape[0]
#     return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(2))


# def logmeanexp(elbo):
#     # [P,B]
#     max_ = torch.max(elbo, 0)[0] #[B]
#     elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
#     return elbo


# from attributes
def get_sentence(list_of_word_idxs, newline_every=[]): #, answer):
    sentence =''
    list_of_word_idxs = list_of_word_idxs.cpu().numpy()#[0]
    for i in range(len(list_of_word_idxs)):
        word = question_idx_to_token[int(list_of_word_idxs[i])]
        sentence += ' ' + word
        if i in newline_every:
            sentence += '\n'

    return sentence


#returns list instead of string
def get_sentence2(list_of_word_idxs): #, answer):
    sentence = []
    list_of_word_idxs = list_of_word_idxs.cpu().numpy()#[0]
    for i in range(len(list_of_word_idxs)):
        word = question_idx_to_token[int(list_of_word_idxs[i])]
        sentence.append(word) 
    return sentence



def smooth_list(x, window_len=5, window='flat'):
    if len(x) < window_len:
        return x
    w = np.ones(window_len,'d') 
    y = np.convolve(w/ w.sum(), x, mode='same')
    return y




color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]

















def get_indexes(question_dataset, word=''):

    # print (len(train_indexes))
    # fsafd

    indexes = []
    for i in range(len(train_indexes)):
        sentence = get_sentence(question_dataset[train_indexes[i]])
        # print (sentence, word, word in sentence)
        if word in sentence:
            indexes.append(train_indexes[i])

        # if i > 10:
        #     fsadf
    return indexes


def make_batch2(image_dataset, question_dataset, batch_size, indexes): #, answer_dataset): #, image_idxs, first_question_idx):

    # if val:
    #     list_ = val_indexes
    # else:
    #     list_ = train_indexes
        
    # assert word != ''

    img_batch = []
    question_batch = []
    for i in range(batch_size):
    # i =0
    # while len(question_batch) < batch_size:

        idx = random.choice(indexes)

        # sentence = get_sentence(question_dataset[idx])
        # if word in sentence:
        img_batch.append(image_dataset[idx])
        question_batch.append(question_dataset[idx])

        # i+=1

    img_batch = np.stack(img_batch) #[B,C,W,H]
    question_batch = np.stack(question_batch) #, 1) #[T,B,L]

    img_batch = torch.from_numpy(img_batch).cuda()
    question_batch = torch.from_numpy(question_batch).cuda()
    # img_batch = rescaling(img_batch)
    img_batch = torch.clamp(img_batch, min=1e-5, max=1-1e-5)

    return img_batch, question_batch #, answer_batch





def make_batch(image_dataset, question_dataset, batch_size, val=False, word=''): #, answer_dataset): #, image_idxs, first_question_idx):

    if val:
        list_ = val_indexes
    else:
        list_ = train_indexes
        
    assert word != ''

    img_batch = []
    question_batch = []
    # for i in range(batch_size):
    i =0
    while len(question_batch) < batch_size:

        # idx = random.choice(list_)

        sentence = get_sentence(question_dataset[i])
        if word in sentence:
            img_batch.append(image_dataset[i])
            question_batch.append(question_dataset[i])

        i+=1

    img_batch = np.stack(img_batch) #[B,C,W,H]
    question_batch = np.stack(question_batch) #, 1) #[T,B,L]

    img_batch = torch.from_numpy(img_batch).cuda()
    question_batch = torch.from_numpy(question_batch).cuda()
    # img_batch = rescaling(img_batch)
    img_batch = torch.clamp(img_batch, min=1e-5, max=1-1e-5)

    return img_batch, question_batch #, answer_batch







# def logposterior_func(model, image_batch, question_batch, dec_type, zs):
def logposterior_func(zs):

    logpz = lognormal(zs, torch.zeros(zs.shape[0], zs.shape[1]).cuda(), 
                            torch.zeros(zs.shape[0], zs.shape[1]).cuda()) #[B] 

    # TODO
    # logws = model.decode(zs, iamge, quesiton, dec_type)
    # print (logpz.shape)

    logpz = logpz #* 34.
    return logpz





# def logposterior_func2(zs):

#     logpz = lognormal(zs, torch.zeros(zs.shape[0], zs.shape[1]).cuda(), 
#                             torch.zeros(zs.shape[0], zs.shape[1]).cuda()) #[B] 

#     return logpz * 32.




def logposterior_func_x(model, x, zs):

    # logpz = lognormal(zs, torch.zeros(zs.shape[0], zs.shape[1]).cuda(), 
    #                         torch.zeros(zs.shape[0], zs.shape[1]).cuda()) #[B] 

    self = model

    n_zs = len(zs)
    batch_size = 80
    # logpys = []
    i = 0

    # q = y.cuda().unsqueeze_(0)
    # embed = self.encoder_embed(q)

    while i <  n_zs:
        cur_batch_size = min(batch_size, n_zs-i)
        batch = zs[i:i+cur_batch_size]

        z_dec = self.z_to_enc(batch)
        
        x_hat = self.image_decoder(z_dec)
        alpha = torch.sigmoid(x_hat)

        beta = Beta(alpha*self.beta_scale, (1.-alpha)*self.beta_scale)
        logpx = beta.log_prob(x) #[120,3,112,112]

        logpx = torch.sum(logpx, 1)
        logpx = torch.sum(logpx, 1)
        logpx = torch.sum(logpx, 1) *.01  #why is this here!
        # print(logpx.shape)
        # fsda



        # word_preds, logpy = self.text_generator.teacher_force(z_dec, embed.repeat(cur_batch_size,1,1), q.repeat(cur_batch_size,1))

        # logpy = logpy *200.

        if i ==0:
            logpys = logpx
        else:
            logpys = torch.cat([logpys, logpx], 0)

        i+=cur_batch_size


    # print (logpys.shape)
    return logpys






def logposterior_func_y(model, y, zs):

    # logpz = lognormal(zs, torch.zeros(zs.shape[0], zs.shape[1]).cuda(), 
    #                         torch.zeros(zs.shape[0], zs.shape[1]).cuda()) #[B] 

    self = model

    n_zs = len(zs)
    batch_size = 80
    # logpys = []
    i = 0

    q = y.cuda().unsqueeze_(0)
    embed = self.encoder_embed(q)

    while i <  n_zs:
        cur_batch_size = min(batch_size, n_zs-i)
        batch = zs[i:i+cur_batch_size]

        z_dec = self.z_to_enc(batch)
        
        word_preds, logpy = self.text_generator.teacher_force(z_dec, embed.repeat(cur_batch_size,1,1), q.repeat(cur_batch_size,1))

        # logpy = logpy *200.

        if i ==0:
            logpys = logpy
        else:
            logpys = torch.cat([logpys, logpy], 0)

        i+=cur_batch_size


    # print (logpys.shape)
    return logpys




def logposterior_func_y_approx(model, y, zs):

    # logpz = lognormal(zs, torch.zeros(zs.shape[0], zs.shape[1]).cuda(), 
    #                         torch.zeros(zs.shape[0], zs.shape[1]).cuda()) #[B] 

    self = model

    n_zs = len(zs)
    batch_size = 80
    # logpys = []
    i = 0

    q = y.cuda().unsqueeze_(0)
    embed = self.encoder_embed(q)
    y_enc = self.encode_attributes2(embed)
    # print (y_enc.shape)
    mu_y, logvar_y = self.inference_net_y(y_enc)

    while i <  n_zs:
        cur_batch_size = min(batch_size, n_zs-i)
        batch = zs[i:i+cur_batch_size]

        logpy = self.flow.logprob(batch, mu_y.repeat(cur_batch_size,1), logvar_y.repeat(cur_batch_size,1))

        if i ==0:
            logpys = logpy
        else:
            logpys = torch.cat([logpys, logpy], 0)

        i+=cur_batch_size


    # print (logpys.shape)
    return logpys




def logposterior_func_x_approx(model, x, zs):

    self = model

    n_zs = len(zs)
    batch_size = 80
    i = 0

    # q = y.cuda().unsqueeze_(0)
    # embed = self.encoder_embed(q)
    # y_enc = self.encode_attributes2(embed)
    # # print (y_enc.shape)
    # mu_y, logvar_y = self.inference_net_y(y_enc)

    x_enc = self.image_encoder2(x)
    mu, logvar = self.inference_net_x(x_enc)

    while i <  n_zs:
        cur_batch_size = min(batch_size, n_zs-i)
        batch = zs[i:i+cur_batch_size]

        logqz = lognormal(batch, mu.detach(), logvar.detach()) 

        # logpy = self.flow.logprob(batch, mu_y.repeat(cur_batch_size,1), logvar_y.repeat(cur_batch_size,1))

        if i ==0:
            logqzs = logqz
        else:
            logqzs = torch.cat([logqzs, logqz], 0)

        i+=cur_batch_size


    # print (logpys.shape)
    return logqzs






def logposterior_func_true(model, y, zs):

    # logpz = lognormal(zs, torch.zeros(zs.shape[0], zs.shape[1]).cuda(), 
    #                         torch.zeros(zs.shape[0], zs.shape[1]).cuda()) #[B] 

    self = model

    n_zs = len(zs)
    batch_size = 80
    # logpys = []
    i = 0

    q = y.cuda().unsqueeze_(0)
    embed = self.encoder_embed(q)


    logpz = lognormal(zs, torch.zeros(zs.shape[0], zs.shape[1]).cuda(), 
                            torch.zeros(zs.shape[0], zs.shape[1]).cuda()) #[B] 

    while i <  n_zs:
        cur_batch_size = min(batch_size, n_zs-i)
        batch = zs[i:i+cur_batch_size]

        z_dec = self.z_to_enc(batch)
        
        word_preds, logpy = self.text_generator.teacher_force(z_dec, embed.repeat(cur_batch_size,1,1), q.repeat(cur_batch_size,1))

        if i ==0:
            logpys = logpy
        else:
            logpys = torch.cat([logpys, logpy], 0)

        i+=cur_batch_size



    # logpys = logpz*32. + logpys*200.
    logpys = logpz + logpys
    # logpys = logpz *(32./200.) + logpys#*200.

    # print (logpys.shape)
    return logpys











def SIR(model, question_batch, inf_type=2, dec_type=2, k=1): #, k=1): #, answer_dataset, image_idxs, first_question_idx):

    # batch_size = 20
    batch_zs = []
    model.eval()
    with torch.no_grad():
        for i in range(len(question_batch)):
            # img_batch, question_batch = make_batch(image_dataset, question_dataset, batch_size, indexes=indexes)#, 
            data1 = question_batch[i]
            # print (data1.shape)
            data1 = data1.view(1,data1.shape[0])
            logws = []
            zs=[]
            for j in range(k):
                outputs = model.forward(x=None, q=data1, inf_type=inf_type, dec_type=dec_type) #, k=k) #, answer_batch)
                logws.append(outputs['logws'])
                zs.append(outputs['z'])

            logws = torch.stack(logws,0) #[k,1]
            # print (logws.shape)

            #log normalize
            max_ = torch.max(logws, 0)[0] 
            lse = torch.log(torch.sum(torch.exp(logws-max_), 0)) + max_
            log_norm_ws = logws - lse
            log_norm_ws = torch.transpose(log_norm_ws, 0,1)

            # print (log_norm_ws.shape)
            # fsdf

            # m = Multinomial(100, torch.exp(log_norm_ws))
            m = Categorical(probs=torch.exp(log_norm_ws))
            idxs = m.sample()

            # print (idxs.shape) 
            # print (idxs) 
            # fofsfsdasd
            # print (zs[idxs.data.item()][0].shape)
            batch_zs.append(zs[idxs.data.item()][0].data.cpu().numpy())



            # fdsfas

            # if k ==1:
            #     LLs.append( torch.mean(logws).data.item())
            # else:
            #     LLs.append( torch.mean(logmeanexp(logws)).data.item())




    # model.train()
    # LL = np.mean(LLs)
    batch_zs = np.array(batch_zs)

    return batch_zs











def SIR_x(model, image_batch, inf_type=1, dec_type=1, k=1): #, k=1): #, answer_dataset, image_idxs, first_question_idx):

    # batch_size = 20
    batch_zs = []
    model.eval()
    with torch.no_grad():
        for i in range(len(image_batch)):
            # img_batch, question_batch = make_batch(image_dataset, question_dataset, batch_size, indexes=indexes)#, 
            data1 = image_batch[i]
            # print (data1.shape)
            data1.unsqueeze_(0)
            # print (data1.shape)
            # fdfa
            # data1 = data1.view(1,data1.shape[0])
            logws = []
            zs=[]
            for j in range(k):
                outputs = model.forward(x=data1, q=None, inf_type=inf_type, dec_type=dec_type) #, k=k) #, answer_batch)
                logws.append(outputs['logws'])
                zs.append(outputs['z'])

            logws = torch.stack(logws,0) #[k,1]
            # print (logws.shape)

            #log normalize
            max_ = torch.max(logws, 0)[0] 
            lse = torch.log(torch.sum(torch.exp(logws-max_), 0)) + max_
            log_norm_ws = logws - lse
            log_norm_ws = torch.transpose(log_norm_ws, 0,1)

            # print (log_norm_ws.shape)
            # fsdf

            # m = Multinomial(100, torch.exp(log_norm_ws))
            m = Categorical(probs=torch.exp(log_norm_ws))
            idxs = m.sample()

            # print (idxs.shape) 
            # print (idxs) 
            # fofsfsdasd
            # print (zs[idxs.data.item()][0].shape)
            batch_zs.append(zs[idxs.data.item()][0].data.cpu().numpy())



            # fdsfas

            # if k ==1:
            #     LLs.append( torch.mean(logws).data.item())
            # else:
            #     LLs.append( torch.mean(logmeanexp(logws)).data.item())




    # model.train()
    # LL = np.mean(LLs)
    batch_zs = np.array(batch_zs)

    return batch_zs




def compute_log_like2(model, image_dataset, question_dataset, n_batches, indexes, inf_type=1, dec_type=0, k=1): #, k=1): #, answer_dataset, image_idxs, first_question_idx):

    batch_size = 20
    LLs = []
    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            img_batch, question_batch = make_batch(image_dataset, question_dataset, batch_size, indexes=indexes)#, 

            logws = []
            for j in range(k):
                outputs = model.forward(x=None, q=question_batch, inf_type=inf_type, dec_type=dec_type) #, k=k) #, answer_batch)
                logws.append(outputs['logws'])

            logws = torch.stack(logws, 0) #[k,B]
            if k ==1:
                LLs.append( torch.mean(logws).data.item())
            else:
                LLs.append( torch.mean(logmeanexp(logws)).data.item())
    model.train()
    LL = np.mean(LLs)

    return LL




def compute_log_like(model, image_dataset, question_dataset, n_batches, val=False, inf_type=1): #, k=1): #, answer_dataset, image_idxs, first_question_idx):

    batch_size = 20
    # n_batches = 11
    LLs = []
    LLs_shuffled = []

    model.eval()
    with torch.no_grad():

        for i in range(n_batches):

            img_batch, question_batch = make_batch(image_dataset, question_dataset, batch_size, val=val)#, 
            shuffled_image = img_batch[torch.randperm(img_batch.shape[0])]

            outputs_shuf = model.forward(shuffled_image, question_batch, inf_type=inf_type, dec_type=0) #, k=k) #, answer_batch)
            outputs = model.forward(img_batch, question_batch, inf_type=inf_type, dec_type=0) #, k=k) #, answer_batch)

            LLs.append(outputs['elbo'].data.item())
            LLs_shuffled.append(outputs_shuf['elbo'].data.item())

    model.train()


    LL = np.mean(LLs)
    LL_shuffle = np.mean(LLs_shuffled)

    return LL, LL_shuffle












def plot_isocontours2_exp_norm(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False, contourtype='f', levels=[]):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    # zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor).cuda()
    bbb = func(aaa)
    zs = bbb.data.cpu().numpy()

    max_ = np.max(zs)
    zs_sum = np.log(np.sum(np.exp(zs-max_))) + max_
    zs = zs - zs_sum
    zs = np.exp(zs)

    # print (zs[:10])
    # print (np.min(zs))
    # print (np.max(zs))
    # print (np.sort(zs)[::-1][:10])
    # fsda
    Z = zs.reshape(X.shape)

    # levels = [0.0, 0.0006, 0.0012, 0.0018, 0.0024, 0.003, 0.0036, 0.0042, 0.0048, ]
    # levels = np.linspace(0,.01, 10)
    # print (levels)
    # fsd

    # print (np.sum(Z))

    # Z = np.clip(Z, a_min=.00051, a_max=.02)
    Z = np.clip(Z, a_min=0., a_max=.01)

    if len(levels) > 0:

        if contourtype == 'notf':
            cs = plt.contour(X, Y, Z, levels, cmap=cmap, alpha=alpha)
        else:
            cs = plt.contourf(X, Y, Z, levels, cmap=cmap, alpha=alpha)

    else:
        if contourtype == 'notf':
            cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)
        else:
            cs = plt.contourf(X, Y, Z, cmap=cmap, alpha=alpha)



    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4) 

        # print (nm)
        # print (lbl)

        # fsdafs

    # ax.set_yticks([])
    # ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')



    # ax.set_xlim(left=-8, right=8)
    # ax.set_ylim([-8,8])
    # ax.set_title('a) z~N(0,1)', fontsize=title_size)
    # ax.xaxis.set_tick_params(labelsize=5)
    # ax.yaxis.set_tick_params(labelsize=5)
    # plt.gca().set_aspect('equal', adjustable='box')
    # ax.tick_params(axis='x', colors='white')
    # ax.grid(True, alpha=.3)








def plot_isocontours2_agg(ax, indexes, model, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1., legend=False, contourtype='f', levels=[]):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    aaa = torch.from_numpy(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T).type(torch.FloatTensor).cuda()



    # Zs = []
    for i in range(len(indexes)):
    # for i in range(10):


    # print (len(indexes))
    # fsdf
    # for i in range(10):

        img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, 1, [indexes[i]])

        # func = lambda zs: logposterior_func_x_approx(model, img_batch, zs) *.5
        func = lambda zs: logposterior_func_x(model, img_batch, zs) 
    
        bbb = func(aaa)
        zs = bbb.data.cpu().numpy()

        max_ = np.max(zs)
        zs_sum = np.log(np.sum(np.exp(zs-max_))) + max_
        zs = zs - zs_sum
        zs = np.exp(zs)
        Z = zs.reshape(X.shape)

        # Zs.append(Zs)
        if i == 0:
            Zs = Z
        else:
            Zs += Z

        print(i, len(indexes))



    Z = Zs / len(indexes)

    # print (np.sum(Z))
    # fdsa

    # fdsfad

    # print (np.sum(Z))



    # levels = np.linspace(0,.01, 10)

    # Z = np.clip(Z, a_min=.00051, a_max=.02)
    Z = np.clip(Z, a_min=0., a_max=.01)



    if len(levels) > 0:

        if contourtype == 'notf':
            cs = plt.contour(X, Y, Z, levels, cmap=cmap, alpha=alpha)
        else:
            cs = plt.contourf(X, Y, Z, levels, cmap=cmap, alpha=alpha)

    else:
        if contourtype == 'notf':
            cs = plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)
        else:
            cs = plt.contourf(X, Y, Z, cmap=cmap, alpha=alpha)



    if legend:
        nm, lbl = cs.legend_elements()
        plt.legend(nm, lbl, fontsize=4) 

    # ax.set_yticks([])
    # ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')









def train2(self, max_steps, load_step, 
                train_image_dataset, train_question_dataset,
                val_image_dataset, val_question_dataset,
                save_dir, params_dir, images_dir, batch_size,
                display_step, save_steps, trainingplot_steps, viz_steps,
                classifier, start_storing_data_step, prior_classifier): 

    random.seed( 99 )

    all_dict = {}
    recent_dict = {}

    list_recorded_values = [ 
            'all_train_elbos', 'all_valid_elbos', 
            'all_logpxs', 'all_logpxs_val', 'all_logpys', 'all_logpys_val',
            'all_logpzs', 'all_logqzs', 'all_logqzys',
            'all_logpzs_val', 'all_logqzs_val', 'all_logqzys_val',
            'all_MIs', 'all_MIs_val', 'all_LL', 'all_SLL', 'all_LL_val', 'all_SLL_val', 
            'all_real_acc', 'all_real_val_acc', 'all_recon_acc', 'all_y_recon_acc', 
            'all_recon_acc_val', 'all_y_recon_acc_val', 
            'all_prior_acc', 'all_prior_acc2', 
            'all_x_inf_acc', 'all_y_inf_acc', 'all_x_inf_acc_entangle', 'all_y_inf_acc_entangle',
            'all_y_inf_acc_k1', 'all_x_inf_acc_k1', 
            'all_logvar_max', 'all_logvar_min', 'all_logvar_mean',
            'acc0', 'acc1', 'acc2', 'acc3',
            'acc0_prior', 'acc1_prior', 'acc2_prior', 'acc3_prior',
            'aa_prior_classifier_acc', 'aa_prior_acc'
    ]

    all_dict['all_steps'] = []
    all_dict['all_betas'] = []

    for label in list_recorded_values:
        all_dict[label] = []
        recent_dict[label] = deque(maxlen=10)

    def add_to_dict(dict1, label, value):
        dict1[label].append(value)


    self.eval()
    # self.train()



    self.B = batch_size
    start_time = time.time()
    step_count = 0
    warmup_steps = 10000. 

    if self.joint_inf:
        inf_type = 0
    else:
        inf_type = 1


    v1 = 1
    v2 = 0

    with torch.no_grad():

        if v2:
            #version 2 where I was trying to get better agg plots

            

                #Fig with 1 subplot
                title_size = 6

                levels = np.linspace(0.005,.01, 30)
                # levels = np.linspace(0.0001,.005, 30)
                # levels = np.concatenate([levels, np.array([5.])])

                # print (levels)
                # fdsf
                legend1 = False
                k=2
                x_lims = [-4,4]
                y_lims = [-4,4]
                scatter_alpha = .1
                #PLot the encodings of the dataset 
                rows = 1
                cols = 1
                fig = plt.figure(figsize=(2+cols,4+rows), facecolor='white', dpi=150)

                numticks = 51 #21 #101



                ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

                word1 = 'small red metal sphere'
                red_indexes = get_indexes(train_question_dataset, word=word1)
                # red_indexes = train_indexes
                plot_isocontours2_agg(ax, red_indexes, model=model, xlimits=x_lims, ylimits=y_lims,
                                 numticks=51, cmap='Blues', alpha=1., legend=legend1, levels=[])
                ax.grid(True, alpha=.3)
                ax.tick_params(axis='x', colors='white')
                ax.yaxis.set_tick_params(labelsize=5)
                ax.tick_params(axis='y', colors='white')
                # ax.set_title('Marginal Approx Posterior', fontsize=title_size)

                # levels=[]




                # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[0]])

                # func = lambda zs: logposterior_func_x(self, img_batch[0], zs) #* 200.
                # # func = lambda zs: logposterior_func_y(self, train_question_dataset[0], zs)
                # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
                #                  numticks=101, cmap='Blues', alpha=1., legend=False)
                # ax.grid(True, alpha=.3)
                # ax.tick_params(axis='x', colors='white')
                # ax.yaxis.set_tick_params(labelsize=5)
                # ax.tick_params(axis='y', colors='white')
                # # ax.set_title('p(y|z)', fontsize=title_size)





                # print('doign 1')
                # word1 = 'small red metal sphere'
                # red_indexes = get_indexes(train_question_dataset, word=word1)
                # plot_isocontours2_agg(ax, red_indexes, model=model, xlimits=x_lims, ylimits=y_lims,
                #                  numticks=numticks, cmap='Oranges', alpha=1., legend=legend1, levels=levels)

                # print('doign 2')
                # word1 = 'large red metal sphere'
                # red_indexes = get_indexes(train_question_dataset, word=word1)
                # plot_isocontours2_agg(ax, red_indexes, model=model, xlimits=x_lims, ylimits=y_lims,
                #                  numticks=numticks, cmap='Reds', alpha=1., legend=legend1, levels=levels)

                # print('doign 3')
                # word1 = 'large blue metal sphere'
                # red_indexes = get_indexes(train_question_dataset, word=word1)
                # plot_isocontours2_agg(ax, red_indexes, model=model, xlimits=x_lims, ylimits=y_lims,
                #                  numticks=numticks, cmap='Blues', alpha=1., legend=legend1, levels=levels)

                # print('doign 4')
                # word1 = 'small blue metal sphere'
                # red_indexes = get_indexes(train_question_dataset, word=word1)
                # plot_isocontours2_agg(ax, red_indexes, model=model, xlimits=x_lims, ylimits=y_lims,
                #                  numticks=numticks, cmap='Purples', alpha=1., legend=legend1, levels=levels)

                # ax.grid(True, alpha=.3)
                ax.tick_params(axis='x', colors='white')
                ax.yaxis.set_tick_params(labelsize=5)
                ax.tick_params(axis='y', colors='white')



                print ('making plot')
                plt_path = save_dir+'fig1_2D_prior_2.png'
                plt.savefig(plt_path)
                print ('saved training plot', plt_path)
                plt.close()
                fsdafa

































        else:



            title_size = 6

            levels = np.linspace(0.0001,.01, 30)
            legend1 = False
            k=2
            x_lims = [-4,4]
            y_lims = [-4,4]
            scatter_alpha = .1
            #PLot the encodings of the dataset 
            rows = 5
            cols = 6
            fig = plt.figure(figsize=(2+cols,4+rows), facecolor='white', dpi=150)






            # ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

            # n_batches = 20
            # #Prior samples
            # for i in range(n_batches):
            #     # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, red_indexes)

            #     z = torch.FloatTensor(batch_size, self.z_size).normal_().numpy() #.cuda() * std

            #     if i == 0:
            #         zs = z
            #     else:
            #         zs = np.concatenate([zs, z], 0)

            #     # print (zs.shape)
            #     # fdsf


            # ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            # ax.set_xlim(left=x_lims[0], right=x_lims[1])
            # ax.set_ylim(y_lims)
            # ax.set_title('   z~N(0,1)', fontsize=title_size)
            # ax.xaxis.set_tick_params(labelsize=5)
            # ax.yaxis.set_tick_params(labelsize=5)
            # plt.gca().set_aspect('equal', adjustable='box')
            # # ax.set_xticks([])
            # ax.tick_params(axis='x', colors='white')
            # ax.grid(True, alpha=.3)






            # ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)

            # # n_batches = 10
            # #Dataset encodings
            # for i in range(n_batches):
            #     img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, train_indexes)
            #     z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)

            #     if i == 0:
            #         zs = z
            #     else:
            #         zs = np.concatenate([zs, z], 0)

            # ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            # ax.set_xlim(left=x_lims[0], right=x_lims[1])
            # ax.set_ylim(y_lims)
            # ax.set_title(' Dataset Encodings', fontsize=title_size)
            # ax.xaxis.set_tick_params(labelsize=5)
            # # ax.text(0.3, .9, text, transform=ax.transAxes, family='serif', size=6)
            # plt.gca().set_aspect('equal', adjustable='box')
            # ax.yaxis.set_tick_params(labelsize=5)
            # # ax.set_xticks([])
            # ax.tick_params(axis='x', colors='white')
            # ax.tick_params(axis='y', colors='white')
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='y', colors='white')






            # #Plot prior
            # ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)
            # func = lambda zs: logposterior_func(zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=legend1, levels=levels)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')
            # ax.set_title(' prior', fontsize=title_size)




            # #Plot prior
            # ax = plt.subplot2grid((rows,cols), (0,3), frameon=False)
            # func = lambda zs: logposterior_func(zs)* 34. # (32./200.)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=legend1, levels=levels)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')
            # ax.set_title('prior', fontsize=title_size)




            # #Plot prior
            # ax = plt.subplot2grid((rows,cols), (0,4), frameon=False)

            # x_lims1 = [-2,2]
            # y_lims1 = [-2,2]
            # func = lambda zs: logposterior_func(zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims1, ylimits=y_lims1,
            #                  numticks=51, cmap='Blues', alpha=1., legend=legend1,
            #                  contourtype='notf')

            # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[2]])
            # func = lambda zs: logposterior_func_y_approx(self, question_batch[0], zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims1, ylimits=y_lims1,
            #                  numticks=51, cmap='Greens', alpha=1., legend=legend1,
            #                  contourtype='notf')

            # func = lambda zs: logposterior_func_x_approx(self, img_batch[0].unsqueeze_(0), zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims1, ylimits=y_lims1,
            #                  numticks=51, cmap='Reds', alpha=1., legend=legend1,
            #                  contourtype='notf')

            # # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')



            # print ('(0,4)')




            # #Plot agg

            # # word1 = 'small red metal sphere'
            # # red_indexes = get_indexes(train_question_dataset, word=word1)
            # # print (len(red_indexes))
            # # word1 = 'large red metal sphere'
            # # red_indexes = get_indexes(train_question_dataset, word=word1)
            # # print (len(red_indexes))
            # # word1 = 'large blue metal sphere'
            # # red_indexes = get_indexes(train_question_dataset, word=word1)
            # # print (len(red_indexes))
            # # word1 = 'small blue metal sphere'
            # # red_indexes = get_indexes(train_question_dataset, word=word1)
            # # print (len(red_indexes))
            # # fdsafs



            # ax = plt.subplot2grid((rows,cols), (0,5), frameon=False)

            # # word1 = 'small red metal sphere'
            # # red_indexes = get_indexes(train_question_dataset, word=word1)
            # red_indexes = train_indexes

            # plot_isocontours2_agg(ax, red_indexes, model=model, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=legend1, levels=levels)

            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')
            # ax.set_title('Marginal Approx Posterior', fontsize=title_size)








            # print ('hereeee')



            # ax = plt.subplot2grid((rows,cols), (1,5), frameon=False)

            # word1 = 'small red metal sphere'
            # red_indexes = get_indexes(train_question_dataset, word=word1)

            # plot_isocontours2_agg(ax, red_indexes, model=model, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=legend1, levels=levels)

            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')
            # ax.set_title('Marginal Approx Posterior', fontsize=title_size)














            # ax = plt.subplot2grid((rows,cols), (2,5), frameon=False)

            # word1 = 'large red metal sphere'
            # red_indexes = get_indexes(train_question_dataset, word=word1)

            # plot_isocontours2_agg(ax, red_indexes, model=model, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=legend1, levels=levels)

            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')




            # ax = plt.subplot2grid((rows,cols), (3,5), frameon=False)

            # word1 = 'small blue metal sphere'
            # red_indexes = get_indexes(train_question_dataset, word=word1)

            # plot_isocontours2_agg(ax, red_indexes, model=model, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=legend1, levels=levels)

            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')



            # ax = plt.subplot2grid((rows,cols), (4,5), frameon=False)

            # word1 = 'large blue metal sphere'
            # red_indexes = get_indexes(train_question_dataset, word=word1)

            # plot_isocontours2_agg(ax, red_indexes, model=model, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=legend1, levels=levels)

            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')





            # print ('(4,5)')

            # ax = plt.subplot2grid((rows,cols), (1,0), frameon=False)

            # for i in range(n_batches):
            #     img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, red_indexes)
            #     z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)

            #     if i == 0:
            #         zs = z
            #     else:
            #         zs = np.concatenate([zs, z], 0)

            # ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)






            print ('hereeee2')




            #Plot qz|y
            img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[0]])

            ax = plt.subplot2grid((rows,cols), (1,4), frameon=False)
            # func = lambda zs: logposterior_func(model, datapoints, zs)
            func = lambda zs: logposterior_func_true(self, question_batch[0], zs)
            plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
                             numticks=51, cmap='Blues', alpha=1., legend=False, contourtype='notf')
            ax.grid(True, alpha=.3)
            ax.tick_params(axis='x', colors='white')
            ax.yaxis.set_tick_params(labelsize=5)
            ax.tick_params(axis='y', colors='white')
            ax.set_title('p(z|y) ', fontsize=title_size)



            # #Plot qz|y
            # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[3]])

            # ax = plt.subplot2grid((rows,cols), (2,4), frameon=False)
            # # func = lambda zs: logposterior_func(model, datapoints, zs)
            # func = lambda zs: logposterior_func_true(self, question_batch[0], zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=False)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')
            # # ax.set_title('q(z|y) distribution', fontsize=title_size)


            # #Plot qz|y
            # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[5]])

            # ax = plt.subplot2grid((rows,cols), (3,4), frameon=False)
            # # func = lambda zs: logposterior_func(model, datapoints, zs)
            # func = lambda zs: logposterior_func_true(self, question_batch[0], zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=False)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')
            # # ax.set_title('q(z|y) distribution', fontsize=title_size)


            # #Plot qz|y
            # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[1]])

            # ax = plt.subplot2grid((rows,cols), (4,4), frameon=False)
            # # func = lambda zs: logposterior_func(model, datapoints, zs)
            # func = lambda zs: logposterior_func_true(self, question_batch[0], zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=False)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')
            # # ax.set_title('q(z|y) distribution', fontsize=title_size)



            # print ('(4,4)')





            # #Plot qz|y
            # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[0]])

            # ax = plt.subplot2grid((rows,cols), (1,3), frameon=False)
            # # func = lambda zs: logposterior_func(model, datapoints, zs)
            # func = lambda zs: logposterior_func_y_approx(self, question_batch[0], zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=False)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')
            # ax.set_title('q(z|y) distribution', fontsize=title_size)



            # #Plot qz|y
            # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[3]])

            # ax = plt.subplot2grid((rows,cols), (2,3), frameon=False)
            # # func = lambda zs: logposterior_func(model, datapoints, zs)
            # func = lambda zs: logposterior_func_y_approx(self, question_batch[0], zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=False)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')
            # # ax.set_title('q(z|y) distribution', fontsize=title_size)


            # #Plot qz|y
            # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[5]])

            # ax = plt.subplot2grid((rows,cols), (3,3), frameon=False)
            # # func = lambda zs: logposterior_func(model, datapoints, zs)
            # func = lambda zs: logposterior_func_y_approx(self, question_batch[0], zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=False)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')
            # # ax.set_title('q(z|y) distribution', fontsize=title_size)


            # #Plot qz|y
            # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[1]])

            # ax = plt.subplot2grid((rows,cols), (4,3), frameon=False)
            # # func = lambda zs: logposterior_func(model, datapoints, zs)
            # func = lambda zs: logposterior_func_y_approx(self, question_batch[0], zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=False)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')
            # # ax.set_title('q(z|y) distribution', fontsize=title_size)


            

            # print ('(4,3)')




            # #Plot p(y|z)

            # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[0]])

            # # print (question_batch[0].unsqueeze_(0).shape)
            # # print (train_question_dataset[0].shape)
            # # print (question_batch[0].shape)
            # # fdsf

            # ax = plt.subplot2grid((rows,cols), (1,2), frameon=False)
            # func = lambda zs: logposterior_func_y(self, question_batch[0], zs) #* 200.
            # # func = lambda zs: logposterior_func_y(self, train_question_dataset[0], zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=False)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')
            # ax.set_title('p(y|z)', fontsize=title_size)




            # #Plot p(z|y) 2

            # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[3]])

            # # print (question_batch[0].unsqueeze_(0).shape)
            # # print (train_question_dataset[0].shape)
            # # print (question_batch[0].shape)
            # # fdsf

            # ax = plt.subplot2grid((rows,cols), (2,2), frameon=False)
            # func = lambda zs: logposterior_func_y(self, question_batch[0], zs)
            # # func = lambda zs: logposterior_func_y(self, train_question_dataset[0], zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=False)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')




            # #Plot p(z|y) 3

            # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[5]])

            # # print (question_batch[0].unsqueeze_(0).shape)
            # # print (train_question_dataset[0].shape)
            # # print (question_batch[0].shape)
            # # fdsf

            # ax = plt.subplot2grid((rows,cols), (3,2), frameon=False)
            # func = lambda zs: logposterior_func_y(self, question_batch[0], zs)
            # # func = lambda zs: logposterior_func_y(self, train_question_dataset[0], zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=False)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')



            # #Plot p(z|y) 4

            # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[1]])

            # # print (question_batch[0].unsqueeze_(0).shape)
            # # print (train_question_dataset[0].shape)
            # # print (question_batch[0].shape)
            # # fdsf

            # ax = plt.subplot2grid((rows,cols), (4,2), frameon=False)
            # func = lambda zs: logposterior_func_y(self, question_batch[0], zs)
            # # func = lambda zs: logposterior_func_y(self, train_question_dataset[0], zs)
            # plot_isocontours2_exp_norm(ax, func, xlimits=x_lims, ylimits=y_lims,
            #                  numticks=51, cmap='Blues', alpha=1., legend=False)
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.tick_params(axis='y', colors='white')








            # # print ('done p(z|y)')
            # print ('(4,2)')


            







            # ax = plt.subplot2grid((rows,cols), (1,1), frameon=False)

            # for i in range(n_batches):
            #     img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[0]])
            #     # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
            #     # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
            #     z = SIR(self, question_batch, inf_type=2, dec_type=2, k=1)

            #     if i == 0:
            #         zs = z
            #     else:
            #         zs = np.concatenate([zs, z], 0)

            # ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            # ax.set_xlim(left=x_lims[0], right=x_lims[1])
            # ax.set_ylim(y_lims)
            # ax.set_title(' q(z|y) Samples', fontsize=title_size)
            # plt.gca().set_aspect('equal', adjustable='box')
            # ax.yaxis.set_tick_params(labelsize=5)
            # ax.xaxis.set_tick_params(labelsize=5)
            # # ax.set_xticks([])
            # ax.tick_params(axis='x', colors='white')
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='y', colors='white')




            # ax = plt.subplot2grid((rows,cols), (2,1), frameon=False)

            # for i in range(n_batches):
            #     img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[3]])
            #     # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
            #     # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
            #     z = SIR(self, question_batch, inf_type=2, dec_type=2, k=1)

            #     if i == 0:
            #         zs = z
            #     else:
            #         zs = np.concatenate([zs, z], 0)

            # ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            # ax.set_xlim(left=x_lims[0], right=x_lims[1])
            # ax.set_ylim(y_lims)
            # # ax.set_title(' q(z|y) Samples ', fontsize=title_size)
            # plt.gca().set_aspect('equal', adjustable='box')
            # ax.xaxis.set_tick_params(labelsize=5)
            # ax.yaxis.set_tick_params(labelsize=5)
            # # ax.set_xticks([])
            # ax.tick_params(axis='x', colors='white')
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='y', colors='white')


            # ax = plt.subplot2grid((rows,cols), (3,1), frameon=False)

            # for i in range(n_batches):
            #     img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[5]])
            #     # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
            #     # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
            #     z = SIR(self, question_batch, inf_type=2, dec_type=2, k=1)

            #     if i == 0:
            #         zs = z
            #     else:
            #         zs = np.concatenate([zs, z], 0)

            # ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            # ax.set_xlim(left=x_lims[0], right=x_lims[1])
            # ax.set_ylim(y_lims)
            # # ax.set_title(' q(z|y) Samples ', fontsize=title_size)
            # plt.gca().set_aspect('equal', adjustable='box')
            # ax.xaxis.set_tick_params(labelsize=5)
            # ax.yaxis.set_tick_params(labelsize=5)
            # # ax.set_xticks([])
            # ax.tick_params(axis='x', colors='white')
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='y', colors='white')



            # ax = plt.subplot2grid((rows,cols), (4,1), frameon=False)

            # for i in range(n_batches):
            #     img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[1]])
            #     # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
            #     # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
            #     z = SIR(self, question_batch, inf_type=2, dec_type=2, k=1)

            #     if i == 0:
            #         zs = z
            #     else:
            #         zs = np.concatenate([zs, z], 0)

            # ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            # ax.set_xlim(left=x_lims[0], right=x_lims[1])
            # ax.set_ylim(y_lims)
            # # ax.set_title(' q(z|y) Samples ', fontsize=title_size)
            # plt.gca().set_aspect('equal', adjustable='box')
            # ax.xaxis.set_tick_params(labelsize=5)
            # ax.yaxis.set_tick_params(labelsize=5)
            # # ax.set_xticks([])
            # ax.tick_params(axis='x', colors='white')
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='y', colors='white')






            # word1 = 'small red metal sphere'
            # red_indexes = get_indexes(train_question_dataset, word=word1)

            # ax = plt.subplot2grid((rows,cols), (1,0), frameon=False)

            # for i in range(n_batches):
            #     img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, red_indexes)
            #     z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)

            #     if i == 0:
            #         zs = z
            #     else:
            #         zs = np.concatenate([zs, z], 0)

            # ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            # ax.set_xlim(left=x_lims[0], right=x_lims[1])
            # ax.set_ylim(y_lims)
            # ax.set_title('x agg.'+' ' +get_sentence(question_batch[0]), fontsize=title_size)
            # plt.gca().set_aspect('equal', adjustable='box')
            # ax.xaxis.set_tick_params(labelsize=5)
            # ax.yaxis.set_tick_params(labelsize=5)
            # # ax.set_xticks([])
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')





            # word1 = 'large red metal sphere'
            # red_indexes = get_indexes(train_question_dataset, word=word1)

            # ax = plt.subplot2grid((rows,cols), (2,0), frameon=False)

            # for i in range(n_batches):
            #     img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, red_indexes)
            #     z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)

            #     if i == 0:
            #         zs = z
            #     else:
            #         zs = np.concatenate([zs, z], 0)

            # ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            # ax.set_xlim(left=x_lims[0], right=x_lims[1])
            # ax.set_ylim(y_lims)
            # ax.set_title('x agg.'+' ' +get_sentence(question_batch[0]), fontsize=title_size)
            # plt.gca().set_aspect('equal', adjustable='box')
            # ax.xaxis.set_tick_params(labelsize=5)
            # ax.yaxis.set_tick_params(labelsize=5)
            # # ax.set_xticks([])
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')





            # word1 = 'small blue metal sphere'
            # red_indexes = get_indexes(train_question_dataset, word=word1)

            # ax = plt.subplot2grid((rows,cols), (3,0), frameon=False)

            # for i in range(n_batches):
            #     img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, red_indexes)
            #     z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)

            #     if i == 0:
            #         zs = z
            #     else:
            #         zs = np.concatenate([zs, z], 0)

            # ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            # ax.set_xlim(left=x_lims[0], right=x_lims[1])
            # ax.set_ylim(y_lims)
            # ax.set_title('x agg.'+' ' +get_sentence(question_batch[0]), fontsize=title_size)
            # plt.gca().set_aspect('equal', adjustable='box')
            # ax.xaxis.set_tick_params(labelsize=5)
            # ax.yaxis.set_tick_params(labelsize=5)
            # # ax.set_xticks([])
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')



            # word1 = 'large blue metal sphere'
            # red_indexes = get_indexes(train_question_dataset, word=word1)

            # ax = plt.subplot2grid((rows,cols), (4,0), frameon=False)

            # for i in range(n_batches):
            #     img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, red_indexes)
            #     z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)

            #     if i == 0:
            #         zs = z
            #     else:
            #         zs = np.concatenate([zs, z], 0)

            # ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            # ax.set_xlim(left=x_lims[0], right=x_lims[1])
            # ax.set_ylim(y_lims)
            # ax.set_title('x agg.'+' ' +get_sentence(question_batch[0]), fontsize=title_size)
            # plt.gca().set_aspect('equal', adjustable='box')
            # ax.xaxis.set_tick_params(labelsize=5)
            # ax.yaxis.set_tick_params(labelsize=5)
            # # ax.set_xticks([])
            # ax.grid(True, alpha=.3)
            # ax.tick_params(axis='x', colors='white')





            # print ('(4,0)')












        # print ('done e')

        if 0:





            ax = plt.subplot2grid((rows,cols), (3,0), frameon=False)

            for i in range(n_batches):
                img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[1]])
                z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)

                if i == 0:
                    zs = z
                else:
                    zs = np.concatenate([zs, z], 0)

            ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            ax.set_xlim(left=x_lims[0], right=x_lims[1])
            ax.set_ylim(y_lims)
            ax.set_title('g) image '+get_sentence(question_batch[0]), fontsize=title_size)
            plt.gca().set_aspect('equal', adjustable='box')
            # ax.xaxis.set_tick_params(labelsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.tick_params(axis='x', colors='white')
            ax.grid(True, alpha=.3)






            ax = plt.subplot2grid((rows,cols), (3,1), frameon=False)

            for i in range(n_batches):
                img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[5]])
                z = SIR(self, question_batch, inf_type=2, dec_type=2, k=k)

                if i == 0:
                    zs = z
                else:
                    zs = np.concatenate([zs, z], 0)

            ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            zs_11 = zs

            ax.set_xlim(left=x_lims[0], right=x_lims[1])
            ax.set_ylim(y_lims)
            ax.set_title('h) k'+str(k)+' ' +get_sentence(question_batch[0]), fontsize=title_size)
            plt.gca().set_aspect('equal', adjustable='box')
            ax.xaxis.set_tick_params(labelsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            # ax.set_xticks([])
            ax.grid(True, alpha=.3)
            ax.tick_params(axis='x', colors='white')








            ax = plt.subplot2grid((rows,cols), (4,0), frameon=False)

            for i in range(n_batches):
                img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[0]])
                z = SIR(self, question_batch, inf_type=2, dec_type=2, k=k)

                if i == 0:
                    zs = z
                else:
                    zs = np.concatenate([zs, z], 0)

            ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=6, alpha=scatter_alpha)

            ax.set_xlim(left=x_lims[0], right=x_lims[1])
            ax.set_ylim(y_lims)
            ax.set_title('i) k'+str(k)+' ' +get_sentence(question_batch[0]), fontsize=title_size)
            plt.gca().set_aspect('equal', adjustable='box')
            ax.xaxis.set_tick_params(labelsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            # ax.set_xticks([])
            ax.grid(True, alpha=.3)
            ax.tick_params(axis='x', colors='white')







            ax = plt.subplot2grid((rows,cols), (4,1), frameon=False)


            ax.scatter(x=zs_11[:,0],y=zs_11[:,1], c='blue', s=6, alpha=scatter_alpha)
            ax.scatter(x=zs[:,0],y=zs[:,1], c='red', s=6, alpha=scatter_alpha)

            ax.set_xlim(left=x_lims[0], right=x_lims[1])
            ax.set_ylim(y_lims)
            ax.set_title('j) combined the k plots', fontsize=title_size)
            plt.gca().set_aspect('equal', adjustable='box')
            ax.xaxis.set_tick_params(labelsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            # ax.set_xticks([])
            ax.grid(True, alpha=.3)
            ax.tick_params(axis='x', colors='white')











    print ('making plot')
    plt_path = save_dir+'fig1_2D_prior_nov12.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()





    fadsf



























    # pca = PCA(n_components=2)

    #Plot samples
    rows = 1
    cols =1 
    fig = plt.figure(figsize=(4,4), facecolor='white', dpi=150)

    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

    n_batches = 10#0 #0
    k= 100

    word1 = 'small'
    # word1 = 'blue'
    # word1 = 'red'
    # word1 = 'metal'
    # word1 = 'large'
    red_indexes = get_indexes(train_question_dataset, word=word1)

    for i in range(n_batches):
        img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, red_indexes)

        if i % 2 == 0:
            print (i)

        # zs_y_red = self.get_z_samples(q=question_batch, inf_type=2).data.cpu().numpy()
        zs_y_red = SIR(self, question_batch, inf_type=2, dec_type=2, k=k)
        if i ==0:
            zs_y_reds = zs_y_red
        else:
            zs_y_reds = np.concatenate([zs_y_reds, zs_y_red], 0)

        # zs_x_red = self.get_z_samples2(x=img_batch, inf_type=1).data.cpu().numpy()
        zs_x_red = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=k)
        if i ==0:
            zs_x_reds = zs_x_red
        else:
            zs_x_reds = np.concatenate([zs_x_reds, zs_x_red], 0)





    # word2 = 'small'
    # blue_indexes = get_indexes(train_question_dataset, word=word2)

    # for i in range(n_batches):
    #     img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, blue_indexes)

    #     # zs_y_blue = self.get_z_samples(q=question_batch, inf_type=2).data.cpu().numpy()
    #     zs_y_blue = SIR(self, question_batch, inf_type=2, dec_type=2, k=k)
    #     if i ==0:
    #         zs_y_blues = zs_y_blue
    #     else:
    #         zs_y_blues = np.concatenate([zs_y_blues, zs_y_blue], 0)

    #     # zs_x_blue = self.get_z_samples2(x=img_batch, inf_type=1).data.cpu().numpy()
    #     zs_x_blue = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=k)
    #     if i ==0:
    #         zs_x_blues = zs_x_blue
    #     else:
    #         zs_x_blues = np.concatenate([zs_x_blues, zs_x_blue], 0)





    print (zs_x_reds.shape)
    # fsdf
    len_ = len(zs_x_reds)


    # img_batch, question_batch = make_batch(train_image_dataset, train_question_dataset, batch_size, word='blue')

    # zs_y_blue = self.get_z_samples(q=question_batch, inf_type=2)
    # zs_y_blue = zs_y_blue.data.cpu().numpy()

    # zs_x_blue = self.get_z_samples2(x=img_batch, inf_type=1)
    # zs_x_blue = zs_x_blue.data.cpu().numpy()


    # if zs.shape[1] > 2:

    #     zs = np.concatenate([zs_y_reds, zs_x_reds], 0)
    #     zs = np.concatenate([zs, zs_y_blues], 0)
    #     zs = np.concatenate([zs, zs_x_blues], 0)

    #     print (zs.shape)
    #     pca.fit(zs)
    #     print(pca.explained_variance_ratio_)  
    #     zs = pca.transform(zs)
    #     print (zs.shape)

    #     ax.scatter(x=zs[0:len_,0],y=zs[0:len_,1], c='pink', label=word1+' y', marker='x', s=5)
    #     ax.scatter(x=zs[len_:len_*2,0],y=zs[len_:len_*2,1], c='red', label=word1+' x', marker='x', s=5)

    #     # ax.scatter(x=zs[len_*2:len_*3,0],y=zs[len_*2:len_*3,1], c='blue', label=word2+' y', marker='x', s=5)
    #     # ax.scatter(x=zs[len_*3:,0],y=zs[len_*3:,1], c='green', label=word2+' x', marker='x', s=5)


    # ax.scatter(x=zs_y_reds[:,0],y=zs_y_reds[:,1], c='blue', label=word1+' y', marker='x', s=5, alpha=.3)
    # ax.scatter(x=zs_x_reds[:,0],y=zs_x_reds[:,1], c='red', label=word1+' x', marker='x', s=5, alpha=.3)




    # for jj in range(len(zs_y_reds)):


    #     ax.scatter(x=zs_y_reds[jj,0],y=zs_y_reds[jj,1], c='blue', s=6, alpha=.3)
    #     ax.scatter(x=zs_x_reds[jj,0],y=zs_x_reds[jj,1], c='red', s=6, alpha=.3)

    #     # ax.scatter(x=zs_y_reds[jj,0],y=zs_y_reds[jj,1], c='blue', marker='x', s=5, alpha=.3)
    #     # ax.scatter(x=zs_x_reds[jj,0],y=zs_x_reds[jj,1], c='red', marker='x', s=5, alpha=.3)


    # ax.scatter(x=zs_y_reds[jj,0],y=zs_y_reds[jj,1], c='blue', label=word1+' p(z|y)', marker='x', s=5, alpha=.3)
    # ax.scatter(x=zs_x_reds[jj,0],y=zs_x_reds[jj,1], c='red', label=word1+' p(z|x)', marker='x', s=5, alpha=.3)

    bandwidth = .2

    x_ = zs_y_reds[:,0] #zs[0:len_,0] # zs[:, 0] 
    y_ = zs_y_reds[:,1] # zs[0:len_,1] #zs[:, 1]
    values = np.vstack([x_, y_])
    kernel = st.gaussian_kde(values,bandwidth)
    xlimits=[-8, 8]
    ylimits=[-8, 8]
    numticks=101
    # cmap = 'Greys'
    cmap = 'Reds'
    # Peform the kernel density estimate
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    f = np.reshape(kernel(positions).T, X.shape)
    # cfset = ax.contourf(X, Y, f, cmap=cmap)
    # cfset1 = ax.contour(X, Y, f, cmap=cmap, label='p(z|Y_large)', linestyles="--")
    cfset1 = ax.contour(X, Y, f, cmap=cmap)
    h1,_ = cfset1.legend_elements()


    x_ = zs_x_reds[:,0] #zs[len_:len_*2,0] # zs[:, 0] 
    y_ = zs_x_reds[:,1]  #zs[len_:len_*2,1] #zs[:, 1]
    values = np.vstack([x_, y_])
    kernel = st.gaussian_kde(values, bandwidth)
    xlimits=[-8, 8]
    ylimits=[-8, 8]
    numticks=101
    cmap = 'binary'
    # Peform the kernel density estimate
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    f = np.reshape(kernel(positions).T, X.shape)
    # cfset = ax.contourf(X, Y, f, cmap=cmap)
    cfset2 = ax.contour(X, Y, f, cmap=cmap, linestyles="--")
    h2,_ = cfset2.legend_elements()


    # ax.set_yticks([])
    # ax.set_xticks([])


    # ax.legend([h1[4], h2[4]], [r'$p(z|Large_{Y})$', r'$p(z|Large_{X})$'], frameon=False, fontsize=8, loc=[.34,.1]) #, family='serif')
    ax.legend([h1[4], h2[4]], [r'$p(z|Red_{Y})$', r'$p(z|Red_{X})$'], frameon=False, fontsize=8, loc=[.34,.1]) #, family='serif')
    # ax.legend([h1[4], h2[4]], [r'$p(z|Metal_{Y})$', r'$p(z|Metal_{X})$'], frameon=False, fontsize=8, loc=[.34,.1]) #, family='serif')
    # ax.legend()
    # plt.legend()




    # ax.text(0.3, .9, text, transform=ax.transAxes, family='serif', size=6)



    # ax.legend(size=6, family='serif')
    # ax.legend([cfset1, cfset2], ['p(z|Y_large)', 'p(z|X_large)'])

    

    # plt.legend(frameon=False)

    # plt.gca().set_aspect('equal', adjustable='box')

    #Get latents  [B,2]

    # print (zs_y.shape)

    # pca.fit(zs_y)
    # print(pca.explained_variance_ratio_)  
    # zs_y = pca.transform(zs_y)


    # print (zs_y.shape)
    # # dfsfa

    # pca.fit(zs_x)
    # print(pca.explained_variance_ratio_)  
    # zs_x = pca.transform(zs_x)

    # pca.fit(zs_x)
    # print(pca.explained_variance_ratio_)  
    # zs_x = pca.transform(zs_x)



    #Plot samples
    # ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)


    plt_path = save_dir+'fig1_2D'+word1+ '.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()

















    rows = 1
    cols =1 
    fig = plt.figure(figsize=(4,4), facecolor='white', dpi=150)

    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

    for jj in range(len(zs_y_reds)):

        ax.scatter(x=zs_y_reds[jj,0],y=zs_y_reds[jj,1], c='blue', s=6, alpha=.3)
        ax.scatter(x=zs_x_reds[jj,0],y=zs_x_reds[jj,1], c='red', s=6, alpha=.3)

        # ax.scatter(x=zs_y_reds[jj,0],y=zs_y_reds[jj,1], c='blue', marker='x', s=5, alpha=.3)
        # ax.scatter(x=zs_x_reds[jj,0],y=zs_x_reds[jj,1], c='red', marker='x', s=5, alpha=.3)

    ax.scatter(x=zs_y_reds[jj,0],y=zs_y_reds[jj,1], c='blue', label=word1+' p(z|y)', marker='x', s=5, alpha=.3)
    ax.scatter(x=zs_x_reds[jj,0],y=zs_x_reds[jj,1], c='red', label=word1+' p(z|x)', marker='x', s=5, alpha=.3)


    # plt.legend()

    # ax.set_yticks([])
    # ax.set_xticks([])
    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])

    plt_path = save_dir+'fig1_2D_samps_'+word1+ '.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()








    afadaf

























































    fdsafafa
    for step in range(max_steps):

        # warmup = min((step_count+load_step) / float(warmup_steps), self.max_beta)
        if step > warmup_steps:
            warmup = self.max_beta
        else:
            warmup = (step / float(warmup_steps)) * self.max_beta


        if not self.just_classifier:

            if self.joint_inf:

                self.optimizer.zero_grad()
                outputs = self.forward(x=img_batch, q=question_batch, warmup=warmup, inf_type=0, dec_type=0) #, k=k_train) #, marginf_type=marginf_type)
                loss = -outputs['welbo']
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self.optimizer_x.zero_grad()
                outputs_only_x = self.forward(x=img_batch, q=None, warmup=warmup, inf_type=1, dec_type=1) #, k=k_train)
                loss_only_x = -outputs_only_x['welbo']
                loss_only_x.backward()
                self.optimizer_x.step()
                self.scheduler_x.step()

            else:

                self.optimizer_x.zero_grad()
                outputs = self.forward(x=img_batch, q=question_batch, warmup=warmup, inf_type=1, dec_type=0) #, k=k_train) #, marginf_type=marginf_type)
                loss = -outputs['welbo']
                loss.backward()
                self.optimizer_x.step()
                self.scheduler_x.step()









        # if not self.just_classifier:

        #     if self.joint_inf:
        #         self.optimizer.zero_grad()
        #         outputs = self.forward(x=img_batch, q=question_batch, warmup=warmup, inf_type=0, dec_type=0) #, k=k_train) #, marginf_type=marginf_type)
        #         loss = -outputs['welbo']
        #         loss.backward()
        #         self.optimizer.step()
        #         self.scheduler.step()

        #         self.optimizer_x.zero_grad()
        #         outputs_only_x = self.forward(x=img_batch, q=None, warmup=warmup, inf_type=1, dec_type=1) #, k=k_train)
        #         loss_only_x = -outputs_only_x['welbo']
        #         loss_only_x.backward()
        #         self.optimizer_x.step()
        #         self.scheduler_x.step()

        #     else:
        #         self.optimizer_x.zero_grad()
        #         outputs_only_x = self.forward(x=img_batch, q=question_batch, warmup=warmup, inf_type=1, dec_type=0) #, k=k_train)
        #         loss_only_x = -outputs_only_x['welbo']
        #         loss_only_x.backward()
        #         self.optimizer_x.step()
        #         self.scheduler_x.step()







            # self.optimizer.zero_grad()
            # outputs = self.forward(x=img_batch, q=question_batch, warmup=warmup, inf_type=inf_type, dec_type=0) #, k=k_train) #, marginf_type=marginf_type)
            # loss = -outputs['welbo']
            # loss.backward()
            # self.optimizer.step()
            # self.scheduler.step()

            # if self.joint_inf:

            #     self.optimizer_decode1.zero_grad()
            #     outputs_only_x = self.forward(x=img_batch, q=None, warmup=warmup, inf_type=1, dec_type=1) #, k=k_train)
            #     loss_only_x = -outputs_only_x['welbo']
            #     loss_only_x.backward()
            #     self.optimizer_decode1.step()
            #     self.scheduler_decode1.step()


        if self.train_classifier:
            classi_loss, acc = classifier.update(x=img_batch, q=question_batch)


        if self.train_prior_classifier:
            x_hat_prior, y_hat_prior, y_hat_prior_sampled_words = self.sample_prior()
            aa_prior_classi_loss, aa_prior_classifier_acc = prior_classifier.update(x=x_hat_prior.detach(), q=y_hat_prior_sampled_words.detach())

        step_count += 1



        








        if step_count % display_step == 0:

            if self.joint_inf:
                inf_type = 0
            else:
                inf_type = 1

            self.eval()
            with torch.no_grad():
                # make validation and prior batch
                val_img_batch, val_question_batch = make_batch(val_image_dataset, val_question_dataset, batch_size, val=True)
                x_hat_prior, y_hat_prior, y_hat_prior_sampled_words = self.sample_prior()
                x_hat_prior2, y_hat_prior2, y_hat_prior_sampled_words2 = self.sample_prior(std=.5)

                outputs, training_recon_img, training_recon_q, training_recon_sampled_words = self.forward(img_batch, question_batch, generate=True, inf_type=inf_type, dec_type=0, warmup=warmup) #, k=k_gen)
                val_outputs, val_recon_img, val_recon_q, val_recon_sampled_words = self.forward(val_img_batch, val_question_batch, generate=True, inf_type=inf_type, dec_type=0) #, k=k_gen)

                #Accuracies 
                _, y_recon_acc = classifier.classifier_loss(x=img_batch, attributes=training_recon_sampled_words.detach())
                _, y_recon_acc_val = classifier.classifier_loss(x=val_img_batch, attributes=val_recon_sampled_words.detach())

                # y_hat_prior = y_hat_prior.view(self.B, self.q_max_len, self.vocab_size)
                # y_hat_prior = torch.max(y_hat_prior,2)[1]
                classi_loss_prior, prior_acc = classifier.classifier_loss(x=x_hat_prior.detach(), attributes=y_hat_prior_sampled_words.detach())
                acc0_prior, acc1_prior, acc2_prior, acc3_prior = classifier.classifier_attribute_accuracies(x=x_hat_prior.detach(), attributes=y_hat_prior_sampled_words.detach())
                
                classi_loss_prior2, prior_acc2 = classifier.classifier_loss(x=x_hat_prior2.detach(), attributes=y_hat_prior_sampled_words2.detach())


                acc0, acc1, acc2, acc3 = classifier.classifier_attribute_accuracies(x=training_recon_img.detach(), attributes=question_batch)
                classi_loss_recon, recon_acc = classifier.classifier_loss(x=training_recon_img.detach(), attributes=question_batch)

                classi_loss_val, val_acc = classifier.classifier_loss(val_img_batch, val_question_batch)
                classi_loss_val_recon, val_recon_acc = classifier.classifier_loss(val_recon_img.detach(), val_question_batch)

                if not self.train_classifier: #else im already computing this
                    classi_loss, acc = classifier.classifier_loss(x=img_batch, attributes=question_batch)

                if self.train_prior_classifier:
                    aa_classi_loss, aa_prior_acc = prior_classifier.classifier_loss(x=img_batch, attributes=question_batch)


                # Conditional  # q(z|x)
                n_samps = 3 # for viewing
                ys = []
                xs_1 = []
                for i in range(n_samps):
                    outputs_cond, recon_img_cond, recon_q_cond, recon_sampled_words_cond = self.forward(x=val_img_batch, q=None, generate=True, inf_type=1, dec_type=1) #, k=k_cond)
                    ys.append(recon_sampled_words_cond)
                    xs_1.append(recon_img_cond)

                _, x_inf_acc = classifier.classifier_loss(x=val_img_batch, attributes=ys[0].detach())
                _, x_inf_acc_k1 = classifier.classifier_loss(x=xs_1[0].detach(), attributes=val_question_batch)
                _, x_inf_acc_entangle = classifier.classifier_loss(x=xs_1[0].detach(), attributes=ys[0].detach())



                # q(z|y)
                xs = []
                ys_2 = []
                for i in range(n_samps):
                    outputs_cond, recon_img_cond, recon_q_cond, recon_sampled_words_cond = self.forward(x=None, q=val_question_batch, generate=True, inf_type=2, dec_type=2) #, k=k_cond)         
                    xs.append(recon_img_cond)
                    ys_2.append(recon_sampled_words_cond)
                
                _, y_inf_acc = classifier.classifier_loss(x=xs[0].detach(), attributes=val_question_batch)
                _, y_inf_acc_k1 = classifier.classifier_loss(x=val_img_batch, attributes=ys_2[0].detach())
                _, y_inf_acc_entangle = classifier.classifier_loss(x=xs[0].detach(), attributes=ys_2[0].detach())


                LL1, LL_shuffle1 = compute_log_like(model, train_image_dataset, train_question_dataset, n_batches=1, inf_type=1)
                LL2, LL_shuffle2 = compute_log_like(model, train_image_dataset, train_question_dataset, n_batches=1, inf_type=2)
                if self.joint_inf:
                    LL0, LL_shuffle0 = compute_log_like(model, train_image_dataset, train_question_dataset, n_batches=1, inf_type=0)

                LL_val1, LL_shuffle_val1 = compute_log_like(model, val_image_dataset, val_question_dataset, n_batches=1, val=True, inf_type=1)
                LL_val2, LL_shuffle_val2 = compute_log_like(model, val_image_dataset, val_question_dataset, n_batches=1, val=True, inf_type=2)
                if self.joint_inf:
                    LL_val0, LL_shuffle_val0 = compute_log_like(model, val_image_dataset, val_question_dataset, n_batches=1, val=True, inf_type=0)

                if self.joint_inf:
                    LL = np.max([LL1, LL2, LL0])
                    LL_shuffle = np.max([LL_shuffle1, LL_shuffle2, LL_shuffle0])
                    LL_val = np.max([LL_val1, LL_val2, LL_val0])
                    LL_shuffle_val = np.max([LL_shuffle_val1, LL_shuffle_val2, LL_shuffle_val0])
                else:
                    LL = np.max([LL1, LL2])
                    LL_shuffle = np.max([LL_shuffle1, LL_shuffle2])
                    LL_val = np.max([LL_val1, LL_val2])
                    LL_shuffle_val = np.max([LL_shuffle_val1, LL_shuffle_val2])
                # print (LL)
                # fdsa
                
                    
            self.train()


            add_to_dict(recent_dict, 'all_train_elbos', outputs['elbo'].data.item())
            add_to_dict(recent_dict, 'all_valid_elbos', val_outputs['elbo'].data.item())

            add_to_dict(recent_dict, 'all_logpxs', outputs['logpx'].data.item())
            add_to_dict(recent_dict, 'all_logpys', outputs['logpy'].data.item())
            add_to_dict(recent_dict, 'all_logpzs', outputs['logpz'].data.item())
            add_to_dict(recent_dict, 'all_logqzs', outputs['logqz'].data.item())
            add_to_dict(recent_dict, 'all_logqzys', outputs['logqzy'].data.item())

            add_to_dict(recent_dict, 'all_logpxs_val', val_outputs['logpx'].data.item())
            add_to_dict(recent_dict, 'all_logpys_val', val_outputs['logpy'].data.item())
            add_to_dict(recent_dict, 'all_logpzs_val', val_outputs['logpz'].data.item())
            add_to_dict(recent_dict, 'all_logqzs_val', val_outputs['logqz'].data.item())
            add_to_dict(recent_dict, 'all_logqzys_val', val_outputs['logqzy'].data.item())

            add_to_dict(recent_dict, 'all_MIs', LL-LL_shuffle)
            add_to_dict(recent_dict, 'all_MIs_val', LL_val-LL_shuffle_val)
            add_to_dict(recent_dict, 'all_LL', LL)
            add_to_dict(recent_dict, 'all_SLL', LL_shuffle)
            add_to_dict(recent_dict, 'all_LL_val', LL_val)
            add_to_dict(recent_dict, 'all_SLL_val', LL_shuffle_val)

            add_to_dict(recent_dict, 'all_real_acc', acc.data.item())
            add_to_dict(recent_dict, 'all_real_val_acc', val_acc.data.item())
            add_to_dict(recent_dict, 'all_recon_acc', recon_acc.data.item())
            add_to_dict(recent_dict, 'all_y_recon_acc', y_recon_acc.data.item())
            add_to_dict(recent_dict, 'all_recon_acc_val', val_recon_acc.data.item())
            add_to_dict(recent_dict, 'all_y_recon_acc_val', y_recon_acc_val.data.item())
            add_to_dict(recent_dict, 'all_prior_acc', prior_acc.data.item())
            add_to_dict(recent_dict, 'all_prior_acc2', prior_acc2.data.item())

            add_to_dict(recent_dict, 'all_x_inf_acc', x_inf_acc.data.item())
            add_to_dict(recent_dict, 'all_y_inf_acc', y_inf_acc.data.item())
            add_to_dict(recent_dict, 'all_x_inf_acc_entangle', x_inf_acc_entangle.data.item())
            add_to_dict(recent_dict, 'all_y_inf_acc_entangle', y_inf_acc_entangle.data.item())

            add_to_dict(recent_dict, 'all_y_inf_acc_k1', y_inf_acc_k1.data.item())
            add_to_dict(recent_dict, 'all_x_inf_acc_k1', x_inf_acc_k1.data.item())

            add_to_dict(recent_dict, 'all_logvar_max', np.max(outputs['logvar'].data.cpu().numpy()))
            add_to_dict(recent_dict, 'all_logvar_min', np.min(outputs['logvar'].data.cpu().numpy()))
            add_to_dict(recent_dict, 'all_logvar_mean', np.mean(outputs['logvar'].data.cpu().numpy()))

            add_to_dict(recent_dict, 'acc0', acc0)
            add_to_dict(recent_dict, 'acc1', acc1)
            add_to_dict(recent_dict, 'acc2', acc2)
            add_to_dict(recent_dict, 'acc3', acc3)

            add_to_dict(recent_dict, 'acc0_prior', acc0_prior)
            add_to_dict(recent_dict, 'acc1_prior', acc1_prior)
            add_to_dict(recent_dict, 'acc2_prior', acc2_prior)
            add_to_dict(recent_dict, 'acc3_prior', acc3_prior)

            if self.train_prior_classifier:
                add_to_dict(recent_dict, 'aa_prior_classifier_acc', aa_prior_classifier_acc.data.item())
                add_to_dict(recent_dict, 'aa_prior_acc', aa_prior_acc.data.item())



            T = time.time() - start_time
            start_time = time.time()

            print (
                'S:{:5d}'.format(step_count),
                'T:{:.2f}'.format(T),
                'welbo:{:.4f}'.format(outputs['welbo'].data.item()),
                'elbo:{:.4f}'.format(outputs['elbo'].data.item()),
                'lpx:{:.4f}'.format(outputs['logpx'].data.item()),
                'lpy:{:.4f}'.format(outputs['logpy'].data.item()),
                'lpz:{:.4f}'.format(outputs['logpz'].data.item()),
                'lqz:{:.4f}'.format(outputs['logqz'].data.item()),
                'TrainAvg:{:.4f}'.format(recent_dict['all_train_elbos'][-1]),
                'ValidAvg:{:.4f}'.format(recent_dict['all_valid_elbos'][-1]),
                # 'AvgMI:{:.4f}'.format(np.mean(recent_MIs)),
                'warmup:{:.4f}'.format(warmup),
                'prioracc:{:.4f}'.format(recent_dict['all_prior_acc'][-1]),
                # 'yinf_acc_e:{:.4f}'.format(np.mean(recent_y_inf_acc_entangle)),
                )

            #Check for nans 
            def isnan(x):
                return x != x
            if (isnan(outputs['welbo']) or isnan(outputs['elbo']) or isnan(outputs['logpx']) or 
                isnan(outputs['logpy']) or isnan(outputs['logpz']) or isnan(outputs['logqz'])):
                fadsfa



            if trainingplot_steps < 222:
                limit_numb = 0
            else:
                limit_numb = 2001

            if step_count > limit_numb:

                all_dict['all_steps'].append(step_count+load_step)
                all_dict['all_betas'].append(warmup)

                for key in recent_dict.keys():
                    all_dict[key].append(np.mean(recent_dict[key]))
            
            

        if step_count%viz_steps==0:# and total_steps>6:

            self.eval()
            with torch.no_grad():
                # make batch
                # val_img_batch, val_question_batch = self.make_batch(val_image_dataset, val_question_dataset, val=True)

                try:   
                    vizualize(model, img_batch, question_batch, 
                                val_img_batch, val_question_batch, 
                                images_dir, step_count+load_step, classifier,
                                training_recon_img, training_recon_q, training_recon_sampled_words,
                                val_recon_img, val_recon_q, val_recon_sampled_words,
                                x_hat_prior, y_hat_prior, y_hat_prior_sampled_words)
                except:
                    if self.quick_check:
                        raise
                    else:
                        print ('problem with viz plotting')


                try:
                    vizualize_dependence(model,
                                val_img_batch, val_question_batch, 
                                images_dir, step_count+load_step, xs, ys, xs_1, ys_2, n_samps)
                except:
                    if self.quick_check:
                        raise
                    else:
                        print ('problem with dependence viz plotting')

                # print (
                #             'LL:{:.4f}'.format(LL),
                #             'LL_shuffle:{:.4f}'.format(LL_shuffle),
                #             'dif:{:.4f}'.format(LL-LL_shuffle),
                #             'scaled_dif:{:.4f}'.format((LL-LL_shuffle) / np.abs(LL+LL_shuffle)),
                #         )

            # print(self.optimizer)
            self.train()
        


        if step_count % trainingplot_steps==0 and step_count > 0 and len(all_dict['all_train_elbos']) > 2:

            # self.plot_curves(save_dir, all_dict)
            try:
                plot_curves(model, save_dir, all_dict)
            except:
                if self.quick_check:
                    raise
                else:
                    print ('problem with curve plotting')
                
            if self.quick_check == 1:
                fsdfaafd




        if step_count % save_steps==0 and step_count > 0:

            try:
                self.save_params_v3(save_dir=params_dir, step=step_count+load_step)

                if self.train_classifier:
                    classifier.save_params_v3(save_dir=params_dir, step=step_count+load_step)

            except:

                print ('problem with saving params')


            #Save params
            # self.save_params_v3(save_dir=params_dir, step=step_count+load_step)
            # self.load_params_v3(save_dir=params_dir, epochs=total_steps)
            # fdas




            # current_val_elbo = np.mean(recent_valid_elbos)
            # if best_val_elbo == None or best_val_elbo < current_val_elbo:
            #     best_val_elbo = current_val_elbo
            #     self.best_step = step_count+load_step

            # print (self.best_step, best_val_elbo)



            try:
                #save results
                save_to=os.path.join(save_dir, "results.pkl")
                with open(save_to, "wb" ) as f:
                    pickle.dump(all_dict, f)
                print ('saved results', save_to)

            except:

                print ('problem with saving results')

























def plot_curves(self, save_dir, all_dict):


    def make_curve_subplot(self, rows, cols, row, col, steps, values_list, label_list, ylabel, show_ticks, colspan, rowspan, set_xlabel=False, color_list=None, linestyle_list=None, title=0):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=colspan, rowspan=rowspan)

        for i in range(len(values_list)):
            if color_list is not None:
                ax.plot(steps, values_list[i], label=label_list[i], c=color_list[i], linestyle=linestyle_list[i])
            else:
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
            ax.set_title('Exp:'+self.exp_name + '  Dataset:' +str(self.image_type) + '    Joint_Inf:' + str(self.joint_inf) +
                            r'       $\lambda_x$:'+str(self.w_logpx) +  r'     $\lambda_y$:'+str(self.w_logpy)  +  r'     $\lambda_{\beta}$:'+str(self.max_beta)
                             +  r'     $\lambda_{q(z|y)}$:'+str(self.w_logqy)   +  r'     $D_z$:'+str(self.z_size), 
                            size=6, family='serif')


    rows = 13
    if self.train_prior_classifier:
        rows +=1
    cols = 2
    text_col_width = cols
    fig = plt.figure(figsize=(4+cols,4+rows), facecolor='white', dpi=150)
    
    col =0
    row = 0
    steps = all_dict['all_steps']
    

    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_train_elbos'], all_dict['all_valid_elbos']], 
        label_list=['Train: {:.2f}'.format( all_dict['all_train_elbos'][-1]), 
                    'Valid: {:.2f}'.format( all_dict['all_valid_elbos'][-1])], 
        ylabel='ELBO', show_ticks=True, colspan=text_col_width, rowspan=1, title=1)
    row+=1

    if self.joint_inf:
        training_inf_dist = r'$A(y_T,y_{\hat{x}}), q(z|x_T,y_T)$: '
        val_inf_dist = r'$A(y_V,y_{\hat{x}}), q(z|x_V,y_V)$: '
        training_inf_dist_y = r'$A(\hat{y},y_{x_T}), q(z|x_T,y_T)$: '
        val_inf_dist_y = r'$A(\hat{y},y_{x_V}), q(z|x_V,y_V)$: '
    else:
        training_inf_dist = r'$A(y_T,y_{\hat{x}}), q(z|x_T)$: '
        val_inf_dist = r'$A(y_V,y_{\hat{x}}), q(z|x_V)$: '
        training_inf_dist_y = r'$A(\hat{y},y_{x_T}), q(z|x_T)$: '
        val_inf_dist_y = r'$A(\hat{y},y_{x_V}), q(z|x_V)$: '


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_real_acc'], all_dict['all_real_val_acc'], 
                    all_dict['all_recon_acc'], all_dict['all_recon_acc_val'], 
                    all_dict['all_y_recon_acc'], all_dict['all_y_recon_acc_val'], 
                    all_dict['all_x_inf_acc'], all_dict['all_x_inf_acc_k1'], all_dict['all_x_inf_acc_entangle'], 
                    all_dict['all_y_inf_acc_k1'], all_dict['all_y_inf_acc'], all_dict['all_y_inf_acc_entangle'], 
                    all_dict['all_prior_acc'], all_dict['all_prior_acc2'],], 
        label_list=[r'$A(y_T,y_{x_T})$:   '+'{:.2f}'.format(all_dict['all_real_acc'][-1]), 
                    r'$A(y_V,y_{x_V})$:   '+'{:.2f}'.format( all_dict['all_real_val_acc'][-1]),
                    training_inf_dist +'{:.2f}'.format(all_dict['all_recon_acc'][-1]),
                    val_inf_dist +'{:.2f}'.format(all_dict['all_recon_acc_val'][-1]),
                    training_inf_dist_y+'{:.2f}'.format( all_dict['all_y_recon_acc'][-1]),
                    val_inf_dist_y+'{:.2f}'.format( all_dict['all_y_recon_acc_val'][-1]),
                    r'$p(\hat{y}_V|x_V), z \sim q(z|x_V)$: '+'{:.2f}'.format( all_dict['all_x_inf_acc'][-1]),
                    r'$p(y_V|\hat{x}_V), z \sim q(z|x_V)$: '+'{:.2f}'.format( all_dict['all_x_inf_acc_k1'][-1]),
                    r'$p(\hat{y}_V|\hat{x}_V), z \sim q(z|x_V)$: '+'{:.2f}'.format( all_dict['all_x_inf_acc_entangle'][-1]),
                    r'$p(\hat{y}_V|x_V), z \sim q(z|y_V)$: '+'{:.2f}'.format( all_dict['all_y_inf_acc_k1'][-1]),
                    r'$p(y_V|\hat{x}_V), z \sim q(z|y_V)$:'+'{:.2f}'.format( all_dict['all_y_inf_acc'][-1]),
                    r'$p(\hat{y}_V|\hat{x}_V), z \sim q(z|y_V)$: '+'{:.2f}'.format( all_dict['all_y_inf_acc_entangle'][-1]),
                    r'$p(\hat{y}|\hat{x}), z \sim p(z)$:    '+'{:.2f}'.format( all_dict['all_prior_acc'][-1]),
                    r'$p(\hat{y}|\hat{x}), z \sim N(0,.5)$:    '+'{:.2f}'.format( all_dict['all_prior_acc2'][-1]),
                    ], 
        color_list=[color_defaults[0], color_defaults[0],
                    color_defaults[1], color_defaults[1],
                    color_defaults[2], color_defaults[2],
                    color_defaults[3], color_defaults[3], color_defaults[3],
                    color_defaults[4], color_defaults[4], color_defaults[4],
                    color_defaults[5], color_defaults[6],
                    ],
        linestyle_list=['-','--',
                        '-','--',
                        '-','--',
                        '-','--',':',
                        '-','--',':',
                        '-','-',
                    ],
        ylabel='Accuracy', show_ticks=True, colspan=text_col_width, rowspan=2)
    row+=1
    row+=1


    # make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
    #     values_list=[
    #                 # all_dict['all_recon_acc'], all_dict['all_y_recon_acc'], 
    #                 # all_dict['all_recon_acc_val'], all_dict['all_y_recon_acc_val'], 
    #                 all_dict['all_x_inf_acc'], all_dict['all_x_inf_acc_entangle'],
    #                 all_dict['all_y_inf_acc'], 
    #                 # all_dict['all_y_inf_acc_k2'], 
    #                 all_dict['all_y_inf_acc_k1'], all_dict['all_y_inf_acc_entangle'], 
    #                 ], 
    #     label_list=[
    #                 # 'x recon. train current:'+'{:.2f}'.format(all_dict['all_recon_acc'][-1]),
    #                 # 'y recon. train current:'+'{:.2f}'.format( all_dict['all_y_recon_acc'][-1]),
    #                 # 'x recon. val current:'+'{:.2f}'.format(all_dict['all_recon_acc_val'][-1]),
    #                 # 'y recon. val current:'+'{:.2f}'.format( all_dict['all_y_recon_acc_val'][-1]),
    #                 r'$p(\hat{y}_V|x_V), z \sim q(z|x_V)$: '+'{:.2f}'.format( all_dict['all_x_inf_acc'][-1]),
    #                 r'$p(\hat{y}_V|\hat{x}_V), z \sim q(z|x_V)$: '+'{:.2f}'.format( all_dict['all_x_inf_acc_entangle'][-1]),
    #                 r'$p(y_V|\hat{x}_V), z \sim q(z|y_V)$:'+'{:.2f}'.format( all_dict['all_y_inf_acc'][-1]),
    #                 # 'y inf. k2 current:'+'{:.2f}'.format( all_dict['all_y_inf_acc_k2'][-1]),
    #                 r'$p(y_V|\hat{x}_V), z \sim q(z|y_V)$ k10:'+'{:.2f}'.format( all_dict['all_y_inf_acc_k1'][-1]),
    #                 r'$p(\hat{y}_V|\hat{x}_V), z \sim q(z|y_V)$: '+'{:.2f}'.format( all_dict['all_y_inf_acc_entangle'][-1]),
    #                 ], 
    #     color_list=[color_defaults[2], color_defaults[3],
    #                 color_defaults[4], color_defaults[4],
    #                 color_defaults[4], 
    #                 ],
    #     linestyle_list=['-','--',
    #                     '-','-',
    #                     '--'
    #                 ],
    #     ylabel='Accuracy', show_ticks=True, colspan=text_col_width, rowspan=2)
    # row+=1
    # row+=1

    if self.train_prior_classifier:
        make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
            values_list=[all_dict['aa_prior_classifier_acc'], all_dict['aa_prior_acc']], 
            label_list=['Classifier: {:.2f}'.format( all_dict['aa_prior_classifier_acc'][-1]),
                        'Samples: {:.2f}'.format( all_dict['aa_prior_acc'][-1])], 
            ylabel='Prior Classifier Acc', show_ticks=False, colspan=text_col_width, rowspan=1)
        row+=1



    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_logpxs'], all_dict['all_logpxs_val']], 
        label_list=['Train: {:.2f}'.format( all_dict['all_logpxs'][-1]),
                    'Valid: {:.2f}'.format( all_dict['all_logpxs_val'][-1])], 
        ylabel='logp(x|z)', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_logpys'], all_dict['all_logpys_val']], 
        label_list=['Train: {:.2f}'.format( all_dict['all_logpys'][-1]),
                    'Valid: {:.2f}'.format( all_dict['all_logpys_val'][-1])],
        ylabel='logp(y|z)', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_logpzs'], all_dict['all_logpzs_val']], 
        label_list=['Train: {:.2f}'.format( all_dict['all_logpzs'][-1]), 
                    'Valid: {:.2f}'.format( all_dict['all_logpzs_val'][-1])],
        ylabel='logp(z)', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_logqzys'], all_dict['all_logqzys_val']], 
        label_list=['Train: {:.2f}'.format( all_dict['all_logqzys'][-1]),
                    'Valid: {:.2f}'.format( all_dict['all_logqzys_val'][-1])],
        ylabel='logq(z|y)', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1

    if self.joint_inf:
        label='logq(z|x,y)'
    else:
        label='logq(z|x)'
    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_logqzs'], all_dict['all_logqzs_val']], 
        label_list=['Train: {:.2f}'.format( all_dict['all_logqzs'][-1]), 
                    'Valid: {:.2f}'.format( all_dict['all_logqzs_val'][-1])], 
        ylabel=label, show_ticks=True, colspan=text_col_width, rowspan=1)
    row+=1

    if self.joint_inf:
        label='q(z|x,y) logvar'
    else:
        label='q(z|x) logvar'
    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_logvar_max'], all_dict['all_logvar_mean'], all_dict['all_logvar_min']], 
        label_list=['max', 'mean', 'min'], ylabel=label, show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1


    # make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
    #     values_list=[smooth_list(all_dict['all_onlyx']), smooth_list(all_dict['all_onlyx_val']),
    #                    smooth_list(all_dict['all_onlyx_trainrecon']), smooth_list(all_dict['all_onlyx_prior'])], 
    #     label_list=['real train', 'real val', 'recon train', 'prior'], ylabel='logp(x)', show_ticks=False, colspan=text_col_width, rowspan=1)
    # row+=1


    # make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
    #     values_list=[smooth_list(all_dict['all_onlyy']), smooth_list(all_dict['all_onlyy_val'])], 
    #     label_list=['real train', 'real val'], ylabel='logp(y)', show_ticks=False, colspan=text_col_width, rowspan=1)
    # row+=1


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_MIs'], all_dict['all_MIs_val'],
                     all_dict['all_LL'], all_dict['all_SLL'],
                     all_dict['all_LL_val'], all_dict['all_SLL_val'],], 
        label_list=['Train: {:.2f}'.format( all_dict['all_MIs'][-1]), 
                    'Valid: {:.2f}'.format( all_dict['all_MIs_val'][-1]),
                    'LL: {:.2f}'.format( all_dict['all_LL'][-1]), 
                    'SLL: {:.2f}'.format( all_dict['all_SLL'][-1]), 
                    'LL_val: {:.2f}'.format( all_dict['all_LL_val'][-1]), 
                    'SLL_val: {:.2f}'.format( all_dict['all_SLL_val'][-1]), ], 
        ylabel='MI S', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_betas']],# all_dict['all_marginf']], 
        label_list=[r'$Beta$'], ylabel='Hyperparameters', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[smooth_list(all_dict['acc0']), smooth_list(all_dict['acc1']),
                      smooth_list(all_dict['acc2']), smooth_list(all_dict['acc3'])], 
        label_list=['size', 'colour', 'material', 'shape'], ylabel='Acc Recon', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[smooth_list(all_dict['acc0_prior']), smooth_list(all_dict['acc1_prior']),
                      smooth_list(all_dict['acc2_prior']), smooth_list(all_dict['acc3_prior'])], 
        label_list=['size', 'colour', 'material', 'shape'], ylabel='Acc Prior', show_ticks=True, colspan=text_col_width, rowspan=1, set_xlabel=True)
    row+=1


    plt_path = save_dir+'curves_plot.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()









def vizualize(self, images, questions, 
                val_img_batch, val_question_batch,
                images_dir, step_count, classifier,
                training_recon_img, training_recon_q_dist, training_recon_q_sampled_words,
                val_recon_img, val_recon_q_dist, val_recon_q_sampled_words,
                prior_img, prior_q_dist, prior_sampled_words):


    def make_bar_subplot(self, rows, cols, row, col, range_, values_, text, sampled_word):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1)
        ax.text(0.3, 1.04, text, transform=ax.transAxes, family='serif', size=6)
        ax.bar(range_, values_)
        ax.get_children()[sampled_word].set_color('black') 
        ax.set_xticks([])
        # ax.set_yticks([])
        ax.tick_params(labelsize=6)
        ax.set_ylim(bottom=0., top=1.)


    def make_text_subplot(self, rows, cols, row, col, text):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1)
        ax.text(-0.15, .2, text, transform=ax.transAxes, family='serif', size=6)
        ax.set_xticks([])
        ax.set_yticks([])


    def make_image_subplot(self, rows, cols, row, col, image, text):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False)
       
        image = image.data.cpu().numpy() * 255.
        image = np.rollaxis(image, 1, 0)
        image = np.rollaxis(image, 2, 1)# [112,112,3]
        image = np.uint8(image)
        ax.imshow(image) #, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)


    # outputs, training_recon_img, training_recon_q_dist, training_recon_q_sampled_words  = self.forward(images, questions, generate=True) #, answers)
    # outputs_val, val_recon_img, val_recon_q_dist, val_recon_q_sampled_words = self.forward(val_img_batch, val_question_batch, generate=True)#, val_answer_batch)
    # prior_img, prior_q_dist, prior_sampled_words = self.sample_prior()

    training_recon_q_dist = F.softmax(training_recon_q_dist, dim=2)
    val_recon_q_dist = F.softmax(val_recon_q_dist, dim=2)
    prior_q_dist = F.softmax(prior_q_dist, dim=2)


    if self.image_type: # if multi
        training_y_hat = classifier.classifier2(x=training_recon_img, attributes=training_recon_q_sampled_words)
        val_y_hat = classifier.classifier2(x=val_recon_img, attributes=val_recon_q_sampled_words)
        prior_y_hat = classifier.classifier2(x=prior_img, attributes=prior_sampled_words)

        training_y_hat = training_y_hat.view(self.B, self.q_max_len-1, self.vocab_size)
        training_y_hat = F.softmax(training_y_hat,2)

        val_y_hat = val_y_hat.view(self.B, self.q_max_len-1, self.vocab_size)
        val_y_hat = F.softmax(val_y_hat,2)

        prior_y_hat = prior_y_hat.view(self.B, self.q_max_len-1, self.vocab_size)
        prior_y_hat = F.softmax(prior_y_hat,2)

    else:
        training_y_hat = classifier.classifier(x=training_recon_img)
        val_y_hat = classifier.classifier(x=val_recon_img)
        prior_y_hat = classifier.classifier(x=prior_img)

        training_y_hat = training_y_hat.view(self.B, self.q_max_len, self.vocab_size)
        training_y_hat = F.softmax(training_y_hat,2)

        val_y_hat = val_y_hat.view(self.B, self.q_max_len, self.vocab_size)
        val_y_hat = F.softmax(val_y_hat,2)

        prior_y_hat = prior_y_hat.view(self.B, self.q_max_len, self.vocab_size)
        prior_y_hat = F.softmax(prior_y_hat,2)

    rows = 6

    if self.image_type:
        cols= 11
    else:
        cols=6

    text_col_width = 1

    n_vizs = 1
    for img_i in range(n_vizs):

        fig = plt.figure(figsize=(7+cols,2+rows), facecolor='white', dpi=150)

        #Row 0
        make_image_subplot(self, rows, cols, row=0, col=0, image=images[img_i], text='Training Recon')
        make_text_subplot(self, rows, cols, row=0, col=1, text='\n\n\n'+get_sentence(questions[img_i], newline_every=[3,4]))

        #Row 1
        make_image_subplot(self, rows, cols, row=1, col=0, image=training_recon_img[img_i], text='')
        sentence_classifier = get_sentence(torch.max(training_y_hat, 2)[1][img_i], newline_every=[3])
        sentence = get_sentence(training_recon_q_sampled_words[img_i], newline_every=[3,4]) 
        make_text_subplot(self, rows, cols, row=1, col=1, text=sentence+'\n\nClassifier:\n'+sentence_classifier)
        sentence = get_sentence2(training_recon_q_sampled_words[img_i])
        dist = training_recon_q_dist[img_i]
        for i in range(len(dist)):
            make_bar_subplot(self, rows, cols, row=1, col=2+i, range_=range(self.vocab_size), values_=dist[i], text=sentence[i], sampled_word=training_recon_q_sampled_words[img_i][i])

        #Row 2
        make_image_subplot(self, rows, cols, row=2, col=0, image=val_img_batch[img_i], text='Validation Recon')
        make_text_subplot(self, rows, cols, row=2, col=1, text='\n\n\n'+get_sentence(val_question_batch[img_i], newline_every=[3,4]))

        #Row 3
        make_image_subplot(self, rows, cols, row=3, col=0, image=val_recon_img[img_i], text='')
        sentence = get_sentence(val_recon_q_sampled_words[img_i], newline_every=[3,4])
        sentence_classifier = get_sentence(torch.max(val_y_hat, 2)[1][img_i], newline_every=[3])
        make_text_subplot(self, rows, cols, row=3, col=1, text=sentence+'\n\nClassifier:\n'+sentence_classifier)
        sentence = get_sentence2(val_recon_q_sampled_words[img_i])
        dist = val_recon_q_dist[img_i]
        for i in range(len(dist)):
            make_bar_subplot(self, rows, cols, row=3, col=2+i, range_=range(self.vocab_size), values_=dist[i], text=sentence[i], sampled_word=val_recon_q_sampled_words[img_i][i])

        #Row 4
        make_image_subplot(self, rows, cols, row=4, col=0, image=prior_img[img_i], text='Prior Samples')
        sentence_classifier = get_sentence(torch.max(prior_y_hat, 2)[1][img_i], newline_every=[3])
        sentence = get_sentence(prior_sampled_words[img_i], newline_every=[3,4])
        make_text_subplot(self, rows, cols, row=4, col=1, text=sentence+'\n\nClassifier:\n'+sentence_classifier)
        sentence = get_sentence2(prior_sampled_words[img_i])
        dist = prior_q_dist[img_i]
        for i in range(len(dist)):
            make_bar_subplot(self, rows, cols, row=4, col=2+i, range_=range(self.vocab_size), values_=dist[i], text=sentence[i], sampled_word=prior_sampled_words[img_i][i])

        #Row 5
        make_image_subplot(self, rows, cols, row=5, col=0, image=prior_img[img_i+1], text='')
        sentence_classifier = get_sentence(torch.max(prior_y_hat, 2)[1][img_i+1], newline_every=[3])
        sentence = get_sentence(prior_sampled_words[img_i+1], newline_every=[3,4])
        make_text_subplot(self, rows, cols, row=5, col=1, text=sentence+'\n\nClassifier:\n'+sentence_classifier)
        sentence = get_sentence2(prior_sampled_words[img_i+1])
        dist = prior_q_dist[img_i+1]
        for i in range(len(dist)):
            make_bar_subplot(self, rows, cols, row=5, col=2+i, range_=range(self.vocab_size), values_=dist[i], text=sentence[i], sampled_word=prior_sampled_words[img_i+1][i])


        # plt.tight_layout()
        plt_path = images_dir + 'img'+str(step_count)+'_'+str(img_i) +'.png'
        plt.savefig(plt_path)
        print ('saved viz',plt_path)
        plt.close(fig)
















def vizualize_dependence(self,
                val_img_batch, val_question_batch,
                images_dir, step_count, xs, ys, xs_1, ys_2, n_samps):


    def make_text_subplot(self, rows, cols, row, col, text, above_text=''):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1)
        ax.text(-0.15, .2, text, transform=ax.transAxes, family='serif', size=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.19, 1.08, above_text, transform=ax.transAxes, family='serif', size=6)


    def make_image_subplot(self, rows, cols, row, col, image, text=''):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False)
       
        image = image.data.cpu().numpy() * 255.
        image = np.rollaxis(image, 1, 0)
        image = np.rollaxis(image, 2, 1)# [112,112,3]
        image = np.uint8(image)
        ax.imshow(image) #, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.19, 1.08, text, transform=ax.transAxes, family='serif', size=6)


    rows = 6
    cols=  4 + 3
    text_col_width = 1


    n_vizs = 1
    for img_i in range(n_vizs):

        fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=150)

        for j in range(int(rows/2)):

            if j==0:
                make_image_subplot(self, rows, cols, row=j, col=0, image=val_img_batch[j], text= 'q(z|x)')
            else:
                make_image_subplot(self, rows, cols, row=j, col=0, image=val_img_batch[j])

            for samp_i in range(n_samps):
                if j==0:
                    make_text_subplot(self, rows, cols, row=j, col=1+samp_i, text=get_sentence(ys[samp_i][j], newline_every=[3,4]), above_text='p(y|z'+str(samp_i)+')')
                else:
                    make_text_subplot(self, rows, cols, row=j, col=1+samp_i, text=get_sentence(ys[samp_i][j], newline_every=[3,4]))

                if j==0:
                    make_image_subplot(self, rows, cols, row=j, col=1+samp_i+n_samps, image=xs_1[samp_i][j], text='p(x|z'+str(samp_i)+')')
                else:
                    make_image_subplot(self, rows, cols, row=j, col=1+samp_i+n_samps, image=xs_1[samp_i][j])


        for j in range(int(rows/2)):

            if j==0:
                make_text_subplot(self, rows, cols, row=int(rows/2) + j, col=0, text=get_sentence(val_question_batch[j], newline_every=[3,4]), above_text='q(z|y)')
            else:
                make_text_subplot(self, rows, cols, row=int(rows/2) + j, col=0, text=get_sentence(val_question_batch[j], newline_every=[3,4]))


            for samp_i in range(n_samps):
                if j==0:
                    make_image_subplot(self, rows, cols, row=int(rows/2)+j, col=1+samp_i, image=xs[samp_i][j], text='p(x|z'+str(samp_i)+')')
                else:
                    make_image_subplot(self, rows, cols, row=int(rows/2)+j, col=1+samp_i, image=xs[samp_i][j])

                if j==0:
                    make_text_subplot(self, rows, cols, row=int(rows/2)+j, col=1+samp_i+n_samps, text=get_sentence(ys_2[samp_i][j], newline_every=[3,4]), above_text='p(y|z'+str(samp_i)+')')
                else:
                    make_text_subplot(self, rows, cols, row=int(rows/2)+j, col=1+samp_i+n_samps, text=get_sentence(ys_2[samp_i][j], newline_every=[3,4]))


        # plt.tight_layout()
        plt_path = images_dir + 'conditional'+str(step_count)+'_'+str(img_i) +'.png'
        plt.savefig(plt_path)
        print ('saved viz',plt_path)
        plt.close(fig)
        # fdsf







































if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--which_gpu', default='0', type=str)

    parser.add_argument('--input_size', default=112, type=int)
    parser.add_argument('--x_enc_size', default=200, type=int)
    parser.add_argument('--y_enc_size', default=200, type=int)
    parser.add_argument('--z_size', default=50, type=int)

    parser.add_argument('--w_logpx', default=.05, type=float)
    parser.add_argument('--w_logpy', default=1000., type=float)
    parser.add_argument('--w_logqy', default=1., type=float)

    parser.add_argument('--max_beta', default=1., type=float)

    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--embed_size', default=100, type=int)

    parser.add_argument('--train_which_model', default='pxy', choices=['pxy', 'px', 'py'])

    parser.add_argument('--single', default=0, type=int)
    parser.add_argument('--singlev2', default=0, type=int)
    parser.add_argument('--multi', default=0, type=int)

    parser.add_argument('--flow_int', default=0, type=int)
    parser.add_argument('--joint_inf', default=0, type=int)

    parser.add_argument('--exp_name', default='test', type=str)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_to_dir', type=str, required=True)

    parser.add_argument('--just_classifier', default=0, type=int)
    parser.add_argument('--train_classifier', default=0, type=int)
    parser.add_argument('--classifier_load_params_dir', default='', type=str)
    parser.add_argument('--classifier_load_step', default=0, type=int)

    parser.add_argument('--params_load_dir', default='')
    parser.add_argument('--model_load_step', default=0, type=int)

    parser.add_argument('--quick_check', default=0, type=int)
    parser.add_argument('--quick_check_data', default='', type=str)


    parser.add_argument('--display_step', default=500, type=int)
    parser.add_argument('--trainingplot_steps', default=5000, type=int)
    parser.add_argument('--viz_steps', default=5000, type=int)
    parser.add_argument('--start_storing_data_step', default=2001, type=int)
    parser.add_argument('--save_params_step', default=50000, type=int)
    parser.add_argument('--max_steps', default=400000, type=int)

    parser.add_argument('--train_prior_classifier', default=0, type=int)

    parser.add_argument('--textAR', type=int, default=1)
    parser.add_argument('--qy_detach', type=int, default=1)
    parser.add_argument('--ssl_type', type=str, default='0')


    
    
    
    

    args = parser.parse_args()
    args_dict = vars(args) #convert to dict

    os.environ['CUDA_VISIBLE_DEVICES'] = args.which_gpu #  '0' #'1' #


    single_object = args.single
    single_object_v2 = args.singlev2
    multi_object = args.multi


    data_dir = args.data_dir 
    question_file = data_dir+'train.h5'
    image_file = data_dir+'train_images.h5'
    vocab_file = data_dir+'train_vocab.json'




    max_steps = args.max_steps
    # save_steps = 50000
    quick_check = args.quick_check

    if quick_check:
        display_step = 1#00 # 10 # 100 # 20
        trainingplot_steps = 1#000 # 4 #1000# 50 #100# 1000 # 500 #2000
        viz_steps = 3#2000 # 500# 2000
    else:
        # debug_steps = 50
        display_step = args.display_step #500 #debug_steps #500 #50# 100 # 10 # 100 # 20
        trainingplot_steps = args.trainingplot_steps # 5000 # debug_steps #2000 # 4 #1000# 50 #100# 1000 # 500 #2000
        viz_steps = args.viz_steps #5000 #debug_steps #  5000 # 500# 2000



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










    if quick_check:
        print ('loading quick stuff')
        # with open(home + "/vl_data/quick_stuff.pkl", "rb" ) as f:
        with open(args.quick_check_data, "rb" ) as f:
            stuff = pickle.load(f)
            train_image_dataset, train_question_dataset, val_image_dataset, \
                 val_question_dataset, test_image_dataset, test_question_dataset, \
                 train_indexes, val_indexes, question_idx_to_token, question_token_to_idx, q_max_len, vocab_size = stuff
    else:

        #Load data  (3,112,112)
        train_loader_kwargs = {
                                'question_h5': question_file,
                                'feature_h5': image_file,
                                'batch_size': args.batch_size,
                                # 'max_samples': 70000, i dont think this actually does anythn
                                }
        loader = ClevrDataLoader(**train_loader_kwargs)

        if single_object:
            train_image_dataset, train_question_dataset, val_image_dataset, \
                 val_question_dataset, test_image_dataset, test_question_dataset, \
                 train_indexes, val_indexes, question_idx_to_token, question_token_to_idx, q_max_len, vocab_size =  preprocess_v1(loader, vocab_file)
        elif multi_object:
            train_image_dataset, train_question_dataset, val_image_dataset, \
                 val_question_dataset, test_image_dataset, test_question_dataset, \
                 train_indexes, val_indexes, question_idx_to_token, question_token_to_idx, q_max_len, vocab_size =  preprocess_v2(loader, vocab_file)
        elif single_object_v2:
            train_image_dataset, train_question_dataset, val_image_dataset, \
                 val_question_dataset, test_image_dataset, test_question_dataset, \
                 train_indexes, val_indexes, question_idx_to_token, question_token_to_idx, q_max_len, vocab_size =  preprocess_v3(loader, vocab_file)





    print ('Training Set')
    print (train_image_dataset.shape)
    print (train_question_dataset.shape)
    print ('Validation Set')
    print (val_image_dataset.shape)
    print (val_question_dataset.shape)
    print ('Test Set')
    print (test_image_dataset.shape)
    print (test_question_dataset.shape)
    print ()


    # # save stuff for quick loading 
    # quick_n =100
    # stuff = [  train_image_dataset[:quick_n], train_question_dataset[:quick_n], val_image_dataset[:quick_n], \
    #              val_question_dataset[:quick_n], test_image_dataset[:quick_n], test_question_dataset[:quick_n], \
    #                 list(range(quick_n)), list(range(quick_n)), question_idx_to_token, question_token_to_idx, q_max_len, vocab_size]
    # with open( home + "/vl_data/quick_stuff.pkl", "wb" ) as f:
    #     pickle.dump(stuff, f)
    # print ('dumped to pickle')

    # fdsad



    # for i in range()





    if args.singlev2:

        objects = [
                    'blue', 'red', 'large', 'small', 'metal', 'sphere',
                            ]

        unpaired_xs_1_indexes = []
        unpaired_xs_2_indexes = []

        for i in range(len(train_question_dataset)):

            sentence = get_sentence2(train_question_dataset[i])

            keep = True
            for j in range(len(sentence)):

                if sentence[j] not in objects:
                    keep=False
                    break

            if keep:
                unpaired_xs_1_indexes.append(i)
            else:
                unpaired_xs_2_indexes.append(i)
            
        print ('group 1', len(unpaired_xs_1_indexes))
        print ('group 2', len(unpaired_xs_2_indexes))


        train_indexes = unpaired_xs_1_indexes




        unpaired_xs_1_indexes = []
        unpaired_xs_2_indexes = []
        for i in range(len(val_question_dataset)):

            sentence = get_sentence2(val_question_dataset[i])

            keep = True
            for j in range(len(sentence)):

                if sentence[j] not in objects:
                    keep=False
                    break

            if keep:
                unpaired_xs_1_indexes.append(i)
            else:
                unpaired_xs_2_indexes.append(i)
            
        print ('group 1', len(unpaired_xs_1_indexes))
        print ('group 2', len(unpaired_xs_2_indexes))


        val_indexes = unpaired_xs_1_indexes

        print (get_sentence(val_question_dataset[val_indexes[0]]))
    
    # afsdsdf

    for i in range(10):

        print (i, get_sentence(train_question_dataset[train_indexes[i]]))
    # fsddfs



















    print ('\nInit VLVAE')
    args_dict['image_type'] = multi_object
    args_dict['vocab_size'] = vocab_size
    args_dict['q_max_len'] = q_max_len
    model = VLVAE(args_dict)
    model.cuda()
    if args.model_load_step>0:
        model.load_params_v3(save_dir=args.params_load_dir, step=args.model_load_step)
    # print(model)
    # fdsa
    print ('VLVAE Initilized\n')
    




    print ('Init classifier')
    if single_object or single_object_v2:
        classifier = attribute_classifier(args_dict)
    elif multi_object:
        classifier = attribute_classifier_with_relations(args_dict) #, encoder_embed=model.encoder_embed)
    classifier.cuda()
    classifier.train()
    if args.classifier_load_step > 0:
        classifier.load_params_v3(save_dir=args.classifier_load_params_dir, step=args.classifier_load_step)
        classifier.eval()
    print ('classifier Initilized\n')



    if args.train_prior_classifier:

        print ('Init prior classifier')
        if single_object or single_object_v2:
            prior_classifier = attribute_classifier(args_dict)
        elif multi_object:
            prior_classifier = attribute_classifier_with_relations(args_dict) #, encoder_embed=model.encoder_embed)
        prior_classifier.cuda()
        prior_classifier.train()
        # if args.classifier_load_step > 0:
        #     prior_classifier.load_params_v3(save_dir=args.classifier_load_params_dir, step=args.classifier_load_step)
        #     prior_classifier.eval()
        print ('Prior classifier Initilized\n')
    else:
        prior_classifier = None








    print ('Running')
    train2(self=model, max_steps=max_steps, load_step=args.model_load_step,
            train_image_dataset=train_image_dataset, train_question_dataset=train_question_dataset,
            val_image_dataset=val_image_dataset, val_question_dataset=val_question_dataset,
            save_dir=exp_dir, params_dir=params_dir, images_dir=images_dir, batch_size=args.batch_size,
            display_step=display_step, save_steps=args.save_params_step, trainingplot_steps=trainingplot_steps, viz_steps=viz_steps,
            classifier=classifier, start_storing_data_step=args.start_storing_data_step, prior_classifier=prior_classifier) 
    model.save_params_v3(save_dir=params_dir, step=max_steps+args.model_load_step)

    print ('Done.')



# Example:

#python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 0 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --flow 1  --exp_name flow_qy --quick_check 1 --model_load_step 100000

# python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --z_size 20 --exp_name gaus_qy

# python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --z_size 20 --quick_check 1 

# python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --quick_check 1 --flow 1 --z_size 20

# python3 train_jointVAE_v21.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 20 --w_logpx .01 --joint_inf 0 --quick_check 1


