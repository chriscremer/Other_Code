



import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../VLVAE'))

from os.path import expanduser
home = expanduser("~")

import numpy as np
import math
import pickle
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

# try:
#     import h5py
# except Warning:
#     pass 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image

# from vr.data import ClevrDataset, ClevrDataLoader
# from Clevr_data_loader import ClevrDataset, ClevrDataLoader
# from cyclegangenerator_bottleneck import *
# from classifier import attribute_classifier, attribute_classifier_with_relations, classifier_of_prior
# from textRNN import TextRNN
# from priorRNN import priorRNN
# from distributions import Normal, Flow1#, IAF_flow
# from preprocess_statements import *
# from qRNN import qRNN


from VLVAE_model import VLVAE

from celeba_classifier import attribute_classifier

# def lognormal(x, mean, logvar):
#     # x: [P,B,Z]
#     # mean, logvar: [B,Z]
#     # return: [P,B]
#     # assert x.shape[1] == mean.shape[0]
#     return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(2))


def logmeanexp(elbo):
    # [P,B]
    max_ = torch.max(elbo, 0)[0] #[B]
    elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
    return elbo


# # from attributes
# def get_sentence(list_of_word_idxs, newline_every=[]): #, answer):
#     sentence =''
#     list_of_word_idxs = list_of_word_idxs.cpu().numpy()#[0]
#     for i in range(len(list_of_word_idxs)):
#         word = question_idx_to_token[int(list_of_word_idxs[i])]
#         sentence += ' ' + word
#         if i in newline_every:
#             sentence += '\n'

#     return sentence


# #returns list instead of string
# def get_sentence2(list_of_word_idxs): #, answer):
#     sentence = []
#     list_of_word_idxs = list_of_word_idxs.cpu().numpy()#[0]
#     for i in range(len(list_of_word_idxs)):
#         word = question_idx_to_token[int(list_of_word_idxs[i])]
#         sentence.append(word) 
#     return sentence



# def smooth_list(x, window_len=5, window='flat'):
#     if len(x) < window_len:
#         return x
#     w = np.ones(window_len,'d') 
#     y = np.convolve(w/ w.sum(), x, mode='same')
#     return y




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























def compute_log_like2(model, image_dataset, question_dataset, n_batches, indexes, inf_type=1, dec_type=0, k=1): #, k=1): #, answer_dataset, image_idxs, first_question_idx):

    batch_size = 20
    # n_batches = 11
    LLs = []
    # LLs_shuffled = []

    model.eval()
    with torch.no_grad():

        for i in range(n_batches):

            img_batch, question_batch = make_batch(image_dataset, question_dataset, batch_size, indexes=indexes)#, 
            # shuffled_image = img_batch[torch.randperm(img_batch.shape[0])]

            logws = []
            # logws_S = []
            for j in range(k):

                # outputs_shuf = model.forward(shuffled_image, question_batch, inf_type=inf_type, dec_type=0) #, k=k) #, answer_batch)
                outputs = model.forward(x=None, q=question_batch, inf_type=inf_type, dec_type=dec_type) #, k=k) #, answer_batch)
                logws.append(outputs['logws'])

            logws = torch.stack(logws, 0) #[k,B]
            # print (logws.shape)
            # fdfa
            if k ==1:
                LLs.append( torch.mean(logws).data.item())
            else:
                LLs.append( torch.mean(logmeanexp(logws)).data.item())
             # outputs['elbo'].data.item())
                # LLs_shuffled.append(outputs_shuf['elbo'].data.item())

    model.train()


    # fsdafa


    LL = np.mean(LLs)
    # LL_shuffle = np.mean(LLs_shuffled)

    return LL#, LL_shuffle




def compute_log_like(model, image_dataset, question_dataset, n_batches, indexes, inf_type=1): #, k=1): #, answer_dataset, image_idxs, first_question_idx):

    batch_size = 20
    # n_batches = 11
    LLs = []
    LLs_shuffled = []

    model.eval()
    with torch.no_grad():

        for i in range(n_batches):

            img_batch, question_batch = make_batch(image_dataset, question_dataset, batch_size, indexes=indexes)#, 
            shuffled_image = img_batch[torch.randperm(img_batch.shape[0])]

            outputs_shuf = model.forward(shuffled_image, question_batch, inf_type=inf_type, dec_type=0) #, k=k) #, answer_batch)
            outputs = model.forward(img_batch, question_batch, inf_type=inf_type, dec_type=0) #, k=k) #, answer_batch)

            LLs.append(outputs['elbo'].data.item())
            LLs_shuffled.append(outputs_shuf['elbo'].data.item())

    model.train()


    LL = np.mean(LLs)
    LL_shuffle = np.mean(LLs_shuffled)

    return LL, LL_shuffle


def isnan(x):
    return x != x





















x1, y1 = 48, 25
downsample = torch.nn.AvgPool2d(2,2,0)



def make_batch(image_dir, attr_dict, batch_size, indexes, word_to_idx): # val=False, SSL=False): #, answer_dataset): #, image_idxs, first_question_idx):


    img_batch = []
    text_batch = []
    for i in range(batch_size):
        idx = random.choice(indexes)


        numb = str(idx) 
        while len(numb) != 6:
            numb = '0' + numb

        img_file = image_dir+ numb + '.jpg'

        jpgfile = Image.open(img_file)
        img = np.array(jpgfile)

        img = np.rollaxis(img, 2, 1)# [112,112,3]
        img = np.rollaxis(img, 1, 0)
        img = img / 255.

        cropped_image = img[:, x1:x1+128, y1:y1+128]
        cropped_image = torch.Tensor(cropped_image)
        downsampled_image = downsample(cropped_image)
        img = downsampled_image

        # print (np.max(img), np.min(img))
        # print (img.shape)
        img_batch.append(img)


        word_idxs = get_text_idxs(attr_dict[numb+'.jpg'], word_to_idx)
        # print (word_idxs)
        random.shuffle(word_idxs)
        # print (word_idxs)

        while len(word_idxs) < 9:
            word_idxs.append(0)

        text_batch.append(word_idxs)


    img_batch = np.stack(img_batch) #[B,C,W,H]
    text_batch = np.stack(text_batch) #, 1) #[T,B,L]

    # print (text_batch.shape)
    # fas

    img_batch = torch.from_numpy(img_batch).cuda()
    text_batch = torch.from_numpy(text_batch).cuda()
    # img_batch = rescaling(img_batch)
    img_batch = torch.clamp(img_batch, min=1e-5, max=1-1e-5)

    # print (img_batch.shape, text_batch.shape)
    # sdfdafs

    return img_batch, text_batch #, answer_batch
















def train2(self, max_steps, load_step, 
                image_dir, train_indexes, val_indexes, attr_dict, word_to_idx,
                save_dir, params_dir, images_dir, batch_size,
                display_step, save_steps, trainingplot_steps, viz_steps,
                classifier, start_storing_data_step): #, k): #, prior_classifier):

    # random.seed( 99 )
    random.seed( 0 )

    all_dict = {}
    recent_dict = {}

    list_recorded_values = [ 
            'all_train_elbos', 'all_valid_elbos', 
            'all_logpxs', 'all_logpxs_val', 'all_logpys', 'all_logpys_val',
            'all_logpzs', 'all_logqzs', 'all_logqzys',
            'all_logpzs_val', 'all_logqzs_val', 'all_logqzys_val',
            # 'all_MIs', 'all_MIs_val', 'all_LL', 'all_SLL', 'all_LL_val', 'all_SLL_val', 
            'all_real_acc', 'all_real_val_acc', 'all_recon_acc', 'all_y_recon_acc', 
            'all_recon_acc_val', 'all_y_recon_acc_val', 
            'all_prior_acc', 'all_prior_acc2', 
            'all_x_inf_acc', 'all_y_inf_acc', 'all_x_inf_acc_entangle', 'all_y_inf_acc_entangle',
            'all_y_inf_acc_k1', 'all_x_inf_acc_k1', 
            # 'all_logvar_max', 'all_logvar_min', 'all_logvar_mean',
            # 'acc0', 'acc1', 'acc2', 'acc3',
            # 'acc0_prior', 'acc1_prior', 'acc2_prior', 'acc3_prior',
            # 'SSL1', 'SSL2',
            # 'SSL1_k1', 'SSL2_k1',
            # 'SSL1_k100', 'SSL2_k100',
    ]

    all_dict['all_steps'] = []
    all_dict['all_betas'] = []

    for label in list_recorded_values:
        all_dict[label] = []
        recent_dict[label] = deque(maxlen=10)

    def add_to_dict(dict1, label, value):
        dict1[label].append(value)


    self.train()

    self.B = batch_size

    start_time = time.time()
    step_count = 0
    warmup_steps = 20000. 

    if self.joint_inf:
        inf_type = 0
    else:
        inf_type = 1

    for step in range(max_steps):

        # Make batch
        if not self.just_classifier or self.train_classifier:
            img_batch, question_batch = make_batch(image_dir, attr_dict, batch_size, indexes=train_indexes, word_to_idx=word_to_idx)

        warmup = min((step_count+load_step) / float(warmup_steps), self.max_beta)


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
                outputs_only_x = self.forward(x=img_batch, q=question_batch, warmup=warmup, inf_type=1, dec_type=0) #, k=k_train)
                loss_only_x = -outputs_only_x['welbo']
                loss_only_x.backward()
                self.optimizer_x.step()
                self.scheduler_x.step()

        if self.train_classifier:
            classi_loss, acc = classifier.update(x=img_batch, q=question_batch)

        



        step_count += 1

        if step_count % display_step == 0:

            if self.joint_inf:
                inf_type = 0
            else:
                inf_type = 1

            self.eval()
            with torch.no_grad():
                # make validation and prior batch
                img_batch, question_batch = make_batch(image_dir, attr_dict, batch_size, indexes=train_indexes, word_to_idx=word_to_idx)
                val_img_batch, val_question_batch = make_batch(image_dir, attr_dict, batch_size, indexes=val_indexes, word_to_idx=word_to_idx)
                x_hat_prior, y_hat_prior, y_hat_prior_sampled_words = self.sample_prior()
                x_hat_prior2, y_hat_prior2, y_hat_prior_sampled_words2 = self.sample_prior(std=.5)

                outputs, training_recon_img, training_recon_q, training_recon_sampled_words = self.forward(img_batch, question_batch, generate=True, inf_type=inf_type, dec_type=0, warmup=warmup) #, k=k_gen)
                val_outputs, val_recon_img, val_recon_q, val_recon_sampled_words = self.forward(val_img_batch, val_question_batch, generate=True, inf_type=inf_type, dec_type=0) #, k=k_gen)

                # #Accuracies 
                y_recon_acc = classifier.accuracy_sequence(x=img_batch, attributes=training_recon_sampled_words.detach())
                # print (y_recon_acc)
                # fassd
                y_recon_acc_val = classifier.accuracy_sequence(x=val_img_batch, attributes=val_recon_sampled_words.detach())

                prior_acc = classifier.accuracy_sequence(x=x_hat_prior.detach(), attributes=y_hat_prior_sampled_words.detach())
                # acc0_prior, acc1_prior, acc2_prior, acc3_prior = classifier.classifier_attribute_accuracies(x=x_hat_prior.detach(), attributes=y_hat_prior_sampled_words.detach())
                
                prior_acc2 = classifier.accuracy_sequence(x=x_hat_prior2.detach(), attributes=y_hat_prior_sampled_words2.detach())


                # acc0, acc1, acc2, acc3 = classifier.classifier_attribute_accuracies(x=training_recon_img.detach(), attributes=question_batch)
                recon_acc = classifier.accuracy_sequence(x=training_recon_img.detach(), attributes=question_batch)



                if not self.train_classifier: #else im already computing this
                    acc = classifier.accuracy_sequence(x=img_batch, attributes=question_batch)

                val_acc = classifier.accuracy_sequence(val_img_batch, val_question_batch)
                val_recon_acc = classifier.accuracy_sequence(val_recon_img.detach(), val_question_batch)


                


                # Conditional  # q(z|x)
                n_samps = 3 # for viewing
                ys = []
                xs_1 = []
                for i in range(n_samps):
                    outputs_cond, recon_img_cond, recon_q_cond, recon_sampled_words_cond = self.forward(x=val_img_batch, q=None, generate=True, inf_type=1, dec_type=1) #, k=k_cond)
                    ys.append(recon_sampled_words_cond)
                    xs_1.append(recon_img_cond)

                x_inf_acc = classifier.accuracy_sequence(x=val_img_batch, attributes=ys[0].detach())
                x_inf_acc_k1 = classifier.accuracy_sequence(x=xs_1[0].detach(), attributes=val_question_batch)
                x_inf_acc_entangle = classifier.accuracy_sequence(x=xs_1[0].detach(), attributes=ys[0].detach())



                # q(z|y)
                xs = []
                ys_2 = []
                for i in range(n_samps):
                    outputs_cond, recon_img_cond, recon_q_cond, recon_sampled_words_cond = self.forward(x=None, q=val_question_batch, generate=True, inf_type=2, dec_type=2) #, k=k_cond)         
                    xs.append(recon_img_cond)
                    ys_2.append(recon_sampled_words_cond)
                

                y_inf_acc = classifier.accuracy_sequence(x=xs[0].detach(), attributes=val_question_batch)
                y_inf_acc_k1 = classifier.accuracy_sequence(x=val_img_batch, attributes=ys_2[0].detach())
                y_inf_acc_entangle = classifier.accuracy_sequence(x=xs[0].detach(), attributes=ys_2[0].detach())


                # LL1, LL_shuffle1 = compute_log_like(model, train_image_dataset, train_question_dataset, n_batches=1, inf_type=1, indexes=train_indexes)
                # LL2, LL_shuffle2 = compute_log_like(model, train_image_dataset, train_question_dataset, n_batches=1, inf_type=2, indexes=train_indexes)
                # if self.joint_inf:
                #     LL0, LL_shuffle0 = compute_log_like(model, train_image_dataset, train_question_dataset, n_batches=1, inf_type=0, indexes=train_indexes)

                # LL_val1, LL_shuffle_val1 = compute_log_like(model, val_image_dataset, val_question_dataset, n_batches=1, inf_type=1, indexes=val_indexes)
                # LL_val2, LL_shuffle_val2 = compute_log_like(model, val_image_dataset, val_question_dataset, n_batches=1, inf_type=2, indexes=val_indexes)
                # if self.joint_inf:
                #     LL_val0, LL_shuffle_val0 = compute_log_like(model, val_image_dataset, val_question_dataset, n_batches=1, inf_type=0, indexes=val_indexes)

                # # print (LL1, LL2, LL0)
                # # print (LL_shuffle1, LL_shuffle2, LL_shuffle0)
                # if self.joint_inf:
                #     LL = np.max([LL1, LL2, LL0])
                #     LL_shuffle = np.max([LL_shuffle1, LL_shuffle2, LL_shuffle0])
                #     LL_val = np.max([LL_val1, LL_val2, LL_val0])
                #     LL_shuffle_val = np.max([LL_shuffle_val1, LL_shuffle_val2, LL_shuffle_val0])
                # else:
                #     LL = np.max([LL1, LL2])
                #     LL_shuffle = np.max([LL_shuffle1, LL_shuffle2])
                #     LL_val = np.max([LL_val1, LL_val2])
                #     LL_shuffle_val = np.max([LL_shuffle_val1, LL_shuffle_val2])
                # # print (LL)
                # # fdsa

                # # py of unpaired groups
                # SSL_LL1_k1 = compute_log_like2(model, train_image_dataset, train_question_dataset, n_batches=1, inf_type=2, dec_type=2, indexes=unpaired_xs_1_indexes, k=1)
                # SSL_LL2_k1 = compute_log_like2(model, train_image_dataset, train_question_dataset, n_batches=1, inf_type=2, dec_type=2, indexes=unpaired_xs_2_indexes, k=1)

                # SSL_LL1_k100 = compute_log_like2(model, train_image_dataset, train_question_dataset, n_batches=1, inf_type=2, dec_type=2, indexes=unpaired_xs_1_indexes, k=100)
                # SSL_LL2_k100 = compute_log_like2(model, train_image_dataset, train_question_dataset, n_batches=1, inf_type=2, dec_type=2, indexes=unpaired_xs_2_indexes, k=100)

                # self.eval()
                # SSL_img_batch, SSL_question_batch = make_batch(train_image_dataset, train_question_dataset, batch_size, indexes=unpaired_xs_1_indexes)
                # SSL_outputs1 = self.forward(x=None, q=SSL_question_batch, inf_type=2, dec_type=2) 

                # SSL_img_batch, SSL_question_batch = make_batch(train_image_dataset, train_question_dataset, batch_size, indexes=unpaired_xs_2_indexes)
                # SSL_outputs2 = self.forward(x=None, q=SSL_question_batch, inf_type=2, dec_type=2) 

                # # print (SSL_outputs1['elbo'])
                # # print (SSL_outputs2['elbo'])
                # # fadsf








                
                    
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

            # add_to_dict(recent_dict, 'all_MIs', LL-LL_shuffle)
            # add_to_dict(recent_dict, 'all_MIs_val', LL_val-LL_shuffle_val)
            # add_to_dict(recent_dict, 'all_LL', LL)
            # add_to_dict(recent_dict, 'all_SLL', LL_shuffle)
            # add_to_dict(recent_dict, 'all_LL_val', LL_val)
            # add_to_dict(recent_dict, 'all_SLL_val', LL_shuffle_val)

            add_to_dict(recent_dict, 'all_real_acc', acc)
            add_to_dict(recent_dict, 'all_real_val_acc', val_acc)
            add_to_dict(recent_dict, 'all_recon_acc', recon_acc)
            add_to_dict(recent_dict, 'all_y_recon_acc', y_recon_acc)
            add_to_dict(recent_dict, 'all_recon_acc_val', val_recon_acc)
            add_to_dict(recent_dict, 'all_y_recon_acc_val', y_recon_acc_val)
            add_to_dict(recent_dict, 'all_prior_acc', prior_acc)
            add_to_dict(recent_dict, 'all_prior_acc2', prior_acc2)

            add_to_dict(recent_dict, 'all_x_inf_acc', x_inf_acc)
            add_to_dict(recent_dict, 'all_y_inf_acc', y_inf_acc)
            add_to_dict(recent_dict, 'all_x_inf_acc_entangle', x_inf_acc_entangle)
            add_to_dict(recent_dict, 'all_y_inf_acc_entangle', y_inf_acc_entangle)

            add_to_dict(recent_dict, 'all_y_inf_acc_k1', y_inf_acc_k1)
            add_to_dict(recent_dict, 'all_x_inf_acc_k1', x_inf_acc_k1)

            # add_to_dict(recent_dict, 'all_logvar_max', np.max(outputs['logvar'].data.cpu().numpy()))
            # add_to_dict(recent_dict, 'all_logvar_min', np.min(outputs['logvar'].data.cpu().numpy()))
            # add_to_dict(recent_dict, 'all_logvar_mean', np.mean(outputs['logvar'].data.cpu().numpy()))

            # add_to_dict(recent_dict, 'acc0', acc0)
            # add_to_dict(recent_dict, 'acc1', acc1)
            # add_to_dict(recent_dict, 'acc2', acc2)
            # add_to_dict(recent_dict, 'acc3', acc3)

            # add_to_dict(recent_dict, 'acc0_prior', acc0_prior)
            # add_to_dict(recent_dict, 'acc1_prior', acc1_prior)
            # add_to_dict(recent_dict, 'acc2_prior', acc2_prior)
            # add_to_dict(recent_dict, 'acc3_prior', acc3_prior)

            # add_to_dict(recent_dict, 'SSL1', SSL_outputs1['elbo'].data.item())
            # add_to_dict(recent_dict, 'SSL2', SSL_outputs2['elbo'].data.item())
            # add_to_dict(recent_dict, 'SSL1_k1', SSL_LL1_k1)
            # add_to_dict(recent_dict, 'SSL2_k1', SSL_LL2_k1)
            # add_to_dict(recent_dict, 'SSL1_k100', SSL_LL1_k100)
            # add_to_dict(recent_dict, 'SSL2_k100', SSL_LL2_k100)



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
                'warmup:{:.4f}'.format(warmup),
                # 'prioracc:{:.4f}'.format(recent_dict['all_prior_acc'][-1]),
                # 'SSL_elbo:{:.4f}'.format(SSL_outputs1['elbo'].data.item()),
                # 'SSL_pyz:{:.4f}'.format(SSL_outputs1['logpy'].data.item()),
                # 'SSL_pz:{:.4f}'.format(SSL_outputs1['logpz'].data.item()),
                # 'SSL_qz:{:.4f}'.format(SSL_outputs1['logqz'].data.item()),
                # 'SSL_elbo2:{:.4f}'.format(SSL_outputs2['elbo'].data.item()),
                # 'SSL_pyz2:{:.4f}'.format(SSL_outputs2['logpy'].data.item()),
                # 'SSL_pz2:{:.4f}'.format(SSL_outputs2['logpz'].data.item()),
                # 'SSL_qz2:{:.4f}'.format(SSL_outputs2['logqz'].data.item()),
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
                limit_numb = start_storing_data_step

            if step_count > limit_numb:

                all_dict['all_steps'].append(step_count+load_step)
                all_dict['all_betas'].append(warmup)

                for key in recent_dict.keys():
                    # if len(recent_dict[key]) ==0:
                    #     print (key)
                    #     fdas
                    all_dict[key].append(np.mean(recent_dict[key]))

            
            

        if step_count%viz_steps==0:# and total_steps>6:

            self.eval()
            with torch.no_grad():
                #     # make batch
                # val_img_batch, val_question_batch = self.make_batch(val_image_dataset, val_question_dataset, val=True)


                my_list = [2,5,3,4,6]


                # Looking at entropy with increasing info 
                all_atts = []
                all_ents = []
                all_images = []
                atts = []
                for n in range(5):
                    if n > len(atts):
                        #sample word
                        # atts.append(random.choice(list(range(1,20))))

                        atts.append(my_list[n-1])

                    tmp_atts = atts.copy()
                    while len(tmp_atts) < 9:
                        tmp_atts.append(0)

                    tmp_atts = np.array(tmp_atts)
                    tmp_atts = torch.from_numpy(tmp_atts).view(1,9).cuda()


                    entropy, image_recons = self.get_entropy_and_samples(q=tmp_atts, inf_type=2, dec_type=1, k_entropy=100, k_recons=3)

                    # print (tmp_atts.)
                    sentence  = get_sentence2(tmp_atts[0])

                    # sentence  = get_sentence(tmp_atts[0])

                    # print (image_recons[0].shape)

                    # fsda

                    # print (sentence, entropy)
                    # fasf
                    all_atts.append(tmp_atts[0])
                    all_ents.append(entropy.data.item())
                    all_images.append(image_recons)

                viz_entropy(model, images_dir, all_atts, all_ents, all_images)






                vizualize(model, img_batch, question_batch, 
                                    val_img_batch, val_question_batch, 
                                    images_dir, step_count+load_step, classifier,
                                    training_recon_img, training_recon_q, training_recon_sampled_words,
                                    val_recon_img, val_recon_q, val_recon_sampled_words,
                                    x_hat_prior, y_hat_prior, y_hat_prior_sampled_words)


                #     # try:   
                #     #     vizualize(model, img_batch, question_batch, 
                #     #                 val_img_batch, val_question_batch, 
                #     #                 images_dir, step_count+load_step, classifier,
                #     #                 training_recon_img, training_recon_q, training_recon_sampled_words,
                #     #                 val_recon_img, val_recon_q, val_recon_sampled_words,
                #     #                 x_hat_prior, y_hat_prior, y_hat_prior_sampled_words)
                #     # except:
                #     #     if self.quick_check:
                #     #         raise
                #     #     else:
                #     #         print ('problem with viz plotting')

                vizualize_dependence(model,
                                val_img_batch, val_question_batch, 
                                images_dir, step_count+load_step, xs, ys, xs_1, ys_2, n_samps)


                # try:
                #     vizualize_dependence(model,
                #                 val_img_batch, val_question_batch, 
                #                 images_dir, step_count+load_step, xs, ys, xs_1, ys_2, n_samps)
                # except:
                #     if self.quick_check:
                #         raise
                #     else:
                #         print ('problem with dependence viz plotting')

                # print (
                #             'LL:{:.4f}'.format(LL),
                #             'LL_shuffle:{:.4f}'.format(LL_shuffle),
                #             'dif:{:.4f}'.format(LL-LL_shuffle),
                #             'scaled_dif:{:.4f}'.format((LL-LL_shuffle) / np.abs(LL+LL_shuffle)),
                #         )
                # fsas


            # print(self.optimizer)
            self.train()
        


        if step_count % trainingplot_steps==0 and step_count > 0 and len(all_dict['all_train_elbos']) > 2:

            plot_curves(model, save_dir, all_dict)
            

            # try:
            #     plot_curves(model, save_dir, all_dict)
            # except:
            #     if self.quick_check:
            #         raise
            #     else:
            #         print ('problem with curve plotting')
                
            # if self.quick_check == 1:
            #     fsdfaafd




        if step_count % save_steps==0 and step_count > 0:
            #Save params
            self.save_params_v3(save_dir=params_dir, step=step_count+load_step)
            # self.load_params_v3(save_dir=params_dir, epochs=total_steps)
            # fdas

            if self.train_classifier:
                classifier.save_params_v3(save_dir=params_dir, step=step_count+load_step)


            # current_val_elbo = np.mean(recent_valid_elbos)
            # if best_val_elbo == None or best_val_elbo < current_val_elbo:
            #     best_val_elbo = current_val_elbo
            #     self.best_step = step_count+load_step

            # print (self.best_step, best_val_elbo)



            #save results
            save_to=os.path.join(save_dir, "results.pkl")
            with open(save_to, "wb" ) as f:
                pickle.dump(all_dict, f)
            print ('saved results', save_to)






















def viz_entropy(self, images_dir, atts, entropy, images):


    def make_bar_subplot(self, rows, cols, row, col, range_, values_, text, sampled_word):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1)
        ax.text(0.3, 1.04, text, transform=ax.transAxes, family='serif', size=6)
        ax.bar(range_, values_)
        ax.get_children()[sampled_word].set_color('black') 
        ax.set_xticks([])
        # ax.set_yticks([])
        ax.tick_params(labelsize=6)
        ax.set_ylim(bottom=0., top=1.)


    def make_text_subplot(self, rows, cols, row, col, text, ent):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1)
        ax.text(0.15, .09, text, transform=ax.transAxes, family='serif', size=6)
        ax.text(.3, .2, '{:.2f}'.format(ent), transform=ax.transAxes, family='serif', size=6)
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


    rows = 5
    cols= len(atts)

    text_col_width = 1


    newline_every = list(range(10))

    # n_vizs = 1
    # for img_i in range(n_vizs):

    fig = plt.figure(figsize=(7+cols,2+rows), facecolor='white', dpi=150)

    for col in range(cols):


        #Row 0
        # make_image_subplot(self, rows, cols, row=0, col=0, image=images[img_i], text='Training Recon')
        make_text_subplot(self, rows, cols, row=0, col=col, text='\n\n\n'+get_sentence(atts[col], newline_every=newline_every), ent=entropy[col])


        make_image_subplot(self, rows, cols, row=1, col=col, image=images[col][0], text='')
        make_image_subplot(self, rows, cols, row=2, col=col, image=images[col][1], text='')
        make_image_subplot(self, rows, cols, row=3, col=col, image=images[col][2], text='')


        # #Row 1
        # 
        # # sentence_classifier = get_sentence(torch.max(training_y_hat, 2)[1][img_i], newline_every=[3])
        # sentence = get_sentence(training_recon_q_sampled_words[img_i], newline_every=newline_every) 
        # make_text_subplot(self, rows, cols, row=1, col=1, text=sentence) #+'\n\nClassifier:\n'+sentence_classifier)
        # sentence = get_sentence2(training_recon_q_sampled_words[img_i])
        # dist = training_recon_q_dist[img_i]
        # for i in range(len(dist)):
        #     make_bar_subplot(self, rows, cols, row=1, col=2+i, range_=range(self.vocab_size), values_=dist[i], text=sentence[i], sampled_word=training_recon_q_sampled_words[img_i][i])





    plt_path = images_dir+'viz_entropy.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()
















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
            ax.set_title('Exp:'+self.exp_name + '    Joint_Inf:' + str(self.joint_inf) +
                            r'       $\lambda_x$:'+str(self.w_logpx) +  r'     $\lambda_y$:'+str(self.w_logpy)  +  r'     $\lambda_{\beta}$:'+str(self.max_beta)
                             +  r'     $\lambda_{q(z|y)}$:'+str(self.w_logqy)   +  r'     $D_z$:'+str(self.z_size), 
                            size=6, family='serif')


    rows = 9
    cols = 2
    text_col_width = cols
    fig = plt.figure(figsize=(4+cols,4+rows), facecolor='white', dpi=150)
    
    col =0
    row = 0
    steps = all_dict['all_steps']
    



    # make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
    #     values_list=[
    #                 # all_dict['SSL1'], all_dict['SSL2'], 
    #                 all_dict['SSL1_k1'], all_dict['SSL2_k1'], 
    #                 all_dict['SSL1_k100'], all_dict['SSL2_k100']], 
    #     label_list=[
    #                 # 'SSL1: {:.2f}'.format( all_dict['SSL1'][-1]),
    #                 # 'SSL2: {:.2f}'.format( all_dict['SSL2'][-1]),
    #                 'SSL1_k1: {:.2f}'.format( all_dict['SSL1_k1'][-1]),
    #                 'SSL2_k1: {:.2f}'.format( all_dict['SSL2_k1'][-1]),
    #                 'SSL1_k100: {:.2f}'.format( all_dict['SSL1_k100'][-1]),
    #                 'SSL2_k100: {:.2f}'.format( all_dict['SSL2_k100'][-1]),], 
    #     ylabel='weighted logp(y) lower bound', show_ticks=False, colspan=text_col_width, rowspan=1, title=1)
    # row+=1



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


    # if self.joint_inf:
    #     label='q(z|x,y) logvar'
    # else:
    #     label='q(z|x) logvar'
    # make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
    #     values_list=[all_dict['all_logvar_max'], all_dict['all_logvar_mean'], all_dict['all_logvar_min']], 
    #     label_list=['max', 'mean', 'min'], ylabel=label, show_ticks=False, colspan=text_col_width, rowspan=1)
    # row+=1


    # make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
    #     values_list=[smooth_list(all_dict['all_onlyx']), smooth_list(all_dict['all_onlyx_val']),
    #                    smooth_list(all_dict['all_onlyx_trainrecon']), smooth_list(all_dict['all_onlyx_prior'])], 
    #     label_list=['real train', 'real val', 'recon train', 'prior'], ylabel='logp(x)', show_ticks=False, colspan=text_col_width, rowspan=1)
    # row+=1


    # make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
    #     values_list=[smooth_list(all_dict['all_onlyy']), smooth_list(all_dict['all_onlyy_val'])], 
    #     label_list=['real train', 'real val'], ylabel='logp(y)', show_ticks=False, colspan=text_col_width, rowspan=1)
    # row+=1


    # make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
    #     values_list=[all_dict['all_MIs'], all_dict['all_MIs_val'],
    #                  all_dict['all_LL'], all_dict['all_SLL'],
    #                  all_dict['all_LL_val'], all_dict['all_SLL_val'],], 
    #     label_list=['Train: {:.2f}'.format( all_dict['all_MIs'][-1]), 
    #                 'Valid: {:.2f}'.format( all_dict['all_MIs_val'][-1]),
    #                 'LL: {:.2f}'.format( all_dict['all_LL'][-1]), 
    #                 'SLL: {:.2f}'.format( all_dict['all_SLL'][-1]), 
    #                 'LL_val: {:.2f}'.format( all_dict['all_LL_val'][-1]), 
    #                 'SLL_val: {:.2f}'.format( all_dict['all_SLL_val'][-1]), ], 
    #     ylabel='MI S', show_ticks=False, colspan=text_col_width, rowspan=1)
    # row+=1


    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['all_betas']],# all_dict['all_marginf']], 
        label_list=[r'$Beta$'], ylabel='Hyperparameters', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1


    # make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
    #     values_list=[all_dict['acc0'], all_dict['acc1'],
    #                   all_dict['acc2'], all_dict['acc3']], 
    #     label_list=['size', 'colour', 'material', 'shape'], ylabel='Acc Recon', show_ticks=False, colspan=text_col_width, rowspan=1)
    # row+=1


    # make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
    #     values_list=[all_dict['acc0_prior'], all_dict['acc1_prior'],
    #                   all_dict['acc2_prior'], all_dict['acc3_prior']], 
    #     label_list=['size', 'colour', 'material', 'shape'], ylabel='Acc Prior', show_ticks=True, colspan=text_col_width, rowspan=1, set_xlabel=True)
    # row+=1


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
        ax.text(-0.15, .09, text, transform=ax.transAxes, family='serif', size=6)
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


    # if self.image_type: # if multi
    #     training_y_hat = classifier.classifier2(x=training_recon_img, attributes=training_recon_q_sampled_words)
    #     val_y_hat = classifier.classifier2(x=val_recon_img, attributes=val_recon_q_sampled_words)
    #     prior_y_hat = classifier.classifier2(x=prior_img, attributes=prior_sampled_words)

    #     training_y_hat = training_y_hat.view(self.B, self.q_max_len-1, self.vocab_size)
    #     training_y_hat = F.softmax(training_y_hat,2)

    #     val_y_hat = val_y_hat.view(self.B, self.q_max_len-1, self.vocab_size)
    #     val_y_hat = F.softmax(val_y_hat,2)

    #     prior_y_hat = prior_y_hat.view(self.B, self.q_max_len-1, self.vocab_size)
    #     prior_y_hat = F.softmax(prior_y_hat,2)

    # else:
    #     training_y_hat = classifier.classifier(x=training_recon_img)
    #     val_y_hat = classifier.classifier(x=val_recon_img)
    #     prior_y_hat = classifier.classifier(x=prior_img)

    #     training_y_hat = training_y_hat.view(self.B, self.q_max_len, self.vocab_size)
    #     training_y_hat = F.softmax(training_y_hat,2)

    #     val_y_hat = val_y_hat.view(self.B, self.q_max_len, self.vocab_size)
    #     val_y_hat = F.softmax(val_y_hat,2)

    #     prior_y_hat = prior_y_hat.view(self.B, self.q_max_len, self.vocab_size)
    #     prior_y_hat = F.softmax(prior_y_hat,2)

    rows = 6

    # if self.image_type:
    #     cols= 11
    # else:
    # cols=6

    cols= 11

    text_col_width = 1


    newline_every = list(range(10))

    n_vizs = 1
    for img_i in range(n_vizs):

        fig = plt.figure(figsize=(7+cols,2+rows), facecolor='white', dpi=150)

        #Row 0
        make_image_subplot(self, rows, cols, row=0, col=0, image=images[img_i], text='Training Recon')
        make_text_subplot(self, rows, cols, row=0, col=1, text='\n\n\n'+get_sentence(questions[img_i], newline_every=newline_every))

        #Row 1
        make_image_subplot(self, rows, cols, row=1, col=0, image=training_recon_img[img_i], text='')
        # sentence_classifier = get_sentence(torch.max(training_y_hat, 2)[1][img_i], newline_every=[3])
        sentence = get_sentence(training_recon_q_sampled_words[img_i], newline_every=newline_every) 
        make_text_subplot(self, rows, cols, row=1, col=1, text=sentence) #+'\n\nClassifier:\n'+sentence_classifier)
        sentence = get_sentence2(training_recon_q_sampled_words[img_i])
        dist = training_recon_q_dist[img_i]
        for i in range(len(dist)):
            make_bar_subplot(self, rows, cols, row=1, col=2+i, range_=range(self.vocab_size), values_=dist[i], text=sentence[i], sampled_word=training_recon_q_sampled_words[img_i][i])

        #Row 2
        make_image_subplot(self, rows, cols, row=2, col=0, image=val_img_batch[img_i], text='Validation Recon')
        make_text_subplot(self, rows, cols, row=2, col=1, text='\n\n\n'+get_sentence(val_question_batch[img_i], newline_every=newline_every))

        #Row 3
        make_image_subplot(self, rows, cols, row=3, col=0, image=val_recon_img[img_i], text='')
        sentence = get_sentence(val_recon_q_sampled_words[img_i], newline_every=newline_every)
        # sentence_classifier = get_sentence(torch.max(val_y_hat, 2)[1][img_i], newline_every=[3])
        make_text_subplot(self, rows, cols, row=3, col=1, text=sentence) #+'\n\nClassifier:\n'+sentence_classifier)
        sentence = get_sentence2(val_recon_q_sampled_words[img_i])
        dist = val_recon_q_dist[img_i]
        for i in range(len(dist)):
            make_bar_subplot(self, rows, cols, row=3, col=2+i, range_=range(self.vocab_size), values_=dist[i], text=sentence[i], sampled_word=val_recon_q_sampled_words[img_i][i])

        #Row 4
        make_image_subplot(self, rows, cols, row=4, col=0, image=prior_img[img_i], text='Prior Samples')
        # sentence_classifier = get_sentence(torch.max(prior_y_hat, 2)[1][img_i], newline_every=[3])
        sentence = get_sentence(prior_sampled_words[img_i], newline_every=newline_every)
        make_text_subplot(self, rows, cols, row=4, col=1, text=sentence)#+'\n\nClassifier:\n'+sentence_classifier)
        sentence = get_sentence2(prior_sampled_words[img_i])
        dist = prior_q_dist[img_i]
        for i in range(len(dist)):
            make_bar_subplot(self, rows, cols, row=4, col=2+i, range_=range(self.vocab_size), values_=dist[i], text=sentence[i], sampled_word=prior_sampled_words[img_i][i])

        #Row 5
        make_image_subplot(self, rows, cols, row=5, col=0, image=prior_img[img_i+1], text='')
        # sentence_classifier = get_sentence(torch.max(prior_y_hat, 2)[1][img_i+1], newline_every=[3])
        sentence = get_sentence(prior_sampled_words[img_i+1], newline_every=newline_every)
        make_text_subplot(self, rows, cols, row=5, col=1, text=sentence)#+'\n\nClassifier:\n'+sentence_classifier)
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
        ax.text(-0.15, .05, text, transform=ax.transAxes, family='serif', size=6)
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

    newline_every = [0,1,2,3,4,5,6,7,8,9]


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
                    make_text_subplot(self, rows, cols, row=j, col=1+samp_i, text=get_sentence(ys[samp_i][j], newline_every=newline_every), above_text='p(y|z'+str(samp_i)+')')
                else:
                    make_text_subplot(self, rows, cols, row=j, col=1+samp_i, text=get_sentence(ys[samp_i][j], newline_every=newline_every))

                if j==0:
                    make_image_subplot(self, rows, cols, row=j, col=1+samp_i+n_samps, image=xs_1[samp_i][j], text='p(x|z'+str(samp_i)+')')
                else:
                    make_image_subplot(self, rows, cols, row=j, col=1+samp_i+n_samps, image=xs_1[samp_i][j])


        for j in range(int(rows/2)):

            if j==0:
                make_text_subplot(self, rows, cols, row=int(rows/2) + j, col=0, text=get_sentence(val_question_batch[j], newline_every=newline_every), above_text='q(z|y)')
            else:
                make_text_subplot(self, rows, cols, row=int(rows/2) + j, col=0, text=get_sentence(val_question_batch[j], newline_every=newline_every))


            for samp_i in range(n_samps):
                if j==0:
                    make_image_subplot(self, rows, cols, row=int(rows/2)+j, col=1+samp_i, image=xs[samp_i][j], text='p(x|z'+str(samp_i)+')')
                else:
                    make_image_subplot(self, rows, cols, row=int(rows/2)+j, col=1+samp_i, image=xs[samp_i][j])

                if j==0:
                    make_text_subplot(self, rows, cols, row=int(rows/2)+j, col=1+samp_i+n_samps, text=get_sentence(ys_2[samp_i][j], newline_every=newline_every), above_text='p(y|z'+str(samp_i)+')')
                else:
                    make_text_subplot(self, rows, cols, row=int(rows/2)+j, col=1+samp_i+n_samps, text=get_sentence(ys_2[samp_i][j], newline_every=newline_every))


        # plt.tight_layout()
        plt_path = images_dir + 'conditional'+str(step_count)+'_'+str(img_i) +'.png'
        plt.savefig(plt_path)
        print ('saved viz',plt_path)
        plt.close(fig)
        # fdsf











def get_text_len(image_atts):

    count = 0
    for key, value in image_atts.items():

        if key == 'Male':
            count+=1
            # if value == '1':
            #     text += ' ' + key
            # else:
            #     text += ' ' + 'Female'
        elif value == '1':
            count+=1

    return count








def get_text_idxs(image_atts, word_to_idx):

    idxs = []
    for key, value in image_atts.items():

        if key == 'Male':
            if value == '1':
                idxs.append(word_to_idx['Male'])
            else:
                idxs.append(word_to_idx['Female'])
        elif value == '1':
            idxs.append(word_to_idx[key])

    return idxs





def get_sentence(list_of_word_idxs, newline_every=[]):

    sentence =''
    list_of_word_idxs = list_of_word_idxs.cpu().numpy()#[0]
    for i in range(len(list_of_word_idxs)):
        word = idx_to_word[int(list_of_word_idxs[i])]
        sentence += ' ' + word
        if i in newline_every:
            sentence += '\n'
    return sentence





def get_sentence2(list_of_word_idxs):

    sentence =[]
    list_of_word_idxs = list_of_word_idxs.cpu().numpy()#[0]
    for i in range(len(list_of_word_idxs)):
        word = idx_to_word[int(list_of_word_idxs[i])]
        sentence.append(word)
        # if i in newline_every:
        #     sentence += '\n'
    return sentence




































if __name__ == "__main__":


    

    parser = argparse.ArgumentParser()

    # parser.add_argument('--feature_dim', default='3,112,112')
    # parser.add_argument('--num_train_samples', default=None, type=int)
    # parser.add_argument('--num_val_samples', default=None, type=int)

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


    parser.add_argument('--textAR', type=int, default=1)

    parser.add_argument('--ssl_type', type=str, default='0')


    args = parser.parse_args()
    args_dict = vars(args) #convert to dict

    os.environ['CUDA_VISIBLE_DEVICES'] = args.which_gpu #  '0' #'1' #

    data_dir = args.data_dir 
    image_dir = data_dir + 'img_align_celeba/'
    attribute_file = data_dir + 'attr_dict.pkl'

    with open(attribute_file, "rb" ) as f:
        attr_dict = pickle.load(f)


    n_images = len(list(os.listdir(image_dir)))
    print ('n_images', n_images)

    print ('n_attributes', len(attr_dict))

    # #Get max length and vocab
    # attr_lens = []
    # for i in range(1, len(attr_dict)+1):

    #     numb = str(i) 
    #     while len(numb) != 6:
    #         numb = '0' + numb

    #     # img_file = data_dir+ numb + '.jpg'
    #     # jpgfile = Image.open(img_file)
    #     # img = np.array(jpgfile)
    #     # img = np.rollaxis(img, 2, 1)# [112,112,3]
    #     #         img = np.rollaxis(img, 1, 0)
    #     # img = img / 255.

    #     # print (attr_dict[numb + '.jpg'])
    #     attr_len = get_text_len(attr_dict[numb + '.jpg'])
    #     attr_lens.append(attr_len)

    #     # fadsf
    #     # text = get_text(attr_dict[numb + '.jpg'])
    #     # texts.append(text)
    #     # # print (i, numb, attr_dict[numb + '.jpg'])
    #     # print (i, numb, text)
    #     # print ()
    # print (np.max(attr_lens), np.mean(attr_lens), np.min(attr_lens)) # 9,4,1

    args_dict['q_max_len'] = 9 #np.max(attr_lens) 
    args_dict['vocab_size'] = 20 



    attrs = ['_',
    'Bushy_Eyebrows',
    'Male',
    'Female',
    'Mouth_Slightly_Open',
    'Smiling',
    'Bald',
    'Bangs',
    'Black_Hair' ,
    'Blond_Hair' ,
    'Brown_Hair' ,
    'Eyeglasses' ,
    'Gray_Hair',
    'Heavy_Makeup' ,
    'Mustache' ,
    'Pale_Skin',
    'Receding_Hairline' ,
    'Straight_Hair' ,
    'Wavy_Hair',
    'Wearing_Hat']

    idx_to_word = {}
    word_to_idx = {}
    for i in range(len(attrs)):
        idx_to_word[i] = attrs[i]
        word_to_idx[attrs[i]] = i

    # print (idx_to_word)
    # print (word_to_idx)
    # fda



    max_steps = args.max_steps
    save_steps = args.save_params_step
    quick_check = args.quick_check

    if quick_check:
        display_step = 1#00 # 10 # 100 # 20
        trainingplot_steps = 1#000 # 4 #1000# 50 #100# 1000 # 500 #2000
        viz_steps = 3#2000 # 500# 2000
    else:
        display_step = args.display_step #500 #debug_steps #500 #50# 100 # 10 # 100 # 20
        trainingplot_steps = args.trainingplot_steps #500# 5000 # debug_steps #2000 # 4 #1000# 50 #100# 1000 # 500 #2000
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










    print ('\nInit VLVAE')
    model = VLVAE(args_dict)
    model.cuda()
    if args.model_load_step>0:
        model.load_params_v3(save_dir=args.params_load_dir, step=args.model_load_step)
    # print(model)
    # fdsa
    print ('VLVAE Initilized\n')



    





    print ('Init classifier')
    # if single_object or single_object_v2:
    classifier = attribute_classifier(args_dict)
    # elif multi_object:
    #     classifier = attribute_classifier_with_relations(args_dict) #, encoder_embed=model.encoder_embed)
    classifier.cuda()
    classifier.train()
    if args.classifier_load_step > 0:
        classifier.load_params_v3(save_dir=args.classifier_load_params_dir, step=args.classifier_load_step)
        classifier.eval()
    print ('classifier Initilized\n')







    # if train_:
    print ('Training')
    train2(self=model, max_steps=max_steps, load_step=args.model_load_step,
            image_dir=image_dir, train_indexes=list(range(1,180001)), val_indexes=list(range(180001, n_images)), attr_dict=attr_dict, word_to_idx=word_to_idx,
            save_dir=exp_dir, params_dir=params_dir, images_dir=images_dir, batch_size=args.batch_size,
            display_step=display_step, save_steps=save_steps, trainingplot_steps=trainingplot_steps, viz_steps=viz_steps,
            classifier=classifier, start_storing_data_step=args.start_storing_data_step) #, k=args.k)
    model.save_params_v3(save_dir=params_dir, step=max_steps+args.model_load_step)

    print ('Done.')



# Example:

#python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 0 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --flow 1  --exp_name flow_qy --quick_check 1 --model_load_step 100000

# python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --z_size 20 --exp_name gaus_qy

# python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --z_size 20 --quick_check 1 

# python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --quick_check 1 --flow 1 --z_size 20

# python3 train_jointVAE_v21.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 20 --w_logpx .01 --joint_inf 0 --quick_check 1


