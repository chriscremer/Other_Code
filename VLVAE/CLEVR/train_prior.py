




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


# from vr.data import ClevrDataset, ClevrDataLoader
from Clevr_data_loader import ClevrDataset, ClevrDataLoader
# from cyclegangenerator_bottleneck import *
from classifier import attribute_classifier, attribute_classifier_with_relations, classifier_of_prior
# from textRNN import TextRNN
# from priorRNN import priorRNN
# from distributions import Normal, Flow1#, IAF_flow
from preprocess_statements import *
# from qRNN import qRNN


from VLVAE_model import VLVAE

from distributions import lognormal, Flow1, Flow1_Cond




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












# def get_indexes(question_dataset, word=''):

#     indexes = []
#     for i in range(len(train_indexes)):
#         sentence = get_sentence(question_dataset[i])
#         if word in sentence:
#             indexes.append(i)
#     return indexes




def get_indexes(question_dataset, word=''):

    indexes = []
    for i in range(len(train_indexes)):
        sentence = get_sentence(question_dataset[train_indexes[i]])
        # print (sentence, word, word in sentence)
        if word in sentence:
            indexes.append(train_indexes[i])

    return indexes



def SIR(model, question_batch, inf_type=2, dec_type=2, k=1): #, k=1): #, answer_dataset, image_idxs, first_question_idx):

    # batch_size = 20
    batch_zs = []
    model.eval()
    # model.train()
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






# def SIR_prior(model, question_batch, inf_type=2, dec_type=2, k=1, prior=None): #, k=1): #, answer_dataset, image_idxs, first_question_idx):

#     # batch_size = 20
#     batch_zs = []
#     model.eval()
#     # model.train()
#     with torch.no_grad():
#         for i in range(len(question_batch)):
#             # img_batch, question_batch = make_batch(image_dataset, question_dataset, batch_size, indexes=indexes)#, 
#             data1 = question_batch[i]
#             # print (data1.shape)
#             data1 = data1.view(1,data1.shape[0])
#             logws = []
#             zs=[]
#             for j in range(k):
#                 outputs = model.forward_prior(x=None, q=data1, inf_type=inf_type, dec_type=dec_type, prior=prior) #, k=k) #, answer_batch)
#                 logws.append(outputs['logws'])
#                 zs.append(outputs['z'])

#             logws = torch.stack(logws,0) #[k,1]
#             # print (logws.shape)

#             #log normalize
#             max_ = torch.max(logws, 0)[0] 
#             lse = torch.log(torch.sum(torch.exp(logws-max_), 0)) + max_
#             log_norm_ws = logws - lse
#             log_norm_ws = torch.transpose(log_norm_ws, 0,1)

#             # print (log_norm_ws.shape)
#             # fsdf

#             # m = Multinomial(100, torch.exp(log_norm_ws))
#             m = Categorical(probs=torch.exp(log_norm_ws))
#             idxs = m.sample()

#             # print (idxs.shape) 
#             # print (idxs) 
#             # fofsfsdasd
#             # print (zs[idxs.data.item()][0].shape)
#             batch_zs.append(zs[idxs.data.item()][0].data.cpu().numpy())



#             # fdsfas

#             # if k ==1:
#             #     LLs.append( torch.mean(logws).data.item())
#             # else:
#             #     LLs.append( torch.mean(logmeanexp(logws)).data.item())




#     # model.train()
#     # LL = np.mean(LLs)
#     batch_zs = np.array(batch_zs)

#     return batch_zs












# def SIR_x_prior(model, image_batch, inf_type=1, dec_type=1, k=1, prior=None): #, k=1): #, answer_dataset, image_idxs, first_question_idx):

#     # batch_size = 20
#     batch_zs = []
#     model.eval()
#     with torch.no_grad():
#         for i in range(len(image_batch)):
#             # img_batch, question_batch = make_batch(image_dataset, question_dataset, batch_size, indexes=indexes)#, 
#             data1 = image_batch[i]
#             # print (data1.shape)
#             data1.unsqueeze_(0)
#             # print (data1.shape)
#             # fdfa
#             # data1 = data1.view(1,data1.shape[0])
#             logws = []
#             zs=[]
#             for j in range(k):
#                 outputs = model.forward_prior(x=data1, q=None, inf_type=inf_type, dec_type=dec_type, prior=prior) #, k=k) #, answer_batch)
#                 logws.append(outputs['logws'])
#                 zs.append(outputs['z'])

#             logws = torch.stack(logws,0) #[k,1]
#             # print (logws.shape)

#             #log normalize
#             max_ = torch.max(logws, 0)[0] 
#             lse = torch.log(torch.sum(torch.exp(logws-max_), 0)) + max_
#             log_norm_ws = logws - lse
#             log_norm_ws = torch.transpose(log_norm_ws, 0,1)

#             # print (log_norm_ws.shape)
#             # fsdf

#             # m = Multinomial(100, torch.exp(log_norm_ws))
#             m = Categorical(probs=torch.exp(log_norm_ws))
#             idxs = m.sample()

#             # print (idxs.shape) 
#             # print (idxs) 
#             # fofsfsdasd
#             # print (zs[idxs.data.item()][0].shape)
#             batch_zs.append(zs[idxs.data.item()][0].data.cpu().numpy())



#             # fdsfas

#             # if k ==1:
#             #     LLs.append( torch.mean(logws).data.item())
#             # else:
#             #     LLs.append( torch.mean(logmeanexp(logws)).data.item())




#     # model.train()
#     # LL = np.mean(LLs)
#     batch_zs = np.array(batch_zs)

#     return batch_zs





def make_batch_in_order(image_dataset, question_dataset, batch_size, indexes, cur_index): # val=False, SSL=False): #, answer_dataset): #, image_idxs, first_question_idx):

    idx = cur_index

    img_batch = []
    text_batch = []
    for i in range(batch_size):

        if idx >= len(indexes):
            break
        else:
            data_index = indexes[idx]

        img_batch.append(image_dataset[idx])
        text_batch.append(question_dataset[idx])

        idx +=1


    img_batch = np.stack(img_batch) #[B,C,W,H]
    text_batch = np.stack(text_batch) #, 1) #[T,B,L]

    img_batch = torch.from_numpy(img_batch).cuda()
    text_batch = torch.from_numpy(text_batch).cuda()
    img_batch = torch.clamp(img_batch, min=1e-5, max=1-1e-5)

    return img_batch, text_batch, idx 









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














def make_batch(image_dataset, question_dataset, batch_size, val=False): #, answer_dataset): #, image_idxs, first_question_idx):

    if val:
        list_ = val_indexes
    else:
        list_ = train_indexes
        

    img_batch = []
    question_batch = []
    for i in range(batch_size):
        idx = random.choice(list_)
        img_batch.append(image_dataset[idx])
        question_batch.append(question_dataset[idx])

    img_batch = np.stack(img_batch) #[B,C,W,H]
    question_batch = np.stack(question_batch) #, 1) #[T,B,L]

    img_batch = torch.from_numpy(img_batch).cuda()
    question_batch = torch.from_numpy(question_batch).cuda()
    # img_batch = rescaling(img_batch)
    img_batch = torch.clamp(img_batch, min=1e-5, max=1-1e-5)

    return img_batch, question_batch #, answer_batch





def compute_log_like_prior(model, image_dataset, question_dataset, n_batches, indexes, inf_type=1, dec_type=0, k=1, prior=None): #, k=1): #, answer_dataset, image_idxs, first_question_idx):

    batch_size = 20
    LLs = []
    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            img_batch, question_batch = make_batch2(image_dataset, question_dataset, batch_size, indexes=indexes)#, 

            logws = []
            for j in range(k):
                outputs = model.forward_prior(x=img_batch, q=question_batch, inf_type=inf_type, dec_type=dec_type, prior=prior) #, k=k) #, answer_batch)
                logws.append(outputs['logws'])

            logws = torch.stack(logws, 0) #[k,B]
            if k ==1:
                LLs.append( torch.mean(logws).data.item())
            else:
                LLs.append( torch.mean(logmeanexp(logws)).data.item())
    model.train()
    LL = np.mean(LLs)

    return LL





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







def get_entropy(dist, k_entropy):

    zeros = torch.zeros(20, dist.z_size).cuda()
    # get entropy of qz
    logqs = []
    for i in range(k_entropy):

        z, logpz, logqz = dist.sample(zeros, zeros) 
        logqs.append(logqz)

    logqs = torch.stack(logqs)
    entropy = - torch.mean(logqs)

    return entropy

















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
            'aa_prior_classifier_acc', 'aa_prior_acc',
            'learned_prior_acc', 'learned_px', 'learned_py', 'learned_px_train', 'learned_py_train'
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
    warmup_steps = 10000. 



    if self.joint_inf:
        inf_type = 0
    else:
        inf_type = 1


    # if self.learn_prior:
    #     print ('Init Learned Prior')

    #     flow = Flow1(args.z_size, n_flows=6)
    #     flow.cuda()
    #     flow_optimizer = optim.Adam(flow.parameters(), lr=.0004, weight_decay=.0000001)

    #     print ('inited')



    if self.learn_prior:
        print ('Init Learned Prior')

        self.priorflow = Flow1(args.z_size, n_flows=6)
        self.priorflow.cuda()
        flow_optimizer = optim.Adam(self.priorflow.parameters(), lr=.0004, weight_decay=.0000001)

        print ('inited')



    # if self.qy_type =='true' or self.qy_type =='agg':

    #     print ('Init Learned qy')

    #     flow_cond = Flow1_Cond(args.z_size, n_flows=6, h_size=200)
    #     flow_cond.cuda()
    #     flow_optimizer = optim.Adam(flow_cond.parameters(), lr=.0004, weight_decay=.0000001)

    #     print ('qy inited')





    # y_inf_net_params = [list(self.encoder_rnn2.parameters()) + list(self.qzy_fc1.parameters()) 
    #                     + list(self.qzy_bn1.parameters()) + list(self.qzy_fc2.parameters())
    #                         + list(self.qzy_fc3.parameters()) + list(self.flow.parameters())]

    # # y_inf_net_params = [list(self.encoder_rnn2.parameters()) + list(self.qzy_fc1.parameters()) 
    # #                     + list(self.qzy_fc2.parameters())
    # #                         + list(self.qzy_fc3.parameters()) + list(self.flow.parameters())]

    # qy_optimizer = optim.Adam(y_inf_net_params[0], lr=.0001, weight_decay=.0000001)









    #EVAL MODEL

    print (len(train_indexes))
    print (len(val_indexes))



    dir_to_load = home + "/Documents/VLVAE_exps/train_learn_prior_2/params/" #model_params450000.pt"
    model.load_params_v3(save_dir=dir_to_load, step=800000)


    zeros = torch.zeros(batch_size, self.z_size).cuda()

    self.eval()
    with torch.no_grad():



        real_accs = []
        learned_accs = []

        for i in range(10):

            #smple learned prior 
            z, logpz, logqz = self.priorflow.sample(zeros, zeros) 
            x_samp_learned_prior, y_hat_learned_prior, sampled_words_learned_prior = self.decode(z)
            # print (sampled_words_learned_prior.detach().shape)
            _, learned_prior_acc = classifier.classifier_loss(x=x_samp_learned_prior.detach(), attributes=sampled_words_learned_prior.detach())
            # print (learned_prior_acc)
            learned_accs.append(learned_prior_acc)

            x_hat_prior, y_hat_prior, y_hat_prior_sampled_words = self.sample_prior()
            _, prior_acc = classifier.classifier_loss(x=x_hat_prior.detach(), attributes=y_hat_prior_sampled_words.detach())
            real_accs.append(prior_acc)


        # if step_count%viz_steps==0:
        viz_more(x_samp_learned_prior, images_dir, step_count)

        print ('real prior acc', np.mean(real_accs), np.var(real_accs))
        print ('learned prior acc', np.mean(learned_accs), np.var(learned_accs))




        LLs = []
        LLs_realprior = []
        k=10

        # px = 1
        # py = 0
        # if px 
        type_ = 1

        prior1 = lambda xx: self.priorflow.logprob(xx, zeros, zeros)
        prior2 = lambda xx: lognormal(xx, zeros, zeros)

        indexes_ = val_indexes

        idx =0
        # for i in range(10):
        while idx < len(val_indexes):
            val_img_batch, val_question_batch, idx = make_batch_in_order(val_image_dataset, val_question_dataset, batch_size, indexes=indexes_, cur_index=idx)

            # val_img_batch, val_question_batch, idx = make_batch_in_order(image_dir, attr_dict, batch_size, indexes=train_indexes_2, word_to_idx=word_to_idx, cur_index=idx)

            zeros = torch.zeros(len(val_img_batch), self.z_size).cuda()
            logws = []
            for j in range(k):
                # outputs = model.forward_prior(x=img_batch, q=question_batch, inf_type=inf_type, dec_type=dec_type, prior=prior)
                outputs = self.forward_prior(x=val_img_batch, q=val_question_batch, inf_type=type_, dec_type=type_, prior=prior1) #, k=k) #, answer_batch)
                logws.append(outputs['logws'])

            logws = torch.stack(logws, 0) #[k,B]
            LL_x  =torch.mean(logmeanexp(logws)).data.item()
            LLs.append( LL_x)


            logws = []
            for j in range(k):
                # outputs = model.forward_prior(x=img_batch, q=question_batch, inf_type=inf_type, dec_type=dec_type, prior=prior)
                outputs = self.forward_prior(x=val_img_batch, q=val_question_batch, inf_type=type_, dec_type=type_, prior=prior2) #, k=k) #, answer_batch)
                logws.append(outputs['logws'])

            logws = torch.stack(logws, 0) #[k,B]
            # if k ==1:
            #     LLs.append( torch.mean(logws).data.item())
            # else:
            LL_x_real  =torch.mean(logmeanexp(logws)).data.item()
            LLs_realprior.append( LL_x_real)

            if idx % 100==0:
                print(idx,len(val_indexes),  'logpx',  np.mean(LLs), np.std(LLs),  LL_x, np.mean(LLs_realprior), np.std(LLs_realprior),  LL_x_real)

    print ('avgLLx: ', np.mean(LLs))
    print ('avgLLx_real: ', np.mean(LLs_realprior))
    fdafa










    
    dfaafda






    model.load_params_v3(save_dir=dir_to_load, step=800000)

    just_acc = False

    train_indexes_2 = train_indexes[:10000]
    indexes_ = val_indexes

    accs = []
    LLs = []
    k=10
    self.eval()
    with torch.no_grad():

        idx =0
        # for i in range(10):
        # while idx < len(val_indexes):
        while idx < len(indexes_):

            val_img_batch, val_question_batch, idx = make_batch_in_order(val_image_dataset, val_question_dataset, batch_size, indexes=indexes_, cur_index=idx)

            # GET ACC
            outputs, training_recon_img, training_recon_q, training_recon_sampled_words = self.forward_qy2(x=val_img_batch, q=val_question_batch, 
                                                                                    warmup=self.max_beta, inf_type=2, dec_type=2, generate=True)
            # classi_loss_prior, prior_acc = classifier.classifier_loss(x=training_recon_img.detach(), attributes=val_question_batch)
            _, acc = classifier.classifier_loss(x=training_recon_img.detach(), attributes=val_question_batch)
            # acc = classifier.accuracy_sequence(x=training_recon_img.detach(), attributes=training_recon_sampled_words.detach())
            accs.append(acc)

            if not just_acc:
                # GET LL
                embed = self.encoder_embed(val_question_batch)
                y_enc = self.encode_attributes2_2(embed)
                mu, logvar = self.inference_net_y_2(y_enc)
                prior1 = lambda xx: self.flow_cond_2.logprob(xx, mu, logvar, y_enc)

                logws = []
                for j in range(k):
                    outputs = self.forward_prior(x=val_img_batch, q=val_question_batch, inf_type=1, dec_type=1, prior=prior1) #, k=k) #, answer_batch)
                    logws.append(outputs['logws'])

                logws = torch.stack(logws, 0) #[k,B]
                LL_x  =torch.mean(logmeanexp(logws)).data.item()
                LLs.append( LL_x)
            else:
                LLs.append(0)

            if idx % 100==0:
                print(idx,len(val_indexes), 'avg acc:', np.mean(accs), acc,  'logpx',  np.mean(LLs), np.std(LLs))#,  LL_x)


    print ('avg: ', np.mean(accs))
    print ('avgLLx: ', np.mean(LLs))
    fdafa
















    z_avg_var = deque(maxlen=10)

    warmup=0.

    for step in range(max_steps):

        img_batch, question_batch = make_batch(train_image_dataset, train_question_dataset, batch_size)
        # warmup = min((step_count+load_step) / float(warmup_steps), self.max_beta)



        # #see if flwo is right
        # self.eval()
        # embed = self.encoder_embed(question_batch)
        # y_enc = self.encode_attributes2(embed)
        # mu, logvar = self.inference_net_y(y_enc)
        # z, logpz, logqz = self.flow.sample(mu, logvar) 

        # logqzy = self.flow.logprob(z.detach(), mu, logvar)

        # print (logqz, logqzy) # looks good.


        # fasd









        if step > warmup_steps:
            warmup = self.max_beta
            # # difference = 
            # # change = .001* np.maximum(1.-np.mean(z_avg_var), -10.)
            # if np.mean(z_avg_var) > 1.:
            #     change = .01
            # else:
            #     change = -.01
            # # warmup = np.maximum(0., warmup - .001*(1.-np.mean(z_avg_var)))
            # warmup = np.maximum(0., warmup + change)
        else:
            warmup = (step / float(warmup_steps)) * self.max_beta
            # warmup = .001




        # q(z|y)
        # # linear_hidden_size = 500 
        # encoder_rnn2 = nn.GRU(input_size=self.embed_size, hidden_size=self.y_enc_size, num_layers=1, 
        #                             dropout=0, batch_first=True, bidirectional=False)
        # qzy_fc1 = nn.Linear(self.y_enc_size, self.linear_hidden_size)
        # qzy_bn1 = nn.BatchNorm1d(self.linear_hidden_size)
        # qzy_fc2 = nn.Linear(self.linear_hidden_size, self.linear_hidden_size)
        # qzy_fc3 = nn.Linear(self.linear_hidden_size, self.z_size*2)
        # flow = Flow1(self.z_size)  


        # y_inf_net_params = [list(encoder_rnn2.parameters()) + list(qzy_fc1.parameters()) 
        #                         + list(qzy_bn1.parameters()) + list(qzy_fc2.parameters())
        #                             + list(qzy_fc3.parameters()) + list(flow.parameters())]
        # qy_optimizer = optim.Adam(y_inf_net_params[0], lr=.0001, weight_decay=.0000001)


        # encoder_rnn2 = nn.GRU(input_size=self.embed_size, hidden_size=self.y_enc_size, num_layers=1, 
        #                             dropout=0, batch_first=True, bidirectional=False)
        # qzy_fc1 = nn.Linear(self.y_enc_size, self.linear_hidden_size)
        # qzy_bn1 = nn.BatchNorm1d(self.linear_hidden_size)
        # qzy_fc2 = nn.Linear(self.linear_hidden_size, self.linear_hidden_size)
        # qzy_fc3 = nn.Linear(self.linear_hidden_size, self.z_size*2)
        # flow = Flow1(self.z_size)  

        # y_inf_net_params = [list(encoder_rnn2.parameters()) + list(qzy_fc1.parameters()) 
        #                         + list(qzy_bn1.parameters()) + list(qzy_fc2.parameters())
        #                             + list(qzy_fc3.parameters()) + list(flow.parameters())]
        # qy_optimizer = optim.Adam(y_inf_net_params[0], lr=.0001, weight_decay=.0000001)




        # self.encoder_rnn2 = nn.GRU(input_size=self.embed_size, hidden_size=self.y_enc_size, num_layers=1, 
        #                             dropout=0, batch_first=True, bidirectional=False)
        # self.qzy_fc1 = nn.Linear(self.y_enc_size, self.linear_hidden_size)
        # self.qzy_bn1 = nn.BatchNorm1d(self.linear_hidden_size)
        # self.qzy_fc2 = nn.Linear(self.linear_hidden_size, self.linear_hidden_size)
        # self.qzy_fc3 = nn.Linear(self.linear_hidden_size, self.z_size*2)
        # self.flow = Flow1(self.z_size)  
        # y_inf_net_params = [list(self.encoder_rnn2.parameters()) + list(self.qzy_fc1.parameters()) 
        #                     + list(self.qzy_bn1.parameters()) + list(self.qzy_fc2.parameters())
        #                         + list(self.qzy_fc3.parameters()) + list(self.flow.parameters())]
        # qy_optimizer = optim.Adam(y_inf_net_params[0], lr=.0001, weight_decay=.0000001)
        # self.cuda()




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

                # if step % 20==0:
                #     print (step)

                # Beta = torch.from_numpy(warmup)


                # self.optimizer_x.zero_grad()
                # outputs = self.forward(x=img_batch, q=question_batch, warmup=torch.tensor([warmup]).cuda(), inf_type=1, dec_type=0) #, k=k_train) #, marginf_type=marginf_type)
                # loss = -outputs['welbo']
                # loss.backward()
                # self.optimizer_x.step()
                # self.scheduler_x.step()

                if self.qy_type =='true':
                    #Learn true posterior 
                    qy_optimizer.zero_grad()
                    outputs = self.forward(x=None, q=question_batch, warmup=self.max_beta, inf_type=2, dec_type=2) #, k=k_train) #, marginf_type=marginf_type)
                    loss = -outputs['welbo']
                    loss.backward()
                    qy_optimizer.step()



                elif self.qy_type =='agg':

                    #Learn agg posterior 
                    qy_optimizer.zero_grad()
                    # with torch.no_grad():
                    outputs = self.forward(x=img_batch, q=question_batch, warmup=self.max_beta, inf_type=1, dec_type=0)
                    loss = -outputs['logqzy']
                    loss.backward()
                    qy_optimizer.step()

                # else:
                #     problemmfsadfa



                # zs = outputs['z'].detach()
                # zav = np.mean(np.var(zs.data.cpu().numpy()))
                # z_avg_var.append(zav)


                # # print (warmup)
                
                # zav = np.mean(np.var(outputs['z'].detach().data.cpu().numpy()))
                # # zav = 0.
                # z_avg_var.append(zav)
                # # print (z_avg_var)
                

                if self.learn_prior:

                    # self.optimizer_x.zero_grad()
                    with torch.no_grad():
                        outputs = self.forward(x=img_batch, q=question_batch, warmup=self.max_beta, inf_type=1, dec_type=2) #, k=k_train) #, marginf_type=marginf_type)
                        # loss = -outputs['welbo']
                        # loss.backward()
                        # self.optimizer_x.step()
                        # self.scheduler_x.step()

                    zs = outputs['z'].detach()




                    flow_optimizer.zero_grad()
                    zeros = torch.zeros(batch_size, self.z_size).cuda()
                    logqzy = self.priorflow.logprob(zs, zeros, zeros)
                    # print (logqzy.shape)
                    flow_loss = -torch.mean(logqzy)
                    flow_loss.backward()
                    flow_optimizer.step()
                    # self.scheduler_x.step()




                zs = outputs['z'].detach()
                zav = np.mean(np.var(zs.data.cpu().numpy()))
                z_avg_var.append(zav)





                if step_count % display_step == 0:

                    with torch.no_grad():
                        # of qy
                        ent, recons = self.get_entropy_and_samples(q=question_batch, inf_type=2, dec_type=2, k_entropy=100, k_recons=1)
                        


                        if self.learn_prior:

                            # of learned prior
                            entropy_learned_prior = get_entropy(self.priorflow, k_entropy=100)
                            # of real prior 
                            entropy_prior = get_entropy(self, k_entropy=100)

                            #get prob of validation z under q and p
                            val_img_batch, val_question_batch = make_batch(val_image_dataset, val_question_dataset, batch_size, val=True) 
                            val_outputs, val_recon_img, val_recon_q, val_recon_sampled_words = self.forward(val_img_batch, val_question_batch, generate=True, inf_type=inf_type, dec_type=0) 
                            z_val = val_outputs['z'].detach()
                            logpz_learned = torch.mean(self.priorflow.logprob(z_val, zeros, zeros))
                            logpz = torch.mean(lognormal(z_val, zeros, zeros)) #[B]



                            prior1 = lambda xx: self.priorflow.logprob(xx, zeros, zeros)
                            prior2 = lambda xx: lognormal(xx, zeros, zeros)

                            LL_x_pz = compute_log_like_prior(self, val_image_dataset, val_question_dataset, n_batches=20, indexes=val_indexes, inf_type=1, dec_type=1, k=10, prior=prior2)
                            LL_y_pz = compute_log_like_prior(self, val_image_dataset, val_question_dataset, n_batches=20, indexes=val_indexes, inf_type=2, dec_type=2, k=10, prior=prior2)

                            LL_x_qz = compute_log_like_prior(self, val_image_dataset, val_question_dataset, n_batches=20, indexes=val_indexes, inf_type=1, dec_type=1, k=10, prior=prior1)
                            LL_y_qz = compute_log_like_prior(self, val_image_dataset, val_question_dataset, n_batches=20, indexes=val_indexes, inf_type=2, dec_type=2, k=10, prior=prior1)

                            LL_x_qz_train = compute_log_like_prior(self, train_image_dataset, train_question_dataset, n_batches=20, indexes=train_indexes, inf_type=1, dec_type=1, k=10, prior=prior1)
                            LL_y_qz_train = compute_log_like_prior(self, train_image_dataset, train_question_dataset, n_batches=20, indexes=train_indexes, inf_type=2, dec_type=2, k=10, prior=prior1)


                            # outputs1_x = self.forward_prior(val_img_batch, val_question_batch, generate=False, inf_type=1, dec_type=1, prior=prior1)
                            # outputs1_y = self.forward_prior(val_img_batch, val_question_batch, generate=False, inf_type=2, dec_type=2, prior=prior1)

                            # outputs2_x = self.forward_prior(val_img_batch, val_question_batch, generate=False, inf_type=1, dec_type=1, prior=prior2)
                            # outputs2_y = self.forward_prior(val_img_batch, val_question_batch, generate=False, inf_type=2, dec_type=2, prior=prior2)

                            # SIR


                    self.train()


                    if self.learn_prior:
                        print (
                            # 'S:{:5d}'.format(step_count),
                            # 'T:{:.2f}'.format(T),
                            'welbo:{:.2f}'.format(outputs['welbo'].data.item()),
                            # 'elbo:{:.2f}'.format(outputs['elbo'].data.item()),
                            # 'lpx:{:.2f}'.format(outputs['logpx'].data.item()),
                            'lpy:{:.2f}'.format(outputs['logpy'].data.item()),
                            'lpz:{:.2f}'.format(outputs['logpz'].data.item()),
                            'lqz:{:.2f}'.format(outputs['logqz'].data.item()),
                            'ent:{:.2f}'.format(ent.data.item()),
                            'flow:{:.2f}'.format(-flow_loss.data.item()),
                            'learned prior ent:{:.2f}'.format(entropy_learned_prior.data.item()),
                            'real prior ent:{:.2f}'.format(entropy_prior.data.item()),
                            'pz:{:.2f}'.format(logpz.data.item()),
                            'pz_learned:{:.2f}'.format(logpz_learned.data.item()),
                            # 'TrainAvg:{:.2f}'.format(recent_dict['all_train_elbos'][-1]),
                            # 'ValidAvg:{:.2f}'.format(recent_dict['all_valid_elbos'][-1]),
                            # 'AvgMI:{:.4f}'.format(np.mean(recent_MIs)),
                            # 'warmup:{:.2f}'.format(warmup),
                            # 'prioracc:{:.2f}'.format(recent_dict['all_prior_acc'][-1]),
                            # 'zavgvar:{:.2f}'.format(np.mean(z_avg_var)),
                            # 'yinf_acc_e:{:.4f}'.format(np.mean(recent_y_inf_acc_entangle)),
                            )

                        print (
                            'logpx:{:.4f}'.format(LL_x_pz),
                            'logpy:{:.4f}'.format(LL_y_pz),
                            'logpx_learned:{:.4f}'.format(LL_x_qz),
                            'logpy_learned:{:.4f}'.format(LL_y_qz),
                            )


                    # else:


                    # # self.optimizer_x.zero_grad()
                    # with torch.no_grad():
                    #     val_img_batch, val_question_batch = make_batch(val_image_dataset, val_question_dataset, batch_size, val=True) 
                    #     outputs = self.forward(x=val_img_batch, q=val_question_batch, warmup=self.max_beta, inf_type=1, dec_type=0) #, k=k_train) #, marginf_type=marginf_type)
                    #     zs = outputs['z'].detach()

                    #     self.flow.logprob(zs)


                    #     print (
                    #         # 'S:{:5d}'.format(step_count),
                    #         # 'T:{:.2f}'.format(T),
                    #         'welbo:{:.2f}'.format(outputs['welbo'].data.item()),
                    #         # 'elbo:{:.2f}'.format(outputs['elbo'].data.item()),
                    #         # 'lpx:{:.2f}'.format(outputs['logpx'].data.item()),
                    #         'lpy:{:.2f}'.format(outputs['logpy'].data.item()),
                    #         'lpz:{:.2f}'.format(outputs['logpz'].data.item()),
                    #         'lqz:{:.2f}'.format(outputs['logqz'].data.item()),
                    #         'ent:{:.2f}'.format(ent.data.item()),
                    #         # 'flow:{:.2f}'.format(-flow_loss.data.item()),
                    #         # 'learned prior ent:{:.2f}'.format(entropy_learned_prior.data.item()),
                    #         # 'real prior ent:{:.2f}'.format(entropy_prior.data.item()),
                    #         # 'pz:{:.2f}'.format(logpz.data.item()),
                    #         # 'pz_learned:{:.2f}'.format(logpz_learned.data.item()),
                    #         # 'TrainAvg:{:.2f}'.format(recent_dict['all_train_elbos'][-1]),
                    #         # 'ValidAvg:{:.2f}'.format(recent_dict['all_valid_elbos'][-1]),
                    #         # 'AvgMI:{:.4f}'.format(np.mean(recent_MIs)),
                    #         # 'warmup:{:.2f}'.format(warmup),
                    #         # 'prioracc:{:.2f}'.format(recent_dict['all_prior_acc'][-1]),
                    #         # 'zavgvar:{:.2f}'.format(np.mean(z_avg_var)),
                    #         # 'yinf_acc_e:{:.4f}'.format(np.mean(recent_y_inf_acc_entangle)),
                    #         )    




                # #Learn true posterior starting from new qy
                # qy_optimizer.zero_grad()
                # embed = self.encoder_embed(q)
                # # y_enc = self.encode_attributes2(embed)
                # h0 = torch.zeros(1, embed.shape[0], self.y_enc_size).type_as(embed.data)
                # out, ht = self.encoder_rnn2(embed, h0)
                # y_enc = out[:,-1] 
                # # mu, logvar = self.inference_net_y(y_enc)
                # input_ = y_enc
                # padded_input = torch.cat([y_enc, torch.zeros(y_enc.shape[0], self.linear_hidden_size-self.y_enc_size).cuda()], 1)
                # out = self.act_func(qzy_bn1(qzy_fc1(input_)))
                # out = qzy_fc2(out)
                # res = out + padded_input 
                # out = qzy_fc3(res)
                # mean = out[:,:self.z_size]
                # logvar = out[:,self.z_size:]
                # logvar = torch.clamp(logvar, min=-15., max=10.)
                # # return mean, logvar
                # z, logpz, logqz = flow.sample(mu, logvar) 
                # z_dec = self.z_to_enc(z)
                # #Decode Text
                # word_preds, logpy = self.text_generator.teacher_force(z_dec, embed, question_batch)
                # logpy = logpy * self.w_logpy
                # log_ws = logpy + logpz - logqz 
                # elbo = torch.mean(log_ws)
                # # warmed_elbo = torch.mean(logpy + warmup*( logpz - logqz))
                # # outputs['logpy'] = torch.mean(logpy)
                # loss = -elbo
                # loss.backward()
                # qy_optimizer.step()

                # if step % 500 ==0:
                #     print (elbo.data.item(), torch.mean(logpy).data.item(), torch.mean(logpz).data.item(), torch.mean(logqz).data.item())





                # self.scheduler_x.step()




                # print (warmup)

                # print()



                # self.optimizer_y.zero_grad()
                # outputs = self.forward(x=None, q=question_batch, warmup=warmup, inf_type=2, dec_type=2) #, k=k_train) #, marginf_type=marginf_type)
                # loss = -outputs['welbo']
                # loss.backward()
                # self.optimizer_y.step()
                # self.scheduler_y.step()










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

                outputs, training_recon_img, training_recon_q, training_recon_sampled_words = self.forward(img_batch, question_batch, generate=True, inf_type=inf_type, dec_type=0, warmup=torch.tensor([warmup]).cuda()) #, k=k_gen)
                val_outputs, val_recon_img, val_recon_q, val_recon_sampled_words = self.forward(val_img_batch, val_question_batch, generate=True, inf_type=inf_type, dec_type=0) #, k=k_gen)

                #Accuracies 
                _, y_recon_acc = classifier.classifier_loss(x=img_batch, attributes=training_recon_sampled_words.detach())
                _, y_recon_acc_val = classifier.classifier_loss(x=val_img_batch, attributes=val_recon_sampled_words.detach())

                # y_hat_prior = y_hat_prior.view(self.B, self.q_max_len, self.vocab_size)
                # y_hat_prior = torch.max(y_hat_prior,2)[1]

                if self.learn_prior:
                    #smple learned prior 
                    z, logpz, logqz = self.priorflow.sample(zeros, zeros) 
                    x_samp_learned_prior, y_hat_learned_prior, sampled_words_learned_prior = self.decode(z)
                    classi_loss_learned_prior, learned_prior_acc = classifier.classifier_loss(x=x_samp_learned_prior.detach(), attributes=sampled_words_learned_prior.detach())


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
                n_samps = 3 #1 # for viewing
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
                y_inf_acc_list = []
                y_inf_acc_k1_list = []
                y_inf_acc_entangle_list = []
                n_samps = 20
                n_samps_for_plotting = 3
                for i in range(n_samps):
                    outputs_cond, recon_img_cond, recon_q_cond, recon_sampled_words_cond = self.forward(x=None, q=val_question_batch, generate=True, inf_type=2, dec_type=2) #, k=k_cond)   

                    # print (recon_img_cond.shape)
                    # fasd

                    if i < n_samps_for_plotting:
                        xs.append(recon_img_cond)
                        ys_2.append(recon_sampled_words_cond)
                    
                    # _, y_inf_acc = classifier.classifier_loss(x=xs[0].detach(), attributes=val_question_batch)
                    # _, y_inf_acc_k1 = classifier.classifier_loss(x=val_img_batch, attributes=ys_2[0].detach())
                    # _, y_inf_acc_entangle = classifier.classifier_loss(x=xs[0].detach(), attributes=ys_2[0].detach())

                    _, y_inf_acc = classifier.classifier_loss(x=recon_img_cond.detach(), attributes=val_question_batch)
                    _, y_inf_acc_k1 = classifier.classifier_loss(x=val_img_batch, attributes=recon_sampled_words_cond.detach())
                    _, y_inf_acc_entangle = classifier.classifier_loss(x=recon_img_cond.detach(), attributes=recon_sampled_words_cond.detach())

                    y_inf_acc_list.append(y_inf_acc.data.item())
                    y_inf_acc_k1_list.append(y_inf_acc_k1.data.item())
                    y_inf_acc_entangle_list.append(y_inf_acc_entangle.data.item())





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
            add_to_dict(recent_dict, 'all_y_inf_acc', np.mean(y_inf_acc_list))
            add_to_dict(recent_dict, 'all_x_inf_acc_entangle', x_inf_acc_entangle.data.item())
            add_to_dict(recent_dict, 'all_y_inf_acc_entangle', np.mean(y_inf_acc_entangle_list))

            add_to_dict(recent_dict, 'all_y_inf_acc_k1', np.mean(y_inf_acc_k1_list))
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



            add_to_dict(recent_dict, 'learned_prior_acc', learned_prior_acc.data.item())
            add_to_dict(recent_dict, 'learned_px', LL_x_qz)
            add_to_dict(recent_dict, 'learned_py', LL_y_qz)
            add_to_dict(recent_dict, 'learned_px_train', LL_x_qz_train)
            add_to_dict(recent_dict, 'learned_py_train', LL_y_qz_train)


            T = time.time() - start_time
            start_time = time.time()

            print (
                'S:{:5d}'.format(step_count),
                'T:{:.2f}'.format(T),
                'welbo:{:.2f}'.format(outputs['welbo'].data.item()),
                'elbo:{:.2f}'.format(outputs['elbo'].data.item()),
                'lpx:{:.2f}'.format(outputs['logpx'].data.item()),
                'lpy:{:.2f}'.format(outputs['logpy'].data.item()),
                'lpz:{:.2f}'.format(outputs['logpz'].data.item()),
                'lqz:{:.2f}'.format(outputs['logqz'].data.item()),
                'TrainAvg:{:.2f}'.format(recent_dict['all_train_elbos'][-1]),
                'ValidAvg:{:.2f}'.format(recent_dict['all_valid_elbos'][-1]),
                # 'AvgMI:{:.4f}'.format(np.mean(recent_MIs)),
                'warmup:{:.2f}'.format(warmup),
                'prioracc:{:.2f}'.format(recent_dict['all_prior_acc'][-1]),
                'learned_prioracc:{:.2f}'.format(learned_prior_acc.data.item()),
                'zavgvar:{:.2f}'.format(np.mean(z_avg_var)),
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

                if self.z_size == 2:
                    vizualize_2D(model, step)


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

                # print (len(xs))
                # print (len(ys))
                # print (len(xs_1))
                # print (len(ys_2))

                # try:
                vizualize_dependence(model,
                                val_img_batch, val_question_batch, 
                                images_dir, step_count+load_step, xs, ys, xs_1, ys_2, n_samps=3)
                # except:
                #     if self.quick_check:
                #         raise
                #     else:
                #         print ('problem with dependence viz plotting')

                # fadfa

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


    rows = 15
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
                    # all_dict['all_y_recon_acc'], all_dict['all_y_recon_acc_val'], 
                    all_dict['all_x_inf_acc'], all_dict['all_x_inf_acc_k1'], all_dict['all_x_inf_acc_entangle'], 
                    all_dict['all_y_inf_acc_k1'], all_dict['all_y_inf_acc'], all_dict['all_y_inf_acc_entangle'], 
                    all_dict['all_prior_acc'], all_dict['all_prior_acc2'],
                    all_dict['learned_prior_acc'],], 
        label_list=[r'$A(y_T,y_{x_T})$:   '+'{:.4f}'.format(all_dict['all_real_acc'][-1]), 
                    r'$A(y_V,y_{x_V})$:   '+'{:.4f}'.format( all_dict['all_real_val_acc'][-1]),
                    training_inf_dist +'{:.4f}'.format(all_dict['all_recon_acc'][-1]),
                    val_inf_dist +'{:.4f}'.format(all_dict['all_recon_acc_val'][-1]),
                    # training_inf_dist_y+'{:.4f}'.format( all_dict['all_y_recon_acc'][-1]),
                    # val_inf_dist_y+'{:.4f}'.format( all_dict['all_y_recon_acc_val'][-1]),
                    r'$p(\hat{y}_V|x_V), z \sim q(z|x_V)$: '+'{:.4f}'.format( all_dict['all_x_inf_acc'][-1]),
                    r'$p(y_V|\hat{x}_V), z \sim q(z|x_V)$: '+'{:.4f}'.format( all_dict['all_x_inf_acc_k1'][-1]),
                    r'$p(\hat{y}_V|\hat{x}_V), z \sim q(z|x_V)$: '+'{:.4f}'.format( all_dict['all_x_inf_acc_entangle'][-1]),
                    r'$p(\hat{y}_V|x_V), z \sim q(z|y_V)$: '+'{:.4f}'.format( all_dict['all_y_inf_acc_k1'][-1]),
                    r'$p(y_V|\hat{x}_V), z \sim q(z|y_V)$:'+'{:.4f}'.format( all_dict['all_y_inf_acc'][-1]),
                    r'$p(\hat{y}_V|\hat{x}_V), z \sim q(z|y_V)$: '+'{:.4f}'.format( all_dict['all_y_inf_acc_entangle'][-1]),
                    r'$p(\hat{y}|\hat{x}), z \sim p(z)$:    '+'{:.4f}'.format( all_dict['all_prior_acc'][-1]),
                    r'$p(\hat{y}|\hat{x}), z \sim N(0,.5)$:    '+'{:.4f}'.format( all_dict['all_prior_acc2'][-1]),
                    r'$learned prior$:    '+'{:.4f}'.format( all_dict['learned_prior_acc'][-1]),
                    ], 
        color_list=[color_defaults[0], color_defaults[0],
                    color_defaults[1], color_defaults[1],
                    # color_defaults[2], color_defaults[2],
                    color_defaults[3], color_defaults[3], color_defaults[3],
                    color_defaults[4], color_defaults[4], color_defaults[4],
                    color_defaults[5], color_defaults[6],
                    color_defaults[7],
                    ],
        linestyle_list=['-','--',
                        '-','--',
                        # '-','--',
                        '-','--',':',
                        '-','--',':',
                        '-','-',
                        '-',
                    ],
        ylabel='Accuracy', show_ticks=True, colspan=text_col_width, rowspan=2)
    row+=1
    row+=1




    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['learned_px_train'], all_dict['learned_px']], 
        label_list=['train: {:.4f}'.format( all_dict['learned_px_train'][-1]),
                    'valid: {:.4f}'.format( all_dict['learned_px'][-1])], 
        ylabel='logpx', show_ticks=False, colspan=text_col_width, rowspan=1)
    row+=1

    make_curve_subplot(self, rows, cols, row=row, col=col, steps=steps, 
        values_list=[all_dict['learned_py_train'], all_dict['learned_py']], 
        label_list=['train: {:.4f}'.format( all_dict['learned_py_train'][-1]),
                    'valid: {:.4f}'.format( all_dict['learned_py'][-1])], 
        ylabel='logpy', show_ticks=False, colspan=text_col_width, rowspan=1)
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













def vizualize_2D(self, step):

    batch_size = 20


    title_size = 6
    markersize = 4


    k=5


    #PLot the encodings of the dataset 
    rows = 6
    cols = 2
    fig = plt.figure(figsize=(2+cols,4+rows), facecolor='white', dpi=150)




    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

    n_batches = 20
    #Prior samples
    for i in range(n_batches):
        # img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, red_indexes)

        z = torch.FloatTensor(batch_size, self.z_size).normal_().numpy() #.cuda() * std

        if i == 0:
            zs = z
        else:
            zs = np.concatenate([zs, z], 0)

        # print (zs.shape)
        # fdsf


    ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=markersize, alpha=.3)

    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])
    ax.set_title('a) z~N(0,1)', fontsize=title_size)
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    plt.gca().set_aspect('equal', adjustable='box')
    # ax.set_xticks([])
    ax.tick_params(axis='x', colors='white')
    ax.grid(True, alpha=.3)




    ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)

    # n_batches = 10
    #Dataset encodings
    for i in range(n_batches):
        img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, train_indexes)
        z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)

        if i == 0:
            zs = z
        else:
            zs = np.concatenate([zs, z], 0)

        # print (zs.shape)
        # fdsf


    ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=markersize, alpha=.3)

    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])
    ax.set_title('b) Dataset Encodings', fontsize=title_size)
    ax.xaxis.set_tick_params(labelsize=5)
    # ax.text(0.3, .9, text, transform=ax.transAxes, family='serif', size=6)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.yaxis.set_tick_params(labelsize=5)
    # ax.set_xticks([])
    ax.tick_params(axis='x', colors='white')
    ax.grid(True, alpha=.3)




    ax = plt.subplot2grid((rows,cols), (1,0), frameon=False)

    for i in range(n_batches):
        img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[0]])
        # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
        # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
        z = SIR(self, question_batch, inf_type=2, dec_type=2, k=1)

        if i == 0:
            zs = z
        else:
            zs = np.concatenate([zs, z], 0)

    ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=markersize, alpha=.3)

    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])
    ax.set_title('c) '+get_sentence(question_batch[0]), fontsize=title_size)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.yaxis.set_tick_params(labelsize=5)
    ax.xaxis.set_tick_params(labelsize=5)
    # ax.set_xticks([])
    ax.tick_params(axis='x', colors='white')
    ax.grid(True, alpha=.3)




    ax = plt.subplot2grid((rows,cols), (1,1), frameon=False)

    for i in range(n_batches):
        img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[3]])
        # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
        # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
        z = SIR(self, question_batch, inf_type=2, dec_type=2, k=1)

        if i == 0:
            zs = z
        else:
            zs = np.concatenate([zs, z], 0)

    ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=markersize, alpha=.3)

    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])
    ax.set_title('d) '+get_sentence(question_batch[0]), fontsize=title_size)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    # ax.set_xticks([])
    ax.tick_params(axis='x', colors='white')
    ax.grid(True, alpha=.3)


    ax = plt.subplot2grid((rows,cols), (2,0), frameon=False)

    for i in range(n_batches):
        img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[5]])
        # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
        # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
        z = SIR(self, question_batch, inf_type=2, dec_type=2, k=1)

        if i == 0:
            zs = z
        else:
            zs = np.concatenate([zs, z], 0)

    ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=markersize, alpha=.3)

    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])
    ax.set_title('e) '+get_sentence(question_batch[0]), fontsize=title_size)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    # ax.set_xticks([])
    ax.tick_params(axis='x', colors='white')
    ax.grid(True, alpha=.3)


    ax = plt.subplot2grid((rows,cols), (2,1), frameon=False)

    for i in range(n_batches):
        img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[1]])
        # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
        # z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)
        z = SIR(self, question_batch, inf_type=2, dec_type=2, k=1)

        if i == 0:
            zs = z
        else:
            zs = np.concatenate([zs, z], 0)

    ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=markersize, alpha=.3)

    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])
    ax.set_title('f) '+get_sentence(question_batch[0]), fontsize=title_size)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    # ax.set_xticks([])
    ax.tick_params(axis='x', colors='white')
    ax.grid(True, alpha=.3)




    ax = plt.subplot2grid((rows,cols), (3,0), frameon=False)

    for i in range(n_batches):
        img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, [train_indexes[1]])
        z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)

        if i == 0:
            zs = z
        else:
            zs = np.concatenate([zs, z], 0)

    ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=markersize, alpha=.3)

    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])
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

    ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=markersize, alpha=.3)

    zs_11 = zs

    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])
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

    ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=markersize, alpha=.3)

    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])
    ax.set_title('i) k'+str(k)+' ' +get_sentence(question_batch[0]), fontsize=title_size)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    # ax.set_xticks([])
    ax.grid(True, alpha=.3)
    ax.tick_params(axis='x', colors='white')







    ax = plt.subplot2grid((rows,cols), (4,1), frameon=False)


    ax.scatter(x=zs_11[:,0],y=zs_11[:,1], c='blue', s=markersize, alpha=.3)
    ax.scatter(x=zs[:,0],y=zs[:,1], c='red', s=markersize, alpha=.3)

    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])
    ax.set_title('j) combined the k plots', fontsize=title_size)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    # ax.set_xticks([])
    ax.grid(True, alpha=.3)
    ax.tick_params(axis='x', colors='white')





    word1 = 'small red metal sphere'
    red_indexes = get_indexes(train_question_dataset, word=word1)

    ax = plt.subplot2grid((rows,cols), (5,0), frameon=False)

    for i in range(n_batches):
        img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, red_indexes)
        z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)

        if i == 0:
            zs = z
        else:
            zs = np.concatenate([zs, z], 0)

    ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=markersize, alpha=.3)

    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])
    ax.set_title('k) x agg.'+' ' +get_sentence(question_batch[0]), fontsize=title_size)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    # ax.set_xticks([])
    ax.grid(True, alpha=.3)





    word1 = 'small blue metal sphere'
    red_indexes = get_indexes(train_question_dataset, word=word1)

    ax = plt.subplot2grid((rows,cols), (5,1), frameon=False)

    for i in range(n_batches):
        img_batch, question_batch = make_batch2(train_image_dataset, train_question_dataset, batch_size, red_indexes)
        z = SIR_x(self, img_batch, inf_type=1, dec_type=1, k=1)

        if i == 0:
            zs = z
        else:
            zs = np.concatenate([zs, z], 0)

    ax.scatter(x=zs[:,0],y=zs[:,1], c='blue', s=markersize, alpha=.3)

    ax.set_xlim(left=-8, right=8)
    ax.set_ylim([-8,8])
    ax.set_title('l) x agg.'+' ' +get_sentence(question_batch[0]), fontsize=title_size)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    # ax.set_xticks([])
    ax.grid(True, alpha=.3)





    plt_path = images_dir+'fig1_2D_prior'+str(step)+'.png'
    plt.savefig(plt_path)
    print ('saved plot', plt_path)
    plt.close()




    

def viz_more(images, images_dir, step_count):


    rows = 1
    cols= 3
    fig = plt.figure(figsize=(7+cols,2+rows), facecolor='white', dpi=150)


    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)
    image = images[0].data.cpu().numpy() * 255.
    image = np.rollaxis(image, 1, 0)
    image = np.rollaxis(image, 2, 1)# [112,112,3]
    image = np.uint8(image)
    ax.imshow(image) #, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)


    ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)
    image = images[1].data.cpu().numpy() * 255.
    image = np.rollaxis(image, 1, 0)
    image = np.rollaxis(image, 2, 1)# [112,112,3]
    image = np.uint8(image)
    ax.imshow(image) #, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)
    image = images[2].data.cpu().numpy() * 255.
    image = np.rollaxis(image, 1, 0)
    image = np.rollaxis(image, 2, 1)# [112,112,3]
    image = np.uint8(image)
    ax.imshow(image) #, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])


    # plt.tight_layout()
    plt_path = images_dir + 'img'+str(step_count)+'_learnedprior.png'
    plt.savefig(plt_path)
    print ('saved viz',plt_path)
    plt.close(fig)








































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

    parser.add_argument('--qy_type', type=str, default='true')

    parser.add_argument('--learn_prior', type=int, default=0)


    
    
    
    

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






    if args.singlev2 and args.z_size == 2:

        

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








    print ('Training')
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


