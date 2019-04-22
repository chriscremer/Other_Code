




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


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image


from celeba_classifier import attribute_classifier












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
        # random.shuffle(word_idxs)
        # print (word_idxs)

        # while len(word_idxs) < 9:
        #     word_idxs.append(0)

        attributes = np.zeros((19))
        for j in range(len(word_idxs)):
            attributes[word_idxs[j]-1] = 1.

        # print (word_idxs)
        # print (attributes)
        # dsfa
        # print (attributes.shape)
        # fds
        text_batch.append(attributes)


    img_batch = np.stack(img_batch) #[B,C,W,H]
    text_batch = np.stack(text_batch) #, 1) #[T,B,L]

    # print (text_batch.shape)
    # fasd

    img_batch = torch.from_numpy(img_batch).cuda()
    text_batch = torch.from_numpy(text_batch).float().cuda()
    # img_batch = rescaling(img_batch)
    img_batch = torch.clamp(img_batch, min=1e-5, max=1-1e-5)

    # print (img_batch.shape, text_batch.shape)
    # sdfdafs

    return img_batch, text_batch #, answer_batch





def make_batch_with_att(image_dir, attr_dict, batch_size, indexes, word_to_idx, att): # val=False, SSL=False): #, answer_dataset): #, image_idxs, first_question_idx):


    img_batch = []
    text_batch = []
    # for i in range(batch_size):
    # i = 1
    i = 180000
    while len(img_batch) < batch_size:


        idx = i #random.choice(indexes)


        numb = str(idx) 
        while len(numb) != 6:
            numb = '0' + numb


        word_idxs = get_text_idxs(attr_dict[numb+'.jpg'], word_to_idx)

        if att in word_idxs:

            attributes = np.zeros((19))
            for j in range(len(word_idxs)):
                attributes[word_idxs[j]-1] = 1.

            text_batch.append(attributes)



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


        i+=1


    img_batch = np.stack(img_batch) #[B,C,W,H]
    text_batch = np.stack(text_batch) #, 1) #[T,B,L]

    # print (text_batch.shape)
    # fasd

    img_batch = torch.from_numpy(img_batch).cuda()
    text_batch = torch.from_numpy(text_batch).float().cuda()
    # img_batch = rescaling(img_batch)
    img_batch = torch.clamp(img_batch, min=1e-5, max=1-1e-5)

    # print (img_batch.shape, text_batch.shape)
    # sdfdafs

    return img_batch, text_batch #, answer_batch

















def train2(classifier, max_steps, load_step, 
                image_dir, train_indexes, val_indexes, attr_dict, word_to_idx,
                save_dir, params_dir, images_dir, batch_size,
                display_step, save_steps, trainingplot_steps, viz_steps, start_storing_data_step): #, k): #, prior_classifier):

    random.seed( 99 )

    # all_dict = {}
    # recent_dict = {}

    # list_recorded_values = [ 
    #         'all_train_elbos', 'all_valid_elbos', 
    #         'all_logpxs', 'all_logpxs_val', 'all_logpys', 'all_logpys_val',
    #         'all_logpzs', 'all_logqzs', 'all_logqzys',
    #         'all_logpzs_val', 'all_logqzs_val', 'all_logqzys_val',
    #         # 'all_MIs', 'all_MIs_val', 'all_LL', 'all_SLL', 'all_LL_val', 'all_SLL_val', 
    #         # 'all_real_acc', 'all_real_val_acc', 'all_recon_acc', 'all_y_recon_acc', 
    #         # 'all_recon_acc_val', 'all_y_recon_acc_val', 
    #         # 'all_prior_acc', 'all_prior_acc2', 
    #         # 'all_x_inf_acc', 'all_y_inf_acc', 'all_x_inf_acc_entangle', 'all_y_inf_acc_entangle',
    #         # 'all_y_inf_acc_k1', 'all_x_inf_acc_k1', 
    #         # 'all_logvar_max', 'all_logvar_min', 'all_logvar_mean',
    #         # 'acc0', 'acc1', 'acc2', 'acc3',
    #         # 'acc0_prior', 'acc1_prior', 'acc2_prior', 'acc3_prior',
    #         # 'SSL1', 'SSL2',
    #         # 'SSL1_k1', 'SSL2_k1',
    #         # 'SSL1_k100', 'SSL2_k100',
    # ]

    # all_dict['all_steps'] = []
    # all_dict['all_betas'] = []

    # for label in list_recorded_values:
    #     all_dict[label] = []
    #     recent_dict[label] = deque(maxlen=10)

    # def add_to_dict(dict1, label, value):
    #     dict1[label].append(value)


    classifier.train()

    # self.B = batch_size

    start_time = time.time()
    # step_count = 0




    #acc per attribute
    
    for i in range(1,20):
        img_batch, question_batch = make_batch_with_att(image_dir, attr_dict, batch_size, indexes=train_indexes, word_to_idx=word_to_idx, att=i)

        # print (img_batch.shape)
        # print (question_batch.shape)
        # fasfd
        accs = classifier.classifier_attribute_accuracies(x=img_batch, attributes=question_batch)

        # print (accs)
        print (i-1, accs[i-1])

    adfas


    train_accs = []
    val_accs = []
    for ii in range(20):
        img_batch, question_batch = make_batch(image_dir, attr_dict, batch_size, indexes=train_indexes, word_to_idx=word_to_idx)
        classi_loss_val, acc_val = classifier.classifier_loss(x=img_batch, attributes=question_batch)
        train_accs.append(acc_val.data.item())
        img_batch, question_batch = make_batch(image_dir, attr_dict, batch_size, indexes=val_indexes, word_to_idx=word_to_idx)
        classi_loss_val, acc_val = classifier.classifier_loss(x=img_batch, attributes=question_batch)
        val_accs.append(acc_val.data.item())

    print ('train:', np.mean(train_accs))
    print ('val:', np.mean(val_accs))

    fsdaa



    for step in range(max_steps):

        # Make batch
        # if not self.just_classifier or self.train_classifier:
        img_batch, question_batch = make_batch(image_dir, attr_dict, batch_size, indexes=train_indexes, word_to_idx=word_to_idx)

        # warmup = min((step_count+load_step) / float(warmup_steps), self.max_beta)


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



        # if self.train_classifier:
        classi_loss, acc = classifier.update(x=img_batch, q=question_batch)


        if step % display_step==0 and step > 0:

            classifier.eval()

            img_batch, question_batch = make_batch(image_dir, attr_dict, batch_size, indexes=val_indexes, word_to_idx=word_to_idx)
            classi_loss_val, acc_val = classifier.classifier_loss(x=img_batch, attributes=question_batch)

            print (step, classi_loss.data.item(), acc.data.item(), 'Val:', classi_loss_val.data.item(), acc_val.data.item())


            classifier.train()



        if step % save_steps==0 and step > 0:
            #Save params
            # self.save_params_v3(save_dir=params_dir, step=step_count+load_step)
            # # self.load_params_v3(save_dir=params_dir, epochs=total_steps)
            # # fdas

            # if self.train_classifier:
            classifier.save_params_v3(save_dir=params_dir, step=step)


            # current_val_elbo = np.mean(recent_valid_elbos)
            # if best_val_elbo == None or best_val_elbo < current_val_elbo:
            #     best_val_elbo = current_val_elbo
            #     self.best_step = step_count+load_step

        #     # print (self.best_step, best_val_elbo)

        #         self.optimizer_x.step()
        #         self.scheduler_x.step()

        #     else:
        #         self.optimizer_x.zero_grad()
        #         outputs_only_x = self.forward(x=img_batch, q=question_batch, warmup=warmup, inf_type=1, dec_type=0) #, k=k_train)
        #         loss_only_x = -outputs_only_x['welbo']
        #         loss_only_x.backward()
        #         self.optimizer_x.step()
        #         self.scheduler_x.step()

        # if self.train_classifier:
        #     classi_loss, acc = classifier.update(x=img_batch, q=question_batch)

        




        # if step_count % save_steps==0 and step_count > 0:
        #     #Save params
        #     self.save_params_v3(save_dir=params_dir, step=step_count+load_step)
        #     # self.load_params_v3(save_dir=params_dir, epochs=total_steps)
        #     # fdas

        #     if self.train_classifier:
        #         classifier.save_params_v3(save_dir=params_dir, step=step_count+load_step)


        #     # current_val_elbo = np.mean(recent_valid_elbos)
        #     # if best_val_elbo == None or best_val_elbo < current_val_elbo:
        #     #     best_val_elbo = current_val_elbo
        #     #     self.best_step = step_count+load_step

        #     # print (self.best_step, best_val_elbo)















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



    max_steps = 500000
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










    # print ('\nInit VLVAE')
    # model = VLVAE(args_dict)
    # model.cuda()
    # if args.model_load_step>0:
    #     model.load_params_v3(save_dir=args.params_load_dir, step=args.model_load_step)
    # # print(model)
    # # fdsa
    # print ('VLVAE Initilized\n')



    





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
    train2(classifier, max_steps=max_steps, load_step=args.model_load_step,
            image_dir=image_dir, train_indexes=list(range(1,180001)), val_indexes=list(range(180001, n_images)), attr_dict=attr_dict, word_to_idx=word_to_idx,
            save_dir=exp_dir, params_dir=params_dir, images_dir=images_dir, batch_size=args.batch_size,
            display_step=display_step, save_steps=save_steps, trainingplot_steps=trainingplot_steps, viz_steps=viz_steps,
             start_storing_data_step=args.start_storing_data_step) #, k=args.k)
    classifier.save_params_v3(save_dir=params_dir, step=max_steps+args.model_load_step)





    print ('Done.')




# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "celeba_3" \
#                               --input_size 64 \
#                               --joint_inf 0  --flow_int 1 --batch_size 20 \
#                               --w_logpy 100 --w_logpx .1 --max_beta 1 --z_size 50 \
#                               --data_dir "$HOME/VL/data/celebA/" \
#                               --save_to_dir "$HOME/Documents/VLVAE_exps/" \
#                               --just_classifier 0 \
#                               --train_classifier 0 \
#                               --which_gpu '0' \
#                               --quick_check 0 \
#                               --params_load_dir "" \
#                               --model_load_step 0 \
#                               --display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
#                               --start_storing_data_step 2001 --save_params_step 50000 \














