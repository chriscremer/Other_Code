
from os.path import expanduser
home = expanduser("~")

import pickle
import _pickle as cPickle

import numpy as np
import torch
from torch.utils.data import Dataset

# from preprocess_statements import preprocess_v2
# from Clevr_data_loader import ClevrDataset, ClevrDataLoader
from clevr_data_utils import ClevrDataLoader, preprocess_v2

import scipy.io


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os




def load_clevr(data_dir, quick=0):

    print ('Loading CLEVR')

    # if vws:
    #     data_dir = home+ "/vl_data/two_objects_large/"  #vws
    # else:
    #     data_dir = home+ "/VL/data/two_objects_no_occ/" #boltz 

    question_file = data_dir+'train.h5'
    image_file = data_dir+'train_images.h5'
    vocab_file = data_dir+'train_vocab.json'
    #Load data  (3,112,112)
    train_loader_kwargs = {
                            'question_h5': question_file,
                            'feature_h5': image_file,
                            # 'batch_size': batch_size,
                            # 'max_samples': 70000, i dont think this actually does anythn
                            }
    loader = ClevrDataLoader(**train_loader_kwargs)


    # print ('step 1')

    # class MyClevrDataset(Dataset):
    #     """Face Landmarks dataset."""

    #     def __init__(self, data):

    #         self.data = data

    #     def __len__(self):
    #         return len(self.data)

    #     def __getitem__(self, idx):

    #         img_batch = self.data[idx]
    #         img_batch = torch.from_numpy(img_batch) #.cuda()
    #         img_batch = torch.clamp(img_batch, min=1e-5, max=1-1e-5)

    #         return img_batch 


    # # quick = args.quick

    # if quick:
    #     print ('loading quick data')
    #     #For quickly running it
    #     if vws:
    #         quick_check_data = home+ "/vl_data/quick_stuff.pkl"
    #     else:
    #         quick_check_data = home+ "/VL/two_objects_large/quick_stuff.pkl" 

    #     with open(quick_check_data, "rb" ) as f:
    #         stuff = pickle.load(f)
    #         train_image_dataset, train_question_dataset, val_image_dataset, \
    #              val_question_dataset, test_image_dataset, test_question_dataset, \
    #              train_indexes, val_indexes, question_idx_to_token, \
    #         question_token_to_idx, q_max_len, vocab_size = stuff

    #     # img_batch, question_batch = get_batch(train_image_dataset, train_question_dataset, batch_size=4)
    #     # train_image_dataset = train_image_dataset[:1]
    #     # val_image_dataset = val_image_dataset[:1]
    #     # test_image_dataset = test_image_dataset[:1]
    #     test_image_dataset = train_image_dataset



    # else:

    if quick:
        print ('loading quick data')
        n_train = 500
        n_val = 1000
    else:
        n_train = 50000
        n_val = 10000   

    train_image_dataset, train_question_dataset, val_image_dataset, \
         val_question_dataset, test_image_dataset, test_question_dataset, \
         train_indexes, val_indexes, question_idx_to_token, \
         question_token_to_idx, q_max_len, vocab_size =  preprocess_v2(loader, vocab_file, n_train=n_train, n_val=n_val)


    print (train_image_dataset.shape)
    print (val_image_dataset.shape)

    test_image_dataset = val_image_dataset

    # fdasfads

    # train_image_dataset = train_image_dataset[:50000]
    # test_image_dataset = val_image_dataset[:10000]

    # train_image_dataset = train_image_dataset[:22]


    # train_x = MyClevrDataset(train_image_dataset)
    # test_x = MyClevrDataset(test_image_dataset)

    train_x = torch.from_numpy(train_image_dataset)
    test_x = torch.from_numpy(test_image_dataset)

    # img_batch = torch.from_numpy(img_batch) #.cuda()

    return train_x, test_x



def numpy(x):
    return x.data.cpu().numpy()














# exp_dir = args.save_to_dir + args.exp_name + '/'
# params_dir = exp_dir + 'params/'
# images_dir = exp_dir + 'images/'
# code_dir = exp_dir + 'code/'


exp_dir = home+'/Documents/glow_clevr/checkclevr/'

if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    print ('Made dir', exp_dir) 

# if not os.path.exists(params_dir):
#     os.makedirs(params_dir)
#     print ('Made dir', params_dir) 

# if not os.path.exists(images_dir):
#     os.makedirs(images_dir)
#     print ('Made dir', images_dir) 

# if not os.path.exists(code_dir):
#     os.makedirs(code_dir)
#     print ('Made dir', code_dir) 





machine = 'vws'
quick = 0

# # CLEVR DATA
if machine in ['vws', 'vector', 'vaughn']:
    data_dir = home+ "/vl_data/two_objects_large/"  #vws
else:
    data_dir = home+ "/VL/data/two_objects_no_occ/" #boltz 

train_x, test_x = load_clevr(data_dir=data_dir, quick=quick)

# test_x = test_x[:30]

print ('Training Set', train_x.shape)
print ('Test Set',  test_x.shape)


img_idx = []
closest_idx = []
closest_dist = []

for i in range (len (train_x)):

    datapoint = train_x[i].view(1,3,112,112)
    # print (torch.sum(torch.abs(datapoint - train_x[0])))
    # fdasf
    # print (datapoint.shape)
    # print (test_x.shape, datapoint.shape)
    dif = test_x - datapoint
    # print (dif.shape)
    # fas
    dif = torch.abs(dif)
    dif = dif.view(len(test_x),-1).sum(-1)
    

    # print (dif)

    # print (dif[:10])

    # print (i, torch.min(dif), torch.max(dif),)
    # print (dif.shape)

    dist, idx = torch.min(dif,0)
    # print(dif[91])
    # print (dist, idx)
    # print (torch.sum(torch.abs(train_x[0] - test_x[idx])))
    # print (torch.sum(torch.abs(test_x[idx] - train_x[0])))
    # fsd

    closest_dist.append(numpy(dist))
    closest_idx.append(numpy(idx))
    img_idx.append(i)

    # print (torch.min(dif,0))

    # fsad

    if i % 100 ==0:

        print (i, len (train_x), np.min(closest_dist))

    # if i > 100:
    #     break


closest_idx = np.array(closest_idx)
closest_dist = np.array(closest_dist)
img_idx = np.array(img_idx)


# print (torch.sum(torch.abs(train_x[img_idx[0]] - test_x[closest_idx[0]])))
# print (closest_idx[0])
# print (closest_dist[0])
# print (img_idx[0])
# fadfs


sort_idxs = np.argsort(closest_dist)

closest_dist = closest_dist[sort_idxs]
closest_idx = closest_idx[sort_idxs]
img_idx = img_idx[sort_idxs]

print (closest_dist[:10])


def make_image_subplot(rows, cols, row, col, image, text):

    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False)
   
    image = image.data.cpu().numpy() * 255.
    image = np.rollaxis(image, 1, 0)
    image = np.rollaxis(image, 2, 1)# [112,112,3]
    image = np.uint8(image)
    ax.imshow(image) #, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)


cols = 2
rows = 5

fig = plt.figure(figsize=(7+cols,2+rows), facecolor='white', dpi=150)

for i in range(rows):

    # print ( torch.sum(torch.abs(train_x[img_idx[i]] - test_x[closest_idx[i]])))

    make_image_subplot(rows=rows, cols=cols, row=i, col=0, image=train_x[img_idx[i]], text=str(closest_dist[i]))
    make_image_subplot(rows=rows, cols=cols, row=i, col=1, image=test_x[closest_idx[i]], text='')



# make_text_subplot(self, rows, cols, row=0, col=1, text='\n\n\n'+get_sentence(questions[img_i], newline_every=[3,4]))

# #Row 1
# make_image_subplot(self, rows, cols, row=1, col=0, image=training_recon_img[img_i], text='')
# sentence_classifier = get_sentence(torch.max(training_y_hat, 2)[1][img_i], newline_every=[3])
# sentence = get_sentence(training_recon_q_sampled_words[img_i], newline_every=[3,4]) 
# make_text_subplot(self, rows, cols, row=1, col=1, text=sentence+'\n\nClassifier:\n'+sentence_classifier)
# sentence = get_sentence2(training_recon_q_sampled_words[img_i])
# dist = training_recon_q_dist[img_i]
# for i in range(len(dist)):
#     make_bar_subplot(self, rows, cols, row=1, col=2+i, range_=range(self.vocab_size), values_=dist[i], text=sentence[i], sampled_word=training_recon_q_sampled_words[img_i][i])


# plt.tight_layout()
plt_path = exp_dir + 'img.png'
plt.savefig(plt_path)
print ('saved viz',plt_path)
plt.close(fig)










fasdfa
