

from os.path import expanduser
home = expanduser("~")

import pickle
import _pickle as cPickle

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess_statements import preprocess_v2
from Clevr_data_loader import ClevrDataset, ClevrDataLoader



def load_clevr(batch_size, vws, quick=0):

    print ('Loading CLEVR')

    if vws:
        data_dir = home+ "/vl_data/two_objects_large/"  #vws
    else:
        data_dir = home+ "/VL/data/two_objects_no_occ/" #boltz 

    question_file = data_dir+'train.h5'
    image_file = data_dir+'train_images.h5'
    vocab_file = data_dir+'train_vocab.json'
    #Load data  (3,112,112)
    train_loader_kwargs = {
                            'question_h5': question_file,
                            'feature_h5': image_file,
                            'batch_size': batch_size,
                            # 'max_samples': 70000, i dont think this actually does anythn
                            }
    loader = ClevrDataLoader(**train_loader_kwargs)

    class MyClevrDataset(Dataset):
        """Face Landmarks dataset."""

        def __init__(self, data):

            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):

            img_batch = self.data[idx]
            img_batch = torch.from_numpy(img_batch) #.cuda()
            img_batch = torch.clamp(img_batch, min=1e-5, max=1-1e-5)

            return img_batch 


    # quick = args.quick

    if not quick:
        #Full dataset
        train_image_dataset, train_question_dataset, val_image_dataset, \
             val_question_dataset, test_image_dataset, test_question_dataset, \
             train_indexes, val_indexes, question_idx_to_token, \
             question_token_to_idx, q_max_len, vocab_size =  preprocess_v2(loader, vocab_file)

        train_image_dataset = train_image_dataset[:50000]


    else:
        #For quickly running it
        if vws:
            quick_check_data = home+ "/vl_data/quick_stuff.pkl"
        else:
            quick_check_data = home+ "/VL/two_objects_large/quick_stuff.pkl" 

        with open(quick_check_data, "rb" ) as f:
            stuff = pickle.load(f)
            train_image_dataset, train_question_dataset, val_image_dataset, \
                 val_question_dataset, test_image_dataset, test_question_dataset, \
                 train_indexes, val_indexes, question_idx_to_token, \
            question_token_to_idx, q_max_len, vocab_size = stuff

        # img_batch, question_batch = get_batch(train_image_dataset, train_question_dataset, batch_size=4)


    # train_image_dataset = train_image_dataset[:1]
    val_image_dataset = val_image_dataset[:1]
    test_image_dataset = test_image_dataset[:1]



    # train_image_dataset = train_image_dataset[:22]
    dataset = MyClevrDataset(train_image_dataset)


    return dataset













def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def load_cifar(data_dir):

    print ('Loading CIFAR')
    # file_ = home+'/Documents/cifar-10-batches-py/data_batch_'
    file_ = data_dir + '/cifar-10-batches-py/data_batch_'

    for i in range(1,6):
        file__ = file_ + str(i)
        b1 = unpickle(file__)
        if i ==1:
            train_x = b1['data']
            train_y = b1['labels']
        else:
            train_x = np.concatenate([train_x, b1['data']], axis=0)
            train_y = np.concatenate([train_y, b1['labels']], axis=0)

    file__ = data_dir + '/cifar-10-batches-py/test_batch'
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

    # print (train_x.shape)
    # print (test_x.shape)
    # print ()

    class MyCIFARDataset(Dataset):
        """Face Landmarks dataset."""

        def __init__(self, data):

            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):

            img_batch = self.data[idx]
            # img_batch = torch.from_numpy(img_batch) #.cuda()
            # img_batch = torch.clamp(img_batch, min=1e-5, max=1-1e-5)

            return img_batch 

    dataset = MyCIFARDataset(train_x)

    return dataset

































