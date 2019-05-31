

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset


import numpy as np
# import pdb
import argparse
import time
import subprocess

from invertible_layers import * 
from utils import * 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



from os.path import expanduser
home = expanduser("~")

import sys, os
sys.path.insert(0, os.path.abspath(home+'/Other_Code/VLVAE/CLEVR/utils'))


from preprocess_statements import preprocess_v2
from Clevr_data_loader import ClevrDataset, ClevrDataLoader

import pickle



# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# training

parser.add_argument('--exp_name', default='glow_improve3_3_16', type=str)
parser.add_argument('--save_to_dir', default=home+'/Documents/glow_clevr/', type=str)
parser.add_argument('--which_gpu', default='1', type=str)

# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=16)

# parser.add_argument('--depth', type=int, default=32) 
# parser.add_argument('--n_levels', type=int, default=3) 
# parser.add_argument('--depth', type=int, default=8) 
# parser.add_argument('--depth', type=int, default=6)  
parser.add_argument('--n_levels', type=int, default=3) 
parser.add_argument('--depth', type=int, default=16)  
# parser.add_argument('--depth', type=int, default=16) 

# parser.add_argument('--n_levels', type=int, default=5) 

parser.add_argument('--norm', type=str, default='actnorm')
# parser.add_argument('--permutation', type=str, default='conv')
parser.add_argument('--permutation', type=str, default='shuffle')
# parser.add_argument('--coupling', type=str, default='affine')
parser.add_argument('--coupling', type=str, default='additive')


parser.add_argument('--n_bits_x', type=int, default=8)
# parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--learntop', action='store_true')
# parser.add_argument('--n_warmup', type=int, default=20, help='number of warmup epochs')
parser.add_argument('--lr', type=float, default=2e-4)
# parser.add_argument('--lr', type=float, default=1e-5)
# logging
parser.add_argument('--print_every', type=int, default=200, help='print NLL every _ minibatches')
parser.add_argument('--curveplot_every', type=int, default=2000)
parser.add_argument('--plotimages_every', type=int, default=2000)
parser.add_argument('--save_every', type=int, default=10000, help='save model every _ epochs')
parser.add_argument('--max_steps', type=int, default=200000)

# parser.add_argument('--data_dir', type=str, default='../pixelcnn-pp')
parser.add_argument('--data_dir', type=str, default=home +'/Documents')


parser.add_argument('--load_step', type=int, default=0)
# parser.add_argument('--load_dir', type=str, default=None, help='directory to load existing model')
parser.add_argument('--load_dir', type=str, default=home+'/Documents/glow_clevr/glow_improve/params/')
args = parser.parse_args()
args.n_bins = 2 ** args.n_bits_x

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)




def plot_curve2(data_dict, exp_dir):

    rows = len(data_dict) - 1
    cols = 1
    fig = plt.figure(figsize=(8+cols,8+rows), facecolor='white') #, dpi=150)

    steps = data_dict['steps']
    col=0
    row=0
    for k,v in data_dict.items():
        if k == 'steps':
            continue

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

        if type(v) == dict:
            for k2,v2 in v.items():
                ax.plot(steps, v2, label=k2+' '+str(v2[-1]))

        else:
            ax.plot(steps, v, label=v[-1])

        ax.legend()
        ax.grid(True, alpha=.3) 
        ax.set_ylabel(k)

        if row==0:
            ax.set_title(exp_dir)

        row+=1

    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = exp_dir+'curveplot.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()




















###########################################################################################
print ('\nExp:', args.exp_name)
print ('gpu:', args.which_gpu)
os.environ['CUDA_VISIBLE_DEVICES'] =  args.which_gpu #'1' # '0,1' #args.which_gpu #  '0' #'1' #

exp_dir = args.save_to_dir + args.exp_name + '/'
params_dir = exp_dir + 'params/'
images_dir = exp_dir + 'images/'
code_dir = exp_dir + 'code/'

if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    print ('Made dir', exp_dir) 

if not os.path.exists(params_dir):
    os.makedirs(params_dir)
    print ('Made dir', params_dir) 

if not os.path.exists(images_dir):
    os.makedirs(images_dir)
    print ('Made dir', images_dir) 

if not os.path.exists(code_dir):
    os.makedirs(code_dir)
    print ('Made dir', code_dir) 

#Copy code
subprocess.call("(rsync -r --exclude=__pycache__/ . "+code_dir+" )", shell=True)
print ('code saved')
###########################################################################################









###########################################################################################
# Init Model
# ------------------------------------------------------------------------------
sampling_batch_size = 64
model = Glow_((sampling_batch_size, 3, 112, 112), args).cuda()
# print(model)
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
fasdfad
# model = nn.DataParallel(model).cuda()
###########################################################################################







###########################################################################################
# CLEVR DATA
data_dir = home+ "/VL/data/two_objects_no_occ/"
question_file = data_dir+'train.h5'
image_file = data_dir+'train_images.h5'
vocab_file = data_dir+'train_vocab.json'
#Load data  (3,112,112)
train_loader_kwargs = {
                        'question_h5': question_file,
                        'feature_h5': image_file,
                        'batch_size': args.batch_size,
                        # 'max_samples': 70000, i dont think this actually does anythn
                        }
loader = ClevrDataLoader(**train_loader_kwargs)

train_image_dataset, train_question_dataset, val_image_dataset, \
     val_question_dataset, test_image_dataset, test_question_dataset, \
     train_indexes, val_indexes, question_idx_to_token, \
     question_token_to_idx, q_max_len, vocab_size =  preprocess_v2(loader, vocab_file)


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

# quick_check_data = home+ "/VL/two_objects_large/quick_stuff.pkl" 
# with open(quick_check_data, "rb" ) as f:
#     stuff = pickle.load(f)
#     train_image_dataset, train_question_dataset, val_image_dataset, \
#          val_question_dataset, test_image_dataset, test_question_dataset, \
#          train_indexes, val_indexes, question_idx_to_token, \
    # question_token_to_idx, q_max_len, vocab_size = stuff

# img_batch, question_batch = get_batch(train_image_dataset, train_question_dataset, batch_size=4)


train_image_dataset = train_image_dataset[:50000]
# train_image_dataset = train_image_dataset[:22]
dataset = MyClevrDataset(train_image_dataset)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=0, drop_last=True)
print (train_image_dataset.shape)
###########################################################################################





###########################################################################################
# Optimizer, Param Init and Load
# ------------------------------------------------------------------------------
# set up the optimizer
optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=.0001)
# optim = optim.Adam(model.parameters(), lr=1e-5, weight_decay=.0001)
# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=45, gamma=0.1)
lr_sched = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.999)
# lr_sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.999)

# for i in range (200000):
#     lr_sched.step()
#     if i % 1000==0:
#         print (i, lr_sched.get_lr()[0])
# fsfsda


# # data dependant init
init_loader = torch.utils.data.DataLoader(dataset, batch_size=100, 
                    shuffle=True, num_workers=1, drop_last=True)
with torch.no_grad():
    model.eval()
    for img in init_loader:
        img = img.cuda()
        objective = torch.zeros_like(img[:, 0, 0, 0])
        _ = model(img, objective)
        break


if args.load_step > 0:
    model.load_params_v3(load_dir=args.load_dir, step=args.load_step, name='model_params')

# # once init is done, we leverage Data Parallel
# model = nn.DataParallel(model).cuda()
# start_epoch = 0

# # load trained model if necessary (must be done after DataParallel)
# if args.load_dir is not None: 
#     model, optim, start_epoch = load_session(model, optim, args)
###########################################################################################








###########################################################################################
# training loop
# ------------------------------------------------------------------------------
max_steps = args.max_steps
model.train()
t = time.time()

    
data_dict = {}
data_dict['steps'] = []
data_dict['lpx'] = []
data_dict['lr'] = []
data_dict['T'] = []

iter_loader = iter(loader)
for i in range(max_steps+1):
    step = i + args.load_step

    
    try:
        img = next(iter_loader).cuda()
    except:
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=0, drop_last=True)
        iter_loader = iter(loader)
        img = next(iter_loader).cuda()
    
    
    # img = img.cuda() 
    objective = torch.zeros_like(img[:, 0, 0, 0])

    # discretizing cost 
    objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
    
    # log_det_jacobian cost (and some prior from Split OP)
    z, objective = model(img, objective)

    nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
    
    # Generative loss
    nobj = torch.mean(nll)

    optim.zero_grad()
    nobj.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 5)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
    optim.step()
    lr_sched.step()


    if step %args.print_every == 0:
        T=time.time() - t
        print(step, 'T:{:.1f}'.format(T), 
                'C:{:.2f}'.format(float(nobj.data.cpu().numpy())), 
                'lr', lr_sched.get_lr()[0])
        t = time.time()


        if step > 40:

            data_dict['steps'].append(step)
            data_dict['lpx'].append(float(nobj.data.cpu().numpy()))
            data_dict['lr'].append(lr_sched.get_lr()[0])
            data_dict['T'].append(T)


    if step % args.curveplot_every ==0 and step > 0 and len(data_dict['steps']) > 2:
        plot_curve2(data_dict, exp_dir)


    if step % args.plotimages_every == 0 and step!=0:

        sample = model.sample()
        print (torch.min(sample), torch.max(sample), sample.shape)
        if (sample!=sample).any():
            print ('nans!!')
            fafad

        sample = torch.clamp(sample, min=1e-5, max=1-1e-5)
        grid = utils.make_grid(sample)
        img_file = images_dir +'plot'+str(step)+'.png'

        utils.save_image(grid, img_file)
        print ('saved', img_file)


    if step % args.save_every ==0 and i > 0:
        
        model.save_params_v3(params_dir, step, name='model_params')

print ('Done.')
###########################################################################################


























        
