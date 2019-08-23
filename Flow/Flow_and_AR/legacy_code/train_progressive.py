

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils



import numpy as np
# import pdb
import argparse
import time
import subprocess
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from os.path import expanduser
home = expanduser("~")

import sys, os
sys.path.insert(0, os.path.abspath(home+'/Other_Code/VLVAE/CLEVR/utils'))


from invertible_layers import * 
from utils import * 


from load_data import load_clevr, load_cifar, load_svhn, load_cifar_adjustable


# from random import shuffle
# from random import sample
import random

# print (sample([1,2,3,4], 2))
# fadsf

# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# training

parser.add_argument('--exp_name', default='clevr_gettingprobofsamples', type=str)
parser.add_argument('--save_to_dir', default=home+'/Documents/glow_clevr/', type=str)
parser.add_argument('--which_gpu', default='0', type=str)

# parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--batch_size', type=int, default=16)

# parser.add_argument('--depth', type=int, default=32) 
parser.add_argument('--n_levels', type=int, default=3) 
# parser.add_argument('--depth', type=int, default=8) 
# parser.add_argument('--depth', type=int, default=6)  
# parser.add_argument('--n_levels', type=int, default=6) 
parser.add_argument('--depth', type=int, default=16)  
# parser.add_argument('--depth', type=int, default=16) 
parser.add_argument('--hidden_channels', type=int, default=128) 


parser.add_argument('--AR_resnets', type=int, default=5) 
parser.add_argument('--AR_channels', type=int, default=64) 

# parser.add_argument('--n_levels', type=int, default=5) 

parser.add_argument('--norm', type=str, default='actnorm')
# parser.add_argument('--permutation', type=str, default='conv')
parser.add_argument('--permutation', type=str, default='shuffle')
# parser.add_argument('--coupling', type=str, default='affine')
# parser.add_argument('--coupling', type=str, default='additive')
parser.add_argument('--coupling', type=str, default='additive')
parser.add_argument('--base_dist', type=str, default='Gauss')


parser.add_argument('--dataset_size', type=int, default=0)


parser.add_argument('--n_bits_x', type=int, default=8)
# parser.add_argument('--n_epochs', type=int, default=2000)
# parser.add_argument('--learntop', action='store_true')
# parser.add_argument('--learntop', type=bool, default=False)
# parser.add_argument('--n_warmup', type=int, default=20, help='number of warmup epochs')
parser.add_argument('--lr', type=float, default=4e-4)
# parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--lr', type=float, default=1e-5)
# logging
# parser.add_argument('--print_every', type=int, default=200, help='print NLL every _ minibatches')
# parser.add_argument('--print_every', type=int, default=100, help='print NLL every _ minibatches')
parser.add_argument('--print_every', type=int, default=1, help='print NLL every _ minibatches')
parser.add_argument('--curveplot_every', type=int, default=2000)
parser.add_argument('--plotimages_every', type=int, default=2000)
# parser.add_argument('--plotimages_every', type=int, default=200)
# parser.add_argument('--plotimages_every', type=int, default=1)
parser.add_argument('--save_every', type=int, default=10000, help='save model every _ epochs')
parser.add_argument('--max_steps', type=int, default=200000)

parser.add_argument('--dataset', type=str)
parser.add_argument('--quick', type=int, default=0)
parser.add_argument('--vws', type=int, default=1)
parser.add_argument('--save_output', type=int, default=0)

# parser.add_argument('--data_dir', type=str, default='../pixelcnn-pp')
parser.add_argument('--data_dir', type=str, default=home +'/Documents')


parser.add_argument('--load_step', type=int, default=0)
# parser.add_argument('--load_dir', type=str, default=None, help='directory to load existing model')
parser.add_argument('--load_dir', type=str, default=home+'/Documents/glow_clevr/glow_sigmoid/params/')
args = parser.parse_args()
args.n_bins = 2 ** args.n_bits_x

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)



def myprint(list_):
    newtext =''
    for i in range(len(list_)):
        newtext += str(list_[i]) + ' '
    print(newtext)

    if save_output:        
        with open(write_to_file, "a") as f:
            for t in list_:
                f.write(str(t) + ' ')
            f.write('\n')

def myprint_t(text):
    print(text)
    if save_output:        
        with open(write_to_file, "a") as f:
            f.write(str(text))
            f.write('\n')




def numpy(x):
    return x.data.cpu().numpy()

def logmeanexp(x):

    max_ = torch.max(x)
    print (max_.shape)
    lme = torch.log(torch.mean(torch.exp(x-max_))) + max_
    dsfad
    return lme



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





def sample_batch(dataset, indexes, batch_size):
    if len(indexes) <= batch_size:
        idxs = indexes
    else:
        idxs = random.sample(indexes, batch_size)

    batch = dataset[idxs].cuda()
    # print (batch.shape)
    # fdsfa
    return batch, idxs.copy()
















###########################################################################################
print ('\nExp:', args.exp_name)
print ('gpu:', args.which_gpu)
if args.which_gpu != 'all':
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

save_output = args.save_output
if save_output:

    for f in os.listdir(exp_dir):
        if 'exp_stdout.txt' in f:
            os.remove(exp_dir + f)

    write_to_file = exp_dir+'exp_stdout.txt'
###########################################################################################













###########################################################################################
# LOAD DATA

if args.dataset=='clevr':
    # # CLEVR DATA
    dataset = load_clevr(batch_size=args.batch_size, vws=args.vws, quick=args.quick)

elif args.dataset=='cifar':
    # # CIFAR DATA
    # # train_image_dataset = load_cifar(data_dir=args.data_dir)
    # train_x, test_x = load_cifar(data_dir=home+'/Documents/', dataset_size=args.dataset_size)
    # dataset = train_x
    # # print (len(test_x), 'test set len')


    svhn_test_x = load_svhn(data_dir=home+'/Documents/')
    # svhn_test_x = test_x


    # load_cifar_adjustable


    # training_set = dataset.data[:100]

    train_x, test_x = load_cifar_adjustable(data_dir=home+'/Documents/')


N_alldata = len(train_x.full_dataset)
print (len(train_x), train_x[0].shape, N_alldata)
# train_x.current_size = 200
# print (len(train_x), train_x[0].shape, len(train_x.full_dataset))

# sample_batch(dataset=train_x.full_dataset, indexes=list(range(100)), batch_size=args.batch_size)
inti_size = 16
training_indexes_notpassed = list(range(inti_size))
training_indexes = list(range(inti_size))
training_indexes_rest = list(range(inti_size,len(train_x)))
###########################################################################################






###########################################################################################
# Init Model
# ------------------------------------------------------------------------------
sampling_batch_size = 64
shape = train_x[0].shape
model = Glow_((sampling_batch_size, shape[0], shape[1], shape[2]), args).cuda()
# print(model)
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
# fasdfad
# model = nn.DataParallel(model).cuda()
for layer in model.layers:
    if 'AR_P' in str(layer):
        print("number of AR parameters:", sum([np.prod(p.size()) for p in layer.parameters()]))
###########################################################################################






###########################################################################################
# Optimizer, Param Init and Loaders
# ------------------------------------------------------------------------------
# set up the optimizer
optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-10)
# optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=.01)
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
if args.load_step == 0:
    init_loader = torch.utils.data.DataLoader(train_x, batch_size=100, 
                        shuffle=True, num_workers=1, drop_last=True)
    with torch.no_grad():
        model.eval()
        for img in init_loader:
            img = img.cuda()
            objective = torch.zeros_like(img[:, 0, 0, 0])
            _ = model(img, objective)
            break
    print ('data dependent init complete')

if args.load_step > 0:
    model.load_params_v3(load_dir=args.load_dir, step=args.load_step, name='model_params')

#multi gpu
print ('gpu count', torch.cuda.device_count())
model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))

# # once init is done, we leverage Data Parallel
# model = nn.DataParallel(model).cuda()
# start_epoch = 0

# # load trained model if necessary (must be done after DataParallel)
# if args.load_dir is not None: 
#     model, optim, start_epoch = load_session(model, optim, args)


# loader = torch.utils.data.DataLoader(train_x, batch_size=min(args.batch_size,len(train_x)), 
#                                 shuffle=True, num_workers=0, drop_last=True)
# iter_loader = iter(loader)
###########################################################################################


























###########################################################################################
# Training loop
# ------------------------------------------------------------------------------
max_steps = args.max_steps
model.train()
t = time.time()

    
data_dict = {}
data_dict['steps'] = []
data_dict['BPD'] = {}
data_dict['BPD']['Train'] = []
data_dict['BPD']['Test'] = []
data_dict['BPD']['SVHN'] = []
data_dict['BPD']['Generated'] = []
data_dict['lr'] = []
data_dict['T'] = []
data_dict['BPD_sd'] = []
# data_dict['BPD_LME'] = []

# data_dict['LL'] = {}
# data_dict['LL']['real_LL'] = []
# data_dict['LL']['sample_LL'] = []

threshold = 3.5

for i in range(max_steps+1):
    step = i + args.load_step

    img, idxs = sample_batch(dataset=train_x.full_dataset, indexes=training_indexes_notpassed, batch_size=args.batch_size)
    img = img.cuda()

    # try:
    #     img = next(iter_loader).cuda()
    # except:
    #     loader = torch.utils.data.DataLoader(train_x, batch_size=min(args.batch_size,len(train_x)), 
    #                         shuffle=True, num_workers=0, drop_last=True)
    #     iter_loader = iter(loader)
    #     img = next(iter_loader).cuda()


    # if step % 3 !=0:

    # dequantize
    img += torch.zeros_like(img).uniform_(0., 1./args.n_bins)
    img = torch.clamp(img, min=1e-5, max=1-1e-5)
    # print (torch.max(img), torch.min(img))
    # print (args.n_bins)
    # fasfd
    
    # img = img.cuda() 
    objective = torch.zeros_like(img[:, 0, 0, 0])
    # discretizing cost 
    objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
    # log_det_jacobian cost (and some prior from Split OP)
    z, objective = model(img, objective)
    nll_train = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
    # Generative loss
    nobj = torch.mean(nll_train)



    passes = numpy(nll_train) < threshold
    if np.sum(passes) > 0:
        # training_indexes_notpassed = []
        for jj in range(len(passes)):
            if passes[jj]:
                training_indexes_notpassed.remove(idxs[jj])



    # grid = utils.make_grid(img)
    # img_file = images_dir +'plot'+str(step)+'.png'
    # utils.save_image(grid, img_file)
    # print ('saved', img_file)



    if (nobj!=nobj).any():
        print ('step', step, '  -nans in obj!!')
        print ('z', (z!=z).any())
        print (nobj)
        fafad


    # if I dont just want to plot
    if args.plotimages_every != 1: 

        optim.zero_grad()
        nobj.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        optim.step()
        lr_sched.step()


















    if step %args.print_every == 0:
        T=time.time() - t
        # print(step, 'T:{:.1f}'.format(T), 
        #         'C:{:.2f}'.format(float(nobj.data.cpu().numpy())), 
        #         'lr', lr_sched.get_lr()[0])
        myprint( (step, 'T:{:.1f}'.format(T), 
                'C:{:.2f}'.format(float(nobj.data.cpu().numpy())), 
                'lr', lr_sched.get_lr()[0],
                'working on', len(training_indexes_notpassed), '/', len(training_indexes), '/', N_alldata,  ) )
        t = time.time()


        if step > 5:

            if args.curveplot_every != -1:

                # RECORD DATA
                data_dict['steps'].append(step)
                data_dict['BPD']['Train'].append(float(nobj.data.cpu().numpy()))
                data_dict['lr'].append(lr_sched.get_lr()[0])
                data_dict['T'].append(T)

                with torch.no_grad():

                    #Get test set likelihood
                    test_loader = torch.utils.data.DataLoader(test_x, batch_size=min(100,len(train_x)), 
                                        shuffle=True, num_workers=1, drop_last=True)
                    iter_test_loader = iter(test_loader)
                    img = next(iter_test_loader).cuda()
                    img = torch.clamp(img, min=1e-5, max=1-1e-5)
                    objective = torch.zeros_like(img[:, 0, 0, 0])
                    objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
                    z, objective = model(img, objective)
                    nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
                    nobj_test = torch.mean(nll)


                    #Get svhn set likelihood
                    test_loader = torch.utils.data.DataLoader(svhn_test_x, batch_size=min(100,len(train_x)), 
                                        shuffle=True, num_workers=1, drop_last=True)
                    iter_test_loader = iter(test_loader)
                    img = next(iter_test_loader).cuda()
                    img = torch.clamp(img, min=1e-5, max=1-1e-5)
                    objective = torch.zeros_like(img[:, 0, 0, 0])
                    objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
                    z, objective = model(img, objective)
                    nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
                    nobj_test_svhn = torch.mean(nll)


                    #Get generated data likelihood
                    sample = model.module.sample()
                    img = torch.clamp(sample, min=1e-5, max=1-1e-5)
                    objective = torch.zeros_like(img[:, 0, 0, 0])
                    objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
                    z, objective = model(img, objective)
                    nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
                    nobj_generated = torch.mean(nll)




                data_dict['BPD']['Test'].append(numpy(nobj_test))
                data_dict['BPD']['SVHN'].append(numpy(nobj_test_svhn))
                data_dict['BPD']['Generated'].append(numpy(nobj_generated))
                data_dict['BPD_sd'].append(np.std(numpy(nll_train)))












    # Progressive Training
    # if step % (args.print_every * 2) ==0 and step>0:
    # if len(training_indexes_notpassed) == 0:
    if step > 100:


        with torch.no_grad():

            # # Check LL of training set 
            # test_loader = torch.utils.data.DataLoader(train_x, batch_size=min(100,len(train_x)), 
            #                     shuffle=False, num_workers=1, drop_last=True)
            # iter_test_loader = iter(test_loader)

            
            # scores = []
            pass_count =0

            # training_indexes = training_indexes[:150]
            training_indexes_notpassed = []


            ii = 0
            while ii < len(training_indexes):
                upper_idx = ii + min(ii+100, len(training_indexes))
                batch_idx = training_indexes[ii:upper_idx]
                batch = train_x.full_dataset[batch_idx]
                ii = upper_idx


                img = torch.clamp(batch, min=1e-5, max=1-1e-5)
                objective = torch.zeros_like(img[:, 0, 0, 0])
                objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
                z, objective = model(img, objective)
                nll_train = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))

                passes = numpy(nll_train) < threshold
                pass_count += np.sum(passes)

                print (numpy(nll_train))

                for jj in range(len(passes)):
                    if not passes[jj]:
                        training_indexes_notpassed.append(batch_idx[jj])




                # scores.extend(list(numpy(nll_train) < threshold))

                # print (scores)
                # fasdfa
            # passing_datapoints = np.sum(np.array(scores))
            print ('passed', pass_count, 'out of', len(training_indexes), 'Training on', len(training_indexes_notpassed))
            print (training_indexes_notpassed)
            print ()
            

            # if passing_datapoints == len(scores):
            if len(training_indexes_notpassed) == 0:
                training_indexes.extend(training_indexes_rest[:100])
                training_indexes_notpassed = training_indexes.copy()
                training_indexes_rest = training_indexes_rest[100:]
                print ('new train_x size:', len(training_indexes))










    if step % args.curveplot_every ==0 and step > 0 and len(data_dict['steps']) > 2:

        plot_curve2(data_dict, exp_dir)



    if step % args.plotimages_every == 0 and step!=0:

        for f in os.listdir(images_dir):
            if 'plot' in f:
                os.remove(images_dir + f)


        # if os.path.isfile(images_dir +'plot2500_temp3.png'):
        #     print ('yer')
        #     os.remove(images_dir +'plot*')
        # else:
        #     print ('noo')

        # afdsfa

        with torch.no_grad():

            if args.base_dist == 'Gauss':
                model_args = {}
                temps = [.5,1.,1.5,2.]
                for temp_i in range(len(temps)):

                    model_args['temp'] = temps[temp_i]

                    sample = model.sample(model_args)

                    sample = torch.clamp(sample, min=1e-5, max=1-1e-5)


                    # print (torch.min(sample), torch.max(sample), sample.shape)

                    objective = torch.zeros_like(sample[:, 0, 0, 0]) + float(-np.log(args.n_bins) * np.prod(sample.shape[1:]))
                    z, objective = model(sample, objective)
                    nll = (-objective) / float(np.log(2.) * np.prod(sample.shape[1:]))
                    nobj_sample = torch.mean(nll)

                    print (temps[temp_i], nobj_sample)


                    if (sample!=sample).any():
                        print ('nans in sample!!')
                        # fafad

                    sample = torch.clamp(sample, min=1e-5, max=1-1e-5)
                    grid = utils.make_grid(sample)
                    img_file = images_dir +'plot'+str(step)+'_temp' +str(temp_i) + '.png'

                    utils.save_image(grid, img_file)
                    # print ('saved', img_file)
                print ('saved images')

            else:


                sample = model.module.sample()

                # print (torch.min(sample), torch.max(sample), sample.shape)

                grid = utils.make_grid(sample)
                img_file = images_dir +'plot'+str(step)+'.png'

                utils.save_image(grid, img_file)
                # print ('saved', img_file)
                print ('saved images')


            if step / args.plotimages_every ==1:

                init_loader = torch.utils.data.DataLoader(train_x, batch_size=min(64,len(train_x)), 
                                    shuffle=True, num_workers=1, drop_last=True)
                iter_loader_init = iter(init_loader)
                img = next(iter_loader_init)
                grid = utils.make_grid(img)
                img_file = images_dir +'realimages.png'
                utils.save_image(grid, img_file)
                print ('saved', img_file)


            if args.plotimages_every == 1: 
                dfadfdasf









    if step % args.save_every ==0 and i > 0:

        for f in os.listdir(params_dir):
            if 'model_params' in f:
                os.remove(params_dir + f)

        model.module.save_params_v3(params_dir, step, name='model_params')


        print ('Getting Full Test Set PBD')
        with torch.no_grad():

            #Get test set likelihood
            test_loader = torch.utils.data.DataLoader(test_x, batch_size=min(100,len(train_x)), 
                                shuffle=False, num_workers=1, drop_last=True)
            iter_test_loader = iter(test_loader)

            LLs = []
            while 1:
                try:
                    img = next(iter_test_loader).cuda()
                except:
                    break

                img = torch.clamp(img, min=1e-5, max=1-1e-5)
                objective = torch.zeros_like(img[:, 0, 0, 0])
                objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
                z, objective = model(img, objective)
                nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
                nobj = torch.mean(nll)

                LLs.append(numpy(nobj))

            print ('FULL Test set BPD', np.mean(LLs), np.std(LLs))



print ('Done.')
###########################################################################################


























        
