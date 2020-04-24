
# train4 incorporates more flickr dataset changes, like better memory allocation

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

# import sys, os
# sys.path.insert(0, os.path.abspath(home+'/Other_Code/VLVAE/CLEVR/utils'))


# from invertible_layers import * 
# from invertible_layers_2 import * 
from invertible_layers_justgauss import * 




from utils import * 


from load_data import load_clevr, load_cifar, load_svhn, load_flickr


# #https://forums.fast.ai/t/show-gpu-utilization-metrics-inside-training-loop-without-subprocess-call/26594
# import nvidia_smi









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
parser.add_argument('--save_output', type=int, default=0)

# parser.add_argument('--vws', type=int, default=1)
parser.add_argument('--machine', type=str, default='vws', choices=['vws', 'boltz', 'vector', 'vaughn'])  

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


# #https://forums.fast.ai/t/show-gpu-utilization-metrics-inside-training-loop-without-subprocess-call/26594
# nvidia_smi.nvmlInit()
# handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# def get_gpu_mem(concat_string=''):

#     # res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
#     # print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

#     res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#     # print(f'mem: {res.used / (1024**2)} (GiB)') # usage in GiB
#     print(concat_string, f'mem: {100 * (res.used / res.total):.3f}%') # percentage 
#     # print(f'mem: {res.used / (1024**2)} (GiB), {100 * (res.used / res.total):.3f}%') # percentage 
#     # print(f'mem: {res.used, res.total} ') # percentage 
#     # print(f'mem: {res.used/ (1024**2) /1000., res.total/ (1024**2) /1000.}  ') # percentage 


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



def plot_curve2(results_dict, exp_dir):

    rows = len(results_dict) - 1
    cols = 1
    fig = plt.figure(figsize=(8+cols,8+rows), facecolor='white') #, dpi=150)

    steps = results_dict['steps']
    col=0
    row=0
    for k,v in results_dict.items():
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
if args.which_gpu != 'all' and args.machine not in ['vector', 'vaughn']:
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
print ('\nLoading Data')
if args.dataset=='clevr':

    # # CLEVR DATA
    if args.machine in ['vws', 'vector', 'vaughn']:
        data_dir = home+ "/vl_data/two_objects_large/"  #vws
    else:
        data_dir = home+ "/VL/data/two_objects_no_occ/" #boltz 
    
    train_x, test_x = load_clevr(batch_size=args.batch_size, data_dir=data_dir, quick=args.quick)
    shape = train_x[0].shape

elif args.dataset=='cifar':
    # CIFAR DATA
    # train_image_dataset = load_cifar(data_dir=args.data_dir)
    train_x, test_x = load_cifar(data_dir=home+'/Documents/', dataset_size=args.dataset_size)
    shape = train_x[0].shape

    # print (len(test_x), 'test set len')
    svhn_test_x = load_svhn(data_dir=home+'/Documents/')
    # svhn_test_x = test_x


# dataset = train_x
elif args.dataset=='flickr':

    if not args.quick:
        train_x, test_x = load_flickr(batch_size=args.batch_size, data_dir='/scratch/gobi1/ccremer/Flickr_Faces/images1024x1024/',
                                                            data_dir_test='/scratch/gobi1/ccremer/Flickr_Faces/images1024x1024_test/') 
    else:
        train_x, test_x = load_flickr(batch_size=args.batch_size, data_dir='/scratch/gobi1/ccremer/Flickr_Faces/images1024x1024_oneimage/',
                                                            data_dir_test='/scratch/gobi1/ccremer/Flickr_Faces/images1024x1024_test/')          

    shape = train_x[0][0].shape


# print ('Batch size', args.batch_size)
print ('Training Set', len(train_x)) #, train_x[0].shape)
print ('Test Set', len(test_x)) #, test_x[0].shape)
print (shape)
# print (torch.min(train_x[0:100]), torch.max(train_x[0:100]))
# print (torch.min(test_x[0:100]), torch.max(test_x[0:100]))
###########################################################################################






###########################################################################################
# Init Model
# ------------------------------------------------------------------------------
print ('\nInitializing Model')
batch_size = args.batch_size
if shape[2]> 200:
    sampling_batch_size = 16 ##25 #2 #64
else:
    sampling_batch_size = 64
model = Glow_((sampling_batch_size, shape[0], shape[1], shape[2]), args).cuda()
# print(model)
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()])/ 1e6 , 'Mil')
# fasdfad
# model = nn.DataParallel(model).cuda()
# count =0
for layer in model.layers:
    # print (count, str(layer)[:6])
    # count+=1
    if 'Att' in str(layer) or 'AR_P' in str(layer):
        print("number of AR parameters:", sum([np.prod(p.size()) for p in layer.parameters()]) / 1e6, 'Mil')

# get_gpu_mem('Model takes')
###########################################################################################







###########################################################################################
# Optimizer, Param Init and Loaders
# ------------------------------------------------------------------------------
print ('Optimizer, Param Init and Loaders')
# set up the optimizer
optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-10)
# optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=.01)
# optim = optim.Adam(model.parameters(), lr=1e-5, weight_decay=.0001)
# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=45, gamma=0.1)
# lr_sched = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.999)
lr_sched = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.999)
# lr_sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.999)

# for i in range (200000):
#     lr_sched.step()
#     if i % 1000==0:
#         print (i, lr_sched.get_lr()[0])
# fsfsda


# get_gpu_mem()

# # data dependant init
if args.load_step == 0:
    if shape[2] > 200:
        data_init_batchsize = min(8,len(train_x))
    else:
        data_init_batchsize = min(100,len(train_x))
    init_loader = torch.utils.data.DataLoader(train_x, batch_size=data_init_batchsize, 
                        shuffle=True, num_workers=1, drop_last=True)
    with torch.no_grad():
        model.eval()

        
        for img in init_loader:

            if isinstance(img, list):
                img = img[0]
                img = img * 255. / 256. 
                img = torch.clamp(img, min=1e-5, max=1-1e-5)

            img = img.cuda()

            # get_gpu_mem()

            # print (img.shape)
            # print (torch.min(img), torch.max(img))
            # fasdf
            objective = torch.zeros_like(img[:, 0, 0, 0])

            # print (objective.shape)

            # model.test_reversability(img, objective)
            # fdsfas
            _ = model(img, objective)

            # fadfa

            # get_gpu_mem()
            objective = 0
            img = 0
            torch.cuda.empty_cache()
            # get_gpu_mem()

            break
    print ('data dependent init complete with', data_init_batchsize, 'datapoints')

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

loader = torch.utils.data.DataLoader(train_x, batch_size=min(args.batch_size,len(train_x)), 
                                shuffle=True, num_workers=0, drop_last=True)
iter_loader = iter(loader)

# get_gpu_mem()

###########################################################################################






















###########################################################################################
# training loop
# ------------------------------------------------------------------------------
print ('\nTraining Beginning')
print ('Batch Size', args.batch_size)
max_steps = args.max_steps
model.train()
t = time.time()

    
results_dict = {}
results_dict['steps'] = []
results_dict['BPD'] = {}
results_dict['BPD']['Train_Noisy'] = []
results_dict['BPD']['Train'] = []
results_dict['BPD']['Test'] = []
if args.dataset=='cifar':
    results_dict['BPD']['SVHN'] = []
results_dict['BPD']['Generated'] = []
results_dict['lr'] = []
results_dict['T'] = []
results_dict['BPD_sd'] = []


for step_not_including_load in range(max_steps+1):
    step = step_not_including_load + args.load_step

    try:
        img = next(iter_loader) #.cuda()
    except:
        loader = torch.utils.data.DataLoader(train_x, batch_size=min(args.batch_size,len(train_x)), 
                            shuffle=True, num_workers=0, drop_last=True)
        iter_loader = iter(loader)
        img = next(iter_loader) #.cuda()

    if isinstance(img, list):
        img = img[0]
        img = img * 255. / 256. 
        # img = torch.clamp(img, min=1e-5, max=1-1e-5)

    img = img.cuda()
    # if step % 4 != 1:
    # dequantize
    # print (torch.min(img), torch.max(img))

    # print (img.shape)


    img += torch.zeros_like(img).uniform_(0., 1./args.n_bins)


    # print (torch.min(img), torch.max(img))
    # fdasfasd
    img = torch.clamp(img, min=1e-5, max=1-1e-5)
    objective = torch.zeros_like(img[:, 0, 0, 0])
    # discretizing cost - NO! its just from the scaling. 
    # We scale [0,255] to [0,1] so by chage of var formula, we gotta do this 
    objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
    # log_det_jacobian cost (and some prior from Split OP)

    z, objective = model(img, objective)
    
    # get_gpu_mem()
    nll_train = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
    # Generative loss
    nobj = torch.mean(nll_train)

    nobj_opt = nobj



    if (nobj_opt!=nobj_opt).any():
        print ('step', step, '  -nans in obj!!')
        print ('z', (z!=z).any())
        print (nobj_opt)
        fafad


    # if I dont just want to plot
    if args.plotimages_every != 1: 
        optim.zero_grad()
        nobj_opt.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        # torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
        optim.step()
        # lr_sched.step()
        ###########################################################################################





















    ###########################################################################################
    if step %args.print_every == 0:

        # print (img[0][0][0][:10])

        

        T=time.time() - t
        t = time.time()
    
        with torch.no_grad():


            #Get non-noisy train set likelihood
            test_loader = torch.utils.data.DataLoader(train_x, batch_size=min(batch_size,len(train_x)), 
                                shuffle=True, num_workers=1, drop_last=True)
            iter_test_loader = iter(test_loader)
            img = next(iter_test_loader)
            if isinstance(img, list):
                img = img[0]
                img = img * 255. / 256. 
            img = img.cuda()
            img = torch.clamp(img, min=1e-5, max=1-1e-5)
            objective = torch.zeros_like(img[:, 0, 0, 0])
            objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
            z, objective = model(img, objective)
            nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
            nobj_train = torch.mean(nll)


            
            #Get test set likelihood
            test_loader = torch.utils.data.DataLoader(test_x, batch_size=min(batch_size,len(test_x)), 
                                shuffle=True, num_workers=1, drop_last=True)
            iter_test_loader = iter(test_loader)
            img = next(iter_test_loader)
            if isinstance(img, list):
                img = img[0]
                img = img * 255. / 256. 
            img = img.cuda()
            img = torch.clamp(img, min=1e-5, max=1-1e-5)
            objective = torch.zeros_like(img[:, 0, 0, 0])
            objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
            z, objective = model(img, objective)
            nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
            nobj_test = torch.mean(nll)




            if args.dataset=='cifar':
                #Get svhn set likelihood
                test_loader = torch.utils.data.DataLoader(svhn_test_x, batch_size=min(batch_size,len(svhn_test_x)), 
                                    shuffle=True, num_workers=1, drop_last=True)
                iter_test_loader = iter(test_loader)
                img = next(iter_test_loader).cuda()
                img = torch.clamp(img, min=1e-5, max=1-1e-5)
                objective = torch.zeros_like(img[:, 0, 0, 0])
                objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
                z, objective = model(img, objective)
                nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
                nobj_test_svhn = torch.mean(nll)




            if step > 125 and args.plotimages_every != 1:
            # if step > 1 and args.plotimages_every != 1:

            #     #Get generated data likelihood
            #     args.sample_size = batch_size #sampling_batch_size
            #     args.temp = 1.
            #     args.special_sample = False #True #False
            #     print ('sampling model')
            #     sample = model.module.sample(args)
            #     img = torch.clamp(sample, min=1e-5, max=1-1e-5)
            #     objective = torch.zeros_like(img[:, 0, 0, 0])
            #     objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
            #     z, objective = model(img, objective)
            #     nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
            #     nobj_generated = torch.mean(nll)


                # RECORD DATA
                results_dict['steps'].append(step)
                results_dict['BPD']['Train_Noisy'].append(numpy(nobj))
                results_dict['BPD']['Train'].append(numpy(nobj_train))
                results_dict['lr'].append(lr_sched.get_lr()[0])
                results_dict['T'].append(T)
                results_dict['BPD']['Test'].append(numpy(nobj_test))
                if step <= args.plotimages_every:
                    results_dict['BPD']['Generated'].append(None)
                else:
                    results_dict['BPD']['Generated'].append(numpy(nobj_generated))
                results_dict['BPD_sd'].append(np.std(numpy(nll_train)))
                if args.dataset=='cifar':
                    results_dict['BPD']['SVHN'].append(numpy(nobj_test_svhn))


        
        # print(step, 'T:{:.1f}'.format(T), 
        #         'C:{:.2f}'.format(float(nobj.data.cpu().numpy())), 
        #         'lr', lr_sched.get_lr()[0])
        myprint( (step, 'T:{:.1f}'.format(T), 
                'C:{:.2f}'.format(numpy(nobj)), 
                'C_train:{:.2f}'.format(numpy(nobj_train)), 
                'C_test:{:.2f}'.format(numpy(nobj_test)), 
                'lr', lr_sched.get_lr()[0]) ) 





    ###########################################################################################
    if step % args.curveplot_every ==0 and step > 0 and len(results_dict['steps']) > 2:
        plot_curve2(results_dict, exp_dir)






    ###########################################################################################
    if step % args.plotimages_every == 0 and step!=0:

        for f in os.listdir(images_dir):
            if 'plot' in f:
                os.remove(images_dir + f)


        # get_gpu_mem()
        objective = 0
        img = 0
        torch.cuda.empty_cache()
        # get_gpu_mem()

        with torch.no_grad():

            args.special_sample = False #True #False
            args.special_h = 8 #24
            args.temp = 1.
            args.sample_size = sampling_batch_size
            print ('sampling model')
            sample = model.module.sample(args)
            # sample = torch.clamp(sample, min=1e-5, max=1-1e-5)
            grid = utils.make_grid(sample, nrow=int(np.sqrt(sampling_batch_size)))
            img_file = images_dir +'plot'+str(step)+'.png'
            utils.save_image(grid, img_file)
            # print ('saved', img_file)
            print ('saved image', img_file)
            img = torch.clamp(sample, min=1e-5, max=1-1e-5)
            objective = torch.zeros_like(img[:, 0, 0, 0])
            objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
            z, objective = model(img, objective)
            nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
            nobj_generated = torch.mean(nll)


            if ( ((step_not_including_load/args.plotimages_every) ==1 ) or (args.plotimages_every ==1 and step_not_including_load==0)) and args.plotimages_every != 1:

                init_loader = torch.utils.data.DataLoader(train_x, batch_size=min(sampling_batch_size,len(train_x)), 
                                    shuffle=True, num_workers=1, drop_last=True)
                iter_loader_init = iter(init_loader)
                img = next(iter_loader_init)
                if isinstance(img, list):
                    img = img[0]
                    img = img * 255. / 256. 

                img += torch.zeros_like(img).uniform_(0., 1./args.n_bins)
                img = torch.clamp(img, min=1e-5, max=1-1e-5)

                grid = utils.make_grid(img, nrow=int(np.sqrt(sampling_batch_size)))
                img_file = images_dir +'realimages.png'
                utils.save_image(grid, img_file)
                print ('saved', img_file)


                # init_loader = torch.utils.data.DataLoader(test_x, batch_size=min(sampling_batch_size,len(train_x)), 
                #                     shuffle=True, num_workers=1, drop_last=True)
                # iter_loader_init = iter(init_loader)
                # img = next(iter_loader_init)
                # if isinstance(img, list):
                #     img = img[0]
                #     img = img * 255. / 256. 
                # grid = utils.make_grid(img)
                # img_file = images_dir +'realimages_test.png'
                # utils.save_image(grid, img_file)
                # print ('saved', img_file)


            if args.plotimages_every == 1: 
                dfadfdasf




    ###########################################################################################
    if step % args.save_every ==0 and step_not_including_load > 0:

        for f in os.listdir(params_dir):
            if 'model_params' in f:
                os.remove(params_dir + f)

        model.module.save_params_v3(params_dir, step, name='model_params')



        #save results
        save_to=os.path.join(exp_dir, "results.pkl")
        with open(save_to, "wb" ) as f:
            pickle.dump(results_dict, f)
        print ('saved results', save_to)



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





            # print ('Getting Full Test Set PBD')
            # #Get test set likelihood
            # test_loader = torch.utils.data.DataLoader(test_x, batch_size=min(100,len(train_x)), 
            #                     shuffle=False, num_workers=1, drop_last=True)
            # iter_test_loader = iter(test_loader)

            # LLs = []
            # count = 0
            # while 1:
            #     try:
            #         img = next(iter_test_loader).cuda()
            #     except:
            #         break

            #     count += len(img)


            #     img += torch.zeros_like(img).uniform_(0., 1./args.n_bins)


            #     grid = utils.make_grid(img)
            #     img_file = images_dir +'realimages'+str(count)+'.png'
            #     utils.save_image(grid, img_file)
            #     print ('saved', img_file)


            #     img = torch.clamp(img, min=1e-5, max=1-1e-5)
            #     objective = torch.zeros_like(img[:, 0, 0, 0])
            #     objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
            #     z, objective = model(img, objective)
            #     nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
            #     nobj = torch.mean(nll)

            #     LLs.append(numpy(nobj))

            #     print (count, np.mean(LLs), np.std(LLs), numpy(nobj))

            # print ('FULL Test set BPD', np.mean(LLs), np.std(LLs))
            # fsdafa









            # if step_not_including_load == 1:
            #     if (img == img11).all():
            #         print ('yer')
            #         ffas
            #     else:
            #         print ('ner')
            #         fadfad
            # img11 = img.clone()













        
