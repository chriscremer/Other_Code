

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
# sys.path.insert(0, os.path.abspath(home+'/Other_Code/VLVAE/CLEVR/utils'))


# from invertible_layers import * 
# from invertible_layers_2 import * 
# from invertible_layers_justgauss import * 
# from invertible_layers_someflows import * 




# from utils import * 


from load_data import load_clevr, load_cifar, load_svhn, load_flickr


# #https://forums.fast.ai/t/show-gpu-utilization-metrics-inside-training-loop-without-subprocess-call/26594
# pip install nvidia-ml-py3
import nvidia_smi


# from nets import NN
# from nets import NN2
from nets import NN3 as NN
# from unets import UNet 








# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# training

parser.add_argument('--exp_name', default='clevr_gettingprobofsamples', type=str)
parser.add_argument('--save_to_dir', default=home+'/Documents/glow_clevr/', type=str)
parser.add_argument('--which_gpu', default='0', type=str)

# parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--batch_size', type=int, default=16)

# # parser.add_argument('--depth', type=int, default=32) 
# parser.add_argument('--n_levels', type=int, default=3) 
# # parser.add_argument('--depth', type=int, default=8) 
# # parser.add_argument('--depth', type=int, default=6)  
# # parser.add_argument('--n_levels', type=int, default=6) 
# parser.add_argument('--depth', type=int, default=16)  
# # parser.add_argument('--depth', type=int, default=16) 
# parser.add_argument('--hidden_channels', type=int, default=128) 


# parser.add_argument('--AR_resnets', type=int, default=5) 
# parser.add_argument('--AR_channels', type=int, default=64) 

# # parser.add_argument('--n_levels', type=int, default=5) 

# parser.add_argument('--norm', type=str, default='actnorm')
# # parser.add_argument('--permutation', type=str, default='conv')
# parser.add_argument('--permutation', type=str, default='shuffle')
# # parser.add_argument('--coupling', type=str, default='affine')
# # parser.add_argument('--coupling', type=str, default='additive')
# parser.add_argument('--coupling', type=str, default='additive')
# parser.add_argument('--base_dist', type=str, default='Gauss')


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


#https://forums.fast.ai/t/show-gpu-utilization-metrics-inside-training-loop-without-subprocess-call/26594
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
def get_gpu_mem(concat_string=''):

    # res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    # print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

    res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # print(f'mem: {res.used / (1024**2)} (GiB)') # usage in GiB
    print(concat_string, f'mem: {100 * (res.used / res.total):.3f}%') # percentage 
    # print(f'mem: {res.used / (1024**2)} (GiB), {100 * (res.used / res.total):.3f}%') # percentage 
    # print(f'mem: {res.used, res.total} ') # percentage 
    # print(f'mem: {res.used/ (1024**2) /1000., res.total/ (1024**2) /1000.}  ') # percentage 


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


def gauss_log_prob(x, mean, logsd):
    Log2PI = float(np.log(2 * np.pi))
    # return  -0.5 * (Log2PI + 2. * logsd + ((x - mean) ** 2) / torch.exp(2. * logsd))

    aaa = Log2PI + 2. * logsd 
    var = torch.exp(2. * logsd)
    bbb = ((x - mean) ** 2) / var
    return  -0.5 * (aaa + bbb)


def logistic_log_prob(x, mean, logsd):

    half_pixel_value = (1./256.) # /2.

    # return torch.sigmoid( (x +half_pixel_value - mean)/ torch.exp(logsd))  - torch.sigmoid( (x -half_pixel_value - mean)/ torch.exp(logsd)) 
    prob =  torch.sigmoid( (x +half_pixel_value - mean))  - torch.sigmoid( (x -half_pixel_value - mean)) 

    return torch.log(prob)



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

    

    train_x, test_x = load_flickr(data_dir='/scratch/gobi1/ccremer/Flickr_Faces/images1024x1024/',
                                data_dir_test='/scratch/gobi1/ccremer/Flickr_Faces/images1024x1024_test/',
                                   dataset_size=args.dataset_size ) 

    # if not args.quick:
    #     train_x, test_x = load_flickr(data_dir='/scratch/gobi1/ccremer/Flickr_Faces/images1024x1024/',
    #                                                         data_dir_test='/scratch/gobi1/ccremer/Flickr_Faces/images1024x1024_test/') 
    # else:
    #     train_x, test_x = load_flickr(data_dir='/scratch/gobi1/ccremer/Flickr_Faces/images1024x1024_oneimage/',
    #                                                         data_dir_test='/scratch/gobi1/ccremer/Flickr_Faces/images1024x1024_test/')          

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
if shape[2]> 200 or args.dataset=='flickr':
    sampling_batch_size = 4 #9 # 16 ##25 #2 #64
else:
    sampling_batch_size = 64
# model = NN(in_channels=3, hidden_channels=128, channels_out=6).cuda()
model = NN(in_channels=3, hidden_channels=64, channels_out=6).cuda()
# model = UNet(in_channels=3, n_classes=6, padding=False, up_mode='upsample', depth=3).cuda()
# print(model)
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()])/ 1e6 , 'Mil')
# fasdfad
# model = nn.DataParallel(model).cuda()
# count =0
# for layer in model.layers:
#     # print (count, str(layer)[:6])
#     # count+=1
#     if 'Att' in str(layer) or 'AR_P' in str(layer):
#         print("number of AR parameters:", sum([np.prod(p.size()) for p in layer.parameters()]) / 1e6, 'Mil')

get_gpu_mem('Model takes')
###########################################################################################







###########################################################################################
# Optimizer, Param Init and Loaders
# ------------------------------------------------------------------------------
print ('Optimizer, Param Init and Loaders')
# set up the optimizer
optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-10)
# # optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=.01)
# # optim = optim.Adam(model.parameters(), lr=1e-5, weight_decay=.0001)
# # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=45, gamma=0.1)
# # lr_sched = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.999)
# lr_sched = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.999)
# # lr_sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.999)

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

get_gpu_mem()

###########################################################################################






















###########################################################################################
# training loop
# ------------------------------------------------------------------------------
print ('\nTraining Beginning')
args.batch_size = min(args.batch_size,len(train_x))
B = args.batch_size
print ('Batch Size', args.batch_size)
max_steps = args.max_steps
model.train()
t = time.time()

    
results_dict = {}
results_dict['steps'] = []
results_dict['loss'] = []
results_dict['NLL'] = []
results_dict['C2'] = []
# results_dict['BPD'] = {}
# results_dict['BPD']['Train_Noisy'] = []
# results_dict['BPD']['Train'] = []
# results_dict['BPD']['Test'] = []
# if args.dataset=='cifar':
#     results_dict['BPD']['SVHN'] = []
# results_dict['BPD']['Generated'] = []
# results_dict['lr'] = []
# results_dict['T'] = []
# results_dict['BPD_sd'] = []


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

    #NOISE
    img += torch.zeros_like(img).uniform_(0., 1./args.n_bins)
    img = torch.clamp(img, min=1e-5, max=1-1e-5)

    # objective = torch.zeros_like(img[:, 0, 0, 0])
    # # discretizing cost - NO! its just from the scaling. 
    # # We scale [0,255] to [0,1] so by chage of var formula, we gotta do this 
    # objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
    # z, objective = model(img, objective)

    input_ = torch.zeros_like(img, requires_grad=True).cuda()
    allseenpixels_mask = torch.zeros(B,1,112*112).cuda()
    total_cost = 0
    total_NLL = 0
    total_error_pred_cost = 0

    inputs = []
    img_preds = []
    img_logsd_preds = []
    error_preds = []
    true_error = []
    true_error_vals = []

    k = 100
    n_steps = 5

    for i_order in range(n_steps):


        inputs.append(input_)

        input_.requires_grad=True

        # print (input_.requires_grad)
        
        mean, logsd, NLL_pred = model(input_)
        # logsd = torch.ones_like(logsd) * .1



        # grad = torch.autograd.grad(outputs=mean[0][0][55][55], inputs=input_)[0]
        # print (grad.shape)




        # Image prediction
        logprob = gauss_log_prob(img, mean, logsd)
        # logprob = logistic_log_prob(img, mean, logsd)


        NLL_pixelavg = -torch.mean(logprob, dim=1)

        # print (NLL_pixelavg)


        # print (logprob.shape)
        # print (NLL_pixelavg.shape)
        # print (NLL_pred.shape)
        # print (torch.sum(allseenpixels_mask))
        

        # Error prediction
        error_pred_cost = (NLL_pixelavg.detach() - NLL_pred)**2
        error_pred_cost = error_pred_cost * (1.-allseenpixels_mask.view(B,112,112))
        error_pred_cost = torch.sum(error_pred_cost.view(B,-1), dim=1)  
        error_pred_cost = torch.mean(error_pred_cost)  


        # Compute lieklihood on the pixels we sample, if its the last one, youre smapling all remaining pixels
        if i_order != n_steps-1:
            
            # Get top error pixels - ie defining sampling order
            NLL_pred_scaled = NLL_pred - torch.min(NLL_pred) #shift so 0 is lowest
            NLL_pred_scaled = NLL_pred_scaled * (1.-allseenpixels_mask.view(B,112,112))  #remove the pixels that are revealed
            NLL_pred_scaled = NLL_pred_scaled.view(B,-1)

            values, indices = torch.topk(NLL_pred_scaled, k=k*5, dim=1, largest=True, sorted=True) 

            # Take random set of k
            idx = torch.randperm(k*5) 
            indices = indices[:,idx]
            indices = indices[:,:k]

            # Make a mask out of it 
            sampled_pixels_mask = torch.zeros_like(NLL_pred_scaled).cuda()
            sampled_pixels_mask.scatter_(1,indices,1)

            sample_NLL = torch.sum(NLL_pixelavg.view(B,-1) * sampled_pixels_mask, dim=1)
            sample_NLL = torch.mean(sample_NLL)

            # Accumulate mask and mask the image
            allseenpixels_mask = allseenpixels_mask.view(B,112*112) + sampled_pixels_mask
            # allseenpixels_mask = torch.clamp(allseenpixels_mask, max=1.) #if done properly, this shouldnt be needed
            if torch.max(allseenpixels_mask) > 1.5:
                print ('sampled same twice')
                dfafsd
            input_ = (img * allseenpixels_mask.view(B,1,112,112)).detach()




        else:

            # grad = torch.autograd.grad(outputs=mean[0][0][55][55], inputs=input_)[0]
            # print (grad.shape)

            sample_NLL = torch.sum(NLL_pixelavg.view(B,-1) * (1.-allseenpixels_mask.view(B,-1)), dim=1)
            sample_NLL = torch.mean(sample_NLL)

        # cost_withoutmask = - torch.mean(logprob[0])
        # logprob = logprob * (1.-total_mask.view(B,1,112,112)) #only error for non-seen pixels
        # cost = - torch.mean(logprob)

        # # if i_order != 0:
        # if i_order == n_steps-1:
        #     # total_cost += cost *10. + cost2 * .1


        if i_order == n_steps-1:
            total_cost += sample_NLL + error_pred_cost #*.1
            total_NLL += sample_NLL
            total_error_pred_cost += error_pred_cost #*.1


        # if i_order == 0:
        #     total_cost += sample_NLL + error_pred_cost #*.1
        #     total_NLL += sample_NLL
        #     total_error_pred_cost += error_pred_cost #*.1


        # Store results
        img_preds.append( torch.clamp(mean, min=1e-5, max=1-1e-5)) #.detach() )
        img_logsd_preds.append( torch.mean(logsd, dim=1).detach())
        error_preds.append(NLL_pred.view(B,1,112,112))
        true_error.append(NLL_pixelavg.detach().view(B,1,112,112))
        true_error_vals.append(numpy(torch.mean(NLL_pixelavg)))


    # afsda




    # nll_train = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
    # nobj = torch.mean(nll_train)
    nobj = total_cost #cost + cost2
    nobj_opt = total_cost #cost + cost2



    if (nobj_opt!=nobj_opt).any():
        print ('step', step, '  -nans in obj!!')
        print (nobj_opt)
        fafad





    # if I dont just want to plot
    # if args.plotimages_every != 1: 
    optim.zero_grad()
    nobj_opt.backward(retain_graph=True)
    # nobj_opt.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 5)
    # torch.nn.utils.clip_grad_value_(model.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
    optim.step()
    # lr_sched.step()
    ###########################################################################################





















    ###########################################################################################
    if step %args.print_every == 0:

        
        # B = 1
        # mean, logsd = model.module.layers[0].get_mean_logsd(torch.ones(B).cuda().view(B,1))

        # print ()
        # print (img[0][0][0][:5])
        # print (mean[0][0][0][:5])
        # print (logsd[0][0][0][:5])
        # print ()

        

        T=time.time() - t
        t = time.time()
    
        with torch.no_grad():


            if step > 25 and args.plotimages_every != 1:
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
                results_dict['loss'].append(numpy(nobj))
                results_dict['NLL'].append(numpy(total_NLL))
                results_dict['C2'].append(numpy(total_error_pred_cost))
                # results_dict['BPD']['Train_Noisy'].append(numpy(nobj))
                # results_dict['BPD']['Train'].append(numpy(nobj_train))
                # results_dict['lr'].append(lr_sched.get_lr()[0])
                # results_dict['T'].append(T)
                # results_dict['BPD']['Test'].append(numpy(nobj_test))
                # if step <= args.plotimages_every:
                #     results_dict['BPD']['Generated'].append(None)
                # else:
                #     results_dict['BPD']['Generated'].append(numpy(nobj_generated))
                # results_dict['BPD_sd'].append(np.std(numpy(nll_train)))
                # if args.dataset=='cifar':
                #     results_dict['BPD']['SVHN'].append(numpy(nobj_test_svhn))


        
        # print(step, 'T:{:.1f}'.format(T), 
        #         'C:{:.2f}'.format(float(nobj.data.cpu().numpy())), 
        #         'lr', lr_sched.get_lr()[0])
        myprint( (step, 'T:{:.1f}'.format(T), 
                'C:{:.2f}'.format(numpy(nobj)),
                'NLL:{:.2f}'.format(numpy(total_NLL)),
                'C2:{:.2f}'.format(numpy(total_error_pred_cost)), ) ) 
                # 'C_train:{:.2f}'.format(numpy(nobj_train)), 
                # 'C_test:{:.2f}'.format(numpy(nobj_test)), 
                # 'lr', lr_sched.get_lr()[0]) ) 





    ###########################################################################################
    if step % args.curveplot_every ==0 and step > 0 and len(results_dict['steps']) > 2:
        plot_curve2(results_dict, exp_dir)






    ###########################################################################################
    if step % args.plotimages_every == 0 and step!=0:

        for f in os.listdir(images_dir):
            if 'plot' in f:
                os.remove(images_dir + f)


        # get_gpu_mem()
        # objective = 0
        # img = 0
        # torch.cuda.empty_cache()
        # get_gpu_mem()



        # GET RECEPTIVE FIELD
        # print (input_.requires_grad)
        # print (mean[0][0][55][55].requires_grad)
        grad = torch.autograd.grad(outputs=mean[0][0][55][55], inputs=input_)[0]
        # print (grad)
        # fadsf
        # grad_nonzero = torch.mean((grad > 0).float(), dim=1).detach()
        # print (grad_nonzero)
        grad = torch.mean(torch.abs(grad), dim=1).detach()




        with torch.no_grad():



            # SAMPLE MODEL
            input_ = torch.zeros_like(img).cuda()
            allseenpixels_mask = torch.zeros(B,1,112*112).cuda()

            k = 100
            n_steps = 5

            for i_order in range(n_steps):
        
                mean, logsd, NLL_pred = model(input_)


                if i_order != n_steps-1:

                    # Get top error pixels - ie defining sampling order
                    NLL_pred_scaled = NLL_pred - torch.min(NLL_pred) #shift so 0 is lowest
                    NLL_pred_scaled = NLL_pred_scaled * (1.-allseenpixels_mask.view(B,112,112))  #remove the pixels that are revealed
                    NLL_pred_scaled = NLL_pred_scaled.view(B,-1)

                    values, indices = torch.topk(NLL_pred_scaled, k=k*5, dim=1, largest=True, sorted=True) 

                    # Take random set of k
                    idx = torch.randperm(k*5) 
                    indices = indices[:,idx]
                    indices = indices[:,:k]

                    # Make a mask out of it 
                    sampled_pixels_mask = torch.zeros_like(NLL_pred_scaled).cuda()
                    sampled_pixels_mask.scatter_(1,indices,1)

                    # TODO change this to sampling 
                    # sample_NLL = torch.mean(NLL_pixelavg.view(B,-1) * sampled_pixels_mask)
                    eps = torch.zeros_like(mean).normal_().cuda()
                    samp = mean + torch.exp(logsd) * eps


                    new_pixels = samp * sampled_pixels_mask.view(B,1,112,112)

                    # Accumulate mask and mask the image
                    allseenpixels_mask = allseenpixels_mask.view(B,112*112) + sampled_pixels_mask
                    # input_ = (img * allseenpixels_mask.view(B,1,112,112)).detach()
                    input_ = input_ + new_pixels


                else:

                    #sample rest of pixels

                    # TODO chagne to smapling
                    # sample_NLL = torch.mean(NLL_pixelavg.view(B,-1) * (1.-allseenpixels_mask.view(B,-1)))

                    eps = torch.zeros_like(mean).normal_().cuda()
                    samp = mean + torch.exp(logsd) * eps

                    new_pixels = samp * (1.-allseenpixels_mask.view(B,1,112,112))

                    sampled_image = input_ + new_pixels
        
        













            def make_contour_subplot(rows, cols, row, col, image, text, legend=False):

                ax = plt.subplot2grid((rows,cols), (row,col), frameon=False)
               
                # print (image.shape)
                image = image.view(112,112) 
                image = image.data.cpu().numpy() 
                image = np.rot90(image)
                image = np.rot90(image)
                image = np.flip(image,1)
                # image = np.uint8(image)
                cs = ax.contourf(image, cmap='Blues')
                # ax.legend()
                ax.set_aspect('equal')

                # if legend:
                #     h1,l1 = cs.legend_elements()
                #     ax.legend(h1, l1, fontsize = 'x-small', loc=5)



                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)






            def make_image_subplot(rows, cols, row, col, image, text):

                ax = plt.subplot2grid((rows,cols), (row,col), frameon=False)
               
                if image.shape[0] != 1:
                    image = image.data.cpu().numpy() * 255.
                    image = np.rollaxis(image, 1, 0)
                    image = np.rollaxis(image, 2, 1)# [112,112,3]
                    image = np.uint8(image)
                    ax.imshow(image) 

                else:
                    image = image.view(112,112) * 255.
                    image = image.data.cpu().numpy() 
                    image = np.uint8(image)
                    ax.imshow(image, cmap='gray')


                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)




            cols = 5
            rows = 6

            fig = plt.figure(figsize=(7+cols,2+rows), facecolor='white', dpi=150)

            img_idx = 0
            make_image_subplot(rows=rows, cols=cols, row=0, col=0, image=img[img_idx], text='')


            # cosstt = -logprob.detach().view(B,1,112,112)
            # cosstt = torch.mean(cosstt, dim=0)
            # make_image_subplot(rows=rows, cols=cols, row=0, col=1, image=cosstt, text='')

            sampled_image = torch.clamp(sampled_image, min=1e-5, max=1-1e-5)
            make_image_subplot(rows=rows, cols=cols, row=0, col=1, image=sampled_image[0], text='Sample')



            make_contour_subplot(rows=rows, cols=cols, row=0, col=2, image=grad[0].view(1,112,112), text='Grad')
            # make_image_subplot(rows=rows, cols=cols, row=0, col=3, image=grad_nonzero[0].view(1,112,112), text='Grad Non Zero')



            



            for ii in range(cols):

                make_image_subplot(rows=rows, cols=cols, row=1, col=ii, image=inputs[ii][img_idx], text='Masked Input')
                make_image_subplot(rows=rows, cols=cols, row=2, col=ii, image=img_preds[ii][img_idx], text='Img Mean Pred')
                make_contour_subplot(rows=rows, cols=cols, row=3, col=ii, image=error_preds[ii][img_idx], text='Error Pred') 
                make_contour_subplot(rows=rows, cols=cols, row=4, col=ii, image=true_error[ii][img_idx], text='True Error '+'{:.3f}'.format(true_error_vals[ii]))
                make_contour_subplot(rows=rows, cols=cols, row=5, col=ii, image=img_logsd_preds[ii][img_idx], text='Img LogSD Pred')


            # img_idx = 1
            # make_image_subplot(rows=rows, cols=cols, row=0+4, col=0, image=img[img_idx], text='')

            # for ii in range(cols):

            #     make_image_subplot(rows=rows, cols=cols, row=1+4, col=ii, image=inputs[ii][img_idx], text='Masked Input')
            #     make_image_subplot(rows=rows, cols=cols, row=2+4, col=ii, image=img_preds[ii][img_idx], text='Img Pred')
            #     make_image_subplot(rows=rows, cols=cols, row=3+4, col=ii, image=error_preds[ii][img_idx], text='Error Pred')







            # plt.tight_layout()
            # plt_path = exp_dir 
            img_file = images_dir + 'img'+str(step)+'.png'
            plt.savefig(img_file)
            print ('saved viz',img_file)
            plt.close(fig)



            # print (img[0][0][:10][:10])

            # print (img_preds[0][0][0][:10][:10])
            # fasdfa










            if args.plotimages_every == 1: 
                dfadfdasf




    # ###########################################################################################
    # if step % args.save_every ==0 and step_not_including_load > 0:

    #     for f in os.listdir(params_dir):
    #         if 'model_params' in f:
    #             os.remove(params_dir + f)

    #     model.module.save_params_v3(params_dir, step, name='model_params')



    #     #save results
    #     save_to=os.path.join(exp_dir, "results.pkl")
    #     with open(save_to, "wb" ) as f:
    #         pickle.dump(results_dict, f)
    #     print ('saved results', save_to)



    #     print ('Getting Full Test Set PBD')
    #     with torch.no_grad():

    #         #Get test set likelihood
    #         test_loader = torch.utils.data.DataLoader(test_x, batch_size=min(100,len(train_x)), 
    #                             shuffle=False, num_workers=1, drop_last=True)
    #         iter_test_loader = iter(test_loader)

    #         LLs = []
    #         while 1:
    #             try:
    #                 img = next(iter_test_loader).cuda()
    #             except:
    #                 break

    #             img = torch.clamp(img, min=1e-5, max=1-1e-5)
    #             objective = torch.zeros_like(img[:, 0, 0, 0])
    #             objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
    #             z, objective = model(img, objective)
    #             nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
    #             nobj = torch.mean(nll)

    #             LLs.append(numpy(nobj))

    #         print ('FULL Test set BPD', np.mean(LLs), np.std(LLs))



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













        
