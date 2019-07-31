

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


from load_data import load_clevr, load_cifar, load_svhn







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
    # CIFAR DATA
    # train_image_dataset = load_cifar(data_dir=args.data_dir)
    train_x, test_x = load_cifar(data_dir=home+'/Documents/', dataset_size=args.dataset_size)
    dataset = train_x
    # print (len(test_x), 'test set len')


    svhn_test_x = load_svhn(data_dir=home+'/Documents/')
    # svhn_test_x = test_x



print (len(dataset), dataset[0].shape)
###########################################################################################






###########################################################################################
# Init Model
# ------------------------------------------------------------------------------
sampling_batch_size = 64
shape = dataset[0].shape
model = Glow_((sampling_batch_size, shape[0], shape[1], shape[2]), args).cuda()
# print(model)
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
# fasdfad
# model = nn.DataParallel(model).cuda()
###########################################################################################






###########################################################################################
# Optimizer, Param Init and Loaders
# ------------------------------------------------------------------------------
# set up the optimizer
optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-9)
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
    init_loader = torch.utils.data.DataLoader(dataset, batch_size=100, 
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

loader = torch.utils.data.DataLoader(dataset, batch_size=min(args.batch_size,len(dataset)), 
                                shuffle=True, num_workers=0, drop_last=True)
iter_loader = iter(loader)
###########################################################################################








###########################################################################################
# training loop
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
data_dict['lr'] = []
data_dict['T'] = []
# data_dict['LL'] = {}
# data_dict['LL']['real_LL'] = []
# data_dict['LL']['sample_LL'] = []



for i in range(max_steps+1):
    step = i + args.load_step

    try:
        img = next(iter_loader).cuda()
    except:
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(args.batch_size,len(dataset)), 
                            shuffle=True, num_workers=0, drop_last=True)
        iter_loader = iter(loader)
        img = next(iter_loader).cuda()


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
    nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
    # Generative loss
    nobj = torch.mean(nll)

    # print (torch.min(z), torch.max(z))
    # print (z.shape)
    # print (torch.mean(torch.abs(z)))

    # fsadfs


    # print (nobj)




    # objective = torch.zeros_like(img[:, 0, 0, 0])
    # # discretizing cost 
    # objective += float(-np.log(args.n_bins) )#* np.prod(img.shape[1:]))
    # # log_det_jacobian cost (and some prior from Split OP)
    # z, objective = model(img, objective)
    # nll = (-objective) / float(np.log(2.) )#* np.prod(img.shape[1:]))
    # # Generative loss
    # nobj = torch.mean(nll)


    # print (nobj)
    # fasfsa





    # else:

    #     model_args = {}
    #     model_args['temp'] = .5
    #     model_args['batch_size'] = args.batch_size
    #     img = model.sample(model_args)
    #     img = torch.clamp(img, min=1e-5, max=1-1e-5).detach()
    #     objective = torch.zeros_like(img[:, 0, 0, 0]) + float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
    #     z, objective = model(img, objective)
    #     nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
    #     nobj = -torch.mean(nll)






    if (nobj!=nobj).any():
        print ('step', step, '  -nans in obj!!')
        print ('z', (z!=z).any())
        print (nobj)
        # allgrads = []
        # param_list = list(model.parameters())
        # for param_i in range(len(param_list)):
        #     g = numpy(param_list[param_i].grad.view(-1))
        #     allgrads.extend(list(g))
        # print ('Grads: max', np.max(allgrads), 'mean', np.mean(allgrads),
        #          'median', np.median(allgrads),  'min', np.min(allgrads),)

        # with torch.no_grad():

            # img = next(iter_loader).cuda()
            # objective = torch.zeros_like(img[:, 0, 0, 0])
            # objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
            # z, objective = model(img, objective)
            # nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
            # nobj = torch.mean(nll)
            # print ('seocnd', nobj)


            # # allparams = []
            # # allgrads = []
            # param_list = list(model.parameters())
            # for param_i in range(len(param_list)):
            #     print (param_i, param_list[param_i].shape)
            #     p =  numpy(param_list[param_i].view(-1))
            #     g = numpy(param_list[param_i].grad.view(-1))
            #     # allparams.extend(list(p))
            #     # allgrads.extend(list(g))

            #     print (param_i, 'param max', np.max(p), 'mean', np.mean(p), 
            #             'median', np.median(p),  'min', np.min(p),)
            #     print (param_i, 'grad max', np.max(g), 'mean', np.mean(g), 
            #             'median', np.median(g),  'min', np.min(g),)

            # count=0
            # for layer in model.layers:

            #     print (count, str(layer)[:6])
            #     param_list = list(layer.parameters())
            #     for param_i in range(len(param_list)):
            #         p =  numpy(param_list[param_i].view(-1))
            #         print (param_i, 'param max', np.max(p), 'mean', np.mean(p), 
            #              'median', np.median(p),  'min', np.min(p),)
            #     print ()
            #     count+=1





        fafad


    # if I dont just want to plot
    if args.plotimages_every != 1: 



        optim.zero_grad()
        nobj.backward()



        # if step > 609:

        #     with torch.no_grad():

        #         allgrads = []
        #         param_list = list(model.parameters())
        #         for param_i in range(len(param_list)):
        #             g = numpy(param_list[param_i].grad.view(-1))
        #             allgrads.extend(list(g))
        #         print (step, 'Grads: max', np.max(allgrads), 'mean', np.mean(allgrads),
        #                  'median', np.median(allgrads),  'min', np.min(allgrads),)


        # see_grads_every = 20

        # if step %(args.print_every*see_grads_every) == 0:

        #     # print (step)
        #     # forwardlayers, names = model.forward_2(img, objective)

        #     # for ii in range(161):
        #     #     print (str(model[ii]))
        #     #     print (sum([np.prod(p.size()) for p in model[ii].parameters()]))
        #     #     print ()

        #     # print (str(model[148]))  #goes up to 160
        #     # print (list(model[148].parameters()))
        #     allparams = []
        #     allgrads = []
        #     param_list = list(model.parameters())
        #     for param_i in range(len(param_list)):
        #         p =  numpy(param_list[param_i].view(-1))
        #         g = numpy(param_list[param_i].grad.view(-1))
        #         allparams.extend(list(p))
        #         allgrads.extend(list(g))

        #     print ('\nParams: max', np.max(allparams), 'mean', np.mean(allparams), 
        #                 'median', np.median(allparams),  'min', np.min(allparams),)
        #     print ('Grads: max', np.max(allgrads), 'mean', np.mean(allgrads), 
        #                 'median', np.median(allgrads),  'min', np.min(allgrads),)


        torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        # torch.nn.utils.clip_grad_value_(model.parameters(), 1)


        # if step % (args.print_every*see_grads_every) == 0:
        #     allgrads = []
        #     param_list = list(model.parameters())
        #     for param_i in range(len(param_list)):
        #         g = numpy(param_list[param_i].grad.view(-1))
        #         allgrads.extend(list(g))
        #     print ('after clip 5, Grads: max', np.max(allgrads), 'mean', np.mean(allgrads), 
        #                 'median', np.median(allgrads),  'min', np.min(allgrads),)


        torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 20)


        # if step %(args.print_every*see_grads_every) == 0:
        #     allgrads = []
        #     param_list = list(model.parameters())
        #     for param_i in range(len(param_list)):
        #         g = numpy(param_list[param_i].grad.view(-1))
        #         allgrads.extend(list(g))
        #     print ('after clip 10, Grads: max', np.max(allgrads), 'mean', np.mean(allgrads), 
        #                 'median', np.median(allgrads),  'min', np.min(allgrads),)



        optim.step()
        lr_sched.step()





    if step %args.print_every == 0:
        T=time.time() - t
        # print(step, 'T:{:.1f}'.format(T), 
        #         'C:{:.2f}'.format(float(nobj.data.cpu().numpy())), 
        #         'lr', lr_sched.get_lr()[0])
        myprint( (step, 'T:{:.1f}'.format(T), 
                'C:{:.2f}'.format(float(nobj.data.cpu().numpy())), 
                'lr', lr_sched.get_lr()[0]) ) 
        t = time.time()


        if step > 5:

            # RECORD DATA
            data_dict['steps'].append(step)
            data_dict['BPD']['Train'].append(float(nobj.data.cpu().numpy()))
            data_dict['lr'].append(lr_sched.get_lr()[0])
            data_dict['T'].append(T)

            with torch.no_grad():

                #Get test set likelihood
                test_loader = torch.utils.data.DataLoader(test_x, batch_size=min(100,len(dataset)), 
                                    shuffle=True, num_workers=1, drop_last=True)
                iter_test_loader = iter(test_loader)

                img = next(iter_test_loader).cuda()

                img = torch.clamp(img, min=1e-5, max=1-1e-5)
                objective = torch.zeros_like(img[:, 0, 0, 0])
                objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
                z, objective = model(img, objective)
                nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
                nobj_test = torch.mean(nll)


                #Get test set likelihood
                test_loader = torch.utils.data.DataLoader(svhn_test_x, batch_size=min(100,len(dataset)), 
                                    shuffle=True, num_workers=1, drop_last=True)
                iter_test_loader = iter(test_loader)

                img = next(iter_test_loader).cuda()

                img = torch.clamp(img, min=1e-5, max=1-1e-5)
                objective = torch.zeros_like(img[:, 0, 0, 0])
                objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
                z, objective = model(img, objective)
                nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
                nobj_test_svhn = torch.mean(nll)





            data_dict['BPD']['Test'].append(numpy(nobj_test))
            data_dict['BPD']['SVHN'].append(numpy(nobj_test_svhn))


            # data_dict['LL']['real_LL'].append(numpy(-nobj))  #batch is only 32 vs 64 for samples..

            # with torch.no_grad():                

            #     sample = model.sample()
            #     objective = torch.zeros_like(sample[:, 0, 0, 0])
            #     objective += float(-np.log(args.n_bins) * np.prod(sample.shape[1:]))
            #     # print (torch.min(sample), torch.max(sample))
            #     sample = torch.clamp(sample, min=1e-5, max=1-1e-5)
            #     # print (torch.min(sample), torch.max(sample))

            #     z, objective = model(sample, objective)
            #     nll = (-objective) / float(np.log(2.) * np.prod(sample.shape[1:]))
            #     nobj_sample = torch.mean(nll)

            # data_dict['LL']['sample_LL'].append(numpy(-nobj_sample))


            # print (torch.max(img), torch.min(img), torch.max(sample), torch.min(sample))
            # fafdsds



    if step % args.curveplot_every ==0 and step > 0 and len(data_dict['steps']) > 2:
        plot_curve2(data_dict, exp_dir)





    # if step % args.curveplot_every*3 ==0 and step > 0:


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

                print (torch.min(sample), torch.max(sample), sample.shape)

                # sample = torch.clamp(sample, min=1e-5, max=1-1e-5)

                # objective = torch.zeros_like(sample[:, 0, 0, 0]) + float(-np.log(args.n_bins) * np.prod(sample.shape[1:]))
                # z, objective = model(sample, objective)
                # nll = (-objective) / float(np.log(2.) * np.prod(sample.shape[1:]))
                # nobj_sample = torch.mean(nll)

                # # print (temps[temp_i], nobj_sample)
                # print ('sample objective', nobj_sample)


                # if (sample!=sample).any():
                #     print ('nans in sample!!')
                #     # fafad

                # sample = torch.clamp(sample, min=1e-5, max=1-1e-5)
                grid = utils.make_grid(sample)
                img_file = images_dir +'plot'+str(step)+'.png'

                utils.save_image(grid, img_file)
                # print ('saved', img_file)
                print ('saved images')


            if step / args.plotimages_every ==1:

                init_loader = torch.utils.data.DataLoader(dataset, batch_size=min(64,len(dataset)), 
                                    shuffle=True, num_workers=1, drop_last=True)
                iter_loader_init = iter(init_loader)
                img = next(iter_loader_init)
                grid = utils.make_grid(img)
                img_file = images_dir +'realimages.png'
                utils.save_image(grid, img_file)
                print ('saved', img_file)


            if args.plotimages_every == 1: 
                dfadfdasf



            # #confirm perfect recon
            # sampling_loader = torch.utils.data.DataLoader(dataset, batch_size=min(sampling_batch_size,len(dataset)), 
            #             shuffle=True, num_workers=1, drop_last=True)
            # sampling_loader = iter(sampling_loader)
            # img = next(sampling_loader).cuda()

            # objective = torch.zeros_like(img[:, 0, 0, 0])
            # objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))

            # # aa = img.clone()            
            # z, objective = model(img, objective)
            # # print(torch.mean((aa - img)**2))
            # # fafds

            # model_args = {}
            # model_args['use_stored_sample'] = 1
            # # model_args['temp'] = 1
            # recons = model.reverse_(z, 0.)[0]

            # grid = utils.make_grid(recons)
            # img_file = images_dir +'recons'+str(step)+'.png'
            # utils.save_image(grid, img_file)
            # print ('saved', img_file)

            # grid = utils.make_grid(img)
            # img_file = images_dir +'reals'+str(step)+'.png'
            # utils.save_image(grid, img_file)
            # print ('saved', img_file)

            # fada


            # forwardlayers, names = model.forward_2(img, objective)
            # fdsafa


            # forwardlayers, names = model.forward_andgetlayers(img, objective)
            # reverselayers = model.reverse_andgetlayers(forwardlayers[-1], 0.)

            # # aa = z.clone()
            # # reverselayers = model.reverse_andgetlayers(z, 0.)
            # # bb = z.clone()
            # # print(torch.mean((aa - bb)**2))


            # print(torch.mean((z - forwardlayers[-1])**2))
            # # print(torch.mean((reverselayers[0] - forwardlayers[-1])**2))

            # print (len(forwardlayers))
            # print (len(reverselayers))

            # for ii in range(len(forwardlayers)):
            #     f = forwardlayers[ii]
            #     r = reverselayers[-(ii+1)]
            #     dif = torch.mean((f-r)**2)
            #     print (ii, names[ii], numpy(dif), forwardlayers[ii].shape)
            # fasf




    if step % args.save_every ==0 and i > 0:

        for f in os.listdir(params_dir):
            if 'model_params' in f:
                os.remove(params_dir + f)

        model.module.save_params_v3(params_dir, step, name='model_params')


        print ('Getting Full Test Set PBD')
        with torch.no_grad():

            #Get test set likelihood
            test_loader = torch.utils.data.DataLoader(test_x, batch_size=min(100,len(dataset)), 
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


























        
