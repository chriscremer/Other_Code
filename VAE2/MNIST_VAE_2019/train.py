
from os.path import expanduser
home = expanduser("~")

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import _pickle as pickle
import argparse
import time
import subprocess
import json
import gzip

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn


from vae3 import VAE

from distributions import Gauss
from inference_net import Inf_Net
from generator import Generator

from plotting import plot_curve2, plot_images







def to_print_mean(x):
    return torch.mean(x).data.cpu().numpy()
def to_print(x):
    return x.data.cpu().numpy()



def get_batch(data, batch_size):

    N = len(data)
    idx = np.random.randint(N, size=batch_size)
    return data[idx]



def train(exp_dict):

    S = exp_dict

    torch.manual_seed(999)
    
    data_dict = {}
    data_dict['steps'] = []
    data_dict['warmup'] = []
    data_dict['welbo'] = []

    if exp_dict['train_encoder_only']:
        opt_all = optim.Adam(S['vae'].encoder.parameters(), lr=.0001)
        print ('training encoder only')
    else:
        opt_all = optim.Adam(S['vae'].parameters(), lr=.0001)
        print ('training encoder and decoder')

    start_time = time.time()
    step = 0
    for step in range(0, S['max_steps'] + 1):

        batch = get_batch(S['train_x'], S['batch_size'])
        warmup = min((step+S['load_step']) / float(S['warmup_steps']), 1.)

        outputs = S['vae'].forward(batch, warmup=warmup)

        opt_all.zero_grad()
        loss = -outputs['welbo']
        loss.backward()
        opt_all.step()




        if step%S['display_step']==0:
            print(
                # 'S:{:5d}'.format(step+load_step),
                'S:{:5d}'.format(step),
                'T:{:.2f}'.format(time.time() - start_time),
                # 'BPD:{:.4f}'.format(LL_to_BPD(outputs['elbo'].data.item())),
                'welbo:{:.4f}'.format(outputs['welbo'].data.item()),
                # 'elbo:{:.4f}'.format(outputs['elbo'].data.item()),
                # 'lpx:{:.4f}'.format(outputs['logpx'].data.item()),
                # 'lpz:{:.4f}'.format(outputs['logpz'].data.item()),
                # 'lqz:{:.4f}'.format(outputs['logqz'].data.item()),
                # 'lpx_v:{:.4f}'.format(valid_outputs['logpx'].data.item()),
                # 'lpz_v:{:.4f}'.format(valid_outputs['logpz'].data.item()),
                # 'lqz_v:{:.4f}'.format(valid_outputs['logqz'].data.item()),
                # 'warmup:{:.4f}'.format(warmup),
                )

            start_time = time.time()


                # model.eval()
                # with torch.no_grad():
                #     valid_outputs = model.forward(x=valid_x[:50].cuda(), warmup=1., inf_net=infnet_valid)
                #     svhn_outputs = model.forward(x=svhn[:50].cuda(), warmup=1., inf_net=infnet_svhn)
                # model.train()


            if step > S['start_storing_data_step']:

                data_dict['steps'].append(step)
                data_dict['warmup'].append(warmup)
                data_dict['welbo'].append(to_print(outputs['welbo']))


            if step % S['trainingplot_steps'] ==0 and step > 0 and len(data_dict['steps']) > 2:

                plot_curve2(data_dict, S['exp_dir'])

            if step % S['save_params_step'] ==0 and step > 0:
                
                S['vae'].encoder.save_params_v3(S['params_dir'], step, name='encoder_params')
                S['vae'].generator.save_params_v3(S['params_dir'], step, name='generator_params')

                # model.save_params_v3(save_dir=params_dir, step=step+load_step)
                # infnet_valid.save_params_v3(save_dir=params_dir, step=step+load_step, name='valid')
                # infnet_svhn.save_params_v3(save_dir=params_dir, step=step+load_step, name='svhn')

            if step % S['viz_steps']==0 and step > 0: 

                recon = to_print(outputs['x_hat']) #[B,784]

                plot_images(to_print(batch), recon, S['images_dir'], step)


            #     model.eval()
            #     with torch.no_grad():
            #         train_recon = model.forward(x=train_x[:10].cuda(), warmup=1.)['x_recon']
            #         valid_recon = model.forward(x=valid_x[:10].cuda(), warmup=1., inf_net=infnet_valid)['x_recon']
            #         svhn_recon = model.forward(x=svhn[:10].cuda(), warmup=1., inf_net=infnet_svhn)['x_recon']
            #         sample_prior = model.sample_prior(z=z_prior)
            #     model.train()

            #     vizualize(images_dir, step+load_step, train_real=train_x[:10], train_recon=train_recon,
            #                                 valid_real=valid_x[:10], valid_recon=valid_recon,
            #                                 svhn_real=svhn[:10], svhn_recon=svhn_recon,
            #                                 prior_samps=sample_prior)
























if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument('--exp_name', default='test', type=str)
    parser.add_argument('--save_to_dir', type=str) # default=home+'/Documents/')
    parser.add_argument('--which_gpu', default='0', type=str)
    parser.add_argument('--x_size', default=50, type=int)
    parser.add_argument('--z_size', default=50, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--q_dist', default='Gauss', type=str)
    parser.add_argument('--n_flows', default=2, type=int)

    parser.add_argument('--params_load_dir', default='')
    parser.add_argument('--load_step', default=0, type=int)

    parser.add_argument('--train_encoder_only', default=0)
    parser.add_argument('--generator_params_load_dir', default='')
    parser.add_argument('--generator_load_step', default=0, type=int)

    parser.add_argument('--display_step', default=500, type=int)
    parser.add_argument('--trainingplot_steps', default=5000, type=int)
    parser.add_argument('--viz_steps', default=5000, type=int)
    parser.add_argument('--start_storing_data_step', default=2001, type=int)
    parser.add_argument('--save_params_step', default=50000, type=int)
    parser.add_argument('--max_steps', default=400000, type=int)
    parser.add_argument('--warmup_steps', default=20000, type=int)

    parser.add_argument('--continue_training', default=0, type=int)



    # # reproducibility is good
    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)



    args = parser.parse_args()
    args_dict = vars(args) #convert to dict


    print ('Exp:', args.exp_name)
    print ('gpu:', args.which_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.which_gpu #  '0' #'1' #

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

    args_dict['exp_dir'] = exp_dir
    args_dict['params_dir'] = params_dir
    args_dict['images_dir'] = images_dir
    args_dict['code_dir'] = code_dir

    #Save args
    json_path = exp_dir+'args_dict.json'
    with open(json_path, 'w') as outfile:
        json.dump(args_dict, outfile, sort_keys=True, indent=4)
    #Copy code
    subprocess.call("(rsync -r --exclude=__pycache__/ . "+code_dir+" )", shell=True)




    # FASHION
    def load_mnist(path, kind='train'):

        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(-1, 784)

        return images#, labels


    path = home+'/Documents/fashion_MNIST'

    train_x = load_mnist(path=path)
    test_x = load_mnist(path=path, kind='t10k')

    train_x = train_x / 255.
    test_x = test_x / 255.

    train_x = torch.tensor(train_x).float().cuda()
    test_x = torch.tensor(test_x).float().cuda()

    args_dict['train_x'] = torch.tensor(train_x).float().cuda()
    args_dict['test_x'] = torch.tensor(test_x).float().cuda()

    #binarize
    # train_x = (train_x > .5).float()
    # test_x = (test_x > .5).float()

    print (train_x.shape)
    print (test_x.shape)

    # print (np.max(train_x))
    # print (test_x[3])
    # fsda






    print ('\nInit VAE')

    args_dict['act_func'] = torch.tanh # F.relu,
    args_dict['encoder_arch'] = [[args.x_size,200],[200,200],[200,args.z_size*2]]
    args_dict['decoder_arch'] = [[args.z_size,200],[200,200],[200,args.x_size]]


    args_dict['prior'] = Gauss(args.z_size)
    args_dict['encoder'] = Inf_Net(args_dict)
    args_dict['generator'] = Generator(args_dict)
    # print (args_dict['prior'])
    # print (args_dict['encoder'])
    # print (args_dict['generator'])


    args_dict['vae'] = VAE(args_dict).cuda()
    # print (vae)
    # vae.cuda()
    if args.load_step>0:
        args_dict['vae'].load_params_v3(save_dir=args.params_load_dir, step=args.load_step)


    # args_dict['encoder'].load_params_v3(save_dir=args.params_load_dir, step=args.load_step, name='encoder_params')
    if args.generator_load_step>0:
        args_dict['generator'].load_params_v3(save_dir=args.generator_params_load_dir, step=args.generator_load_step, name='generator_params')


    print ('VAE Initilized\n')
    print (args_dict['vae'])






    # #Load data
    # print ('Loading data' )
    # data_location = home + '/Documents/MNIST_data/'
    # # with open(data_location + 'binarized_mnist.pkl', 'rb') as f:
    # #     train_x, valid_x, test_x = pickle.load(f)
    # with open(data_location + 'binarized_mnist.pkl', 'rb') as f:
    #     train_x, valid_x, test_x = pickle.load(f, encoding='latin1')
    # args_dict['train_x'] = torch.tensor(train_x).float().cuda()
    # args_dict['valid_x'] = torch.tensor(valid_x).float().cuda()
    # args_dict['test_x'] = torch.tensor(test_x).float().cuda()
    # print ('Train', args_dict['train_x'].shape)
    # print ('Valid', args_dict['valid_x'].shape)
    # print ('Test', args_dict['test_x'].shape)









    # sd = args_dict['vae'].state_dict()
    # for key, val in sd.items():
    #     print (key)
    # fsadfa
    print('\n training')
    train(args_dict)

    print ('Done.')






































