



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

from collections import deque 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn


from vae3 import VAE

from distributions import Gauss
from distributions import Flow1
from distributions import Flow_Cond

from inference_net import Inf_Net
from generator import Generator





def to_print_mean(x):
    return torch.mean(x).data.cpu().numpy()
def to_print(x):
    return x.data.cpu().numpy()




def logmeanexp(elbo):
    # [P,B]
    max_ = np.max(elbo, axis=0, keepdims=True)
    # print (max_.shape)
    elbo = np.log(np.mean(np.exp(elbo - max_), axis=0)) + max_ #[B]
    return elbo






def amort_q_bounds(exp_dict, k, batch_idx):

    exp_dict['vae'].eval()
    batch = exp_dict['train_x'][batch_idx]

    bounds = []
    for i in range(k):

        outputs = exp_dict['vae'].forward(batch)
        bounds.append(to_print(outputs['elbo_B']))

    bounds = np.array(bounds)
    L_vae = np.mean(bounds, axis=0)
    L_iwae = np.squeeze(logmeanexp(bounds))

    return L_vae, L_iwae






def optimal_q_bounds(exp_dict, k, batch_idx):

    vae = exp_dict['vae']
    vae.eval()
    
    L_vaes = []
    L_iwaes = []
    for j in batch_idx:

        if j%10==0:
            print (j+1, len(batch_idx))

        batch = exp_dict['train_x'][j].view(1,784)

        # if exp_dict['q_dist'] == 'Gauss':
        #     mu, logvar = vae.encoder.encode(batch)
        #     q = Gauss(exp_dict['z_size'], mu=mu, logvar=logvar).cuda()
        # else:
        #     TODO


        mu, logvar = vae.encoder.encode(batch)


        # q = Gauss(exp_dict['z_size'], mu=mu, logvar=logvar).cuda()
        # q = Flow1(exp_dict, mu=mu, logvar=logvar).cuda()

        # exp_dict['n_flows'] = 5
        # q = Flow_Cond(exp_dict, mu=mu, logvar=logvar).cuda()



        if exp_dict['q_dist'] == 'Gauss':
            q = Gauss(exp_dict['z_size'], mu=mu, logvar=logvar).cuda()

        elif exp_dict['q_dist'] == 'Flow':
            q = Flow1(exp_dict, mu=mu, logvar=logvar).cuda()

        elif exp_dict['q_dist'] == 'Flow_Cond':
            q = Flow_Cond(exp_dict, mu=mu, logvar=logvar).cuda()


        opt_q = optimize_local_q_dist(exp_dict, batch, q)

        bounds = []
        for i in range(k):

            z, logqz = opt_q.sample()
            logpz = vae.prior.logprob(z)
            x_hat, logpxz = vae.generator.decode(batch,z)
            obj = logpxz + logpz - logqz

            bounds.append(to_print(obj))

        bounds = np.array(bounds)

        L_vae = np.mean(bounds, axis=0)
        L_iwae = np.squeeze(logmeanexp(bounds))

        L_vaes.append(L_vae)
        L_iwaes.append(L_iwae)

    return np.array(L_vaes), np.array(L_iwaes)





def optimize_local_q_dist(exp_dict, x, q):

    vae = exp_dict['vae']

    opt = optim.Adam(q.parameters(), lr=.001)

    last_100 = deque(maxlen=100)
    best_last_100_avg = -1
    consecutive_worse = 0
    for step in range(50000):

        # objs = 0
        # P = 2
        # for k in range(P):
        #     z, logqz = q.sample()
        #     logpz = vae.prior.logprob(z)
        #     x_hat, logpxz = vae.generator.decode(x,z)
        #     obj = logpxz + logpz - logqz
        #     objs += obj
        # objs = objs/P

        z, logqz = q.sample()
        logpz = vae.prior.logprob(z)
        x_hat, logpxz = vae.generator.decode(x,z)
        obj = logpxz + logpz - logqz

        opt.zero_grad()
        loss = -obj
        loss.backward()
        opt.step()

        loss = to_print(loss)
        last_100.append(loss)
        if step % 100 ==0:
            last_100_avg = np.mean(last_100)
            if last_100_avg< best_last_100_avg or best_last_100_avg == -1:
                consecutive_worse=0
                best_last_100_avg = last_100_avg
            else:
                consecutive_worse +=1 
                if consecutive_worse> 20:
                    break
            # if step % 2000 ==0:
            #     print (step, last_100_avg, consecutive_worse)

    return q














if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', default='test', type=str)
    parser.add_argument('--save_to_dir', type=str) # default=home+'/Documents/')
    parser.add_argument('--which_gpu', default='0', type=str)
    parser.add_argument('--q_dist', default='Gauss', type=str)
    parser.add_argument('--n_flows', default=2, type=int)
    parser.add_argument('--x_size', type=int)
    parser.add_argument('--z_size', type=int)

    parser.add_argument('--encoder_params_load_dir', default='')
    parser.add_argument('--encoder_load_step', default=0, type=int)
    parser.add_argument('--generator_params_load_dir', default='')
    parser.add_argument('--generator_load_step', default=0, type=int)

    # parser.add_argument('--params_load_dir', type=str, required=True)
    # parser.add_argument('--load_step', type=int, required=True)

    parser.add_argument('--display_step', default=500, type=int)



    args = parser.parse_args()
    args_dict = vars(args) #convert to dict


    print ('Exp:', args.exp_name)
    print ('gpu:', args.which_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.which_gpu #  '0' #'1' #




    exp_dir = args.save_to_dir + args.exp_name + '/'
    gaps_dir = exp_dir + 'gaps/'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print ('Made dir', exp_dir) 

    if not os.path.exists(gaps_dir):
        os.makedirs(gaps_dir)
        print ('Made dir', gaps_dir) 

    args_dict['exp_dir'] = exp_dir
    args_dict['gaps_dir'] = gaps_dir


    #Save args
    json_path = gaps_dir+'args_dict.json'
    with open(json_path, 'w') as outfile:
        json.dump(args_dict, outfile, sort_keys=True, indent=4)
    #Copy code
    # subprocess.call("(rsync -r --exclude=__pycache__/ . "+code_dir+" )", shell=True)





    print ('\nInit VAE')

    args_dict['act_func'] = torch.tanh # F.relu,
    args_dict['encoder_arch'] = [[args.x_size,200],[200,200],[200,args.z_size*2]]
    args_dict['decoder_arch'] = [[args.z_size,200],[200,200],[200,args.x_size]]


    args_dict['prior'] = Gauss(args.z_size)
    args_dict['encoder'] = Inf_Net(args_dict)
    args_dict['generator'] = Generator(args_dict)

    # print ('prior', args_dict['prior'])
    # print ('encoder', args_dict['encoder'])
    # print ('generator', args_dict['generator'])

    # print (args_dict['prior'].parameters())
    # print (args_dict['prior'].named_parameters())
    # for key, val in args_dict['prior'].state_dict().items():
    #     print (key)
    # fddsf
    # fasdfs

    # args_dict['encoder'].load_params_v3(save_dir=args.params_load_dir, step=args.load_step, name='encoder_params')
    # args_dict['generator'].load_params_v3(save_dir=args.params_load_dir, step=args.load_step, name='generator_params')

    # args_dict['encoder'].load_params_v3(save_dir=args.params_load_dir, step=args.load_step, name='encoder_params')
    if args.generator_load_step>0:
        args_dict['generator'].load_params_v3(save_dir=args.generator_params_load_dir, step=args.generator_load_step, name='generator_params')

    # args_dict['encoder'].load_params_v3(save_dir=args.params_load_dir, step=args.load_step, name='encoder_params')
    if args.encoder_load_step>0:
        args_dict['encoder'].load_params_v3(save_dir=args.encoder_params_load_dir, step=args.encoder_load_step, name='encoder_params')





    args_dict['vae'] = VAE(args_dict).cuda()
    # print (args_dict['vae'])
    # print ('VAE Initilized\n')






    #Load data
    # print ('Loading data' )
    data_location = home + '/Documents/MNIST_data/'
    # with open(data_location + 'binarized_mnist.pkl', 'rb') as f:
    #     train_x, valid_x, test_x = pickle.load(f)
    with open(data_location + 'binarized_mnist.pkl', 'rb') as f:
        train_x, valid_x, test_x = pickle.load(f, encoding='latin1')
    args_dict['train_x'] = torch.tensor(train_x).float().cuda()
    args_dict['valid_x'] = torch.tensor(valid_x).float().cuda()
    args_dict['test_x'] = torch.tensor(test_x).float().cuda()
    # print ('Train', args_dict['train_x'].shape)
    # print ('Valid', args_dict['valid_x'].shape)
    # print ('Test', args_dict['test_x'].shape)



    torch.manual_seed(999)


    k = 500
    n = 100
    batch_idx = np.array(list(range(n)))


    print ('\nq bounds')
    L_vae_q, L_iwae_q = amort_q_bounds(args_dict, k=k, batch_idx=batch_idx)

    print ('L_vae_q', np.mean(L_vae_q), np.std(L_vae_q))
    print ('L_iwae_q', np.mean(L_iwae_q), np.std(L_iwae_q))




    print ('\nq* bounds')
    L_vae_qstar, L_iwae_qstar = optimal_q_bounds(args_dict, k=k, batch_idx=batch_idx)
    print (L_vae_qstar.shape)
    print ('L_vae_qstar', np.mean(L_vae_qstar), np.std(L_vae_qstar))
    print ('L_iwae_qstar', np.mean(L_iwae_qstar), np.std(L_iwae_qstar))

    print ('amort', np.mean(L_vae_qstar) - np.mean(L_vae_q))
    print ('approx', np.mean(L_iwae_qstar) - np.mean(L_vae_qstar))


    gaps_dict = {}
    gaps_dict['L_vae_q'] = L_vae_q
    gaps_dict['L_iwae_q'] = L_iwae_q
    gaps_dict['L_vae_qstar'] = L_vae_qstar
    gaps_dict['L_iwae_qstar'] = L_iwae_qstar

    save_to=os.path.join(gaps_dir, "results_n"+str(n)+".pkl")
    with open(save_to, "wb" ) as f:
        pickle.dump(gaps_dict, f)
    print ('saved results', save_to)

    # #Save args
    # json_path = gaps_dir+'results_json.json'
    # with open(json_path, 'w') as outfile:
    #     json.dump(gaps_dict, outfile, sort_keys=True, indent=4)












































