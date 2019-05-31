
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
from torch.optim import lr_scheduler
import torch.nn as nn


from vae3 import Conditional_VAE

from distributions import Gauss
from inference_net import Inf_Net
from generator import Conditional_Generator
from text_nets import Text_encoder

from plotting import plot_curve2, plot_images, plot_images_dif_zs, get_sentence





sys.path.insert(0, os.path.abspath(home+'/Other_Code/VLVAE/CLEVR/utils'))

from preprocess_statements import preprocess_v2
from Clevr_data_loader import ClevrDataset, ClevrDataLoader







def to_print_mean(x):
    return torch.mean(x).data.cpu().numpy()
def to_print(x):
    return x.data.cpu().numpy()







def train(exp_dict):

    S = exp_dict
    model = S['vae']

    # torch.manual_seed(999)
    
    data_dict = {}
    data_dict['steps'] = []
    data_dict['welbo'] = []
    data_dict['lpx'] = []
    data_dict['lpz'] = []
    data_dict['lqz'] = []
    data_dict['warmup'] = []
    data_dict['lr'] = []

    lr=.004

    if exp_dict['train_encoder_only']:
        opt_all = optim.Adam(model.encoder.parameters(), lr=lr, weight_decay=.0000001)
        print ('training encoder only')
    else:
        opt_all = optim.Adam(model.parameters(), lr=lr, weight_decay=.0000001)
        lr_sched = lr_scheduler.StepLR(opt_all, step_size=1, gamma=0.999995)

    start_time = time.time()
    step = 0
    for step in range(0, S['max_steps'] + 1):

        img_batch, text_batch = get_batch(image_dataset=S['train_x'], text_dataset=S['train_y'], batch_size=S['batch_size'])
        # warmup = min((step+S['load_step']) / float(S['warmup_steps']), 1.)
        warmup = max(50. - (step / float(S['warmup_steps'])), 1.)

        outputs = model.forward(img_batch, text_batch, warmup=warmup)

        opt_all.zero_grad()
        loss = -outputs['welbo']
        loss.backward()
        opt_all.step()
        lr_sched.step()




        if step%S['display_step']==0:
            print(
                # 'S:{:5d}'.format(step+load_step),
                'S:{:5d}'.format(step),
                'T:{:.2f}'.format(time.time() - start_time),
                # 'BPD:{:.4f}'.format(LL_to_BPD(outputs['elbo'].data.item())),
                'welbo:{:.4f}'.format(outputs['welbo'].data.item()),
                'elbo:{:.4f}'.format(outputs['elbo'].data.item()),
                'lpx:{:.4f}'.format(outputs['logpxz'].data.item()),
                'lpz:{:.4f}'.format(outputs['logpz'].data.item()),
                'lqz:{:.4f}'.format(outputs['logqz'].data.item()),
                # 'lpx_v:{:.4f}'.format(valid_outputs['logpx'].data.item()),
                # 'lpz_v:{:.4f}'.format(valid_outputs['logpz'].data.item()),
                # 'lqz_v:{:.4f}'.format(valid_outputs['logqz'].data.item()),
                'warmup:{:.2f}'.format(warmup),
                )

            start_time = time.time()


                # model.eval()
                # with torch.no_grad():
                #     valid_outputs = model.forward(x=valid_x[:50].cuda(), 
                            # warmup=1., inf_net=infnet_valid)
                #     svhn_outputs = model.forward(x=svhn[:50].cuda(), 
                            # warmup=1., inf_net=infnet_svhn)
                # model.train()


            if step > S['start_storing_data_step']:

                data_dict['steps'].append(step)
                data_dict['warmup'].append(warmup)
                data_dict['welbo'].append(to_print(outputs['welbo']))
                data_dict['lpx'].append(to_print(outputs['logpxz']))
                data_dict['lpz'].append(to_print(outputs['logpz']))
                data_dict['lqz'].append(to_print(outputs['logqz']))
                data_dict['lr'].append(lr_sched.get_lr()[0])



            if step % S['trainingplot_steps'] ==0 and step > 0 and len(data_dict['steps']) > 2:

                plot_curve2(data_dict, S['exp_dir'])

            if step % S['save_params_step'] ==0 and step > 0:
                
                # model.encoder.save_params_v3(S['params_dir'], step, name='encoder_params')
                # model.generator.save_params_v3(S['params_dir'], step, name='generator_params')
                model.save_params_v3(S['params_dir'], step, name='model_params')

                # model.save_params_v3(save_dir=params_dir, step=step+load_step)
                # infnet_valid.save_params_v3(save_dir=params_dir, step=step+load_step, name='valid')
                # infnet_svhn.save_params_v3(save_dir=params_dir, step=step+load_step, name='svhn')

            if step % S['viz_steps']==0 and step > 0: 

                recon = to_print(outputs['x_hat']) #[B,784]
                plot_images(to_print(img_batch), recon, S['images_dir'], step)


            if (step % (int(S['viz_steps']/5))) ==0 and step > 0: 

                image1 = img_batch[0].view(1,3,112,112)
                image2 = img_batch[1].view(1,3,112,112)
                text1 = text_batch[0].view(1,9)
                text2 = text_batch[1].view(1,9)
                z = model.get_z(image1)
                new_image1 = model.generate_given_z_y(y=text1, z=z)
                new_image2 = model.generate_given_z_y(y=text2, z=z)

                plot_images_dif_zs(to_print(image1), to_print(image2), 
                                    get_sentence(question_idx_to_token, text1[0], [3,4]), 
                                    get_sentence(question_idx_to_token, text2[0], [3,4]), 
                                    to_print(new_image1), to_print(new_image2),
                                    image_dir=S['images_dir'], step=step)



            #     model.eval()
            #     with torch.no_grad():
            #         train_recon = model.forward(x=train_x[:10].cuda(), warmup=1.)['x_recon']
            #         valid_recon = model.forward(x=valid_x[:10].cuda(), 
                                        # warmup=1., inf_net=infnet_valid)['x_recon']
            #         svhn_recon = model.forward(x=svhn[:10].cuda(), 
                                        # warmup=1., inf_net=infnet_svhn)['x_recon']
            #         sample_prior = model.sample_prior(z=z_prior)
            #     model.train()

            #     vizualize(images_dir, step+load_step, train_real=train_x[:10], 
                                                # train_recon=train_recon,
            #                                 valid_real=valid_x[:10], valid_recon=valid_recon,
            #                                 svhn_real=svhn[:10], svhn_recon=svhn_recon,
            #                                 prior_samps=sample_prior)
























if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument('--exp_name', default='test', type=str)
    parser.add_argument('--save_to_dir', type=str) # default=home+'/Documents/')
    parser.add_argument('--which_gpu', default='0', type=str)
    parser.add_argument('--x_size', type=int)
    parser.add_argument('--img_dim', type=int)
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



    # reproducibility is good
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


    args = parser.parse_args()
    args_dict = vars(args) #convert to dict



    ################################################################################
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
    ################################################################################








    ################################################################################
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



    def get_batch(image_dataset, text_dataset, batch_size):

        N = len(image_dataset)
        idx = np.random.randint(N, size=batch_size)
        img_batch = image_dataset[idx]
        question_batch = text_dataset[idx]

        img_batch = torch.from_numpy(img_batch).cuda()
        # question_batch = torch.from_numpy(question_batch).cuda()
        question_batch = question_batch.cuda()

        img_batch = torch.clamp(img_batch, min=1e-5, max=1-1e-5)

        # print (torch.max(img_batch))
        # print (torch.min(img_batch))
        # fsdafsa

        # img_batch += torch.zeros_like(img_batch).uniform_(0., 1./args.n_bins)

        return img_batch, question_batch



    # def get_batch(data, batch_size):

    #     N = len(data)
    #     idx = np.random.randint(N, size=batch_size)
    #     batch = data[idx]

    #     batch = torch.from_numpy(batch).cuda() 

    #     # print (torch.max(batch))
    #     # print (torch.min(batch))
    #     # fsadf
    #     batch = torch.clamp(batch, min=1e-5, max=1-1e-5)

    #     return batch




    # quick_check_data = home+ "/VL/two_objects_large/quick_stuff.pkl" 
    # with open(quick_check_data, "rb" ) as f:
    #     stuff = pickle.load(f)
    #     train_image_dataset, train_question_dataset, val_image_dataset, \
    #          val_question_dataset, test_image_dataset, test_question_dataset, \
    #          train_indexes, val_indexes, question_idx_to_token, \
    #             question_token_to_idx, q_max_len, vocab_size = stuff

    # img_batch, question_batch = get_batch(train_image_dataset, 
                                # train_question_dataset, batch_size=4)
    # train_image_dataset = train_image_dataset[:50000]

    
    args_dict['train_x'] = train_image_dataset[:50000] 
    args_dict['train_y'] = train_question_dataset[:50000] 
    print (args_dict['train_x'].shape)
    print (args_dict['train_y'].shape)

    args_dict['vocab_size'] = vocab_size
    args_dict['word_embedding_size'] = 50
    args_dict['y_enc_size'] = 50
    # fasdfa
    ################################################################################




    







    ################################################################################
    #INIT MODEL
    print ('\nInit VAE')

    # args_dict['act_func'] = torch.tanh # F.relu,
    # args_dict['encoder_arch'] = [[args.x_size,200],[200,200],[200,args.z_size*2]]
    # args_dict['decoder_arch'] = [[args.z_size,200],[200,200],[200,args.x_size]]


    args_dict['prior'] = Gauss(args.z_size)
    args_dict['encoder'] = Inf_Net(args_dict)
    args_dict['generator'] = Conditional_Generator(args_dict)
    args_dict['text_encoder'] = Text_encoder(args_dict) 
    # print (args_dict['prior'])
    # print (args_dict['encoder'])
    # print (args_dict['generator'])
    args_dict['vae'] = Conditional_VAE(args_dict).cuda()
    # print (vae)

    #LOAD PARAMS
    if args.load_step>0:
        args_dict['vae'].load_params_v3(save_dir=args.params_load_dir, step=args.load_step)
    # args_dict['encoder'].load_params_v3(save_dir=args.params_load_dir, 
                # step=args.load_step, name='encoder_params')
    if args.generator_load_step>0:
        args_dict['generator'].load_params_v3(save_dir=args.generator_params_load_dir, 
                                                step=args.generator_load_step, 
                                                name='generator_params')

    print ('VAE Initilized')
    # print (args_dict['vae'])
    print("number of model parameters:", sum([np.prod(p.size()) for p in args_dict['vae'].parameters()]))
    ################################################################################











    ################################################################################
    # sd = args_dict['vae'].state_dict()
    # for key, val in sd.items():
    #     print (key)
    # fsadfa
    print('\nTraining')
    train(args_dict)

    print ('Done.')
    ################################################################################






































