



# a2c with rmsprop had sudden spike drops in performance
# i want to see if adam fixes it


import os
from os.path import expanduser
home = expanduser("~")

import json
import subprocess

import models as models_file
# from train import train

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print ('Made dir', path) 








# Experiment 
##################
exp_name = 'a2c_opt_smaller_lr'
envs = ['BreakoutNoFrameskip-v4']#,'SeaquestNoFrameskip-v4','PongNoFrameskip-v4'] #'BeamRiderNoFrameskip-v4',
models = [models_file.a2c_rms, models_file.a2c_adam]
iters = 2
num_frames = 6e6
#####################



exp_path = home + '/Documents/tmp/' + exp_name
make_dir(exp_path)

for env in envs:

    env_path = exp_path +'/'+env
    make_dir(env_path)

    for model_dict in models:

        model_path = env_path +'/'+ model_dict['name']
        make_dir(model_path)

        for iter_i in range(iters):

            iter_path = model_path +'/'+ str(iter_i)
            make_dir(iter_path)

            model_dict['seed']=iter_i
            model_dict['save_to']=iter_path
            model_dict['num_frames']=num_frames
            model_dict['env'] = env

            print (env, model_dict['name'], iter_i)

            #write to json
            json_path = iter_path+'/model.json'
            with open(json_path, 'w') as outfile:
                json.dump(model_dict, outfile)

            #train model
            subprocess.call("(cd "+home+"/Other_Code/RL4/pytorch-a2c-ppo-acktr/ && python train.py --m {})".format(json_path), shell=True) 
            print('')


print ('Done.')







