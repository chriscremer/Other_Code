

# one experiment: multiple envs, multple models, multiple iterations


import os
from os.path import expanduser
home = expanduser("~")

import models as models_file
from train import train

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print ('Made dir', path) 







# Experiment 
##################
exp_name = 'first_exp_4'
envs = ['BreakoutNoFrameskip-v4','SeaquestNoFrameskip-v4','PongNoFrameskip-v4'] #'BeamriderNoFrameskip-v4',
models = [models_file.model1, models_file.model2]
iters = 1
num_frames = 1e4#6e6
#####################




exp_path = home + '/Documents/tmp/' + exp_name
make_dir(exp_path)

for env in envs:

    env_path = exp_path +'/'+env
    make_dir(env_path)

    for model in models:

        model_path = env_path +'/'+ model['name']
        make_dir(model_path)

        for iter_i in range(iters):

            iter_path = model_path +'/'+ str(iter_i)
            make_dir(iter_path)

            model['seed']=iter_i
            model['save_to']=iter_path
            model['num_frames']=num_frames
            model['env'] = env

            print (env, model['name'], iter_i)
            train(model)
            print('')


print ('Done.')







