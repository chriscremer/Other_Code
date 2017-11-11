



import os
from os.path import expanduser
home = expanduser("~")

import json
import subprocess

import models as mf
# from train import train

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print ('Made dir', path) 


def print_stuff():
    print ()
    print ('Exp Name:', exp_name)
    print (envs)
    print ([m['name'] for m in models_list])
    print ('Noframskip', noFrameSkip)
    print ('Iters', iters)
    print ('Num_frames', num_frames)
    print ('save_interval', save_interval)
    print ('which_gpu', which_gpu)
    print ('num_processes', num_processes)
    print ()
    print ()





# Experiment 
##################
exp_name = 'confirm_a2c_works_enduro'
envs = ['Enduro']# ['Breakout'] #['Freeway'] #['Pong']##,'Seaquest',,,'Kangaroo', 'BeamRider', 'Alien', 
            # 'Amidar','Assault', 'Freeway',
            # 'MontezumaRevenge','Venture','Zaxxon','PrivateEye', 'Gopher']
models_list = [mf.a2c_long]  #[mf.ppo_v1]# ##  #  ##  ##  ##, 
which_gpu = 1
noFrameSkip = True
iters = 1
num_frames = 6e6
save_interval=1e6 #save model params and videos
num_processes=20
#####################









print_stuff()


if noFrameSkip:
    envs = [x+'NoFrameskip-v4' for x in envs]


exp_path = home + '/Documents/tmp/' + exp_name+'/'
make_dir(exp_path)

for env in envs:

    env_path = exp_path +env
    make_dir(env_path)

    for model_dict in models_list:

        model_path = env_path +'/'+ model_dict['name']
        make_dir(model_path)

        for iter_i in range(iters):

            iter_path = model_path +'/'+ 'seed'+str(iter_i)
            make_dir(iter_path)

            model_dict['exp_name'] = exp_name
            model_dict['exp_path'] = exp_path
            model_dict['seed']=iter_i
            model_dict['save_to']=iter_path
            model_dict['num_frames']=num_frames
            model_dict['env'] = env
            model_dict['cuda'] = True
            model_dict['which_gpu'] = which_gpu
            model_dict['save_interval'] = save_interval
            model_dict['num_processes'] = num_processes

            print (env, model_dict['name'], iter_i)

            #write to json
            json_path = iter_path+'/model_dict.json'
            with open(json_path, 'w') as outfile:
                json.dump(model_dict, outfile,sort_keys=True, indent=4)

            #train model
            subprocess.call("(cd "+home+"/Other_Code/RL4/pytorch-a2c-ppo-acktr/ && python train3.py --m {})".format(json_path), shell=True) 
            print('')

    print_stuff()

print ('Done.')






#ENVIRONMENT LIST

# Alien
# Amidar
# Assault
# Asterix
# Asteroids
# Atlantis
# BankHeist
# BattleZone
# BeamRider
# Bowling
# Boxing
# Breakout
# Centipede
# ChopperCommand
# CrazyClimber
# DemonAttack
# DoubleDunk
# Enduro
# FishingDerby
# Freeway
# Frostbite
# Gopher
# Gravitar
# IceHockey
# Jamesbond
# Kangaroo
# Skipping
# Krull
# KungFuMaster
# MontezumaRevenge
# MsPacman
# NameThisGame
# Pitfall
# Pong
# PrivateEye
# Qbert
# Riverraid
# RoadRunner
# Robotank
# Seaquest
# SpaceInvaders
# StarGunner
# Tennis
# TimePilot
# Tutankham
# UpNDown
# Venture
# VideoPinball
# WizardOfWor
# Zaxxon






