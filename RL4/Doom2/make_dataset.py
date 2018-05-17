
from vizdoom import *

import os, sys
from os.path import expanduser
home = expanduser("~")

import numpy as np

import torch
import torch.nn.functional as F

import json
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import imageio 

def preprocess2(img):
    # numpy to pytoch, float, scale, cuda, downsample
    img = torch.from_numpy(img).float() / 255.
    img = F.avg_pool2d(img, kernel_size=2, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
    # print (img.shape) #[3,240,320]
    img = np.uint8(img.numpy()*255.)
    return img 





make_ = 1
view_ = 0


save_dir =  home+'/Documents/tmp/Doom2/' 
frame_repeat = 1 #2 #6 #1


if make_:

    # Create Doom instance
    print("Initializing doom...")
    # config_file_path = "../../scenarios/simpler_basic.cfg"
    # config_file_path = "../../scenarios/rocket_basic.cfg"
    # config_file_path = "../../scenarios/basic.cfg"
    config_file_path = home + "/ViZDoom/scenarios/take_cover.cfg"
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    # game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.\n")
    


    # Action = which buttons are pressed
    # n = game.get_available_buttons_size()
    print (game.get_available_buttons()) #_size()
    # fasf
    # actions = [list(a) for a in it.product([0, 1], repeat=n)]
    actions = [[0,0],[0,1],[1,0]]
    print ('Actions', actions)
    n_actions = len(actions)
    print ('n_actions', n_actions)
    print()
    # fadsfsa




    # Run game with random actions, record trajectories
    # list of lists, where lists are trajectories. trajectories have actinos and states 
    dataset = []
    tmp_traj = []
    dataset_count = 0
    done = 0

    print ('Make dataset')
    #run agent, store states, save dataset
    game.new_episode()
    step_count = 0
    while 1:

        # print (step_count)

        state = game.get_state()

        s1 = state.screen_buffer  #[3,480,640] uint8
        s1 = preprocess2(s1) #[3,240,320]

        # TO SEE HEALTH
        # print (game.get_available_game_variables())
        # print (game.get_game_variable(GameVariable.HEALTH))
        # print (state.game_variables[0])
        # fasd


        action_i = np.random.randint(0,n_actions)
        action_cpu = actions[action_i]

        # print (action_i, reward, isterminal)
        reward = game.make_action(action_cpu, frame_repeat) / float(frame_repeat)

        isterminal = game.is_episode_finished()*1 #converts bool to int

        tmp_traj.append([s1, action_cpu, isterminal])
        step_count+=1
        dataset_count+=1

        # if game.is_episode_finished():
        if isterminal:

            print (step_count*12, 'steps', step_count, 'frames', dataset_count, 'total')

            fsdfa/

            dataset.append(tmp_traj)
            tmp_traj=[]
            # fadfa
            # score = game.get_total_reward()
            game.new_episode()
            step_count = 0

            if dataset_count > 10000:
                break





    save_to = save_dir+'doom_dataset_10000.pkl'
    pickle.dump( dataset, open(save_to, "wb" ) )
    print('saved ', save_to)


                    













if view_:

    load_from = save_dir+'doom_dataset_10000.pkl'


    print ("Load dataset")
    # dir_ = 'RoadRunner_colour_4'
    # dataset = pickle.load( open( home+'/Documents/tmp/breakout_2frames/breakout_trajectories_10000.pkl', "rb" ) )
    dataset = pickle.load( open( load_from, "rb" ) )

    print ('numb trajectories', len(dataset))
    print ([len(x) for x in dataset])
    print ('Avg len', np.mean([len(x) for x in dataset]), np.mean([len(x) for x in dataset])*12)
    print()



    rows = 1
    cols = 1

    traj_numb = 100
    print ('traj', str(traj_numb), 'len', str(len(dataset[traj_numb])))

    gif_dir = save_dir + 'view_dataset_traj'+str(traj_numb)+'/'


    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
        print ('Made dir', gif_dir) 
    else:
        print (gif_dir, 'exists')
        # fasdf 

    max_count = 100000 #1000
    # count = 0    

    cols = 1
    rows = 1

    traj = dataset[traj_numb]
    for i in range(len(traj)):

        s = traj[i][0]
        a = traj[i][1]
        t = traj[i][2]

        plt_path = gif_dir+'frame'+str(i)+'.png'
        # fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=640*rows)
        fig = plt.figure(figsize=(5+cols,3+rows), facecolor='white', dpi=150)

        s = np.rollaxis(s, 1, 0)
        frame = np.rollaxis(s, 2, 1)

        #Plot Frame
        ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.1, 1.04, 'Real Frame '+str(i)+'  a:'+str(a)+'  t:'+str(t), transform=ax.transAxes, family='serif', size=6)


        # plt.tight_layout()
        plt.savefig(plt_path)
        print ('saved viz',plt_path)
        plt.close(fig)

        # frame_count+=1

        # count+=1

    

    # print ('game over', game.is_episode_finished() )
    # print (count)
    # print (len(frames))


    print('Making Gif')
    # frames_path = save_dir+'gif/'
    images = []
    for i in range(len(traj)):
        images.append(imageio.imread(gif_dir+'frame'+str(i)+'.png'))

    gif_path_this = gif_dir+ 'gif_'+str(traj_numb)+'.gif'
    # imageio.mimsave(gif_path_this, images)
    imageio.mimsave(gif_path_this, images, duration=.1)
    print ('made gif', gif_path_this)
    







print ('Done.')


























