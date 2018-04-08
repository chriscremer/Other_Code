

# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})

from os.path import expanduser
home = expanduser("~")


















def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)  #didnt see any change from doing this
    return [x, y]


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]


# def visdom_plot(viz, win, folder, game, name, bin_size=100, smooth=1):
#     tx, ty = load_data(folder, smooth, bin_size)
#     if tx is None or ty is None:
#         return win

#     fig = plt.figure()
#     plt.plot(tx, ty, label="{}".format(name))

#     # Ugly hack to detect atari
#     # if game.find('NoFrameskip') > -1:
#     plt.xticks([1e6, 2e6, 4e6, 6e6, 8e6, 10e6],
#                ["1M", "2M", "4M", "6M", "8M", "10M"])
#     plt.xlim(0, 10e6)
#     # else:
#     #     plt.xticks([1e5, 2e5, 4e5, 6e5, 8e5, 1e5],
#     #                ["0.1M", "0.2M", "0.4M", "0.6M", "0.8M", "1M"])
#     #     plt.xlim(0, 1e6)

#     plt.xlabel('Number of Timesteps')
#     plt.ylabel('Rewards')


#     plt.title(game)
#     plt.legend(loc=4)

#     # plt.show()
#     # plt.draw()
#     # image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

#     # plt.savefig('/home/ccremer/Documents/tmp/pytorch_pong_2017-11-01_17-16/plot.png')
#     plt.savefig(folder+'/plot.png')
#     print('made fig',folder+'/plot.png')
#     # plt.close(fig)
#     # fadfa

#     # # Show it in visdom
#     # image = np.transpose(image, (2, 0, 1))
#     # return viz.image(image, win=win)



# def plot_multiple_iterations(dir_all):

#     # dir_all contains dirs of each run of the same algo with different seeds

#     # print (os.listdir(dir_all))
#     txs, tys=[],[]
#     for dir_i in os.listdir(dir_all):
        
#         if os.path.isdir(dir_all+dir_i):
#             # print (dir_all+dir_i)
#             tx, ty = load_data(dir_all+dir_i, smooth=1, bin_size=100)
#             txs.append(tx)
#             tys.append(ty)
#             # break

#     # print (txs)
#     length = max([len(t) for t in txs])
#     longest = None
#     for j in range(len(txs)):
#         if len(txs[j]) == length:
#             longest = txs[j]
#     # For line with less data point, the last value will be repeated and appended
#     # Until it get the same size with longest one
#     for j in range(len(txs)):
#         if len(txs[j]) < length:
#             repeaty = np.ones(length - len(txs[j])) * tys[j][-1]
#             addindex = len(txs[j]) - length
#             addx = longest[addindex:]
#             tys[j] = np.append(tys[j], repeaty)
#             txs[j] = np.append(txs[j], addx)

#     x = np.mean(np.array(txs), axis=0)
#     y_mean = np.mean(np.array(tys), axis=0)
#     y_std = np.std(np.array(tys), axis=0)

#     color = color_defaults[0] 

#     fig = plt.figure()

#     y_upper = y_mean + y_std
#     y_lower = y_mean - y_std
#     plt.fill_between(x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3)
#     plt.plot(x, list(y_mean), label='a2c', color=color, rasterized=True)  


#     # plt.plot(tx, ty, label="{}".format('a2c'))
#     # plt.xticks([1e6, 2e6, 4e6, 6e6, 8e6, 10e6],
#     #            ["1M", "2M", "4M", "6M", "8M", "10M"])
#     # plt.xlim(0, 10e6)

#     plt.xticks([1e6, 2e6, 4e6, 6e6],
#                ["1M", "2M", "4M", "6M"])
#     plt.xlim(0, 6e6)

#     plt.xlabel('Number of Timesteps')
#     plt.ylabel('Rewards')

#     plt.title('pong')
#     plt.legend(loc=4)

#     plt.savefig(dir_all+'plot.png')
#     print('made fig',dir_all+'plot.png')





















def plot_multiple_iterations2(dir_all, ax, color, m_i):

    # dir_all contains dirs of each run of the same algo with different seeds

    # print (os.listdir(dir_all))
    txs, tys=[],[]
    for dir_i in os.listdir(dir_all):

        if os.path.isdir(dir_all+dir_i):
                
            monitor_dir = os.path.join(dir_all+dir_i, 'monitor_rewards')
            # print (monitor_dir)

            if os.path.isdir(monitor_dir):
                # print ('YESS')
                # print (dir_all+dir_i)
                tx, ty = load_data(monitor_dir, smooth=1, bin_size=100)
                txs.append(tx)
                tys.append(ty)
                # break
            # else:
            #     print ('NO')
            # fdsafa

    # if txs[0] != None:

    continue_ = True
    for i in range(len(txs)):
        if txs[i] == None:
            continue_ = False


    # if txs[0] != None:
    if continue_:


        
        # print (txs)
        length = max([len(t) for t in txs])
        longest = None
        for j in range(len(txs)):
            if len(txs[j]) == length:
                longest = txs[j]
        # For line with less data point, the last value will be repeated and appended
        # Until it get the same size with longest one
        for j in range(len(txs)):
            if len(txs[j]) < length:
                repeaty = np.ones(length - len(txs[j])) * tys[j][-1]
                addindex = len(txs[j]) - length
                addx = longest[addindex:]
                tys[j] = np.append(tys[j], repeaty)
                txs[j] = np.append(txs[j], addx)

        x = np.mean(np.array(txs), axis=0)
        y_mean = np.mean(np.array(tys), axis=0)
        y_std = np.std(np.array(tys), axis=0)

        # if m_i =='a2c':
        #     print (y_mean.shape)
        #     fad

        # color = color_defaults[0] 

        # fig = plt.figure()

        y_upper = y_mean + y_std
        y_lower = y_mean - y_std
        plt.fill_between(x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3)
        plt.plot(x, list(y_mean), label=m_i, color=color, rasterized=True)  
        plt.legend(loc=4)

        # plt.plot(tx, ty, label="{}".format('a2c'))
        # plt.xticks([1e6, 2e6, 4e6, 6e6, 8e6, 10e6],
        #            ["1M", "2M", "4M", "6M", "8M", "10M"])
        # plt.xlim(0, 10e6)


        # plt.savefig(dir_all+'plot.png')
        # print('made fig',dir_all+'plot.png')

















def make_plots(model_dict):
    #this gets called during training

    exp_path = model_dict['exp_path']
    num_frames = model_dict['num_frames']


    # Get number of envs to define subplots
    n_envs = 0
    for dir_i in os.listdir(exp_path):
        if os.path.isdir(exp_path+dir_i):
            # print (exp_path+dir_i)
            n_envs +=1

    cols = min(n_envs,3)
    # rows = max(n_envs // 3, 1)
    rows = int(np.ceil(n_envs /3))
    # print (rows, cols)

    fig = plt.figure(figsize=(8+cols,3+rows), facecolor='white')

    cur_col = 0
    cur_row = 0
    for env_i in os.listdir(exp_path):
        if os.path.isdir(exp_path+env_i):

            # print (cur_row, cur_col)
            ax = plt.subplot2grid((rows,cols), (cur_row,cur_col), frameon=False)#, aspect=0.3)# adjustable='box', )

            
            plt.xlim(0, num_frames)
            if cur_row == rows-1:
                if num_frames == 6e6:
                    plt.xticks([1e6, 2e6, 4e6, 6e6],["1M", "2M", "4M", "6M"])
                elif num_frames == 10e6:
                    plt.xticks([2e6, 4e6, 6e6, 8e6, 10e6],["2M", "4M", "6M", "8M", "10M"])
                plt.xlabel('Number of Timesteps',family='serif')
            else:
                plt.xticks([])
            if cur_col == 0:
                plt.ylabel('Rewards',family='serif')
            plt.title(env_i.split('NoF')[0],family='serif')

            ax.yaxis.grid(alpha=.1)
            # ax.set(aspect=1)
            # plt.gca().set_aspect('equal', adjustable='box')
            # ax.set_aspect(.5, adjustable='box')

            m_count =0
            for m_i in os.listdir(exp_path+env_i):
                m_dir = exp_path+env_i+'/'+m_i+'/'
                if os.path.isdir(m_dir):
                    
                    # print (cur_row, cur_col, m_dir)
                    color = color_defaults[m_count] 
                    plot_multiple_iterations2(m_dir, ax, color, m_i)
                    m_count+=1

            cur_col+=1
            if cur_col >= cols:
                cur_col = 0
                cur_row+=1


    fig_path = exp_path + model_dict['exp_name'] #+ 'exp_plot' 
    plt.savefig(fig_path+'.png')
    # print('made fig', fig_path+'.png')

    plt.savefig(fig_path+'.pdf')
    # print('made fig', fig_path+'.pdf')




    plt.close(fig)
















if __name__ == "__main__":


    exp_name =  'run_envs_6M_7' #'a2c_opt_smaller_lr' #, 'a2c_opt'# 'first_full_run/'
    exp_path = home + '/Documents/tmp/' + exp_name +'/'



    # Get number of envs to define subplots
    n_envs = 0
    for dir_i in os.listdir(exp_path):
        if os.path.isdir(exp_path+dir_i):
            print (exp_path+dir_i)
            n_envs +=1

    cols = min(n_envs,3)
    # rows = max(n_envs // 3, 1)
    rows = int(np.ceil(n_envs /3))
    print (rows, cols)

    fig = plt.figure(figsize=(8+cols,3+rows), facecolor='white')

    cur_col = 0
    cur_row = 0
    for env_i in os.listdir(exp_path):
        if os.path.isdir(exp_path+env_i):

            # print (cur_row, cur_col)
            ax = plt.subplot2grid((rows,cols), (cur_row,cur_col), frameon=False)#, aspect=0.3)# adjustable='box', )

            
            plt.xlim(0, 6e6)
            if cur_row == rows-1:
                plt.xticks([1e6, 2e6, 4e6, 6e6],["1M", "2M", "4M", "6M"])
                plt.xlabel('Number of Timesteps',family='serif')
            else:
                plt.xticks([])
            if cur_col == 0:
                plt.ylabel('Rewards',family='serif')
            plt.title(env_i.split('NoF')[0],family='serif')

            ax.yaxis.grid(alpha=.1)
            # ax.set(aspect=1)
            # plt.gca().set_aspect('equal', adjustable='box')
            # ax.set_aspect(.5, adjustable='box')

            m_count =0
            for m_i in os.listdir(exp_path+env_i):
                m_dir = exp_path+env_i+'/'+m_i+'/'
                if os.path.isdir(m_dir):

                    print (cur_row, cur_col, m_dir)
                    color = color_defaults[m_count] 
                    plot_multiple_iterations2(m_dir, ax, color, m_i)
                    m_count+=1

            

            cur_col+=1
            if cur_col >= cols:
                cur_col = 0
                cur_row+=1


    fig_path = exp_path + exp_name 
    plt.savefig(fig_path+'.png')
    print('made fig', fig_path+'.png')

    plt.savefig(fig_path+'.pdf')
    print('made fig', fig_path+'.pdf')
























#     # lim_val = .24
#     # xlimits=[-lim_val, lim_val]
#     # ylimits=[-lim_val, lim_val]

#     x_text = .05
#     y_text = .4

#     #annotate
#     
#     # ax.annotate('True Posterior', xytext=(.1, .5), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Blue', size='large')
#     ax.annotate('True\nPosterior', xytext=(x_text, y_text), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Black')#, size='large')
#     ax.set_yticks([])
#     ax.set_xticks([])
#     plt.gca().set_aspect('equal', adjustable='box')


#     ax = plt.subplot2grid((rows,cols), (1,0), frameon=False)



# fsadfa


#FOr each model






#     # plot_multiple_iterations(home+'/Documents/tmp/a2c_pong/')
#     plot_multiple_iterations(home+'/Documents/tmp/complete/')


#     fada




#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--algo', default='a2c',
#                         help='algorithm to use: a2c | ppo | acktr')
#     parser.add_argument('--dir', default='a2c',
#                         help='dir to load')
#     parser.add_argument('--env-name', default='PongNoFrameskip-v4',
#                         help='environment to train on (default: PongNoFrameskip-v4)')
#     args = parser.parse_args()



#     # from visdom import Visdom
#     # viz = Visdom()
#     # visdom_plot(viz, None, '/tmp/gym/', 'BreakOut', 'a2c', bin_size=100, smooth=1)
#     viz=None
#     visdom_plot(viz, None, args.dir, args.env_name, args.algo, bin_size=100, smooth=1)
#     print ('Done')






