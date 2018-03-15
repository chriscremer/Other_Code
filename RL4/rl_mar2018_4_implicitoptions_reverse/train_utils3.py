





import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import subprocess
import os
import shutil

import imageio

import copy

import csv  


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


def makedir(dir_, print_=True, rm=False):

    if rm and os.path.exists(dir_):
        shutil.rmtree(dir_)
        os.makedirs(dir_)
        print ('rm dir and made', dir_) 
        
        # if print_:
        #     print ('Made dir', dir_) 
    elif not os.path.exists(dir_):
        os.makedirs(dir_)
        if print_:
            print ('Made dir', dir_) 



def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    # halfwidth = min(31, int(np.ceil(len(x) / 30)))
    halfwidth = min(21, int(np.ceil(len(x) / 20)))

    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


















def do_grad_var(envs, agent, model_dict, total_num_steps, update_current_state, update_rewards):

    save_dir = model_dict['save_to']
    shape_dim0 = model_dict['shape_dim0']
    num_processes = model_dict['num_processes']
    obs_shape = model_dict['obs_shape']
    dtype = model_dict['dtype']
    num_steps = model_dict['num_steps']
    gamma = model_dict['gamma']

    # print ('ls')

    # ls_path = save_dir+'/learning_signal/'
    ls_file = save_dir + '/monitor_other_info.csv'

    # if os.path.exists(ls_path):
    #     try:
    #         os.remove(ls_file)
    #     except OSError:
    #         pass

    # makedir(ls_path, print_=False)



    
    num_processes = 1
    #just need these so it can be passed to update other stuff, like masks
    episode_rewards = torch.zeros([num_processes, 1]) #keeps track of current episode cumulative reward
    final_rewards = torch.zeros([num_processes, 1])


    avg_over = 10   #over number of trajectories/episodes
    #things to record for each episode
    dist_entropies = []
    reward_sums = []
    # next_frame_errors = []
    grads = []
    value_pred_error = []



    # get data
    for j in range(avg_over):

        # print (j)



        # Init state
        state = envs.reset()  # (channels, height, width)
        state = np.expand_dims(state,0) # (1, channels, height, width)
        current_state = torch.zeros(num_processes, *obs_shape)  # (processes, channels*stack, height, width)
        current_state = update_current_state(current_state, state, shape_dim0).type(dtype) #add the new frame, remove oldest
        # agent.insert_first_state(current_state) #storage has states: (num_steps + 1, num_processes, *obs_shape), set first step 


        # #store in agent list
        # state_frames = []
        # value_frames = []
        # actions_frames = []

        agent.rollouts_list.reset() #empty lists
        agent.rollouts_list.states = [current_state]

        max_steps = 30


        step = 0
        done_ = False
        # while not done_ and step < 400:
        while not done_ and step < max_steps:

            # state1 = np.squeeze(state[0])
            # state_frames.append(state1)

            # Act, [P,1], [P], [P,1], [P]
            # state_pytorch = Variable(agent.rollouts.states[step])
            # state_pytorch = Variable(agent.rollouts_list.states[-1], volatile=True)
            state_pytorch = Variable(agent.rollouts_list.states[-1], volatile=False)
            value, action, action_log_probs, dist_entropy = agent.act(state_pytorch)
            # value_frames.append([value.data.cpu().numpy()[0][0]])

            # action_prob = agent.actor_critic.action_dist(Variable(agent.rollouts_list.states[-1], volatile=True))[0]
            # action_prob = np.squeeze(action_prob.data.cpu().numpy())  # [A]
            # actions_frames.append(action_prob)

            # value, action = agent.act(Variable(agent.rollouts_list.states[-1], volatile=True))
            # Step, S:[P,C,H,W], R:[P], D:[P]
            cpu_actions = action.data.squeeze(1).cpu().numpy() #[P]
            state, reward, done, info = envs.step(cpu_actions) 

            #because it only has one process
            state = np.expand_dims(state,0) # (1, channels, height, width)
            reward = np.expand_dims(reward,0) # (1, 1)
            done = np.expand_dims(done,0) # (1, 1)

            # Record rewards, Update state, Agent record step
            reward, masks, final_rewards, episode_rewards, current_state = update_rewards(reward, done, final_rewards, episode_rewards, current_state)
            current_state = update_current_state(current_state, state, shape_dim0)

            agent.rollouts_list.insert(step, current_state, action.data, value, reward.numpy()[0][0], masks, action_logprob=action_log_probs)





            done_ = done[0]
            step+=1
            dist_entropies.append(dist_entropy.data.cpu().numpy())
            # next_frame_errors.append(agent.state_pred_error.data.cpu().numpy()[0])
            



        next_value = agent.actor_critic(Variable(agent.rollouts_list.states[-1], volatile=True))[0].data
        agent.rollouts_list.compute_returns(next_value, gamma)
        # print (agent.rollouts_list.returns)#.cpu().numpy())

        # print (agent.rollouts_list.value_preds)

        # print (agent.rollouts_list.returns[:-1])

        values = torch.cat(agent.rollouts_list.value_preds, 0).view(step, 1, 1) 
        action_log_probs = torch.cat(agent.rollouts_list.action_log_probs).view(step, 1, 1)
        # dist_entropy = torch.cat(agent.rollouts_list.dist_entropy).view(max_steps, 1, 1)
        returns = torch.cat(agent.rollouts_list.returns[:-1], 0).view(step, 1, 1) 

        agent.rollouts_list.value_preds = []
        agent.rollouts_list.action_log_probs = []
        # agent.rollouts_list.dist_entropy = []
        agent.rollouts_list.state_preds = []

        advantages = Variable(returns).cuda() - values
        value_loss = advantages.pow(2).mean()


        action_loss = -(advantages.detach() * action_log_probs).mean()
        gradients = torch.autograd.grad(outputs=action_loss, inputs=(agent.actor_critic.conv1.parameters()), retain_graph=True, create_graph=True)
        # print (gradients[0])
        # print (gradients[1])  #what is this!!>!>
        # print(gradients.size())
        # print (torch.mean(gradients[0]))




        #Store info

        reward_sums.append(np.mean(agent.rollouts_list.rewards))
        grads.append(torch.mean(gradients[0]).data.cpu().numpy()[0])
        value_pred_error.append(value_loss.data.cpu().numpy()[0])


    # print (grads)
    # fsda

    # print (dist_entropy)
    # dist_entropies = 
    avg_ent = np.mean(dist_entropies)

    # if np.var(reward_sums) > 0:
    #     var_reward_sum = np.var(reward_sums/np.var(reward_sums)) #[10] -> 1
    var_reward_sum = np.var(reward_sums)  #/np.var(reward_sums)) #[10] -> 1

    var_grads = np.var(grads)

    avg_value_pred_loss = np.mean(value_pred_error)

    # avg_next_state_pred_error = np.mean(next_frame_errors)



    with open(ls_file,'a') as f:
        writer = csv.writer(f)
        # writer.writerow([total_num_steps, avg_ent, var_reward_sum, avg_next_state_pred_error])
        writer.writerow([total_num_steps, avg_ent, var_reward_sum, var_grads, avg_value_pred_loss])

  
















def plot_multiple_iterations2(dir_all, ax, color, m_i):

    # dir_all contains dirs of each run of the same algo with different seeds

    # print (dir_all)

    # print (os.listdir(dir_all))
    txs, tys=[],[]
    #for seed in agent in env
    for dir_i in os.listdir(dir_all):

        # print (dir_i)

        if os.path.isdir(dir_all+dir_i):
                
            # monitor_dir = os.path.join(dir_all+dir_i, 'monitor_rewards')
            # monitor_dir = os.path.join(dir_all+dir_i, 'monitor_other_info')
            ls_file = os.path.join(dir_all+dir_i, 'monitor_other_info.csv')
            # print (monitor_dir)

            # if os.path.isdir(monitor_dir):
            # print ('YESS')
            # print (dir_all+dir_i)

            # tx, ty = load_data(monitor_dir, smooth=1, bin_size=100)

            # load data
            timesteps = []
            ents = []
            var_reward_sums = []
            grad_var = []
            pred_error = []
            # next_state_errors = []
            with open(ls_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    # print (row)
                    timesteps.append(float(row[0]))
                    ents.append(float(row[1]))
                    var_reward_sums.append(float(row[2]))
                    # next_state_errors.append(float(row[3]))
                    grad_var.append(float(row[3]))
                    pred_error.append(float(row[4]))

            # print (ents)
            # print (grad_var)
            # selected_info = ents
            # selected_info = grad_var
            selected_info = pred_error

            # my smoothing
            selected_info_smooth = []
            selected_info_smooth.append(selected_info[0])
            for i in range(1,len(selected_info)-1):
                selected_info_smooth.append(np.mean(selected_info[i-1:i+2]))
            selected_info_smooth.append(selected_info[-1])

            txs.append(timesteps)
            tys.append(selected_info_smooth)
            # txs.append(tx)
            # tys.append(ty)
            # break
            # else:
            #     print ('NO')
            # fdsafa
    if len(txs[0]) <3:
        return

    if txs[0] != None:

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

        if len(x) != len(y_mean):
            print (len(x))
            print (len(y_mean))

        # if m_i =='a2c':
        #     print (y_mean.shape)
        #     fad

        # color = color_defaults[0] 

        # fig = plt.figure()

        # print (x)
        # print (y_mean)

        y_upper = y_mean + y_std
        y_lower = y_mean - y_std

        # print (x)
        # print (y_mean)
        # print (y_std)
        # print (y_upper)
        # print (y_lower)
        plt.fill_between(x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3)
        plt.plot(x, list(y_mean), label=m_i, color=color, rasterized=True)  
        plt.legend(loc=4)

        # plt.plot(tx, ty, label="{}".format('a2c'))
        # plt.xticks([1e6, 2e6, 4e6, 6e6, 8e6, 10e6],
        #            ["1M", "2M", "4M", "6M", "8M", "10M"])
        # plt.xlim(0, 10e6)


        # plt.savefig(dir_all+'plot.png')
        # print('made fig',dir_all+'plot.png')








def update_grad_plot(model_dict):

    #need to check for other agents, plot them all together. .
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
                # plt.ylabel('Rewards',family='serif')
                # plt.ylabel('Action Entropy',family='serif')
                # plt.ylabel('Gradient Variance',family='serif')
                plt.ylabel('Value Prediction Error',family='serif')

            plt.title(env_i.split('NoF')[0],family='serif')
            ax.yaxis.grid(alpha=.1)

            m_count =0
            #for each agent in env
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



    # fig_path = exp_path + model_dict['exp_name'] #+ 'exp_plot' 
    fig_path = exp_path + 'info_plot' #+ 'exp_plot' 
    plt.savefig(fig_path+'.png')
    # print('made fig', fig_path+'.png')

    plt.savefig(fig_path+'.pdf')
    # print('made fig', fig_path+'.pdf')




    plt.close(fig)










    # num_frames = model_dict['num_frames']
    # save_dir = model_dict['save_to']
    # env = model_dict['env']

    # # ls_path = save_dir+'/learning_signal/'
    # ls_file = save_dir + '/monitor_other_info.csv'





    # # load data
    # timesteps = []
    # ents = []
    # var_reward_sums = []
    # # next_state_errors = []
    # with open(ls_file, 'r') as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:
    #         # print (row)
    #         timesteps.append(float(row[0]))
    #         ents.append(float(row[1]))
    #         var_reward_sums.append(float(row[2]))
    #         # next_state_errors.append(float(row[3]))


    # if len(timesteps) < 10:
    #     return





    # # plot data
    # rows =1
    # cols=1
    # fig = plt.figure(figsize=(8+cols,3+rows), facecolor='white')

    # # cur_col = 0
    # # cur_row = 0
    # # for env_i in os.listdir(exp_path):
    # #     if os.path.isdir(exp_path+env_i):

    #         # print (cur_row, cur_col)
    # ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)#, aspect=0.3)# adjustable='box', )

    # plt.xlim(0, num_frames)
    # # if cur_row == rows-1:
    # if num_frames == 6e6:
    #     plt.xticks([1e6, 2e6, 4e6, 6e6],["1M", "2M", "4M", "6M"])
    # elif num_frames == 10e6:
    #     plt.xticks([2e6, 4e6, 6e6, 8e6, 10e6],["2M", "4M", "6M", "8M", "10M"])
    # plt.xlabel('Number of Timesteps',family='serif')
    # # else:
    # #     plt.xticks([])
    # # if cur_col == 0:
    # plt.ylabel('Action Entropy',family='serif')
    # # plt.ylabel('V[R]',family='serif')
    # # plt.ylabel('State Prediction Error',family='serif')


    # plt.title(env.split('NoF')[0],family='serif')
    # # plt.title(env,family='serif')

    # ax.yaxis.grid(alpha=.1)
    # # ax.set(aspect=1)
    # # plt.gca().set_aspect('equal', adjustable='box')
    # # ax.set_aspect(.5, adjustable='box')

    # m_count =0
    # # for m_i in os.listdir(exp_path+env_i):
    # #     m_dir = exp_path+env_i+'/'+m_i+'/'
    # #     if os.path.isdir(m_dir):
            
    # # print (cur_row, cur_col, m_dir)
    # color = color_defaults[m_count] 
    # # plot_multiple_iterations2(m_dir, ax, color, m_i)
    # # m_count+=1

    # # print (timesteps)
    # # print (var_reward_sums)
    # # if len(timesteps) > 30:
    # #     timesteps, var_reward_sums = smooth_reward_curve(timesteps, var_reward_sums)

    # # plt.plot(timesteps, ents)

    # # var_reward_sums = next_state_errors

    # selected_info = ents

    # # my smoothing
    # selected_info_smooth = []
    # selected_info_smooth.append(selected_info[0])
    # for i in range(1,len(selected_info)-1):
    #     selected_info_smooth.append(np.mean(selected_info[i-1:i+2]))
    # selected_info_smooth.append(selected_info[-1])


    # # plt.plot(timesteps, var_reward_sums)
    # # plt.plot(timesteps, var_reward_sums_smooth)
    # plt.plot(timesteps[6:], selected_info_smooth[6:])
    # #why skip the first 5????
    #     #because its really high early on?? 



    # # cur_col+=1
    # # if cur_col >= cols:
    # #     cur_col = 0
    # #     cur_row+=1


    # # fig_path = exp_path + model_dict['exp_name'] #+ 'exp_plot' 

    # # fig_path = ls_path + 'learning_signal'


    # # exp_path = model_dict['exp_path']

    # fig_path = exp_path + '/info_plot'


    

    # plt.savefig(fig_path+'.png')
    # # print('made fig', fig_path+'.png')
    # plt.savefig(fig_path+'.pdf')
    # # print('made fig', fig_path+'.pdf')
    # plt.close(fig)




























