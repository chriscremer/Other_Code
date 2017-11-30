



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




def do_params(save_dir, agent, total_num_steps, model_dict):
    cuda = model_dict['cuda']

    save_path = os.path.join(save_dir, 'model_params')
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    # A really ugly way to save a model to CPU
    save_model = agent.actor_critic
    if cuda:
        save_model = copy.deepcopy(agent.actor_critic).cpu()
    # torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
    # steps_sci_nota = '{e}'.format(total_num_steps)
    save_to=os.path.join(save_path, "model_params" + str(total_num_steps)+".pt")
    # save_to=os.path.join(save_path, "model_params" + steps_sci_nota+".pt")
    torch.save(save_model, save_to)
    print ('saved', save_to)





def do_vid(envs_video, update_current_state, shape_dim0, dtype, agent, model_dict, total_num_steps):
    env_name = model_dict['env']
    save_dir = model_dict['save_to']
    algo = model_dict['algo']
    obs_shape = model_dict['obs_shape']



    n_vids=3
    for i in range(n_vids):
        done=False
        state = envs_video.reset()
        # state = torch.from_numpy(state).float().type(dtype)
        current_state = torch.zeros(1, *obs_shape)
        current_state = update_current_state(current_state, state, shape_dim0).type(dtype)
        # print ('Recording')
        # count=0
        while not done:
            # print (count)
            # count +=1
            # Act
            state_var = Variable(current_state, volatile=True) 
            # print (state_var.size())
            value, action, action_log_probs, dist_entropy = agent.act(state_var)
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Observe reward and next state
            state, reward, done, info = envs_video.step(cpu_actions) # state:[nProcesss, ndims, height, width]
            # state = torch.from_numpy(state).float().type(dtype)
            # current_state = torch.zeros(1, *obs_shape)
            current_state = update_current_state(current_state, state, shape_dim0).type(dtype)
    state = envs_video.reset()
    
    vid_path = save_dir+'/videos/'
    count =0
    for aaa in os.listdir(vid_path):

        if 'openaigym' in aaa and '.mp4' in aaa:
            #os.rename(vid_path+aaa, vid_path+'vid_t'+str(total_num_steps)+'.mp4')
            subprocess.call("(cd "+vid_path+" && mv "+ vid_path+aaa +" "+ vid_path+env_name+'_'+algo+'_vid_t'+str(total_num_steps)+'_'+str(count) +".mp4)", shell=True) 
            count+=1
        if '.json' in aaa:
            os.remove(vid_path+aaa)








#with list rollout
# and single process env


def do_gifs(envs, agent, model_dict, update_current_state, update_rewards, total_num_steps):
    save_dir = model_dict['save_to']
    shape_dim0 = model_dict['shape_dim0']
    num_processes = model_dict['num_processes']
    obs_shape = model_dict['obs_shape']
    dtype = model_dict['dtype']
    num_steps = model_dict['num_steps']
    gamma = model_dict['gamma']


    num_processes = 1
    

    gif_path = save_dir+'/gifs/'
    makedir(gif_path, print_=False)

    gif_epoch_path = save_dir+'/gifs/gif'+str(total_num_steps)+'/'
    makedir(gif_epoch_path, print_=False, rm=True)

    n_gifs = 1


    episode_rewards = torch.zeros([num_processes, 1]) #keeps track of current episode cumulative reward
    final_rewards = torch.zeros([num_processes, 1])




    # get data
    for j in range(n_gifs):

        state_frames = []
        value_frames = []
        actions_frames = []

        # Init state
        state = envs.reset()  # (channels, height, width)

        state = np.expand_dims(state,0) # (1, channels, height, width)

        current_state = torch.zeros(num_processes, *obs_shape)  # (processes, channels*stack, height, width)
        current_state = update_current_state(current_state, state, shape_dim0).type(dtype) #add the new frame, remove oldest
        # agent.insert_first_state(current_state) #storage has states: (num_steps + 1, num_processes, *obs_shape), set first step 


        agent.rollouts_list.reset()
        agent.rollouts_list.states = [current_state]


        step = 0
        done_ = False
        while not done_ and step < 400:

            state1 = np.squeeze(state[0])
            state_frames.append(state1)

            value, action, action_log_probs, dist_entropy = agent.act(Variable(agent.rollouts_list.states[-1], volatile=True))
            value_frames.append([value.data.cpu().numpy()[0][0]])

            action_prob = agent.actor_critic.action_dist(Variable(agent.rollouts_list.states[-1], volatile=True))[0]
            action_prob = np.squeeze(action_prob.data.cpu().numpy())  # [A]
            actions_frames.append(action_prob)

            # value, action = agent.act(Variable(agent.rollouts_list.states[-1], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy() #[P]
            # Step, S:[P,C,H,W], R:[P], D:[P]
            state, reward, done, info = envs.step(cpu_actions) 

            state = np.expand_dims(state,0) # (1, channels, height, width)
            reward = np.expand_dims(reward,0) # (1, 1)
            done = np.expand_dims(done,0) # (1, 1)

            # Record rewards
            reward, masks, final_rewards, episode_rewards, current_state = update_rewards(reward, done, final_rewards, episode_rewards, current_state)
            # Update state
            current_state = update_current_state(current_state, state, shape_dim0)
            # Agent record step
            # agent.insert_data(step, current_state, action.data, value.data, reward, masks)
            agent.rollouts_list.insert(step, current_state, action.data, value.data, reward.numpy()[0][0], masks)


            done_ = done[0]
            # print (done)

            step+=1
            # print ('step', step)






        next_value = agent.actor_critic(Variable(agent.rollouts_list.states[-1], volatile=True))[0].data
        agent.rollouts_list.compute_returns(next_value, gamma)
        # print (agent.rollouts_list.returns)#.cpu().numpy())

        # print ('steps', step)
        # print ('reward_length', len(agent.rollouts_list.rewards))
        # print ('return length', len(agent.rollouts_list.returns))
        # print ('state_frames', len(state_frames))


        # if sum(agent.rollouts_list.rewards) == 0.:
        #     continue


        #make figs
        # for j in range(n_gifs):

        frames_path = gif_epoch_path+'frames'+str(j)+'/'
        makedir(frames_path, print_=False)

        # for step in range(num_steps):
        for step in range(len(state_frames)-1):


            rows = 1
            cols = 3

            fig = plt.figure(figsize=(8,4), facecolor='white')



            # plot frame
            ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

            # state1 = np.squeeze(state[0])
            state1 = state_frames[step]
            ax.imshow(state1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('State '+str(step),family='serif')





            #plot values histogram
            ax = plt.subplot2grid((rows,cols), (0,2), frameon=False)

            values = value_frames[step]#[0]#.cpu().numpy()
            weights = np.ones_like(values)/float(len(values))
            ax.hist(values, 50, range=[0., 8.], weights=weights)
            
            ax.set_ylim([0.,1.])
            ax.set_title('Value',family='serif')

            if step %10 == 0:
                print (step, len(agent.rollouts_list.returns)-1)
            # print ()
            val_return = agent.rollouts_list.returns[step] #.cpu().numpy()#[0][0]
            # print(val_return)
            ax.plot([val_return,val_return],[0,1])
            

            #plot actions
            ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)

            # action_prob = agent.actor_critic.action_dist(Variable(agent.rollouts_list.states[-1], volatile=True))[0]
            # action_prob = np.squeeze(action_prob.data.cpu().numpy())
            action_prob = actions_frames[step]
            action_size = envs.action_space.n
            # print (action_size)
            
            ax.bar(range(action_size), action_prob)
            ax.set_title('Action',family='serif')
            plt.xticks(range(action_size),['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'R_FIRE', 'L_FIRE'], fontsize=6)
            ax.set_ylim([0.,1.])


            #plot fig
            plt.tight_layout(pad=3., w_pad=2.5, h_pad=1.0)
            plt_path = frames_path+'plt' +str(step)+'.png'
            plt.savefig(plt_path)
            # print ('saved',plt_path)
            plt.close(fig)




        # Make gif 

        
        # dir_ = home+ '/Documents/tmp/a2c_reg_and_dropout_pong2/PongNoFrameskip-v4/a2c_dropout/seed0/frames_a2c_dropout_PongNoFrameskip-v4_6000000'
        # print('making gif')
        max_epoch = 0
        for file_ in os.listdir(frames_path):
            if 'plt' in file_:
                numb = file_.split('plt')[1].split('.')[0]
                numb = int(numb)
                if numb > max_epoch:
                    max_epoch = numb
        # print ('max epoch in dir', max_epoch)

        images = []
        for i in range(max_epoch+1):
            images.append(imageio.imread(frames_path+'plt'+str(i)+'.png'))
            
        gif_path_this = gif_epoch_path + str(j) + '.gif'
        imageio.mimsave(gif_path_this, images)
        print ('made gif', gif_path_this)




















def do_ls(envs, agent, model_dict, total_num_steps, update_current_state, update_rewards):

    save_dir = model_dict['save_to']
    shape_dim0 = model_dict['shape_dim0']
    num_processes = model_dict['num_processes']
    obs_shape = model_dict['obs_shape']
    dtype = model_dict['dtype']
    num_steps = model_dict['num_steps']
    gamma = model_dict['gamma']

    # print ('ls')

    ls_path = save_dir+'/learning_signal/'
    ls_file = ls_path + 'ls_monitor.csv'

    # if os.path.exists(ls_path):
    #     try:
    #         os.remove(ls_file)
    #     except OSError:
    #         pass

    makedir(ls_path, print_=False)



    avg_over = 10
    num_processes = 1
    

    episode_rewards = torch.zeros([num_processes, 1]) #keeps track of current episode cumulative reward
    final_rewards = torch.zeros([num_processes, 1])


    dist_entropies = []

    reward_sums = []

    next_frame_errors = []

    # get data
    for j in range(avg_over):

        state_frames = []
        value_frames = []
        actions_frames = []

        # Init state
        state = envs.reset()  # (channels, height, width)

        state = np.expand_dims(state,0) # (1, channels, height, width)

        current_state = torch.zeros(num_processes, *obs_shape)  # (processes, channels*stack, height, width)
        current_state = update_current_state(current_state, state, shape_dim0).type(dtype) #add the new frame, remove oldest
        # agent.insert_first_state(current_state) #storage has states: (num_steps + 1, num_processes, *obs_shape), set first step 


        agent.rollouts_list.reset()
        agent.rollouts_list.states = [current_state]


        step = 0
        done_ = False
        while not done_ and step < 400:

            state1 = np.squeeze(state[0])
            state_frames.append(state1)

            value, action, action_log_probs, dist_entropy = agent.act(Variable(agent.rollouts_list.states[-1], volatile=True))
            value_frames.append([value.data.cpu().numpy()[0][0]])

            action_prob = agent.actor_critic.action_dist(Variable(agent.rollouts_list.states[-1], volatile=True))[0]
            action_prob = np.squeeze(action_prob.data.cpu().numpy())  # [A]
            actions_frames.append(action_prob)

            # value, action = agent.act(Variable(agent.rollouts_list.states[-1], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy() #[P]
            # Step, S:[P,C,H,W], R:[P], D:[P]
            state, reward, done, info = envs.step(cpu_actions) 

            state = np.expand_dims(state,0) # (1, channels, height, width)
            reward = np.expand_dims(reward,0) # (1, 1)
            done = np.expand_dims(done,0) # (1, 1)

            # Record rewards
            reward, masks, final_rewards, episode_rewards, current_state = update_rewards(reward, done, final_rewards, episode_rewards, current_state)
            # Update state
            current_state = update_current_state(current_state, state, shape_dim0)
            # Agent record step
            # agent.insert_data(step, current_state, action.data, value.data, reward, masks)
            agent.rollouts_list.insert(step, current_state, action.data, value.data, reward.numpy()[0][0], masks)


            done_ = done[0]
            # print (done)

            step+=1
            # print ('step', step)

            dist_entropies.append(dist_entropy.data.cpu().numpy())

            next_frame_errors.append(agent.state_pred_error.data.cpu().numpy()[0])
            



        next_value = agent.actor_critic(Variable(agent.rollouts_list.states[-1], volatile=True))[0].data
        agent.rollouts_list.compute_returns(next_value, gamma)
        # print (agent.rollouts_list.returns)#.cpu().numpy())

        # reward_sums.append(np.sum(agent.rollouts_list.rewards))
        reward_sums.append(np.mean(agent.rollouts_list.rewards))


    # print (dist_entropy)
    # dist_entropies = 
    avg_ent = np.mean(dist_entropies)

    # if np.var(reward_sums) > 0:
    #     var_reward_sum = np.var(reward_sums/np.var(reward_sums)) #[10] -> 1
    var_reward_sum = np.var(reward_sums)  #/np.var(reward_sums)) #[10] -> 1

    avg_next_state_pred_error = np.mean(next_frame_errors)



    with open(ls_file,'a') as f:
        writer = csv.writer(f)
        writer.writerow([total_num_steps, avg_ent, var_reward_sum, avg_next_state_pred_error])

    return





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



def update_ls_plot(model_dict):


    num_frames = model_dict['num_frames']
    save_dir = model_dict['save_to']
    env = model_dict['env']

    ls_path = save_dir+'/learning_signal/'
    ls_file = ls_path + 'ls_monitor.csv'


    # load data
    timesteps = []
    ents = []
    var_reward_sums = []
    next_state_errors = []
    with open(ls_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # print (row)
            timesteps.append(float(row[0]))
            ents.append(float(row[1]))
            var_reward_sums.append(float(row[2]))
            next_state_errors.append(float(row[3]))


    if len(timesteps) < 10:
        return

    # plot data

    rows =1
    cols=1
    fig = plt.figure(figsize=(8+cols,3+rows), facecolor='white')

    # cur_col = 0
    # cur_row = 0
    # for env_i in os.listdir(exp_path):
    #     if os.path.isdir(exp_path+env_i):

            # print (cur_row, cur_col)
    ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)#, aspect=0.3)# adjustable='box', )

    plt.xlim(0, num_frames)
    # if cur_row == rows-1:
    if num_frames == 6e6:
        plt.xticks([1e6, 2e6, 4e6, 6e6],["1M", "2M", "4M", "6M"])
    elif num_frames == 10e6:
        plt.xticks([2e6, 4e6, 6e6, 8e6, 10e6],["2M", "4M", "6M", "8M", "10M"])
    plt.xlabel('Number of Timesteps',family='serif')
    # else:
    #     plt.xticks([])
    # if cur_col == 0:
    # plt.ylabel('Entropy',family='serif')
    # plt.ylabel('V[R]',family='serif')

    plt.ylabel('State Prediction Error',family='serif')


    plt.title(env.split('NoF')[0],family='serif')
    # plt.title(env,family='serif')

    ax.yaxis.grid(alpha=.1)
    # ax.set(aspect=1)
    # plt.gca().set_aspect('equal', adjustable='box')
    # ax.set_aspect(.5, adjustable='box')

    m_count =0
    # for m_i in os.listdir(exp_path+env_i):
    #     m_dir = exp_path+env_i+'/'+m_i+'/'
    #     if os.path.isdir(m_dir):
            
    # print (cur_row, cur_col, m_dir)
    color = color_defaults[m_count] 
    # plot_multiple_iterations2(m_dir, ax, color, m_i)
    # m_count+=1

    # print (timesteps)
    # print (var_reward_sums)
    # if len(timesteps) > 30:
    #     timesteps, var_reward_sums = smooth_reward_curve(timesteps, var_reward_sums)

    # plt.plot(timesteps, ents)

    var_reward_sums = next_state_errors

    # my smoothing
    var_reward_sums_smooth = []
    var_reward_sums_smooth.append(var_reward_sums[0])
    for i in range(1,len(var_reward_sums)-1):
        var_reward_sums_smooth.append(np.mean(var_reward_sums[i-1:i+2]))
    var_reward_sums_smooth.append(var_reward_sums[-1])


    # plt.plot(timesteps, var_reward_sums)
    # plt.plot(timesteps, var_reward_sums_smooth)
    plt.plot(timesteps[6:], var_reward_sums_smooth[6:])



    # cur_col+=1
    # if cur_col >= cols:
    #     cur_col = 0
    #     cur_row+=1


    # fig_path = exp_path + model_dict['exp_name'] #+ 'exp_plot' 

    fig_path = ls_path + 'learning_signal'

    plt.savefig(fig_path+'.png')
    # print('made fig', fig_path+'.png')
    plt.savefig(fig_path+'.pdf')
    # print('made fig', fig_path+'.pdf')
    plt.close(fig)



















def view_reward_episode(model_dict, frames):

    state_frames = frames

    save_dir = model_dict['save_to']

    rewardstates_path = save_dir+'/reward_episode/'
    makedir(rewardstates_path, print_=False)


    #make gif

    max_epoch = 0
    for file_ in os.listdir(rewardstates_path):
        if 'plt' in file_:
            numb = file_.split('plt')[1].split('.')[0]
            numb = int(numb)
            if numb > max_epoch:
                max_epoch = numb
    # print ('max epoch in dir', max_epoch)

    frames_path = rewardstates_path

    images = []
    for i in range(max_epoch+1):
        images.append(imageio.imread(frames_path+'plt'+str(i)+'.png'))
        
    gif_path_this = rewardstates_path + 'GIF.gif'
    imageio.mimsave(gif_path_this, images)
    print ('made gif', gif_path_this)











def do_prob_state(envs, agent, model_dict, vae, update_current_state, total_num_steps):

    #run for x epochs, compute probs of each state, make figs, make gif

    save_dir = model_dict['save_to']
    shape_dim0 = model_dict['shape_dim0']
    # num_processes = model_dict['num_processes']
    obs_shape = model_dict['obs_shape']
    dtype = model_dict['dtype']
    # num_steps = model_dict['num_steps']
    # gamma = model_dict['gamma']

    # print ('ls')

    save_path = save_dir+'/state_probs/timestep'+str(total_num_steps)+'/'
    # save_file = ls_path + 'ls_monitor.csv'

    # if os.path.exists(ls_path):
    #     try:
    #         os.remove(ls_file)
    #     except OSError:
    #         pass

    makedir(save_path, print_=False)




    # avg_over = 10
    num_processes = 1
    

    # episode_rewards = torch.zeros([num_processes, 1]) #keeps track of current episode cumulative reward
    # final_rewards = torch.zeros([num_processes, 1])


    # dist_entropies = []

    # reward_sums = []

    # next_frame_errors = []

    # get data
    # for j in range(avg_over):

    #     state_frames = []
    #     value_frames = []
    #     actions_frames = []

    # Init state
    state = envs.reset()  # (channels, height, width)

    state = np.expand_dims(state,0) # (1, channels, height, width)

    current_state = torch.zeros(num_processes, *obs_shape)  # (processes, channels*stack, height, width)
    current_state = update_current_state(current_state, state, shape_dim0).type(dtype) #add the new frame, remove oldest
    # agent.insert_first_state(current_state) #storage has states: (num_steps + 1, num_processes, *obs_shape), set first step 


    # agent.rollouts_list.reset()
    # agent.rollouts_list.states = [current_state]

    frames = []
    probs = []

    done_ = 0
    step = 0
    while not done_ and step < 400:

        # state1 = np.squeeze(state[0])
        # state_frames.append(state1)

        value, action, action_log_probs, dist_entropy = agent.act(Variable(current_state, volatile=True))
        # value_frames.append([value.data.cpu().numpy()[0][0]])

        # action_prob = agent.actor_critic.action_dist(Variable(agent.rollouts_list.states[-1], volatile=True))[0]
        # action_prob = np.squeeze(action_prob.data.cpu().numpy())  # [A]
        # actions_frames.append(action_prob)

        # value, action = agent.act(Variable(agent.rollouts_list.states[-1], volatile=True))
        cpu_actions = action.data.squeeze(1).cpu().numpy() #[P]
        # Step, S:[P,C,H,W], R:[P], D:[P]
        state, reward, done, info = envs.step(cpu_actions) 

        state = np.expand_dims(state,0) # (1, channels, height, width)
        reward = np.expand_dims(reward,0) # (1, 1)
        done = np.expand_dims(done,0) # (1, 1)

        # Record rewards
        # reward, masks, final_rewards, episode_rewards, current_state = update_rewards(reward, done, final_rewards, episode_rewards, current_state)
        # Update state
        current_state = update_current_state(current_state, state, shape_dim0)
        # Agent record step
        # agent.insert_data(step, current_state, action.data, value.data, reward, masks)
        # agent.rollouts_list.insert(step, current_state, action.data, value.data, reward.numpy()[0][0], masks)


        done_ = done[0]
        # print (done)
        step+=1
        # print ('step', step)

        # dist_entropies.append(dist_entropy.data.cpu().numpy())

        # next_frame_errors.append(agent.state_pred_error.data.cpu().numpy()[0])
        
        state_get_prob = Variable(torch.from_numpy(state).float().view(1,84,84)).cuda()

        state_get_prob = state_get_prob  / 255.0
        # print (state_get_prob.size())

        # batch = batch[1:]  # [Steps,Processes,Stack,84,84]
        # batch = current_state[:,0] # [Processes,84,84]
        # batch = batch.contiguous().view(-1,84,84)   # [Steps*Processes,84,84]


        elbo, logpx, logpz, logqz = vae.forward(state_get_prob, k=100)


        frames.append(state)
        probs.append(elbo.data.cpu().numpy())

        # print (state.shape)



    for i in range(len(frames)):

        state = frames[i]
        state1 = np.squeeze(state)
        # fdsa

        rows = 1
        cols = 2

        fig = plt.figure(figsize=(8,4), facecolor='white')

        # plot frame
        ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

        # state1 = np.squeeze(state[0])
        # state1 = state_frames[step]
        ax.imshow(state1, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('State '+str(i),family='serif')


        #plot log prob
        
        min_logprob = np.min(probs)

        probs = np.array(probs) - min_logprob 
        max_logprob = np.max(probs)
        probs = probs  / max_logprob

        ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)

        # action_prob = agent.actor_critic.action_dist(Variable(agent.rollouts_list.states[-1], volatile=True))[0]
        # action_prob = np.squeeze(action_prob.data.cpu().numpy())
        # action_prob = actions_frames[step]
        # action_size = envs.action_space.n
        # # print (action_size)
        
        ax.bar(1, probs[i])
        # ax.set_title('Action',family='serif')
        # plt.xticks(range(action_size),['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'R_FIRE', 'L_FIRE'], fontsize=6)

        # ax.set_ylim([min_logprob,max_logprob])
        ax.set_ylim([0.,1.])






        #plot fig
        # plt.tight_layout(pad=3., w_pad=2.5, h_pad=1.0)
        plt_path = save_path+'plt' +str(i)+'.png'
        plt.savefig(plt_path)
        # print ('saved',plt_path)
        plt.close(fig)




    #make gif
    images = []
    for i in range(len(frames)):
        images.append(imageio.imread(save_path+'plt'+str(i)+'.png'))
        
    gif_path_this = save_path + 'GIF.gif'
    imageio.mimsave(gif_path_this, images)
    print ('made gif', gif_path_this)






    # print ('donethis')
    # fsdfa

































































# show state, actions, value, state prob

def do_gifs2(envs, agent, vae, model_dict, update_current_state, update_rewards, total_num_steps):
    save_dir = model_dict['save_to']
    shape_dim0 = model_dict['shape_dim0']
    num_processes = model_dict['num_processes']
    obs_shape = model_dict['obs_shape']
    dtype = model_dict['dtype']
    num_steps = model_dict['num_steps']
    gamma = model_dict['gamma']

    action_names = envs.unwrapped.get_action_meanings()

    vow = ["A", "E", "I", "O", "U"]
    # ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
    # print (action_names)
    for aa in range(len(action_names)):
        for v in vow:
            action_names[aa] = action_names[aa].replace(v, "")
    # print (action_names)
    # fads

    num_processes = 1
    

    gif_path = save_dir+'/gifs/'
    makedir(gif_path, print_=False)

    gif_epoch_path = save_dir+'/gifs/gif'+str(total_num_steps)+'/'
    makedir(gif_epoch_path, print_=False, rm=True)

    n_gifs = 1


    episode_rewards = torch.zeros([num_processes, 1]) #keeps track of current episode cumulative reward
    final_rewards = torch.zeros([num_processes, 1])




    # get data
    for j in range(n_gifs):

        state_frames = []
        value_frames = []
        actions_frames = []
        probs = []

        # Init state
        state = envs.reset()  # (channels, height, width)

        state = np.expand_dims(state,0) # (1, channels, height, width)

        current_state = torch.zeros(num_processes, *obs_shape)  # (processes, channels*stack, height, width)
        current_state = update_current_state(current_state, state, shape_dim0).type(dtype) #add the new frame, remove oldest
        # agent.insert_first_state(current_state) #storage has states: (num_steps + 1, num_processes, *obs_shape), set first step 


        agent.rollouts_list.reset()
        agent.rollouts_list.states = [current_state]


        step = 0
        done_ = False
        while not done_ and step < 400:

            state1 = np.squeeze(state[0])
            state_frames.append(state1)

            value, action, action_log_probs, dist_entropy = agent.act(Variable(agent.rollouts_list.states[-1], volatile=True))
            value_frames.append([value.data.cpu().numpy()[0][0]])

            action_prob = agent.actor_critic.action_dist(Variable(agent.rollouts_list.states[-1], volatile=True))[0]
            action_prob = np.squeeze(action_prob.data.cpu().numpy())  # [A]
            actions_frames.append(action_prob)

            # value, action = agent.act(Variable(agent.rollouts_list.states[-1], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy() #[P]
            # Step, S:[P,C,H,W], R:[P], D:[P]
            state, reward, done, info = envs.step(cpu_actions) 

            state = np.expand_dims(state,0) # (1, channels, height, width)
            reward = np.expand_dims(reward,0) # (1, 1)
            done = np.expand_dims(done,0) # (1, 1)

            # Record rewards
            reward, masks, final_rewards, episode_rewards, current_state = update_rewards(reward, done, final_rewards, episode_rewards, current_state)
            # Update state
            current_state = update_current_state(current_state, state, shape_dim0)
            # Agent record step
            # agent.insert_data(step, current_state, action.data, value.data, reward, masks)
            agent.rollouts_list.insert(step, current_state, action.data, value.data, reward.numpy()[0][0], masks)


            done_ = done[0]
            # print (done)

            step+=1
            # print ('step', step)




            state_get_prob = Variable(torch.from_numpy(state).float().view(1,84,84)).cuda()
            state_get_prob = state_get_prob  / 255.0
            elbo, logpx, logpz, logqz = vae.forward(state_get_prob, k=100)
            probs.append(elbo.data.cpu().numpy())




        next_value = agent.actor_critic(Variable(agent.rollouts_list.states[-1], volatile=True))[0].data
        agent.rollouts_list.compute_returns(next_value, gamma)
        # print (agent.rollouts_list.returns)#.cpu().numpy())

        # print ('steps', step)
        # print ('reward_length', len(agent.rollouts_list.rewards))
        # print ('return length', len(agent.rollouts_list.returns))
        # print ('state_frames', len(state_frames))


        # if sum(agent.rollouts_list.rewards) == 0.:
        #     continue


        #make figs
        # for j in range(n_gifs):

        frames_path = gif_epoch_path+'frames'+str(j)+'/'
        makedir(frames_path, print_=False)

        # for step in range(num_steps):
        for step in range(len(state_frames)-1):

            if step %10 == 0:
                print (step, len(state_frames)-1)

            # if step > 30:
            #     break


            rows = 1
            cols = 9

            fig = plt.figure(figsize=(12,4), facecolor='white')



            #Plot probs
            ax = plt.subplot2grid((rows,cols), (0,0), frameon=False, colspan=1)
            min_logprob = np.min(probs)
            probs = np.array(probs) - min_logprob 
            max_logprob = np.max(probs)
            probs = probs  / max_logprob
            ax.bar(1, probs[step])
            ax.set_ylim([0.,1.])
            ax.set_title('State Prob',family='serif')
            ax.set_yticks([])
            ax.set_xticks([])



            # plot frame
            ax = plt.subplot2grid((rows,cols), (0,1), frameon=False, colspan=3)
            # state1 = np.squeeze(state[0])
            state1 = state_frames[step]
            ax.imshow(state1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('State '+str(step),family='serif')


            #plot actions
            ax = plt.subplot2grid((rows,cols), (0,4), frameon=False, colspan=3)
            action_prob = actions_frames[step]
            action_size = envs.action_space.n
            # print (action_size)
            ax.bar(range(action_size), action_prob)
            ax.set_title('Action',family='serif')
            # plt.xticks(range(action_size),['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'R_FIRE', 'L_FIRE'], fontsize=6)
            plt.xticks(range(action_size),action_names, fontsize=5)
            ax.set_ylim([0.,1.])




            #plot values histogram
            ax = plt.subplot2grid((rows,cols), (0,7), frameon=False, colspan=2)
            values = value_frames[step]#[0]#.cpu().numpy()
            weights = np.ones_like(values)/float(len(values))
            ax.hist(values, 50, range=[-2., 2.], weights=weights)
            ax.set_ylim([0.,1.])
            ax.set_title('Value',family='serif')
            val_return = agent.rollouts_list.returns[step] #.cpu().numpy()#[0][0]
            # print(val_return)
            ax.plot([val_return,val_return],[0,1])
            ax.set_yticks([])
            








            #plot fig
            plt.tight_layout(pad=1.5, w_pad=.4, h_pad=1.)
            plt_path = frames_path+'plt' +str(step)+'.png'
            plt.savefig(plt_path)
            # print ('saved',plt_path)
            plt.close(fig)

            





        # Make gif 

        
        # dir_ = home+ '/Documents/tmp/a2c_reg_and_dropout_pong2/PongNoFrameskip-v4/a2c_dropout/seed0/frames_a2c_dropout_PongNoFrameskip-v4_6000000'
        # print('making gif')
        max_epoch = 0
        for file_ in os.listdir(frames_path):
            if 'plt' in file_:
                numb = file_.split('plt')[1].split('.')[0]
                numb = int(numb)
                if numb > max_epoch:
                    max_epoch = numb
        # print ('max epoch in dir', max_epoch)

        images = []
        for i in range(max_epoch+1):
            images.append(imageio.imread(frames_path+'plt'+str(i)+'.png'))
            
        gif_path_this = gif_epoch_path + str(j) + '.gif'
        imageio.mimsave(gif_path_this, images)
        print ('made gif', gif_path_this)



























