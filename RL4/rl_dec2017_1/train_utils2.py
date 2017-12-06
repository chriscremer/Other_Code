







# show state, actions, value, state prob, reconstruction

def do_gifs3(envs, agent, vae, model_dict, update_current_state, update_rewards, total_num_steps):
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

        recons = []

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
            elbo, logpx, logpz, logqz, recon = vae.forward3(state_get_prob, k=100)
            probs.append(elbo.data.cpu().numpy())

            recons.append(recon.data.cpu().numpy())


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


            rows = 2
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



            # plot recon
            ax = plt.subplot2grid((rows,cols), (1,1), frameon=False, colspan=3)
            # state1 = np.squeeze(state[0])
            state1 = recons[step]
            ax.imshow(state1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_title('State '+str(step),family='serif')



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

