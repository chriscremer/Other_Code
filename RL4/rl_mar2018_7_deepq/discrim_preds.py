






import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler







def discrim_predictions(model_dict, states, actions, discriminator, reverse=False):

    #rollouts has states, actions, dones of n-step
    #discriminator takes two frames, predicts an action
    #this func will use discrim to make preds and compute the error of those preds 
        #then return the errors for each timestep
        #then those errors will be used to optimze the discrim and agent 


    #How to deal with last timestep if doing 2-timestep pred? 
    #Maybe ill just combine with 1-step
    #Ya make the step number be changeable 
    #note that this is different from the n-step for computing the return

    #will be averaging over n-step prediction errors

    n_steps = model_dict['num_steps'] #this is the episode length basically
    n_processes = model_dict['num_processes'] #this is the episode length basically

    max_pred_step = 3 #5
    #list for each n-step
    errors = [[] for i in range(max_pred_step)]



    # states = torch.cuda.FloatTensor(states) #.cuda()   #[S+1,P,stack,84,84]
    # actions = torch.cuda.LongTensor(actions) #.cuda()   #[S,P,1]

    # print (states.size())
    # print (actions.size())
    # faads

    if reverse:
        idx = [i for i in range(n_steps, -1, -1)]
        idx = torch.LongTensor(idx).cuda()
        states = states.index_select(0, idx)
        idx = [i for i in range(n_steps-1, -1, -1)]
        idx = torch.LongTensor(idx).cuda()
        actions = actions.index_select(0, idx)
    # else:
    #     states = rollouts.states
    #     actions = rollouts.actions

    # print (torch.sum(states[0]))
    # print (torch.sum(states[-1]))

    # print (states.size())

    for s in range(1,max_pred_step+1):

        for t in range(n_steps):

            # get states for this action and n-step
                #first state is state of action, final state is n-step away
            # print (rollouts.states[t].size())  #[P,stack,84,84]
            # print (rollouts.actions[t].size())  #[P,1]
            

            if t+s <= n_steps:

                # print (t, 't')

                #take last one in the stack, which is the newest frame
                first_frame = states[t][:,-1].contiguous().view(n_processes,1,84,84) #[P,1,84,84]
                # print (first_frame.size())
                final_frame = states[t+s][:,-1].contiguous().view(n_processes,1,84,84)

                action = Variable(actions[t])

                discrim_error = discriminator.forward(first_frame, final_frame, action)

                errors[s-1].append(discrim_error)

            # else:
            #     fdsfas
                #insert zero if less than n-step states left
                #maybe not, it just wont be averaged over this step


            #LOOK at episode completion!!-
    # print ('len')
    # print (len(errors))
    # print (len(errors[0]))
    # print (len(errors[1]))
    

    #average the errors
    avg_errors = []
    for t in range(n_steps):

        # print (t, 't')

        count = 0
        for s in range(max_pred_step):

            # print (s, 's')


            if s == 0:
                errors_sum = errors[s][t]
                count = 1
            else:
                if t < len(errors[s]):
                    errors_sum += errors[s][t]
                    count+=1

        # print (count)
        avg_error = errors_sum / count
        avg_errors.append(avg_error)

    # print (len(avg_errors))  #[S,P]
    avg_errors = torch.stack(avg_errors)#[S,P]
    # print (avg_errors.size())#[S,P]
    # fdsfa

    #still missing that Im not looking at episode completion. ..

    if reverse:
        idx = [i for i in range(n_steps-1, -1, -1)]
        idx = torch.LongTensor(idx).cuda()
        avg_errors = avg_errors.index_select(0, Variable(idx))

    return avg_errors


            




















