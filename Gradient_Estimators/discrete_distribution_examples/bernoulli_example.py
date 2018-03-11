



import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import math


from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli



# print (torch)
# print (dir(torch))
# print (torch.multinomial)
# print (dir(torch.multinomial))
# fsda





# class Categorical_v1(nn.Module):
#     def __init__(self, seed=1):
#         super(Gaus_1D, self).__init__()

#         torch.manual_seed(seed)

#         self.n_classes = 3

#         self.class_probs_pre_softmax = Variable(torch.FloatTensor([1.,.2,3.]), requires_grad = True)


#     def log_prob(self, x):
#         '''
#         x: [B,K]  one hot vectors 
#         output: [B,K]
#         '''

#         logprobs = torch.logsoftmax(self.class_probs_pre_softmax) #[K]

#         return x * logprobs



#     def sample(self, k):
#         '''
#         k: # of samples
#         output: [k,X]
#         '''

#         eps = Variable(torch.FloatTensor(k, self.x_size).normal_()) #.type(self.dtype)) #[P,B,Z]
#         z = eps.mul(torch.exp(.5*self.logvar)) + self.mean  #[P,B,Z]
#         return z




# def return_1d_distribution(distribution, xlimits, numticks):

#     x = np.expand_dims(np.linspace(*xlimits, num=numticks),1) #[B,1]

#     x_pytorch = Variable(torch.FloatTensor(x))
#     log_probs_pytorch = distribution.log_prob(x_pytorch)

#     log_probs = log_probs_pytorch.data.numpy()

#     px = np.exp(log_probs)

#     return x, px




# def return_1d_evaluation(eval_this, xlimits, numticks):

#     x = np.expand_dims(np.linspace(*xlimits, num=numticks),1) #[B,1]

#     x_pytorch = Variable(torch.FloatTensor(x))
#     log_probs_pytorch = eval_this(x_pytorch)

#     # print (log_probs_pytorch)
#     # fsadf

#     log_probs = log_probs_pytorch.data.numpy()

#     # px = np.exp(log_probs)

#     return x, log_probs






class NN(nn.Module):
    def __init__(self, seed=1):
        super(NN, self).__init__()

        torch.manual_seed(seed)

        self.input_size = 1
        self.output_size = 1
        h_size = 500

        # self.net = nn.Sequential(
        #   nn.Linear(self.input_size,h_size),
        #   nn.ReLU(),
        #   nn.Linear(h_size,self.output_size)
        # )
        self.net = nn.Sequential(
          nn.Linear(self.input_size,h_size),
          # nn.Tanh(),
          # nn.Linear(h_size,h_size),
          nn.Tanh(),
          nn.Linear(h_size,self.output_size)
        )

        # self.optimizer = optim.Adam(self.parameters(), lr=.01)
        self.optimizer = optim.Adam(self.parameters(), lr=.0001)




    def train(self, objective, dist):

        done = 0
        step_count = 0
        last_100 = []
        best_100 = 0
        not_better_counter = 0
        first = 1
        while not done:

            #sample batch
            # samps = sampler(20) #[B,X]
            samps = dist.sample(1)

            # print (samps.size())
            obj_value = objective(samps).unsqueeze(1) #[B,1]
            pred = self.net(samps) #[B,1]

            loss = torch.mean(((obj_value - pred)/.1)**2 ) #[1]

            # print (samps)
            # print (pred)
            # print (loss)

            # fdsa

            # print(step_count, loss.data.numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            

            if len(last_100) <  200:
                last_100.append(loss)
            else:
                last_100_avg = torch.mean(torch.stack(last_100))
                # print(step_count, loss.data.numpy())
                print(step_count, last_100_avg.data.numpy(), not_better_counter)
                # print(best_100, last_100_avg)
                # print (last_100_avg< best_100)
                if first or (last_100_avg< best_100).data.numpy():
                    first = 0
                    best_100 = last_100_avg
                    not_better_counter = 0
                else:
                    not_better_counter +=1
                    if not_better_counter == 200:
                        break
                last_100 = []

            step_count+=1







    # def train2(self, objective, dist):

    #     done = 0
    #     step_count = 0
    #     last_100 = []
    #     best_100 = 0
    #     not_better_counter = 0
    #     first = 1
    #     while not done:

    #         #Sample
    #         samps = dist.sample(1)  #.detach()  #this is critical, or else its 0


    #         # dist_params = torch.stack([dist.mean, dist.logvar])
    #         # print (dist_params)

    #         obj_value = objective(samps).unsqueeze(1) #[B,1]


    #         pred = self.net(samps) #[B,1]

    #         gradients = torch.autograd.grad(outputs=pred, inputs=(dist.mean,dist.logvar), retain_graph=True, create_graph=True)
    #         # gradients = torch.autograd.grad(outputs=pred, inputs=(dist.mean), retain_graph=True, create_graph=True)
    #                       # grad_outputs= torch.ones(1),
    #                       # create_graph=True, retain_graph=True, only_inputs=True)[0]
    #         # gradients = torch.sum(torch.stack(gradients))
    #         gradients = torch.stack(gradients)

    #         samps = samps.detach()

    #         log_probs = dist.log_prob(samps)

    #         gradients2 = torch.autograd.grad(outputs=log_probs, inputs=(dist.mean,dist.logvar), retain_graph=True, create_graph=True)
    #         # gradients2 = torch.autograd.grad(outputs=log_probs, inputs=(dist.mean), retain_graph=True, create_graph=True)
    #                       # grad_outputs= torch.ones(1),
    #                       # create_graph=True, retain_graph=True, only_inputs=True)[0]
    #         # gradients2 = torch.sum(torch.stack(gradients2))
    #         gradients2 = torch.stack(gradients2)

    #         g_lax = (obj_value - pred) * gradients2 + gradients

    #         loss = torch.mean((obj_value - pred)**2) #[1]
    #         # loss = torch.mean((obj_value - pred)**2  + .9*gradients**2) #[1]
    #         # loss = torch.mean((obj_value - pred)**2 * gradients2**2  + gradients**2) #[1]
    #         # loss = torch.mean(g_lax)**2

    #         # print(step_count, loss.data.numpy())

    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

            

    #         if len(last_100) <= 100:
    #             last_100.append(loss)
    #         else:
    #             last_100_avg = torch.mean(torch.stack(last_100))
    #             # print(step_count, loss.data.numpy())
    #             print(step_count, last_100_avg.data.numpy())
    #             # print(best_100, last_100_avg)
    #             # print (last_100_avg< best_100)
    #             if first or (last_100_avg< best_100).data.numpy():
    #                 first = 0
    #                 best_100 = last_100_avg
    #                 not_better_counter = 0
    #             else:
    #                 not_better_counter +=1
    #                 if not_better_counter == 30:#3:
    #                     break
    #             last_100 = []

    #         step_count+=1








# class NN_drop(NN):
#     def __init__(self, seed=1):
#         super(NN_drop, self).__init__()

#         torch.manual_seed(seed)

#         self.input_size = 1
#         self.output_size = 1
#         h_size = 50


#         # #this samples a mask for each datapoint in the batch
#         # self.net = nn.Sequential(
#         #   nn.Linear(self.input_size,h_size),
#         #   nn.ReLU(),
#         #   nn.Dropout(p=0.5),
#         #   nn.Linear(h_size,self.output_size)
#         # )

#         #want to keep mask constant for batch

#         self.l1 = nn.Linear(self.input_size,h_size)
#         self.a1 = nn.ReLU()
#         # nn.Dropout(p=0.5),
#         self.l2 = nn.Linear(h_size,self.output_size)

        



#         self.optimizer = optim.Adam(self.parameters(), lr=.01)



#     def net(self, x_input):

#         output = self.l1(x_input)
#         output = self.a1(output)


#         # mask = Variable(torch.bernoulli(output.data.new(output.data.size()).fill_(0.5)))
#         mask = Variable(torch.bernoulli(output.data.new(1,50).fill_(0.5)))

#         # print (mask)
#         # fsad

#         output = output*mask

#         output = self.l2(output)


#         return output














# def printnp(pytorch_vec):

#     return pytorch_vec.









if __name__ == "__main__":


    #Plot distributions
    rows = 6
    cols = 2
    # fig = plt.figure(figsize=(4+cols,5+rows), facecolor='white')
    fig = plt.figure(figsize=(4+cols,1+rows), facecolor='white')
    # viz_range = [-3,3]
    # numticks = 300

    # # samps = samps.data.numpy()
    # #Plot samples
    # for i in range(len(samps)):
    #     ax.plot([samps[i],samps[i]], [0,.1], linewidth=2, label=r'$z_q$')

    cur_row = 0

    #Plot discrete distribution
    ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=2)
    # ax.axis('off')
    # ax.set_yticks([])
    # ax.set_xticks([])

    # x, y = return_1d_distribution(distribution=p_x, xlimits=viz_range, numticks=numticks)
    # ax.plot(x, y, linewidth=2, label=r'$p(z)$')

    # x, y = return_1d_distribution(distribution=q_x, xlimits=viz_range, numticks=numticks)
    # ax.plot(x, y, linewidth=2, label=r'$q(z)$')

    # ax.legend(fontsize=9, loc=2)

    probs = [.2,.1,.5,.2]
    n_samples = 1000
    

    #Distrbution
    # print (probs)
    ax.bar(['0','1','2','3'],probs)
    ax.text(-.5, .5, s=r'Distribution', fontsize=10, family='serif')
    cur_row+=1


    #Samples using multinomial
    samples = np.random.multinomial(n_samples, probs) / n_samples
    # print (samples)
    ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=2)
    ax.bar(['0','1','2','3'],samples)
    ax.text(-.5, .5, s=r'Samples Categorical', fontsize=10, family='serif')
    cur_row+=1

    #Samples using Gumbel
    logits = np.log(probs)
    noise = np.random.gumbel(size=(n_samples, len(logits)))
    samples = np.argmax(logits + noise, axis=1)
    # print (samples.shape)
    # print (samples)
    # samples = np.sum(samples, axis=)
    zeros = np.zeros((n_samples, len(logits)))
    zeros[np.arange(len(zeros)),samples] = 1.
    # print (zeros)
    # print (zeros.shape)
    samples = np.sum(zeros, axis=0) / n_samples
    # print (samples)
    ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=2)
    ax.bar(['0','1','2','3'],samples)
    ax.text(-.5, .5, s=r'Samples Hard Gumbel', fontsize=10, family='serif')
    cur_row+=1



    #Plot gumbel pdf
    ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=2)

    # plt.plot(x,y)

    
    dist = RelaxedBernoulli(probs=torch.Tensor([0.5]), temperature=torch.Tensor([0.5]))
    # samp = dist.sample()
    # print (samp)
    # logprob = dist.log_prob(samp)


    x = [[.01], [.1], [.3], [.5], [.7], [.9], [.99]]
    x_len = len(x)
    logprob = dist.log_prob(torch.Tensor(x))
    logprob = np.reshape(logprob.numpy(), [x_len]) #[5]
    # print (samp, torch.exp(logprob))
    x = np.reshape(np.array(x), [x_len])
    plt.plot(x,logprob)
    cur_row+=1



    # ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=2)
    # plt.plot(x,np.exp(logprob))
    # cur_row+=1




    obj_value = .4

    objective = lambda x:  (x - obj_value)**2 

    objective_hard = lambda x:  (0. - obj_value)**2  if x<.5 else (1. - obj_value)**2 


    ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=2, rowspan=2)
    plt.plot(x,objective(x), label='Obj')

    plt.plot(x,np.exp(logprob), label='z prob')

    new_x = np.arange(.01,.99,.01)
    # print (new_x)
    # fsdaf
    xx = [objective_hard(a) for a in new_x]
    plt.plot(new_x,xx,label='Obj after hard')

    plt.legend()

    cur_row+=1










    plt.tight_layout()
    plt.show()

    fad



















    print('REINFORCE')
   
    # samp = dist.sample()
    # logprob = dist.log_prob(samp)
    # print (samp, torch.exp(logprob))

    n_samps = 1000

    bern_param_value = .7
    bern_param = Variable(torch.Tensor([bern_param_value]), requires_grad=True)
    # dist = Bernoulli(torch.Tensor([bern_param]))
    dist = Bernoulli(bern_param)
    losses = []
    grads = []
    for i in range(n_samps):
        samp = dist.sample()
        logprob = dist.log_prob(samp)
        logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(logparam), retain_graph=True)[0]
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param))[0]
        loss = objective(samp)
        losses.append(loss.numpy()[0])
        # print (logprobgrad)
        # print (loss)
        grads.append((logprobgrad*loss).numpy()[0])
        # print (bern_param.grad)
        # bern_param.grad.data.zero_()
    print (bern_param.data.numpy()[0], np.mean(losses), np.var(grads),  np.log(np.var(grads)))


    bern_param_value = .5
    bern_param = Variable(torch.Tensor([bern_param_value]), requires_grad=True)
    # dist = Bernoulli(torch.Tensor([bern_param]))
    dist = Bernoulli(bern_param)
    losses = []
    grads = []
    for i in range(n_samps):
        samp = dist.sample()
        logprob = dist.log_prob(samp)
        logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param))[0]
        loss = objective(samp)
        losses.append(loss.numpy()[0])
        # print (logprobgrad)
        # print (loss)
        grads.append((logprobgrad*loss).numpy()[0])
        # print (bern_param.grad)
        # bern_param.grad.data.zero_()
    print (bern_param.data.numpy()[0], np.mean(losses), np.var(grads),  np.log(np.var(grads)))



    bern_param_value = .2
    bern_param = Variable(torch.Tensor([bern_param_value]), requires_grad=True)
    # dist = Bernoulli(torch.Tensor([bern_param]))
    dist = Bernoulli(bern_param)
    losses = []
    grads = []
    for i in range(n_samps):
        samp = dist.sample()
        logprob = dist.log_prob(samp)
        logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param))[0]
        loss = objective(samp)
        losses.append(loss.numpy()[0])
        # print (logprobgrad)
        # print (loss)
        grads.append((logprobgrad*loss).numpy()[0])
        # print (bern_param.grad)
        # bern_param.grad.data.zero_()
    print (bern_param.data.numpy()[0], np.mean(losses), np.var(grads),  np.log(np.var(grads)))


    bern_param_value = .1
    bern_param = Variable(torch.Tensor([bern_param_value]), requires_grad=True)
    # dist = Bernoulli(torch.Tensor([bern_param]))
    dist = Bernoulli(bern_param)
    losses = []
    grads = []
    for i in range(n_samps):
        samp = dist.sample()
        logprob = dist.log_prob(samp)
        logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param))[0]
        loss = objective(samp)
        losses.append(loss.numpy()[0])
        # print (logprobgrad)
        # print (loss)
        grads.append((logprobgrad*loss).numpy()[0])
        # print (bern_param.grad)
        # bern_param.grad.data.zero_()
    print (bern_param.data.numpy()[0], np.mean(losses), np.var(grads),  np.log(np.var(grads)))





    print()
    print('REINFORCE with Baseline')



    #With baseline

    baseline = .25





    bern_param_value = .7
    bern_param = Variable(torch.Tensor([bern_param_value]), requires_grad=True)
    # dist = Bernoulli(torch.Tensor([bern_param]))
    dist = Bernoulli(bern_param)
    losses = []
    grads = []
    for i in range(n_samps):
        samp = dist.sample()
        logprob = dist.log_prob(samp)
        logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(logparam), retain_graph=True)[0]
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param))[0]
        loss = objective(samp)
        losses.append(loss.numpy()[0])
        # print (logprobgrad)
        # print (loss)
        grads.append((logprobgrad*(loss-baseline)).numpy()[0])
        # print (bern_param.grad)
        # bern_param.grad.data.zero_()
    print (bern_param.data.numpy()[0], np.mean(losses), np.var(grads),  np.log(np.var(grads)))


    bern_param_value = .5
    bern_param = Variable(torch.Tensor([bern_param_value]), requires_grad=True)
    # dist = Bernoulli(torch.Tensor([bern_param]))
    dist = Bernoulli(bern_param)
    losses = []
    grads = []
    for i in range(n_samps):
        samp = dist.sample()
        logprob = dist.log_prob(samp)
        logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param))[0]
        loss = objective(samp)
        losses.append(loss.numpy()[0])
        # print (logprobgrad)
        # print (loss)
        grads.append((logprobgrad*(loss-baseline)).numpy()[0])
        # print (bern_param.grad)
        # bern_param.grad.data.zero_()
    print (bern_param.data.numpy()[0], np.mean(losses), np.var(grads),  np.log(np.var(grads)))



    bern_param_value = .2
    bern_param = Variable(torch.Tensor([bern_param_value]), requires_grad=True)
    # dist = Bernoulli(torch.Tensor([bern_param]))
    dist = Bernoulli(bern_param)
    losses = []
    grads = []
    for i in range(n_samps):
        samp = dist.sample()
        logprob = dist.log_prob(samp)
        logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param))[0]
        loss = objective(samp)
        losses.append(loss.numpy()[0])
        # print (logprobgrad)
        # print (loss)
        grads.append((logprobgrad*(loss-baseline)).numpy()[0])
        # print (bern_param.grad)
        # bern_param.grad.data.zero_()
    print (bern_param.data.numpy()[0], np.mean(losses), np.var(grads),  np.log(np.var(grads)))


    bern_param_value = .1
    bern_param = Variable(torch.Tensor([bern_param_value]), requires_grad=True)
    # dist = Bernoulli(torch.Tensor([bern_param]))
    dist = Bernoulli(bern_param)
    losses = []
    grads = []
    for i in range(n_samps):
        samp = dist.sample()
        logprob = dist.log_prob(samp)
        logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param))[0]
        loss = objective(samp)
        losses.append(loss.numpy()[0])
        # print (logprobgrad)
        # print (loss)
        grads.append((logprobgrad*(loss-baseline)).numpy()[0])
        # print (bern_param.grad)
        # bern_param.grad.data.zero_()
    print (bern_param.data.numpy()[0], np.mean(losses), np.var(grads),  np.log(np.var(grads)))





    print()
    print ('Done.')


    # fdsfad






    # # name_file = home+'/Downloads/bern_1.pdf'


    # # print (torch)
    # # print (torch.distributions_dir)

    # # import distributions_dir



    # # import torch.distributions_dir
    # # from torch.distributions_dir.bernoulli import Bernoulli
    # # from torch.distributions_dir import bernoulli

    # # fdas

    # # print (dir(torch))
    # # print (torch)


    # # dist = torch.distributions.bernoulli.Bernoulli([.5])
    
    # # dist = torch.distributions.Bernoulli(torch.Tensor([0.3]))
    # # dist = torch.distributions.Relaxed_Bernoulli(torch.Tensor([0.3]))
    # # print (samp, logprob, torch.exp(logprob))


    # bern_param = .5
    # dist = Bernoulli(torch.Tensor([bern_param]))
    # losses = []
    # for i in range(n_samps):
    #     samp = dist.sample()
    #     losses.append(objective(samp).numpy()[0])
    # print (bern_param, np.mean(losses))


    # bern_param = .3
    # dist = Bernoulli(torch.Tensor([bern_param]))
    # losses = []
    # for i in range(n_samps):
    #     samp = dist.sample()
    #     losses.append(objective(samp).numpy()[0])
    # print (bern_param, np.mean(losses))

    # bern_param = .1
    # dist = Bernoulli(torch.Tensor([bern_param]))
    # losses = []
    # for i in range(n_samps):
    #     samp = dist.sample()
    #     losses.append(objective(samp).numpy()[0])
    # print (bern_param, np.mean(losses))

    # bern_param = .05
    # dist = Bernoulli(torch.Tensor([bern_param]))
    # losses = []
    # for i in range(n_samps):
    #     samp = dist.sample()
    #     losses.append(objective(samp).numpy()[0])
    # print (bern_param, np.mean(losses))

    # bern_param = .001
    # dist = Bernoulli(torch.Tensor([bern_param]))
    # losses = []
    # for i in range(n_samps):
    #     samp = dist.sample()
    #     losses.append(objective(samp).numpy()[0])
    # print (bern_param, np.mean(losses))


    # bern_param = .000001
    # dist = Bernoulli(torch.Tensor([bern_param]))
    # losses = []
    # for i in range(n_samps):
    #     samp = dist.sample()
    #     losses.append(objective(samp).numpy()[0])
    # print (bern_param, np.mean(losses))



    # bern_param = Variable(torch.Tensor([.5]), requires_grad=True)
    # # print (bern_param)
    # dist = Bernoulli(bern_param)
    # samp = dist.sample()
    # logprob = dist.log_prob(Variable(samp))
    # print (samp, torch.exp(logprob))


    # gradients = torch.autograd.grad(outputs=logprob, inputs=(bern_param)) #, retain_graph=True, create_graph=True)

    # print (gradients)

    # fdsa






    # fdsa





    
    # # from torch.distributions import Relaxed_Bernoulli
    # dist = RelaxedBernoulli(probs=torch.Tensor([0.5]), temperature=torch.Tensor([0.5]))
    # samp = dist.sample()
    # logprob = dist.log_prob(samp)
    # # print (samp, logprob, torch.exp(logprob))
    # print (samp, torch.exp(logprob))




    # FDDSA




    # Define 2 distribution p and q
    #PLot them

    # #ORIGINAL
    # p_mean = [-2.5]
    # p_logvar = [1.]
    # q_mean = [4.5]
    # q_logvar = [.5]
    # mu_x_range = [0,5] #for plotting
    # baseline = 8.8

    # #NEW
    # p_mean = [-2.5]
    # p_logvar = [1.]
    # q_mean = [-1.5]
    # q_logvar = [.5]
    # mu_x_range = [-3,3] #for plotting
    # baseline = .68


    # #MoG taget
    # q_mean = [0] #[-1.5]
    # q_logvar = [-1.]#[-8.]#[0.]#[.5]
    # mu_x_range = [-3,3] #for plotting
    # # baseline = .68
    # # p_x = target_MoG_1D()


    # p_x = Gaus_1D(p_mean, p_logvar)
    # q_x = Gaus_1D(q_mean, q_logvar)





    # objective = lambda x: (q_x.log_prob(x) - p_x.log_prob(x)) # * torch.exp(q_x.log_prob(x))


    # objective = lambda x: x/5. + torch.sin(x*50.)/3. # * torch.exp(q_x.log_prob(x))
    # objective = lambda x: x/5. + torch.sin(x*10.)/3. # * torch.exp(q_x.log_prob(x))
    # objective = lambda x: x/5. + torch.sin(x*8.)/3. # * torch.exp(q_x.log_prob(x))
    # objective = lambda x: torch.sign(x) # * torch.exp(q_x.log_prob(x))





    # x, y = return_1d_distribution(distribution=p_x, xlimits=viz_range, numticks=numticks)


    # objective_for_plot = lambda x: (q_x.log_prob(x) - p_x.log_prob(x)) * torch.exp(q_x.log_prob(x))

    # x1, y1 = return_1d_evaluation(eval_this=objective_for_plot, xlimits=[-10,10], numticks=5)
    # x2, y2 = return_1d_evaluation(eval_this=q_x.log_prob, xlimits=[-10,10], numticks=5)
    # x3, y3 = return_1d_evaluation(eval_this=p_x.log_prob, xlimits=[-10,10], numticks=5)


    # print ('x')
    # print (x1)
    # print ('q-p')
    # print (y1)
    # print ('q')
    # print (y2)
    # print ('p')
    # print (y3)
    # # print (y2-y3)

    # fadd


    # print (surrogate_model.parameters())
    # print (surrogate_model)
    # fdasa


    # surrogate_model = NN()
    # # surrogate_model = NN_drop()


    # # surrogate_model.train(objective=objective, sampler=q_x.sample) 
    # # surrogate_model.train2(objective=objective, dist=q_x) 
    # surrogate_model.train(objective=objective, dist=q_x) 
    # q_x.mean.grad.data.zero_()
    # q_x.logvar.grad.data.zero_()








    # # Get distribution of gradients
    # n_grads = 500
    # n_samps = 1
    # mean_grads = []
    # logvar_grads = []
    # for i in range(n_grads):
    #     #Sample
    #     samps = q_x.sample(n_samps)

    #     # log_qx = q_x.log_prob(samps)
    #     # log_px = p_x.log_prob(samps)
    #     # #Compute KL
    #     # log_qp = log_qx - log_px
    #     # # print (log_qp)

    #     obj = objective(samps)

    #     #Compute grad
    #     log_qp_avg = torch.mean(obj)
    #     log_qp_avg.backward()
    #     # print(q_x.mean.grad, q_x.logvar.grad)
    #     mean_grads.append(q_x.mean.grad.data.numpy()[0])
    #     logvar_grads.append(q_x.logvar.grad.data.numpy()[0])

    #     q_x.mean.grad.data.zero_()
    #     q_x.logvar.grad.data.zero_()




    # # Get distribution of gradients, differnet number of MC samples
    # n_grads = 500
    # n_samps = 10
    # mean_grads_10MC = []
    # logvar_grads_10MC = []
    # for i in range(n_grads):
    #     #Sample
    #     samps = q_x.sample(n_samps)
    #     log_qx = q_x.log_prob(samps)
    #     log_px = p_x.log_prob(samps)

    #     #Compute KL
    #     log_qp = log_qx - log_px
    #     # print (log_qp)

    #     #Compute grad
    #     log_qp_avg = torch.mean(log_qp)
    #     log_qp_avg.backward()
    #     # print(q_x.mean.grad, q_x.logvar.grad)
    #     mean_grads_10MC.append(q_x.mean.grad.data.numpy()[0])
    #     logvar_grads_10MC.append(q_x.logvar.grad.data.numpy()[0])

    #     q_x.mean.grad.data.zero_()
    #     q_x.logvar.grad.data.zero_()


    # # Get distribution of gradients, SF samples
    # n_grads = 5000
    # n_samps = 1
    # mean_grads_10MC_SF = []
    # logvar_grads_10MC_SF = []
    # for i in range(n_grads):
    #     #Sample
    #     samps = q_x.sample(n_samps).detach()  #this is critical, or else its 0
    #     log_qx = q_x.log_prob(samps) #.detach()
    #     # print (log_qx)
    #     log_px = p_x.log_prob(samps) #.detach() 

    #     #Compute KL
    #     log_qp = log_qx - log_px
    #     # print (log_qp)

    #     #Compute grad
    #     log_qp_avg = torch.mean(log_qp) 
    #     # print (log_qp_avg)
    #     # fsd
    #     # log_qp_avg.backward()
    #     # print (log_q_avg)
    #     log_q_avg = torch.mean(log_qx)
    #     # print (log_q_avg)
    #     # fsda
    #     log_q_avg.backward()

    #     # print(q_x.mean.grad, q_x.logvar.grad)
    #     mean_grads_10MC_SF.append(q_x.mean.grad.data.numpy()[0]*log_qp_avg.data.numpy()[0])
    #     logvar_grads_10MC_SF.append(q_x.logvar.grad.data.numpy()[0]*log_qp_avg.data.numpy()[0])
    #     # mean_grads_10MC_SF.append(q_x.mean.grad.data.numpy()[0])
    #     # logvar_grads_10MC_SF.append(q_x.logvar.grad.data.numpy()[0])

    #     q_x.mean.grad.data.zero_()
    #     q_x.logvar.grad.data.zero_()




    # # Get distribution of gradients, SF samples and baseline
    # n_grads = 5000
    # n_samps = 1
    # mean_grads_10MC_SF_Base = []
    # logvar_grads_10MC_SF_Base = []
    # for i in range(n_grads):
    #     #Sample
    #     samps = q_x.sample(n_samps).detach()  #this is critical, or else its 0
    #     log_qx = q_x.log_prob(samps)
    #     # print (log_qx)
    #     log_px = p_x.log_prob(samps) #.detach() 

    #     #Compute KL
    #     log_qp = log_qx - log_px
    #     # print (log_qp)

    #     #Compute grad
    #     log_qp_avg = torch.mean(log_qp) - baseline
    #     # print (log_qp_avg)
    #     # fsd
    #     # log_qp_avg.backward()
    #     # print (log_q_avg)
    #     log_q_avg = torch.mean(log_qx)
    #     # print (log_q_avg)
    #     # fsda
    #     log_q_avg.backward()

    #     # print(q_x.mean.grad, q_x.logvar.grad)
    #     mean_grads_10MC_SF_Base.append(q_x.mean.grad.data.numpy()[0]*log_qp_avg.data.numpy()[0])
    #     logvar_grads_10MC_SF_Base.append(q_x.logvar.grad.data.numpy()[0]*log_qp_avg.data.numpy()[0])
    #     # mean_grads_10MC_SF.append(q_x.mean.grad.data.numpy()[0])
    #     # logvar_grads_10MC_SF.append(q_x.logvar.grad.data.numpy()[0])

    #     q_x.mean.grad.data.zero_()
    #     q_x.logvar.grad.data.zero_()









    # # Get distribution of gradients, SF + LAX
    # n_grads = 500
    # n_samps = 1
    # mean_grads_LAX = []
    # logvar_grads_LAX = []
    # for i in range(n_grads):
    #     #Sample
    #     samps = q_x.sample(n_samps)  #this is critical, or else its 0

    #     pred = surrogate_model.net(samps) #[B,1]

    #     pred.backward()
    #     aa = q_x.mean.grad.data.numpy()[0]  #the indexing is important for not letting it change!
    #     bb = q_x.logvar.grad.data.numpy()[0]
    #     # print(aa)

    #     # print (aa, bb)
    #     # fdsaf
    #     q_x.mean.grad.data.zero_()
    #     q_x.logvar.grad.data.zero_()


    #     samps = samps.detach()


    #     log_qx = q_x.log_prob(samps)
    #     log_px = p_x.log_prob(samps) #.detach() 
    #     #Compute KL
    #     log_qp = log_qx - log_px - pred


    #     obj = objective(samps)
    #     #Compute grad
    #     log_qp_avg = torch.mean(obj)
    #     log_qp_avg.backward()

    #     #Compute grad
    #     log_qp_avg = torch.mean(log_qp) 
    #     log_q_avg = torch.mean(log_qx)
    #     log_q_avg.backward()

    #     # print(q_x.mean.grad, q_x.logvar.grad)
    #     # print (aa)
    #     # fsd
    #     mean_grads_LAX.append(q_x.mean.grad.data.numpy()[0]*log_qp_avg.data.numpy()[0] + aa)
    #     logvar_grads_LAX.append(q_x.logvar.grad.data.numpy()[0]*log_qp_avg.data.numpy()[0] + bb)

    #     q_x.mean.grad.data.zero_()
    #     q_x.logvar.grad.data.zero_()














    # # # Get distribution of gradients, SF + LAX + Dropout
    # # n_grads = 500
    # # n_samps = 1
    # # mean_grads_LAX = []
    # # logvar_grads_LAX = []
    # # for i in range(n_grads):
    # #     #Sample
    # #     samps = q_x.sample(n_samps)  #this is critical, or else its 0

    # #     pred = surrogate_model.net(samps) #[B,1]

    # #     pred.backward()
    # #     aa = q_x.mean.grad.data.numpy()[0]  #the indexing is important for not letting it change!
    # #     bb = q_x.logvar.grad.data.numpy()[0]
    # #     # print(aa)


    # #     grad_of_dif_masks_mean = []
    # #     grad_of_dif_masks_logvar = []
    # #     for j in range(5):

    # #         # print (aa, bb)
    # #         # fdsaf
    # #         q_x.mean.grad.data.zero_()
    # #         q_x.logvar.grad.data.zero_()


    # #         samps = samps.detach()


    # #         log_qx = q_x.log_prob(samps)
    # #         log_px = p_x.log_prob(samps) #.detach() 

    # #         # print (log_qx)



    # #         #Compute KL
    # #         log_qp = log_qx - log_px - pred

    # #         #Compute grad
    # #         log_qp_avg = torch.mean(log_qp) 
    # #         log_q_avg = torch.mean(log_qx)
    # #         log_q_avg.backward()

    # #         # (grad param)*(R-pred) + grad param
    # #         mean_grad = q_x.mean.grad.data.numpy()[0]*log_qp_avg.data.numpy()[0] + aa
    # #         logvar_grad = q_x.logvar.grad.data.numpy()[0]*log_qp_avg.data.numpy()[0] + bb

    # #         grad_of_dif_masks_mean.append(mean_grad)
    # #         grad_of_dif_masks_logvar.append(logvar_grad)

    # #         q_x.mean.grad.data.zero_()
    # #         q_x.logvar.grad.data.zero_()


    # #     # grad_of_dif_masks_mean = torch.stack(grad_of_dif_masks_mean)
    # #     # grad_of_dif_masks_logvar = torch.stack(grad_of_dif_masks_logvar)

    # #     # print(grad_of_dif_masks_mean)

    # #     # print (grad_of_dif_masks_mean[0].shape)
    # #     # fasd


    # #     # print(q_x.mean.grad, q_x.logvar.grad)
    # #     # print (aa)
    # #     # fsd
    # #     # mean_grads_LAX.append(q_x.mean.grad.data.numpy()[0]*log_qp_avg.data.numpy()[0] + aa)
    # #     # logvar_grads_LAX.append(q_x.logvar.grad.data.numpy()[0]*log_qp_avg.data.numpy()[0] + bb)

    # #     mean_grads_LAX.append( np.mean(grad_of_dif_masks_mean))
    # #     logvar_grads_LAX.append(np.mean(grad_of_dif_masks_logvar) ) 

    # #     # q_x.mean.grad.data.zero_()
    # #     # q_x.logvar.grad.data.zero_()













    # # Get distribution of gradients, SF + LAX + Dropout  version 2, take average of surrogate grads
    # n_grads = 500
    # n_samps = 1
    # mean_grads_LAX = []
    # logvar_grads_LAX = []
    # for i in range(n_grads):
    #     #Sample
    #     samps = q_x.sample(n_samps)  #this is critical, or else its 0

    #     grad_of_dif_masks_mean = []
    #     grad_of_dif_masks_logvar = []
    #     for j in range(5):

    #         pred = surrogate_model.net(samps) #[B,1]

    #         pred.backward(retain_graph=True)
    #         # aa = q_x.mean.grad.data.numpy()[0]  #the indexing is important for not letting it change!
    #         # bb = q_x.logvar.grad.data.numpy()[0]

    #         # grad_of_dif_masks_mean.append(aa)
    #         # grad_of_dif_masks_logvar.append(bb)

    #     # print (q_x.mean.grad / 5.)
    #     aa = q_x.mean.grad.data.numpy()[0]  / 5.
    #     bb = q_x.logvar.grad.data.numpy()[0] / 5.
    #     # print(aa)
    #     # ffdsa

    #     q_x.mean.grad.data.zero_()
    #     q_x.logvar.grad.data.zero_()

    #     # aa = np.mean(grad_of_dif_masks_mean)
    #     # bb = np.mean(grad_of_dif_masks_logvar)

    #     samps = samps.detach()

    #     log_qx = q_x.log_prob(samps)
    #     log_px = p_x.log_prob(samps) #.detach() 

    #     #Compute KL
    #     log_qp = log_qx - log_px - pred

    #     #Compute grad
    #     log_qp_avg = torch.mean(log_qp) 
    #     log_q_avg = torch.mean(log_qx)
    #     log_q_avg.backward()

    #     # (grad param)*(R-pred) + grad param
    #     mean_grad = q_x.mean.grad.data.numpy()[0]*log_qp_avg.data.numpy()[0] + aa
    #     logvar_grad = q_x.logvar.grad.data.numpy()[0]*log_qp_avg.data.numpy()[0] + bb


    #         # q_x.mean.grad.data.zero_()
    #         # q_x.logvar.grad.data.zero_()

    #     mean_grads_LAX.append(mean_grad)
    #     logvar_grads_LAX.append(logvar_grad) 



























    # # print (mean_grads_10MC_SF)
    # # fdsa

    # #Plot distributions
    # rows = 7
    # cols = 2
    # # fig = plt.figure(figsize=(4+cols,5+rows), facecolor='white')
    # fig = plt.figure(figsize=(4+cols,1+rows), facecolor='white')
    # # viz_range = [-3,3]
    # # numticks = 300

    # # # samps = samps.data.numpy()
    # # #Plot samples
    # # for i in range(len(samps)):
    # #     ax.plot([samps[i],samps[i]], [0,.1], linewidth=2, label=r'$z_q$')

    # cur_row = 0

    # #Plot discrete distribution
    # ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=2)
    # # ax.axis('off')
    # # ax.set_yticks([])
    # # ax.set_xticks([])

    # # x, y = return_1d_distribution(distribution=p_x, xlimits=viz_range, numticks=numticks)
    # # ax.plot(x, y, linewidth=2, label=r'$p(z)$')

    # # x, y = return_1d_distribution(distribution=q_x, xlimits=viz_range, numticks=numticks)
    # # ax.plot(x, y, linewidth=2, label=r'$q(z)$')

    # # ax.legend(fontsize=9, loc=2)

    # probs = [.2,.1,.5,.2]
    # n_samples = 1000
    

    # #Distrbution
    # print (probs)
    # ax.bar(['0','1','2','3'],probs)
    # ax.text(-.5, .5, s=r'Distribution', fontsize=10, family='serif')
    # cur_row+=1


    # #Samples using multinomial
    # samples = np.random.multinomial(n_samples, probs) / n_samples
    # print (samples)
    # ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=2)
    # ax.bar(['0','1','2','3'],samples)
    # ax.text(-.5, .5, s=r'Samples v1', fontsize=10, family='serif')
    # cur_row+=1

    # #Samples using Gumbel
    # logits = np.log(probs)
    # noise = np.random.gumbel(size=(n_samples, len(logits)))
    # samples = np.argmax(logits + noise, axis=1)
    # # print (samples.shape)
    # # print (samples)
    # # samples = np.sum(samples, axis=)
    # zeros = np.zeros((n_samples, len(logits)))
    # zeros[np.arange(len(zeros)),samples] = 1.
    # # print (zeros)
    # # print (zeros.shape)
    # samples = np.sum(zeros, axis=0) / n_samples
    # print (samples)
    # ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=2)
    # ax.bar(['0','1','2','3'],samples)
    # ax.text(-.5, .5, s=r'Samples v2', fontsize=10, family='serif')
    # cur_row+=1

    # # fasda






    # matplotlib.pyplot.hist(x, bins=None, range=None, density=None, weights=None, 


    # # Plot kl q||p
    # ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=2)
    # # ax.axis('off')
    # # ax.set_yticks([])
    # # ax.set_xticks([])

    # x, y = return_1d_evaluation(eval_this=objective, xlimits=viz_range, numticks=numticks)
    # # x, y = return_1d_evaluation(eval_this=objective_for_plot, xlimits=viz_range, numticks=numticks)
    # # ax.plot(x, y, linewidth=2, label=r'$KL(q||p)$')
    # # ax.plot(x, y, linewidth=2, label=r'$\log \frac{q}{p}$')
    # # ax.plot(x, y, linewidth=2, label=r'$\log q - \log p$')
    # ax.plot(x, y, linewidth=2, label=r'$Objective$')

    # # surrogate_for_plot = lambda x: surrogate_model.net(x) * torch.exp(q_x.log_prob(x))
 
    # x, y = return_1d_evaluation(eval_this=surrogate_model.net, xlimits=viz_range, numticks=numticks)
    # # x, y = return_1d_evaluation(eval_this=surrogate_for_plot, xlimits=viz_range, numticks=numticks)
    # ax.plot(x, y, linewidth=1, label=r'$Surrogate1$')

    # x, y = return_1d_evaluation(eval_this=surrogate_model.net, xlimits=viz_range, numticks=numticks)
    # # x, y = return_1d_evaluation(eval_this=surrogate_for_plot, xlimits=viz_range, numticks=numticks)
    # ax.plot(x, y, linewidth=1, label=r'$Surrogate2$')

    # x, y = return_1d_evaluation(eval_this=surrogate_model.net, xlimits=viz_range, numticks=numticks)
    # # x, y = return_1d_evaluation(eval_this=surrogate_for_plot, xlimits=viz_range, numticks=numticks)
    # ax.plot(x, y, linewidth=1, label=r'$Surrogate3$')


    # ax.legend(fontsize=6, loc=2)
    # cur_row+=1














    # bins = 50
    # # mu_x_range = [0,5]
    # # mu_x_range = [-3,3]

    # # weights = np.empty_like(mean_grads)
    # # weights.fill(bins / 7 / len(mean_grads))

    # #Plot mean grads histogram
    # ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=1)
    # ax.hist(mean_grads, bins=bins, normed=True, range=mu_x_range)
    # # ax.set_xlabel(r'$\nabla \mu$')
    # ax.set_ylim(top=3.)
    # ax.set_ylabel('REPARAM\n1 MC Sample', family='serif')
    # ax.set_yticks([0,1,2])

    # mean_ = np.mean(mean_grads)
    # ax.plot([mean_,mean_], [0,.2], linewidth=2, label=r'$mean$')


    # #Plot logvar grads histogram
    # ax = plt.subplot2grid((rows,cols), (cur_row,1), frameon=False, colspan=1)
    # ax.hist(logvar_grads, bins=bins, normed=True, range=[-4,6])
    # # ax.set_xlabel(r'$\nabla \log \sigma$')
    # ax.set_ylim(top=1.)
    # ax.set_yticks([0.,.25,.50,.75])

    # mean_ = np.mean(logvar_grads)
    # ax.plot([mean_,mean_], [0,.1], linewidth=2, label=r'$mean$')
    # cur_row+=1


    # #Plot mean grads histogram
    # ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=1)
    # ax.hist(mean_grads_10MC, bins=bins, normed=True, range=mu_x_range)
    # # ax.set_xlabel(r'$\nabla \mu$')
    # ax.set_ylim(top=3.)
    # ax.set_ylabel('REPARAM\n10 MC Sample', family='serif')
    # ax.set_yticks([0,1,2])

    # mean_ = np.mean(mean_grads_10MC)
    # ax.plot([mean_,mean_], [0,.2], linewidth=2, label=r'$mean$')


    # #Plot logvar grads histogram
    # ax = plt.subplot2grid((rows,cols), (cur_row,1), frameon=False, colspan=1)
    # ax.hist(logvar_grads_10MC, bins=bins, normed=True, range=[-4,6])
    # # ax.set_xlabel(r'$\nabla \log \sigma$')
    # ax.set_ylim(top=1.)
    # ax.set_yticks([0.,.25,.50,.75])

    # mean_ = np.mean(logvar_grads_10MC)
    # ax.plot([mean_,mean_], [0,.1], linewidth=2, label=r'$mean$')
    # cur_row+=1



    # #Plot mean grads histogram
    # ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=1)
    # ax.hist(mean_grads_10MC_SF, bins=bins, normed=True, range=mu_x_range)
    # # ax.set_xlabel(r'$\nabla \mu$')
    # ax.set_ylim(top=3.)
    # ax.set_ylabel('SF\n1 MC Sample', family='serif')
    # ax.set_yticks([0,1,2])

    # mean_ = np.mean(mean_grads_10MC_SF)
    # ax.plot([mean_,mean_], [0,.2], linewidth=2, label=r'$mean$')


    # #Plot logvar grads histogram
    # ax = plt.subplot2grid((rows,cols), (cur_row,1), frameon=False, colspan=1)
    # ax.hist(logvar_grads_10MC_SF, bins=bins, normed=True, range=[-4,6])
    # # ax.set_xlabel(r'$\nabla \log \sigma$')
    # ax.set_ylim(top=1.)
    # ax.set_yticks([0.,.25,.50,.75])

    # mean_ = np.mean(logvar_grads_10MC_SF)
    # ax.plot([mean_,mean_], [0,.1], linewidth=2, label=r'$mean$')
    # cur_row+=1




    # #Plot mean grads histogram
    # ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=1)
    # ax.hist(mean_grads_10MC_SF_Base, bins=bins, normed=True, range=mu_x_range)
    # # ax.set_xlabel(r'$\nabla \mu$')
    # ax.set_ylim(top=3.)
    # ax.set_ylabel('SF+Baseline\n1 MC Sample', family='serif')
    # ax.set_yticks([0,1,2])

    # mean_ = np.mean(mean_grads_10MC_SF_Base)
    # ax.plot([mean_,mean_], [0,.2], linewidth=2, label=r'$mean$')


    # #Plot logvar grads histogram
    # ax = plt.subplot2grid((rows,cols), (cur_row,1), frameon=False, colspan=1)
    # ax.hist(logvar_grads_10MC_SF_Base, bins=bins, normed=True, range=[-4,6])
    # # ax.set_xlabel(r'$\nabla \log \sigma$')
    # ax.set_ylim(top=1.)
    # ax.set_yticks([0.,.25,.50,.75])

    # mean_ = np.mean(logvar_grads_10MC_SF_Base)
    # ax.plot([mean_,mean_], [0,.1], linewidth=2, label=r'$mean$')
    # cur_row+=1






    # #Plot mean grads histogram  LAX
    # ax = plt.subplot2grid((rows,cols), (cur_row,0), frameon=False, colspan=1)
    # ax.hist(mean_grads_LAX, bins=bins, normed=True, range=mu_x_range)
    # ax.set_xlabel(r'$\nabla \mu$')
    # ax.set_ylim(top=3.)
    # ax.set_ylabel('SF+LAX\n1 MC Sample', family='serif')
    # ax.set_yticks([0,1,2])

    # mean_ = np.mean(mean_grads_LAX)
    # ax.plot([mean_,mean_], [0,.2], linewidth=2, label=r'$mean$')


    # #Plot logvar grads histogram
    # ax = plt.subplot2grid((rows,cols), (cur_row,1), frameon=False, colspan=1)
    # ax.hist(logvar_grads_LAX, bins=bins, normed=True, range=[-4,6])
    # ax.set_xlabel(r'$\nabla \log \sigma$')
    # ax.set_ylim(top=1.)
    # ax.set_yticks([0.,.25,.50,.75])

    # mean_ = np.mean(logvar_grads_LAX)
    # ax.plot([mean_,mean_], [0,.1], linewidth=2, label=r'$mean$')
    # cur_row+=1






    # plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=.5)
    plt.tight_layout()


    plt.show()


    # # name_file = home+'/Documents/tmp/plot.pdf'

    # name_file = home+'/Downloads/squiggle_example_2.pdf'


    # plt.savefig(name_file)
    # print ('Saved fig', name_file)

    print ('Done.')

















