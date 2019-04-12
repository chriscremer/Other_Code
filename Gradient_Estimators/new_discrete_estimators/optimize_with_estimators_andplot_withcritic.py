


from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import sys, os
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('./VAE'))

import pickle

import numpy as np

import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from NN import NN
from NN_forrelax import NN as NN2



def H(x):
    if x > .5:
        return 1
    else:
        return 0

def Hpy(x):
    # if x > .5:
    #     return torch.tensor([1])
    # else:
    #     return torch.tensor([0])

    return (x > .5).float()


def prob_to_logit(prob):
    return torch.log(prob) - torch.log(1-prob)

def logit_to_prob(logit):
    return torch.sigmoid(logit)



def smooth_list(x, window_len=5, window='flat'):
    if len(x) < window_len:
        return x
    w = np.ones(window_len,'d') 
    y = np.convolve(w/ w.sum(), x, mode='same')
    y[-1] = x[-1]
    y[-2] = x[-2]
    y[0] = x[0]
    y[1] = x[1]
    return y


# n= 10000
# theta = .5
f = lambda x: (x-val)**2

total_steps = 10000


# save_dir = home+'/Downloads/tmmpp/'
save_dir = home+'/Documents/Grad_Estimators/new/'








# with open(save_dir+"data.p", "rb" ) as f:

#     favorite_color = pickle.load(f)


# print(len(favorite_color[0]))

# fsdfa








# with open( save_dir+"data.p", "wb" ) as f:
#     pickle.dump([losses], f)
# print ('saved data')


# fsadfs










logits = 0.


trian_ =0  # if 0, it loads data

if trian_:



    # playing with relax to make it unbiased...

    # # logits = 0.
    # bern_param = torch.tensor([logits], requires_grad=True)
    # val=.4


    # print()
    # print ('RELAX')
    # print ('Value:', val)
    # print()

    # net = NN()


    # # print (len(net.parameters()))

    # # optim = torch.optim.Adam([bern_param] + list(net.parameters()), lr=.004)
    # optim = torch.optim.Adam([bern_param], lr=.004)
    # optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


    # steps = []
    # losses8 = []
    # for step in range(total_steps):

    #     dist = RelaxedBernoulli(torch.Tensor([1.]), logits=bern_param)
    #     dist_bern = Bernoulli(logits=bern_param)

    #     optim.zero_grad()

    #     zs = []
    #     for i in range(20):
    #         z = dist.rsample()
    #         zs.append(z)
    #     zs = torch.stack(zs)

    #     b = Hpy(zs)

    #     # EHRERE, I think the bug is above, see grad of stuff 

    #     # #p(z|b)
    #     # theta = logit_to_prob(bern_param)
    #     # v = torch.rand(zs.shape[0], zs.shape[1]) 
    #     # # v= (1-b)*v*(1-theta) + b*v*theta+(1-theta)
    #     # # z_tilde = torch.log(theta/(1-theta)) + torch.log(v/(1-v))
    #     # v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
    #     # z_tilde = bern_param.detach() + torch.log(v_prime) - torch.log1p(-v_prime)
    #     # z_tilde = torch.sigmoid(z_tilde)
    #     # # z_tilde = torch.tensor(z_tilde, requires_grad=True)
    #     # # print (z_tilde)
    #     # # fadfsa

    #     # p(z|b) v2 - just removed detach()
    #     theta = logit_to_prob(bern_param) #bern_param #logit_to_prob(bern_param)
    #     v = torch.rand(zs.shape[0], zs.shape[1]) 
    #     v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
    #     z_tilde = bern_param + torch.log(v_prime) - torch.log1p(-v_prime)
    #     z_tilde = torch.sigmoid(z_tilde)


    #     # v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
    #     # z_tilde = bern_param.detach() + torch.log(v_prime) - torch.log1p(-v_prime)
    #     # z_tilde = torch.sigmoid(z_tilde).detach()
    #     # z_tilde = torch.tensor(z_tilde, requires_grad=True)

    #     logprob = dist_bern.log_prob(b)
        
    #     pred = net.net(z_tilde)

    #     # logprobgrad = torch.autograd.grad(outputs=torch.mean(pred), inputs=(bern_param), retain_graph=True)[0]
    #     # print (logprobgrad.shape, torch.max(logprobgrad), torch.min(logprobgrad))
    #     # fadaf

        
    #     pred2 = net.net(zs)
    #     f_b = f(b)

    #     # print (bern_param.detach().numpy())

    #     loss = torch.mean((f_b-pred.detach()) * logprob - pred + pred2 )


    #     # logprobgrad_2 = torch.autograd.grad(outputs=torch.mean(logprob), inputs=(bern_param), retain_graph=True)[0]
    #     # print (logprobgrad_2)

    #     # logprobgrad_2 = torch.autograd.grad(outputs=torch.mean(pred), inputs=(bern_param), retain_graph=True)[0]
    #     # print (logprobgrad_2)

    #     # logprobgrad_2 = torch.autograd.grad(outputs=torch.mean(pred2), inputs=(bern_param), retain_graph=True)[0]
    #     # print (logprobgrad_2)

    #     # fdfa

    #     loss.backward(retain_graph=True)  
    #     optim.step()

    #     optim_NN.zero_grad()
    #     NN_loss = torch.mean((f_b - pred)**2) 
    #     NN_loss.backward()  
    #     optim_NN.step()

    #     if step%50 ==0:
    #         if step %500==0:
    #             print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
    #         losses8.append(torch.mean(f_b).numpy())
    #         steps.append(step)

    #     # if step%1 ==0:
    #     #     if step %1==0:
    #     #         print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
    #     #     losses8.append(torch.mean(f_b).numpy())
    #     #     steps.append(step)


    # fdsfsad












    # logits = 0
    bern_param = torch.tensor([logits], requires_grad=True)
    val=.4


    print()
    print ('REINFORCE with critic')
    print ('Value:', val)
    print (f(0), f(1))
    print()

    net = NN()

    optim = torch.optim.Adam([bern_param], lr=.004)
    optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)

    # total_steps = 999999

    steps = []
    losses4= []
    for step in range(total_steps):

        dist = Bernoulli(logits=bern_param)

        optim.zero_grad()

        zs = []
        for i in range(20):
            z = dist.sample()
            zs.append(z)
        zs = torch.FloatTensor(zs).unsqueeze(1)

        logprob = dist.log_prob(zs.detach())

        # print (zs)
        f_b = f(zs)

        pred = net.net(zs*0.)

        loss = torch.mean( (f_b-pred) * logprob)

        loss.backward(retain_graph=True)  
        optim.step()

        optim_NN.zero_grad()
        # pred = net.net(zs)

        # print (logprob)
        # fdasf
        # NN_loss = torch.mean((f_b - pred)**2)
        # print (f_b.shape, pred.shape, logprob.shape)
        # fdsf
        NN_loss = torch.mean(  ((f_b - pred)*logprob)**2   )
        # print (logprob)

        NN_loss.backward()  
        optim_NN.step()


        if step%50 ==0:
            if step %500==0:
                print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), pred[0], NN_loss)
            losses4.append(torch.mean(f_b).numpy())
            steps.append(step)


    


    # fdsfa
    




    # logits = 0
    bern_param = torch.tensor([logits], requires_grad=True)
    val=.49


    print()
    print ('REINFORCE with critic')
    print ('Value:', val)
    print (f(0), f(1))
    print()

    net = NN()

    optim = torch.optim.Adam([bern_param], lr=.004)
    optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


    steps = []
    losses11= []
    for step in range(total_steps):

        dist = Bernoulli(logits=bern_param)

        optim.zero_grad()

        zs = []
        for i in range(20):
            z = dist.sample()
            zs.append(z)
        zs = torch.FloatTensor(zs).unsqueeze(1)

        logprob = dist.log_prob(zs.detach())

        f_b = f(zs)

        pred = net.net(zs*0.)

        loss = torch.mean( (f_b-pred) * logprob)

        loss.backward(retain_graph=True)  
        optim.step()

        optim_NN.zero_grad()
        # pred = net.net(zs)
        # NN_loss = torch.mean((f_b - pred)**2)
        NN_loss = torch.mean(  ((f_b - pred)*logprob)**2   )

        NN_loss.backward()  
        optim_NN.step()

        if step%50 ==0:
            if step %500==0:
                print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy())
            losses11.append(torch.mean(f_b).numpy())
            steps.append(step)








    # logits = 0
    bern_param = torch.tensor([logits], requires_grad=True)
    val=.499


    print()
    print ('REINFORCE with critic')
    print ('Value:', val)
    print (f(0), f(1))
    print()

    net = NN()

    optim = torch.optim.Adam([bern_param], lr=.004)
    optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


    steps = []
    losses12= []
    for step in range(total_steps):

        dist = Bernoulli(logits=bern_param)

        optim.zero_grad()

        zs = []
        for i in range(20):
            z = dist.sample()
            zs.append(z)
        zs = torch.FloatTensor(zs).unsqueeze(1)

        logprob = dist.log_prob(zs.detach())

        f_b = f(zs)

        pred = net.net(zs*0.)

        loss = torch.mean( (f_b-pred) * logprob)

        loss.backward(retain_graph=True)  
        optim.step()

        optim_NN.zero_grad()
        # pred = net.net(zs)
        # NN_loss = torch.mean((f_b - pred)**2)
        NN_loss = torch.mean(  ((f_b - pred)*logprob)**2   )

        NN_loss.backward()  
        optim_NN.step()



        if step%50 ==0:
            if step %500==0:
                print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy())
            losses12.append(torch.mean(f_b).numpy())
            steps.append(step)






























    # logits = 0
    bern_param = torch.tensor([logits], requires_grad=True)
    val=.4

    print()
    print('REINFORCE')
    print ('Value:', val)
    # print ('n:', n)
    # print ('theta:', theta)
    print()


    optim = torch.optim.Adam([bern_param], lr=.004)

    steps = []
    losses= []
    for step in range(total_steps):

        dist = Bernoulli(logits=bern_param)

        optim.zero_grad()

        bs = []
        for i in range(20):
            samps = dist.sample()
            bs.append(H(samps))
        bs = torch.FloatTensor(bs).unsqueeze(1)

        logprob = dist.log_prob(bs)
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

        loss = torch.mean(f(bs) * logprob)

        #review the pytorch_toy and the RL code to see how PG was done 

        loss.backward()  
        optim.step()

        if step%50 ==0:
            if step %500==0:
                print (step, torch.mean(f(bs)).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy())
            losses.append(torch.mean(f(bs)).numpy())
            steps.append(step)






    # logits = 0
    bern_param = torch.tensor([logits], requires_grad=True)
    val=.49

    print()
    print('REINFORCE')
    print ('Value:', val)
    # print ('n:', n)
    # print ('theta:', theta)
    print()

    optim = torch.optim.Adam([bern_param], lr=.004)

    steps = []
    losses2= []
    for step in range(total_steps):

        dist = Bernoulli(logits=bern_param)

        optim.zero_grad()

        bs = []
        for i in range(20):
            samps = dist.sample()
            bs.append(H(samps))
        bs = torch.FloatTensor(bs).unsqueeze(1)

        logprob = dist.log_prob(bs)
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

        loss = torch.mean(f(bs) * logprob)

        #review the pytorch_toy and the RL code to see how PG was done 

        loss.backward()  
        optim.step()

        if step%50 ==0:
            if step %500==0:
                print (step, torch.mean(f(bs)).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy())
            losses2.append(torch.mean(f(bs)).numpy())
            steps.append(step)








    # logits = 0
    bern_param = torch.tensor([logits], requires_grad=True)
    val=.499

    print()
    print('REINFORCE')
    print ('Value:', val)
    # print ('n:', n)
    # print ('theta:', theta)
    print()

    optim = torch.optim.Adam([bern_param], lr=.004)

    steps = []
    losses3= []
    for step in range(total_steps):

        dist = Bernoulli(logits=bern_param)

        optim.zero_grad()

        bs = []
        for i in range(20):
            samps = dist.sample()
            bs.append(H(samps))
        bs = torch.FloatTensor(bs).unsqueeze(1)

        logprob = dist.log_prob(bs)
        # logprobgrad = torch.autograd.grad(outputs=logprob, inputs=(bern_param), retain_graph=True)[0]

        loss = torch.mean(f(bs) * logprob)

        #review the pytorch_toy and the RL code to see how PG was done 

        loss.backward()  
        optim.step()

        if step%50 ==0:
            if step %500==0:
                print (step, torch.mean(f(bs)).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy())
            losses3.append(torch.mean(f(bs)).numpy())
            steps.append(step)











































































    # logits = 0
    bern_param = torch.tensor([logits], requires_grad=True)
    val=.4


    print()
    print ('SimpLAX')
    print ('Value:', val)
    print()

    net = NN()


    # print (len(net.parameters()))

    # optim = torch.optim.Adam([bern_param] + list(net.parameters()), lr=.004)
    optim = torch.optim.Adam([bern_param], lr=.004)
    optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


    steps = []
    losses5= []
    for step in range(total_steps):

        dist = RelaxedBernoulli(torch.Tensor([1.]), logits=bern_param)

        optim.zero_grad()

        zs = []
        for i in range(20):
            z = dist.rsample()
            zs.append(z)
        # zs = torch.FloatTensor(zs).unsqueeze(1)
        zs = torch.stack(zs)

        logprob = dist.log_prob(zs.detach())
        
        pred = net.net(zs)
        H_z = Hpy(zs)
        f_b = f(H_z)

        loss = torch.mean((f_b-pred.detach()) * logprob + pred )

        loss.backward(retain_graph=True)  
        # loss.backward()  
        optim.step()


        optim_NN.zero_grad()
        # pred = net.net(zs)
        NN_loss = torch.mean((f_b - pred)**2)
        NN_loss.backward()  
        optim_NN.step()


        if step%50 ==0:
            if step %500==0:
                print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
            losses5.append(torch.mean(f_b).numpy())
            steps.append(step)








    # logits = 0
    bern_param = torch.tensor([logits], requires_grad=True)
    val=.49


    print()
    print ('SimpLAX')
    print ('Value:', val)
    print()

    net = NN()


    # print (len(net.parameters()))

    # optim = torch.optim.Adam([bern_param] + list(net.parameters()), lr=.004)
    optim = torch.optim.Adam([bern_param], lr=.004)
    optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


    steps = []
    losses6= []
    for step in range(total_steps):

        dist = RelaxedBernoulli(torch.Tensor([1.]), logits=bern_param)

        optim.zero_grad()

        zs = []
        for i in range(20):
            z = dist.rsample()
            zs.append(z)
        # zs = torch.FloatTensor(zs).unsqueeze(1)
        zs = torch.stack(zs)

        logprob = dist.log_prob(zs.detach())
        
        pred = net.net(zs)
        H_z = Hpy(zs)
        f_b = f(H_z)

        loss = torch.mean((f_b-pred.detach()) * logprob + pred )

        loss.backward(retain_graph=True)  
        # loss.backward()  
        optim.step()


        optim_NN.zero_grad()
        # pred = net.net(zs)
        NN_loss = torch.mean((f_b - pred)**2)
        NN_loss.backward()  
        optim_NN.step()


        if step%50 ==0:
            if step %500==0:
                print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
            losses6.append(torch.mean(f_b).numpy())
            steps.append(step)












    # logits = 0
    bern_param = torch.tensor([logits], requires_grad=True)
    val=.499


    print()
    print ('SimpLAX')
    print ('Value:', val)
    print()

    net = NN()


    # print (len(net.parameters()))

    # optim = torch.optim.Adam([bern_param] + list(net.parameters()), lr=.004)
    optim = torch.optim.Adam([bern_param], lr=.004)
    optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


    steps = []
    losses7= []
    for step in range(total_steps):

        dist = RelaxedBernoulli(torch.Tensor([1.]), logits=bern_param)

        optim.zero_grad()

        zs = []
        for i in range(20):
            z = dist.rsample()
            zs.append(z)
        # zs = torch.FloatTensor(zs).unsqueeze(1)
        zs = torch.stack(zs)

        logprob = dist.log_prob(zs.detach())
        
        pred = net.net(zs)
        H_z = Hpy(zs)
        f_b = f(H_z)

        loss = torch.mean((f_b-pred.detach()) * logprob + pred )

        loss.backward(retain_graph=True)  
        # loss.backward()  
        optim.step()


        optim_NN.zero_grad()
        # pred = net.net(zs)
        NN_loss = torch.mean((f_b - pred)**2)
        NN_loss.backward()  
        optim_NN.step()


        if step%50 ==0:
            if step %500==0:
                print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
            losses7.append(torch.mean(f_b).numpy())
            steps.append(step)




























    # logits = 0
    bern_param = torch.tensor([logits], requires_grad=True)
    val=.4


    print()
    print ('RELAX')
    print ('Value:', val)
    print()

    net = NN()


    # print (len(net.parameters()))

    # optim = torch.optim.Adam([bern_param] + list(net.parameters()), lr=.004)
    optim = torch.optim.Adam([bern_param], lr=.004)
    optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


    steps = []
    losses8 = []
    for step in range(total_steps):

        dist = RelaxedBernoulli(torch.Tensor([1.]), logits=bern_param)
        dist_bern = Bernoulli(logits=bern_param)

        optim.zero_grad()

        zs = []
        for i in range(20):
            z = dist.rsample()
            zs.append(z)
        zs = torch.stack(zs)

        b = Hpy(zs)

        # EHRERE, I think the bug is above, see grad of stuff 

        #p(z|b)
        theta = logit_to_prob(bern_param)
        v = torch.rand(zs.shape[0], zs.shape[1]) 

        # v= (1-b)*v*(1-theta) + b*v*theta+(1-theta)
        # z_tilde = torch.log(theta/(1-theta)) + torch.log(v/(1-v))

        v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
        # z_tilde = bern_param.detach() + torch.log(v_prime) - torch.log1p(-v_prime)
        z_tilde = bern_param + torch.log(v_prime) - torch.log1p(-v_prime)

        z_tilde = torch.sigmoid(z_tilde)
        # z_tilde = torch.tensor(z_tilde, requires_grad=True)

        # print (z_tilde)
        # fadfsa


        # v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
        # z_tilde = bern_param.detach() + torch.log(v_prime) - torch.log1p(-v_prime)
        # z_tilde = torch.sigmoid(z_tilde).detach()
        # z_tilde = torch.tensor(z_tilde, requires_grad=True)

        logprob = dist_bern.log_prob(b)
        
        pred = net.net(z_tilde)

        # logprobgrad = torch.autograd.grad(outputs=torch.mean(pred), inputs=(logits), retain_graph=True)[0]
        # print (logprobgrad.shape, torch.max(logprobgrad), torch.min(logprobgrad))
        # fadaf


        pred2 = net.net(zs)
        f_b = f(b)

        # print (bern_param.detach().numpy())

        loss = torch.mean((f_b-pred.detach()) * logprob - pred + pred2 )


        # logprobgrad_2 = torch.autograd.grad(outputs=torch.mean(logprob), inputs=(bern_param), retain_graph=True)[0]
        # print (logprobgrad_2)

        # logprobgrad_2 = torch.autograd.grad(outputs=torch.mean(pred), inputs=(bern_param), retain_graph=True)[0]
        # print (logprobgrad_2)

        # logprobgrad_2 = torch.autograd.grad(outputs=torch.mean(pred2), inputs=(bern_param), retain_graph=True)[0]
        # print (logprobgrad_2)

        # fdfa

        loss.backward(retain_graph=True)  
        optim.step()

        optim_NN.zero_grad()
        NN_loss = torch.mean((f_b - pred)**2) 
        NN_loss.backward()  
        optim_NN.step()

        if step%50 ==0:
            if step %500==0:
                print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
            losses8.append(torch.mean(f_b).numpy())
            steps.append(step)

        # if step%1 ==0:
        #     if step %1==0:
        #         print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
        #     losses8.append(torch.mean(f_b).numpy())
        #     steps.append(step)











    # logits = 0
    bern_param = torch.tensor([logits], requires_grad=True)
    val=.49


    print()
    print ('RELAX')
    print ('Value:', val)
    print()

    net = NN()


    # print (len(net.parameters()))

    # optim = torch.optim.Adam([bern_param] + list(net.parameters()), lr=.004)
    optim = torch.optim.Adam([bern_param], lr=.004)
    optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


    steps = []
    losses9 = []
    for step in range(total_steps):

        dist = RelaxedBernoulli(torch.Tensor([1.]), logits=bern_param)
        dist_bern = Bernoulli(logits=bern_param)

        optim.zero_grad()

        zs = []
        for i in range(20):
            z = dist.rsample()
            zs.append(z)
        zs = torch.stack(zs)

        b = Hpy(zs)

        #p(z|b)
        theta = logit_to_prob(bern_param)
        v = torch.rand(zs.shape[0], zs.shape[1]) 

        # v= (1-b)*v*(1-theta) + b*v*theta+(1-theta)
        # z_tilde = torch.log(theta/(1-theta)) + torch.log(v/(1-v))

        v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
        # z_tilde = bern_param.detach() + torch.log(v_prime) - torch.log1p(-v_prime)
        z_tilde = bern_param + torch.log(v_prime) - torch.log1p(-v_prime)

        z_tilde = torch.sigmoid(z_tilde)

        logprob = dist_bern.log_prob(b)
        
        pred = net.net(z_tilde)

        pred2 = net.net(zs)
        f_b = f(b)

        loss = torch.mean((f_b-pred.detach()) * logprob - pred + pred2 )

        loss.backward(retain_graph=True)  
        optim.step()

        optim_NN.zero_grad()
        NN_loss = torch.mean((f_b - pred)**2) 
        NN_loss.backward()  
        optim_NN.step()

        if step%50 ==0:
            if step %500==0:
                print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
            losses9.append(torch.mean(f_b).numpy())
            steps.append(step)

















    # logits = 0
    bern_param = torch.tensor([logits], requires_grad=True)
    val=.499


    print()
    print ('RELAX')
    print ('Value:', val)
    print()

    net = NN()


    # print (len(net.parameters()))

    # optim = torch.optim.Adam([bern_param] + list(net.parameters()), lr=.004)
    optim = torch.optim.Adam([bern_param], lr=.004)
    optim_NN = torch.optim.Adam(net.parameters(), lr=.0004)


    steps = []
    losses10 = []
    for step in range(total_steps):

        dist = RelaxedBernoulli(torch.Tensor([1.]), logits=bern_param)
        dist_bern = Bernoulli(logits=bern_param)

        optim.zero_grad()

        zs = []
        for i in range(20):
            z = dist.rsample()
            zs.append(z)
        zs = torch.stack(zs)

        b = Hpy(zs)

        #p(z|b)
        theta = logit_to_prob(bern_param)
        v = torch.rand(zs.shape[0], zs.shape[1]) 

        # v= (1-b)*v*(1-theta) + b*v*theta+(1-theta)
        # z_tilde = torch.log(theta/(1-theta)) + torch.log(v/(1-v))

        v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
        # z_tilde = bern_param.detach() + torch.log(v_prime) - torch.log1p(-v_prime)
        z_tilde = bern_param + torch.log(v_prime) - torch.log1p(-v_prime)

        z_tilde = torch.sigmoid(z_tilde)

        logprob = dist_bern.log_prob(b)
        
        pred = net.net(z_tilde)
        pred2 = net.net(zs)
        f_b = f(b)

        loss = torch.mean((f_b-pred.detach()) * logprob - pred + pred2 )

        loss.backward(retain_graph=True)  
        optim.step()

        optim_NN.zero_grad()
        NN_loss = torch.mean((f_b - pred)**2) 
        NN_loss.backward()  
        optim_NN.step()

        if step%50 ==0:
            if step %500==0:
                print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
            losses10.append(torch.mean(f_b).numpy())
            steps.append(step)






    data_dict = {}

    data_dict['steps'] = steps
    data_dict['losses'] = losses
    data_dict['losses2'] = losses2
    data_dict['losses3'] = losses3
    data_dict['losses4'] = losses4
    data_dict['losses5'] = losses5
    data_dict['losses6'] = losses6
    data_dict['losses7'] = losses7
    data_dict['losses8'] = losses8
    data_dict['losses9'] = losses9
    data_dict['losses10'] = losses10
    data_dict['losses11'] = losses11
    data_dict['losses12'] = losses12


    with open( save_dir+"data.p", "wb" ) as f:
        pickle.dump(data_dict, f)
    print ('saved data')





else:

    with open(save_dir+"data.p", "rb" ) as f:

        data_dict = pickle.load(f)
    print ('loaded data')


    # print(len(favorite_color[0]))

    # fsdfa



    steps = data_dict['steps']
    losses = data_dict['losses']
    losses2 = data_dict['losses2']
    losses3 = data_dict['losses3']
    losses4 = data_dict['losses4']
    losses5 = data_dict['losses5']
    losses6 = data_dict['losses6']
    losses7 = data_dict['losses7']
    losses8 = data_dict['losses8']
    losses9 = data_dict['losses9']
    losses10 = data_dict['losses10']
    losses11 = data_dict['losses11']
    losses12 = data_dict['losses12']

















# print (len(steps))
# print (len(losses))
# print (len(losses2))
# print (len(losses3))

ylabel = 'f(b)'
xlabel = 'Steps'



rows = 1
cols = 3
# text_col_width = cols
fig = plt.figure(figsize=(10+cols,2+rows), facecolor='white') #, dpi=150)
# fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

col =0
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(steps, smooth_list(losses), label='REINFORCE', alpha=.8)
ax.plot(steps, smooth_list(losses4), label='REINFORCE with critic', alpha=.8)
ax.plot(steps, smooth_list(losses8), label='RELAX', alpha=.8)
ax.plot(steps, smooth_list(losses5), label='SimpLAX', alpha=.8)

ax.grid(True, alpha=.3)
ax.text(x=0.7,y=1.01,s='f(0)=.16\nf(1)=.36', size=8, family='serif', transform=ax.transAxes)
ax.text(x=0.3,y=1.03,s=r'$f(b)=(b-.4)^2$', size=8, family='serif', transform=ax.transAxes)
# ax.set_title(r'$f(b)=(b-.4)^2$' + '\nf(0)=.16\nf(1)=.36', size=8, family='serif')
ax.tick_params(labelsize=6)
ax.set_ylabel(ylabel, size=6, family='serif')
ax.set_xlabel(xlabel, size=6, family='serif')
ax.legend(prop={'size':7}) #, loc=2)  #upper left


col =1
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(steps, smooth_list(losses2), label='REINFORCE')
ax.plot(steps, smooth_list(losses11), label='REINFORCE with critic')
ax.plot(steps, smooth_list(losses9), label='RELAX')
ax.plot(steps, smooth_list(losses6), label='SimpLAX')

ax.grid(True, alpha=.3)
ax.text(x=0.7,y=1.01,s='f(0)=.2401\nf(1)=.2601', size=8, family='serif', transform=ax.transAxes)
ax.text(x=0.3,y=1.03,s=r'$f(b)=(b-.49)^2$', size=8, family='serif', transform=ax.transAxes)
# ax.set_title(r'$f(b)=(b-.49)^2$' +  '\n f(0)=.2401\nf(1)=.2601', size=8, family='serif')
ax.tick_params(labelsize=6)
# ax.set_ylabel(ylabel, size=6, family='serif')
ax.set_xlabel(xlabel, size=6, family='serif')
ax.legend(prop={'size':7})

col =2
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(steps, smooth_list(losses3), label='REINFORCE')
ax.plot(steps, smooth_list(losses12), label='REINFORCE with critic')
ax.plot(steps, smooth_list(losses10), label='RELAX')
ax.plot(steps, smooth_list(losses7), label='SimpLAX')

ax.grid(True, alpha=.3)
ax.text(x=0.7,y=1.01,s='f(0)=.249\nf(1)=.251', size=8, family='serif', transform=ax.transAxes)
ax.text(x=0.3,y=1.03,s=r'$f(b)=(b-.499)^2$', size=8, family='serif', transform=ax.transAxes)
# ax.set_title(r'$f(b)=(b-.499)^2$' +'\nf(0)=.249\nf(1)=.251', size=8, family='serif')
ax.tick_params(labelsize=6)
# ax.set_ylabel(ylabel, size=6, family='serif')
ax.set_xlabel(xlabel, size=6, family='serif')
ax.legend(prop={'size':7})














# plt_path = save_dir+'curves_plot_withcritic2.png'
plt_path = save_dir+'curves_plot_withcritic3.pdf'
plt.savefig(plt_path)
print ('saved training plot', plt_path)

plt_path = save_dir+'curves_plot_withcritic3.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)

plt.close()




# data = [losses, losses2, losses3, losses4,
#         losses5, losses6, losses7, losses8,
#         losses9, losses10, ]









