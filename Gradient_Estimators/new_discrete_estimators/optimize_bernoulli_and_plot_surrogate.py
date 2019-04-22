
import numpy as np
import os
import subprocess
from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import sys, os
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('./VAE'))

import pickle
import imageio


import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

# from NN import NN
# from NN_forrelax import NN as NN2

from NN2 import NN3 as NN


def to_print_mean(x):
    return torch.mean(x).data.cpu().numpy()
def to_print(x):
    return x.data.cpu().numpy()


# def H(x):
#     if x > .5:
#         return 1
#     else:
#         return 0

def H(x):
    # if x > .5:
    #     return torch.tensor([1])
    # else:
    #     return torch.tensor([0])

    return (x > 0.).float()


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


def numb_to_string(numb, str_length=5):

    string = str(numb)
    while len(string) < str_length:
        string = '0'+string
    return string


def sample_Gumbel(probs):

    u = torch.rand(probs.shape).clamp(1e-10, 1.-1e-10).float()
    z = torch.log(probs) - torch.log(1.-probs) + torch.log(u) - torch.log(1.-u)
    return z


def sample_conditional_Gumbel(probs, b):

    v = torch.rand(probs.shape).clamp(1e-10, 1.-1e-10).float()
    if b==0:
        return -torch.log( v/(1-v) * 1/probs +1)
    else:
        return torch.log( v/(1-v) * 1/(1-probs) +1)







def save_plot(step, save_dir):


    #PLOT


    rows = 1
    cols = 1
    # text_col_width = cols
    fig = plt.figure(figsize=(8+cols,2+rows), facecolor='white') #, dpi=150)
    # fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

    col =0
    row = 0

    z = torch.tensor(np.linspace(-4,4,300)).view(300,1).float()

    f_H_z = f(H(z))
    cz = surrogate.net(z)


    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.plot(to_print(z), to_print(f_H_z), label='f(H(z))', alpha=.8)
    ax.plot(to_print(z), to_print(cz), label='c(z)', alpha=.8)

    ax.grid(True, alpha=.3)
    # ax.text(x=0.7,y=1.01,s='f(0)=.16\nf(1)=.36', size=8, family='serif', transform=ax.transAxes)
    # ax.text(x=0.3,y=1.03,s=r'$f(b)=(b-.4)^2$', size=8, family='serif', transform=ax.transAxes)
    # ax.set_title(r'$f(b)=(b-.4)^2$' + '\nf(0)=.16\nf(1)=.36', size=8, family='serif')
    ax.tick_params(labelsize=6)
    # ax.set_ylabel(ylabel, size=6, family='serif')
    # ax.set_xlabel(xlabel, size=6, family='serif')
    ax.legend(prop={'size':7}) #, loc=2)  #upper left
    # ax.set_ylim(.1,.4) for .4
    ax.set_ylim(.235,.265) #for .49
    ax.text(x=0.1,y=1.0,s=numb_to_string(step), size=8, family='serif', transform=ax.transAxes)




    # row=1
    # ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)
    # ax.hist(zs[:100])
    # ax.set_xlim(-4,4)



    plt_path = save_dir+'plot'+numb_to_string(step)+'.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)

    plt.close()



































if __name__ == "__main__":

    # n= 10000
    # theta = .5
    val=.49
    f = lambda x: (x-val)**2

    total_steps = 20000
    exp_name = 'surr_plot_8'


    print ('Exp:', exp_name)

    save_dir = home+'/Documents/Grad_Estimators/new/'
    exp_dir = save_dir + exp_name + '/'
    # params_dir = exp_dir + 'params/'
    images_dir = exp_dir + 'images/'
    code_dir = exp_dir + 'code/'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print ('Made dir', exp_dir) 

    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
        print ('Made dir', code_dir) 

    # if not os.path.exists(params_dir):
    #     os.makedirs(params_dir)
    #     print ('Made dir', params_dir) 

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print ('Made dir', images_dir) 


    #Save code
    subprocess.call("(rsync -r --exclude=__pycache__/ . "+code_dir+" )", shell=True)




    train_ =1  # if 0, it loads data

    if train_:

        logits = torch.log(torch.tensor(np.array([.5]))).float()
        logits.requires_grad_(True)
        

        print()
        print ('RELAX')
        print ('Value:', val)
        print()

        # net = NN()
        surrogate = NN(input_size=1, output_size=1, n_residual_blocks=2)
        optim = torch.optim.Adam([logits], lr=.004)
        optim_NN = torch.optim.Adam(surrogate.parameters(), lr=.00005)


        steps = []
        losses10 = []
        zs = []
        for step in range(total_steps+1):

            # batch=[]

            optim_NN.zero_grad()

            losses = 0
            for ii in range(10):
                #Sample p(z)
                z = sample_Gumbel(probs=torch.exp(logits))

                b = H(z)

                #Sample p(z|b)
                z_tilde = sample_conditional_Gumbel(probs=torch.exp(logits), b=b)

                dist_bern = Bernoulli(logits=logits)
                logpb = dist_bern.log_prob(b)

                f_b = f(b)
                pred = surrogate.net(z)
                # print (z)

                NN_loss = torch.mean((f_b - pred)**2) 

                losses+=NN_loss

            losses.backward()  
            optim_NN.step()


            zs.append(to_print(z)[0])


            if step%50 ==0:
                if step %500==0:
                #     print (step, torch.mean(f_b).numpy(), bern_param.detach().numpy(), logit_to_prob(bern_param).detach().numpy(), NN_loss.detach().numpy())
                # losses10.append(torch.mean(f_b).numpy())
                # steps.append(step)
                    print(step, NN_loss)

                    save_plot(step=step, save_dir=images_dir)









    # MAKE GIF
    print ('Making gif')
    images = []
    image_names = os.listdir(images_dir)
    # print (image_names)
    image_names.sort()
    # print (image_names)

    # print (os.listdir(images_dir))

    for i in range(len(image_names)):
    # for i in range(max_epoch+1):
    # for step in range(total_steps+1):
        # print(file_)
        # fsdfa

        # images.append(imageio.imread(dir_+'/'+file_))
        images.append(imageio.imread(images_dir+image_names[i]))

    #hold the last frame a bit longer
    for j in range(5):
        images.append(imageio.imread(images_dir+image_names[i]))


    kargs = { 'duration': .4 }
    imageio.mimsave(images_dir+'movie.gif', images, **kargs)
    # imageio.mimsave(exportname, frames, 'GIF', **kargs)
    print ('made gif', images_dir+'movie.gif')

    fasad



print ('Done.')





