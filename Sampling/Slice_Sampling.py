

# Slice Sampling

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def target_prob(x):
    #square root scale to make var into std
    y = (.3*norm.pdf(x, loc=-2, scale=.5**(.5)) + 
        # (.25*norm.pdf(x, loc=-3, scale=.5**(.5))) + 
        (.4*norm.pdf(x, loc=1, scale=.5**(.5))) +
        (.3*norm.pdf(x, loc=3, scale=.5**(.5))) )
    return y


def distribution_plot_lists(dist_prob):
    x = np.linspace(-12, 12, 200)
    y = [dist_prob(x_i) for x_i in x]
    return x, y




n_samples = 1000
display_step = 10

step_out_increment = .1



fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)


samps = []
# ws = []

#Init sample
x = 0.

for i in range(n_samples):

    #Get prob of the sample
    # print x
    p_x = target_prob(x)

    # print p_x

    #Sample u from Uniform (0,p(x))
    u = np.random.uniform(low=0., high=p_x)

    #Step out to get interval where u > p(x)
    x_min = x
    x_max = x
    while target_prob(x_min - step_out_increment) > u:
        x_min = x_min - step_out_increment
    while target_prob(x_max + step_out_increment) > u:
        x_max = x_max + step_out_increment  

    #Sample x from interval
    x = np.random.uniform(low=x_min, high=x_max)

    samps.append(x)


    if i %display_step == 0:
    
        #Clear plot
        plt.cla()
        ax.set_yticks([])
        ax.set_xticks([])


        #Plot u
        plt.plot([-12,12], [u,u], label='u')

        #Plot interval
        plt.plot([x_min, x_max], [.1, .1], label='interval')

        #Plot sample
        plt.plot([x,x], [0,.3], label='x')

        x_values, y_values = distribution_plot_lists(target_prob)
        ax.plot(x_values, y_values, linewidth=2, label="Target")


        ax.plot([],[],label='Sample '+str(i))#+' Weight ' + str(("%.2f" % w)))
        plt.legend(fontsize=6)

        # plt.draw()
        plt.pause(1./10.)
        # plt.pause(5.)




#Plot Slice samples histogram


plt.cla()
plt.ioff()

x, y = distribution_plot_lists(target_prob)
ax.plot(x, y, linewidth=2, label="Target")

# x, y = distribution_plot_lists(proposal_prob)
# ax.plot(x, y, linewidth=2, label="Proposal")

ax.hist(samps, bins=200, normed=True, range=[-12,12], alpha=.6, label='Slice, k='+str(len(samps)))

ax.set_yticks([])
ax.set_xticks([])
plt.legend(fontsize=6)
# plt.pause(100.)
plt.show()






















