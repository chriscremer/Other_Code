

# Metropolis Sampling

#Metropolis-Hastings is used when the proposal is not symmetric


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

def proposal_prob(x, mean):
    #square root scale to make var into std
    y = norm.pdf(x, loc=mean, scale=1.**(.5))
    return y

def distribution_plot_lists(dist_prob):
    x = np.linspace(-12, 12, 200)
    y = [dist_prob(x_i) for x_i in x]
    return x, y




n_samples = 1000
display_step = 10



fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)


samps = []
# ws = []
mean = 0
prev_samp = 0

for i in range(n_samples):

    #Sample from the proposal 
    samp = np.random.randn() * 1.**.5 + mean
    # samp = samp[0]
    
    #Calc sample weight
    # q_x = proposal_prob(samp)
    p_x = target_prob(samp)
    p_x_prev = target_prob(prev_samp)

    ratio = p_x / p_x_prev

    if ratio >= 1.:
        #accept
        samps.append(samp)
        mean = samp
        prev_samp = samp
        acc_rej = 'Accept'
        # print i, ratio
    else:
        r = np.random.rand()

        #accept
        if ratio > r:
            samps.append(samp)
            mean = samp
            prev_samp = samp
            acc_rej = 'Accept'
        #reject
        else:
            samps.append(prev_samp)
            acc_rej = 'Reject'
            # mean = prev_samp

        # print i, ratio, r, acc_rej


    # samps.append(samp)
    # ws.append(w)

    if i %display_step == 0:
    
        #Clear plot
        plt.cla()

        #Plot sample
        plt.plot([samp,samp], [0,.5])
        ax.set_yticks([])
        ax.set_xticks([])

        x, y = distribution_plot_lists(target_prob)
        ax.plot(x, y, linewidth=2, label="Target")

        proposal_prob2 = lambda x: proposal_prob(x, mean)

        x, y = distribution_plot_lists(proposal_prob2)
        ax.plot(x, y, linewidth=2, label="Proposal")

        ax.plot([],[],label='Sample '+str(i) +' '+ acc_rej)
        plt.legend(fontsize=6)

        # plt.draw()
        plt.pause(1./10.)
        # plt.pause(5.)



#Plot Metropolis samples histogram


plt.cla()
plt.ioff()

x, y = distribution_plot_lists(target_prob)
ax.plot(x, y, linewidth=2, label="Target")

# x, y = distribution_plot_lists(proposal_prob)
# ax.plot(x, y, linewidth=2, label="Proposal")

ax.hist(samps, bins=200, normed=True, range=[-12,12], alpha=.6, label='Metropolis, k='+str(len(samps)))

ax.set_yticks([])
ax.set_xticks([])
plt.legend(fontsize=6)
# plt.pause(100.)
plt.show()






















