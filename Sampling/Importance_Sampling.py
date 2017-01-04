

#Importance Sampling

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

def proposal_prob(x):
    #square root scale to make var into std
    y = norm.pdf(x, loc=0, scale=1.**(.5))
    return y

def distribution_plot_lists(dist_prob):
    x = np.linspace(-12, 12, 200)
    y = [dist_prob(x_i) for x_i in x]
    return x, y




n_samples = 1000



fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)


samps = []
ws = []

for i in range(n_samples):

    #Sample from the proposal 
    samp = np.random.randn() * 1.**.5 + 0
    
    #Calc sample weight
    q_x = proposal_prob(samp)
    p_x = target_prob(samp)
    w = p_x / q_x

    samps.append(samp)
    ws.append(w)

    if i %100 == 0:
    
        #Clear plot
        plt.cla()

        #Plot sample
        plt.plot([samp,samp], [0,.5])
        ax.set_yticks([])
        ax.set_xticks([])

        x, y = distribution_plot_lists(target_prob)
        ax.plot(x, y, linewidth=2, label="Target")

        x, y = distribution_plot_lists(proposal_prob)
        ax.plot(x, y, linewidth=2, label="Proposal")

        ax.plot([],[],label='Sample '+str(i)+' Weight ' + str(("%.2f" % w)))
        plt.legend(fontsize=6)

        # plt.draw()
        plt.pause(1./100.)
        # plt.pause(5.)



# #Importance Sampling: Resample the samples from a categorical parameterized by their weights
# samps = np.array(samps)
# ws = np.array(ws)

# normalized_ws = ws / np.sum(ws)
# # normalized_ws = np.ones([n_samples]) / n_samples

# importacne_samples_counts = np.random.multinomial(n_samples, normalized_ws)

# # importacne_samples = samps[importacne_samples_indexes]
# importacne_samples = []
# for i in range(n_samples):
#     for j in range(importacne_samples_counts[i]):
#         importacne_samples.append(samps[i])

# # print importacne_samples_indexes.shape
# # print importacne_samples_indexes
# # print samps
# # print importacne_samples


#Plot Importance Weighted samples histogram

plt.cla()
plt.ioff()

x, y = distribution_plot_lists(target_prob)
ax.plot(x, y, linewidth=2, label="Target")

x, y = distribution_plot_lists(proposal_prob)
ax.plot(x, y, linewidth=2, label="Proposal")

ax.hist(samps, bins=200, normed=True, weights=ws, range=[-12,12], alpha=.6, label='Importance, k='+str(len(samps)))

ax.set_yticks([])
ax.set_xticks([])
plt.legend(fontsize=6)
# plt.pause(100.)
plt.show()






















