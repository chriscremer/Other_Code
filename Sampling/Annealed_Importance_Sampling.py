


# Annealed Importacne Sampling (AIS)

# using Metropolis transitions

# TODO: plot the intermediate distributions


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

def proposal_prob2(x, mean):
    #square root scale to make var into std
    y = norm.pdf(x, loc=mean, scale=1.**(.5))
    return y

def distribution_plot_lists(dist_prob):
    x = np.linspace(-12, 12, 200)
    y = [dist_prob(x_i) for x_i in x]
    return x, y

def intermediate_dist(x, beta, mean):

    #geometric average of prior and posterior

    prior = proposal_prob2(x, mean)
    posterior = target_prob(x)

    intermediate = prior**(1-beta) * posterior**(beta)

    return intermediate


def intermediate_dist2(x, beta):

    #geometric average of prior and posterior

    prior = proposal_prob(x)
    posterior = target_prob(x)

    intermediate = prior**(1-beta) * posterior**(beta)

    return intermediate




n_samples = 1000

n_transitions = 50

betas = [x/float(n_transitions) for x in range(n_transitions)]


fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)


samps = []
ws = []

prev_samp = 0.
# mean = 0.

#One sample for each annealing chain
for i in range(n_samples):

    # Annealing chain
    for t in range(n_transitions):

        # - Start by sampling the prior
        # - Then make a Metropolis transition BUT in this case,
        # the target distribution isnt the same for the first 
        # and second sample. The target starts as the prior then 
        # becomes a geometric average of the prior and the actual 
        # target (psoterior). So need to compute the ratio of the 
        # prob of intermediate distribution. So if it doesnt 
        # in probability, I might reject it and use the previous 
        # one. 
        # - Once the annealed Metropolis chain is complete, I'll 
        # have a sequence of samples. I only care about the last
        # one. 
        # However, to compute the weight of that sample, I need 
        # to look back at the ratio of the probs of the 
        # intermediate distributions. see ais paper for equation. 
        # - In the end Ill have samples and weights, then just do
        # importance sampling. The advantage is that I can get to 
        # other modes to do the annealing. 

        #If transitions is 0/1 then it should just be IS

        if t==0:
            #Sample from the initial proposal 
            samp = np.random.randn() * 1.**.5 + prev_samp
            w = 1 #since Z of normal is 1
            # mean = samp

            prev_samp = samp

        else:

            #Metropolis sample, where the mean is the previous sample
            samp = np.random.randn() * 1.**.5 + prev_samp

            p_x_t = intermediate_dist2(samp, betas[t])
            p_prevx_tminus1 = intermediate_dist2(prev_samp, betas[t-1])

            ratio = p_x_t / p_prevx_tminus1

            if ratio >= 1.:
                #accept
                prev_samp = samp
            else:
                r = np.random.rand()
                if ratio > r:
                    #accept
                    prev_samp = samp
                else:
                    #reject
                    prev_samp = prev_samp # this does nothing


            w = w * intermediate_dist2(prev_samp, betas[t]) / intermediate_dist2(prev_samp, betas[t-1])


    samps.append(prev_samp)
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

        # x, y = distribution_plot_lists(proposal_prob)
        # ax.plot(x, y, linewidth=2, label="Proposal")

        ax.plot([],[],label='Sample '+str(i)+' Weight ' + str(("%.2f" % w)))
        plt.legend(fontsize=6)

        # plt.draw()
        plt.pause(1./100.)
        # plt.pause(5.)




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






















