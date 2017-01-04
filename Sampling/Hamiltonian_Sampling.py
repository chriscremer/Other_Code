

#Hamiltonian/Hybrid Monte Carlo (HMC)

# Produces distant proposals for the Metropolis algorithm

#Need gradient info, its tricky for my MoG posterior. So Im using a N(0,1) posterior 



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def target_prob(x):
    #square root scale to make var into std
    # y = (.3*norm.pdf(x, loc=-2, scale=.5**(.5)) + 
    #     # (.25*norm.pdf(x, loc=-3, scale=.5**(.5))) + 
    #     (.4*norm.pdf(x, loc=1, scale=.5**(.5))) +
    #     (.3*norm.pdf(x, loc=3, scale=.5**(.5))) )

    y = norm.pdf(x, loc=0, scale=1.)

    return y

# def log_target_prob_gradient(x):
    
#     # grad = .3*(x-(-2))/.5**(.5) + .4*(x-(1))/.5**(.5) + .3*(x-(3))/.5**(.5)
#     return grad

def distribution_plot_lists(dist_prob):
    x = np.linspace(-12, 12, 200)
    y = [dist_prob(x_i) for x_i in x]
    return x, y



#HMC methods

def leap_frog_steps(x, v):

    # The gradients are just x and v because of the standard gaussian posterior

    step_size = .1
    n_steps = 10

    #Update velocity/momemtum a half step
    v = v - .5*step_size * x

    #Update the position by a full step
    x = x + step_size * v

    for t in range(n_steps):

        #Update velocity/momemtum a full step
        v = v - step_size * x

        #Update the position by a full step
        x = x + step_size * v

    #Update velocity/momemtum a half step
    v = v - .5*step_size * x

    return x, v


def hamiltonian(x,v):

    # H(q,p) = U(q) + K(p)
    # where K(p) = p^2 / 2
    # where U(q) = -log posterior

    return .5*(x**2) + .5*(v**2)


def hmc(init_x):
    '''
    Returns a HMC sample
    '''

    # Sample momemtum
    init_mom = np.random.randn()

    # Run a number of leapfrog steps
    x, mom = leap_frog_steps(init_x, init_mom)

    # Calc hamiltonian of original state
    hamil_0 = hamiltonian(init_x, init_mom)

    # Calc hamiltonian of new state
    hamil_T = hamiltonian(x, mom)

    # accept with prob min(1, exp(orig-current))
    p_accept = min(1., np.exp(hamil_0 - hamil_T))

    if p_accept > np.random.uniform():
        return x
    else:
        return init_x




n_samples = 1000
display_step = 10


fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)


samps = []
# ws = []

#Init sample
x = 0.

for i in range(n_samples):

    x = hmc(x)
    samps.append(x)


    if i %display_step == 0:
    
        #Clear plot
        plt.cla()
        ax.set_yticks([])
        ax.set_xticks([])

        #Plot sample
        plt.plot([x,x], [0,.3], label='x')

        #Plot target
        x_values, y_values = distribution_plot_lists(target_prob)
        ax.plot(x_values, y_values, linewidth=2, label="Target")

        ax.plot([],[],label='Sample '+str(i))#+' Weight ' + str(("%.2f" % w)))
        plt.legend(fontsize=6)

        plt.pause(1./10.)
        # plt.pause(5.)




#Plot final distribution
plt.cla()
plt.ioff()

x, y = distribution_plot_lists(target_prob)
ax.plot(x, y, linewidth=2, label="Target")

# x, y = distribution_plot_lists(proposal_prob)
# ax.plot(x, y, linewidth=2, label="Proposal")

ax.hist(samps, bins=200, normed=True, range=[-12,12], alpha=.6, label='Hamiltonian, k='+str(len(samps)))

ax.set_yticks([])
ax.set_xticks([])
plt.legend(fontsize=6)
# plt.pause(100.)
plt.show()






















