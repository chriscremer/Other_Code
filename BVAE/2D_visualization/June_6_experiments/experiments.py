
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
home = expanduser("~")

import tf_true_posteriors as ttp
from plotting_functions import plot_isocontours
from plotting_functions import plot_kde

from variational_q_models import Factorized_Gaussian_model as FGM
from variational_q_models import IW_model as IWM


# from scipy.stats import norm









if __name__ == '__main__':

    random_seed = 1

    posteriors = [ttp.log_posterior_0, ttp.log_posterior_1, 
                    ttp.log_posterior_2, ttp.log_posterior_3]


    fig = plt.figure(figsize=(8,8), facecolor='white')

    alpha=.2

    for p_i in range(len(posteriors)):

        print '\nPosterior', p_i

        posterior = ttp.posterior_class(posteriors[p_i])
        ax = plt.subplot2grid((4,4), (p_i,0), frameon=False)#, colspan=3)
        plot_isocontours(ax, posterior.run_log_post, cmap='Blues')

        # proposal = ttp.posterior_class(ttp.log_proposal)
        # ax1 = plt.subplot2grid((4,4), (p_i,1), frameon=False)#, colspan=3)
        # plot_isocontours(ax1, proposal.run_log_post, cmap='Reds')

        print 'FG'
        ax1 = plt.subplot2grid((4,4), (p_i,1), frameon=False)#, colspan=3)
        model = FGM(posteriors[p_i])
        model.train(10000)
        samps = model.sample(1000)
        plot_kde(ax1, samps, cmap='Reds')
        plot_isocontours(ax1, posterior.run_log_post, cmap='Blues', alpha=alpha)

        print 'IW'
        ax2 = plt.subplot2grid((4,4), (p_i,2), frameon=False)#, colspan=3)
        model = IWM(posteriors[p_i])
        model.train(10000)
        samps = model.sample(1000)
        plot_kde(ax2, samps, cmap='Reds')
        plot_isocontours(ax2, posterior.run_log_post, cmap='Blues', alpha=alpha)

        if p_i == 0:
            ax.annotate('Posterior', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')
            ax1.annotate('FG', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')
            ax2.annotate('IW', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')


    plt.show()























