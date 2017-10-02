


import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from os.path import expanduser
home = expanduser("~")

import tf_true_posteriors as ttp
from plotting_functions import plot_isocontours
from plotting_functions import plot_kde
from variational_q_models import Factorized_Gaussian_model as FGM
from variational_q_models import IW_model as IWM
from variational_q_models import AV_model as AVM
from variational_q_models import Norm_Flow_model as NFM
from variational_q_models import Hamiltonian_Variational_model as HVM
from variational_q_models import Auxiliary_Flow_model as AFM
from variational_q_models import Hamiltonian_Flow_model as HFM
from variational_q_models import Hamiltonian_Flow_model2 as HFM2
from variational_q_models import Hamiltonian_Flow_model3 as HFM3

from variational_q_models import Hamiltonian_Flow_model4 as HFM4
from variational_q_models import Norm_Flow_model4 as NFM4
from variational_q_models import Hamiltonian_Flow_model5 as HFM5


from variational_q_models import New_NormFlow as NNF



from variational_q_models import HNF as HNF













if __name__ == '__main__':

    # random_seed = 1

    # posteriors = [ttp.log_posterior_0, ttp.log_posterior_1, 
    #                 ttp.log_posterior_2, ttp.log_posterior_3]
    # posterior_names = ['RIWA', 'top_bot', 'vert_hori', 'top_left']

    # posteriors = []
    # posterior_names = ['two_moons', 'wiggles']


    #ALL
    # posteriors = [ttp.log_posterior_0, ttp.log_posterior_1, 
    #                 ttp.log_posterior_2, ttp.log_posterior_3,
    #                     ttp.logprob_two_moons, ttp.logprob_wiggle]
    # posterior_names = ['RIWA', 'top_bot', 'vert_hori', 'top_left', 'two_moons', 'wiggles']





    # posteriors = [ttp.log_posterior_0, ttp.logprob_two_moons, ttp.logprob_wiggle]
    # posterior_names = ['RIWA', 'two_moons', 'wiggles']

    # posteriors = [ttp.logprob_wiggle]
    # posterior_names = ['wiggles']


    posteriors = [ttp.log_posterior_0]
    posterior_names = ['RIWA']

    # posteriors = [ttp.log_posterior_0, ttp.log_posterior_1, ttp.log_posterior_3, ttp.logprob_two_moons]
    # posterior_names = ['RIWA', 'top_bot',  'top_left', 'two_moons']

    # posteriors = [ttp.log_posterior_3, ttp.logprob_two_moons]
    # posterior_names = ['top_left', 'two_moons']

    # posteriors = [ttp.log_posterior_3]
    # posterior_names = ['top_left']

    #ALL
    # models = [FGM, IWM, AVM, NFM, HVM, AFM, HFM]
    # model_names = ['FG', 'IW', 'AV', 'NF', 'HV', 'AF', 'HF']

    # models = [FGM, IWM, NFM, HVM, HFM]
    # model_names = ['FFG', 'IW', 'NF', 'HVI',  'HF']

    # models = [HFM4]
    # model_names = ['HFM4']

    # models = [NFM4]
    # model_names = ['NFM4']

    # models = [HFM5]
    # model_names = ['HFM5']



    models = [NNF]
    model_names = ['NNF']

    # models = [HVM]
    # model_names = ['HVM']

    # models = [NFM, HVM, HFM]
    # model_names = ['NF', 'HVI', 'HF']

    # models = [HNF]
    # model_names = ['HNF']

    # models = [NFM]
    # model_names = ['NF']

    # models = [AVM, NFM, AFM]
    # model_names = ['AV', 'NF', 'AF']

    # models = [AVM]
    # model_names = ['AV']

    # models = [HFM, HFM2, HFM3]
    # model_names = ['HF','HF2','HF3']

    # models = [HFM2]
    # model_names = ['HF2']
    # models = []
    # model_names = []
    
    alpha=.2
    rows = len(posteriors)
    columns = len(models) +1 #+1 for posteriors

    n_flows_range = range(1,7,2)
    columns += len(n_flows_range)

    fig = plt.figure(figsize=(6+columns,2+rows), facecolor='white')

    for p_i in range(len(posteriors)):

        print '\nPosterior', p_i, posterior_names[p_i]

        posterior = ttp.posterior_class(posteriors[p_i])
        ax = plt.subplot2grid((rows,columns), (p_i,0), frameon=False)#, colspan=3)
        plot_isocontours(ax, posterior.run_log_post, cmap='Blues')
        if p_i == 0: ax.annotate('Posterior', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

        for q_i in range(len(models)):

            for i in range(len(n_flows_range)):

                print model_names[q_i], n_flows_range[i]
                ax = plt.subplot2grid((rows,columns), (p_i,q_i+1+i), frameon=False)#, colspan=3)
                model = models[q_i](posteriors[p_i], n_flows=n_flows_range[i])
                # print 'train'
                # model.train(10000, save_to=home+'/Documents/tmp/vars.ckpt')
                model.train(20000, save_to='')
                #to debug
                # model.train(1, save_to='')

                samps = model.sample(1000)
                plot_kde(ax, samps, cmap='Blues')
                plot_isocontours(ax, posterior.run_log_post, cmap='Greys', alpha=alpha)
                if p_i == 0: ax.annotate(model_names[q_i]+str(n_flows_range[i]), xytext=(.38, 1.1), xy=(0, 1), textcoords='axes fraction')

    # plt.show()
    plt.savefig(home+'/Documents/tmp/plots_'+model_names[0]+'.png')
    print 'saved'









# if __name__ == '__main__':

#     # random_seed = 1

#     posteriors = [ttp.log_posterior_0, ttp.log_posterior_1, 
#                     ttp.log_posterior_2, ttp.log_posterior_3]

#     models = []

#     fig = plt.figure(figsize=(8,8), facecolor='white')

#     alpha=.2
#     rows = len(posteriors)
#     columns = len(models)

#     for p_i in range(len(posteriors)):

#         print '\nPosterior', p_i

#         posterior = ttp.posterior_class(posteriors[p_i])
#         ax = plt.subplot2grid((rows,columns), (p_i,0), frameon=False)#, colspan=3)
#         plot_isocontours(ax, posterior.run_log_post, cmap='Blues')

#         print 'FG'
#         ax1 = plt.subplot2grid((rows,columns), (p_i,1), frameon=False)#, colspan=3)
#         # model = FGM(posteriors[p_i])
#         # model.train(10000)
#         # samps = model.sample(1000)
#         # plot_kde(ax1, samps, cmap='Reds')
#         # plot_isocontours(ax1, posterior.run_log_post, cmap='Blues', alpha=alpha)

#         print 'IW'
#         ax2 = plt.subplot2grid((rows,columns), (p_i,2), frameon=False)#, colspan=3)
#         # model = IWM(posteriors[p_i])
#         # model.train(10000)
#         # samps = model.sample(1000)
#         # plot_kde(ax2, samps, cmap='Reds')
#         # plot_isocontours(ax2, posterior.run_log_post, cmap='Blues', alpha=alpha)

#         print 'AV'
#         ax3 = plt.subplot2grid((rows,columns), (p_i,3), frameon=False)#, colspan=3)
#         # model = AVM(posteriors[p_i])
#         # model.train(10000)
#         # samps = model.sample(1000)
#         # plot_kde(ax3, samps, cmap='Reds')
#         # plot_isocontours(ax3, posterior.run_log_post, cmap='Blues', alpha=alpha)

#         print 'NF'
#         ax4 = plt.subplot2grid((rows,columns), (p_i,4), frameon=False)#, colspan=3)
#         # model = NFM(posteriors[p_i])
#         # model.train(10000)
#         # samps = model.sample(1000)
#         # plot_kde(ax4, samps, cmap='Reds')
#         # plot_isocontours(ax4, posterior.run_log_post, cmap='Blues', alpha=alpha)

#         print 'HV'
#         ax5 = plt.subplot2grid((rows,columns), (p_i,5), frameon=False)#, colspan=3)
#         model = HVM(posteriors[p_i])
#         model.train(10000)
#         samps = model.sample(1000)
#         plot_kde(ax5, samps, cmap='Reds')
#         plot_isocontours(ax5, posterior.run_log_post, cmap='Blues', alpha=alpha)

#         if p_i == 0:
#             ax.annotate('Posterior', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')
#             ax1.annotate('FG', xytext=(.38, 1.1), xy=(0, 1), textcoords='axes fraction')
#             ax2.annotate('IW', xytext=(.38, 1.1), xy=(0, 1), textcoords='axes fraction')
#             ax3.annotate('AV', xytext=(.38, 1.1), xy=(0, 1), textcoords='axes fraction')
#             ax4.annotate('NF', xytext=(.38, 1.1), xy=(0, 1), textcoords='axes fraction')
#             ax5.annotate('HV', xytext=(.38, 1.1), xy=(0, 1), textcoords='axes fraction')

    # plt.show()














