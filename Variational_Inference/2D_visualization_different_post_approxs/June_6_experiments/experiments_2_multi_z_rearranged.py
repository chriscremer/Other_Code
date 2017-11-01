





import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
from os.path import expanduser
home = expanduser("~")

import tf_true_posteriors as ttp
from plotting_functions import plot_isocontours
from plotting_functions import plot_kde

from variational_q_models_multi_z import Factorized_Gaussian_model as FGM
from variational_q_models_multi_z import IW_model as IWM
from variational_q_models_multi_z import AV_model as AVM
from variational_q_models_multi_z import Norm_Flow_model as NFM
from variational_q_models_multi_z import Hamiltonian_Variational_model as HVM
from variational_q_models_multi_z import Auxiliary_Flow_model as AFM
from variational_q_models_multi_z import Hamiltonian_Flow_model as HFM
from variational_q_models_multi_z import Hamiltonian_Flow_model2 as HFM2
from variational_q_models_multi_z import Hamiltonian_Flow_model3 as HFM3











if __name__ == '__main__':

    # random_seed = 1

    # posteriors = [ttp.log_posterior_0, ttp.log_posterior_1, 
    #                 ttp.log_posterior_2, ttp.log_posterior_3]
    # posterior_names = ['RIWA', 'top_bot', 'vert_hori', 'top_left']

    # posteriors = []
    # posterior_names = ['two_moons', 'wiggles']

    # posteriors = [ttp.log_posterior_0, ttp.log_posterior_1, 
    #                 ttp.log_posterior_2, ttp.log_posterior_3,
    #                     ttp.logprob_two_moons, ttp.logprob_wiggle]
    # posterior_names = ['RIWA', 'top_bot', 'vert_hori', 'top_left', 'two_moons', 'wiggles']
    

    # posteriors = [ttp.log_posterior_0]
    # posterior_names = ['RIWA']


    # posteriors = [ttp.log_posterior_0, ttp.log_posterior_1, ttp.log_posterior_3, ttp.logprob_two_moons]
    # posterior_names = ['RIWA', 'top_bot',  'top_left', 'two_moons']

    posteriors = [ttp.log_posterior_0, ttp.logprob_two_moons]
    posterior_names = ['RIWA', 'two_moons']

    # posteriors = [ttp.log_posterior_3]
    # posterior_names = ['top_left']

    # models = [FGM, IWM, AVM, NFM, HVM, AFM, HFM]
    # model_names = ['FG', 'IW', 'AV', 'NF', 'HV', 'AF', 'HF']

    models = [FGM, AFM]
    model_names = ['FG','AF']

    # models = [FGM]
    # model_names = ['FG']

    # models = [AFM]
    # model_names = ['AF']

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







    # train_ = 0
    
    # alpha=.2
    # rows = len(posteriors)
    # columns = len(models) +1 #+1 for posteriors

    # fig = plt.figure(figsize=(6+columns,2+rows), facecolor='white')

    # for p_i in range(len(posteriors)):

    #     print '\nPosterior', p_i, posterior_names[p_i]

    #     posterior = ttp.posterior_class(posteriors[p_i])
    #     ax = plt.subplot2grid((rows,columns), (p_i,0), frameon=False)#, colspan=3)
    #     plot_isocontours(ax, posterior.run_log_post, cmap='Blues')
    #     if p_i == 0: ax.annotate('Posterior', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

    #     for q_i in range(len(models)):

    #         save_to = home+'/Documents/tmp/' + posterior_names[p_i]+ '_'+ model_names[q_i]+'.ckpt'
    #         # print os.path.isfile(save_to)
    #         # fasd
    #         if os.path.isfile(save_to+'.meta'): 
    #             load_from = home+'/Documents/tmp/' + posterior_names[p_i]+ '_'+ model_names[q_i]+'.ckpt'
    #         else:
    #             load_from = ''

    #         print model_names[q_i]
    #         ax = plt.subplot2grid((rows,columns), (p_i,q_i+1), frameon=False)#, colspan=3)
    #         model = models[q_i](posteriors[p_i])
            
    #         if train_:            
    #             model.train(100000, save_to=save_to, load_from=load_from)

    #         model.load_params(load_from)
    #         samps = model.sample(100)
    #         plot_kde(ax, samps, cmap='Blues')
    #         plot_isocontours(ax, posterior.run_log_post, cmap='Greys', alpha=alpha)
    #         if p_i == 0: ax.annotate(model_names[q_i], xytext=(.38, 1.1), xy=(0, 1), textcoords='axes fraction')

    # # plt.show()
    # plt.savefig(home+'/Documents/tmp/plots.png')
    # print 'saved'










    train_ = 0
    
    alpha=.2
    rows = len(models) +1 #+1 for posteriors 
    columns = len(posteriors)  +1 # for annotation

    fig = plt.figure(figsize=(2+columns,4+rows), facecolor='white')




    #annotate
    ax = plt.subplot2grid((rows,columns), (0,0), frameon=False)
    ax.annotate('True', xytext=(.5, .5), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Black', size='large')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')


    ax = plt.subplot2grid((rows,columns), (1,0), frameon=False)
    ax.annotate('FFG', xytext=(.5, .5), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Black', size='large')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')

    ax = plt.subplot2grid((rows,columns), (2,0), frameon=False)
    ax.annotate('Flow', xytext=(.5, .5), xy=(.5, .5), textcoords='axes fraction', family='serif', color='Black', size='large')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')






    for p_i in range(len(posteriors)):

        print '\nPosterior', p_i, posterior_names[p_i]

        posterior = ttp.posterior_class(posteriors[p_i])
        ax = plt.subplot2grid((rows,columns), (0,p_i+1), frameon=False)#, colspan=3)
        plot_isocontours(ax, posterior.run_log_post, cmap='Blues')
        # if p_i == 0: ax.annotate('Posterior', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')



        for q_i in range(len(models)):



            save_to = home+'/Documents/tmp/' + posterior_names[p_i]+ '_'+ model_names[q_i]+'.ckpt'
            # print os.path.isfile(save_to)
            # fasd
            if os.path.isfile(save_to+'.meta'): 
                load_from = home+'/Documents/tmp/' + posterior_names[p_i]+ '_'+ model_names[q_i]+'.ckpt'
            else:
                load_from = ''

            print model_names[q_i]
            ax = plt.subplot2grid((rows,columns), (q_i+1,p_i+1), frameon=False)#, colspan=3)
            model = models[q_i](posteriors[p_i])
            
            if train_:            
                model.train(100000, save_to=save_to, load_from=load_from)

            model.load_params(load_from)
            samps = model.sample(100)
            plot_kde(ax, samps, cmap='Blues')
            plot_isocontours(ax, posterior.run_log_post, cmap='Greys', alpha=alpha)
            # if p_i == 0: ax.annotate(model_names[q_i], xytext=(.38, 1.1), xy=(0, 1), textcoords='axes fraction')

    # plt.show()
    plt.savefig(home+'/Documents/tmp/plots.png')
    print 'saved'
    plt.savefig(home+'/Documents/tmp/plots.eps')
    print 'saved'
    plt.savefig(home+'/Documents/tmp/plots.pdf')
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














