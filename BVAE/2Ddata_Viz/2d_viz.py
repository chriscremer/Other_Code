

#start by making plot - done
# then smaples from plot - done
# then train vae on smaples
# show p(x)



import numpy as np

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

from os.path import expanduser
home = expanduser("~")


import tf_true_posteriors as ttp

from plotting_functions import plot_isocontours
from plotting_functions import plot_isocontours_mog
from plotting_functions import plot_scatter
from plotting_functions import return_z_values


import sys
sys.path.insert(0, './BVAE_11jul2017')

from BVAE_2 import BVAE


file_type = '.png' #'.eps'






n_samps = 5
n_test_samps = 100
n_z_samps = 50


x_size = 2   #f_height=28f_width=28
z_size = 2
epochs = 500000 / (n_samps/5)
n_batch = 5
display_step = epochs / 10 # / 25
lr = .001
S_training = 1  #number of weight samples
k_training = 1 #number of z samples
lmba = 0. #l2 weight on the encoder
# n_qz_transformations = 0
# list_of_decoders = ['BNN', 'MNF']
# decoder = 'BNN'
# decoder = 'BNN'
ga = 'hypo_net'
# ga = 'none'
# list_of_ga = ['none', 'hypo_net']
# Test settings
S_evaluation = 5 #2  
k_evaluation = 100 #500
n_batch_eval = 1 #2

decoder_type = 'bayesian'


if decoder_type == 'no_reg':
    qW_weight = .00000000001
    pW_weight = .00000000001
elif decoder_type == 'weight_decay':
    qW_weight = .00000000001
    pW_weight = 1.
elif decoder_type == 'bayesian':
    qW_weight = 1.
    pW_weight = 1.

hyperparams = {
    'learning_rate': lr,
    'x_size': x_size,
    'z_size': z_size,
    'encoder_hidden_layers': [20],
    'decoder_hidden_layers': [20],
    'likelihood_distribution': 'Gaussian', #'Bernoulli',
    'ga': ga, #'hypo_net',# 'none',# # #,
    'n_qz_transformations': 3,
    'decoder': 'BNN', #'MNF', ##'BNN_sparse', #   #'BNN', # #
    'n_W_particles': S_training,
    'n_z_particles': k_training,
    'qW_weight': qW_weight,
    'pW_weight': pW_weight,
    'lmba': lmba,
    'logprob_type': 2, #ignores multiple n_W
    'scale_log_probs': True
    }


param_file = home+'/Documents/tmp/vars_GA_2.ckpt'
# param_file = home+'/Documents/tmp/vars2.ckpt'



# #Just basic Gaussian
# dist = ttp.G_class()
# samps = np.array([dist.run_sample_post() for x in range(n_samps)])
# print samps.shape

rs = np.random.RandomState(0)

p1 = ttp.G_class2([-2.,2.], [-2., 1.5])
p2 = ttp.G_class2([2.,-1.], [1.5, -2.])
dist = ttp.MoG_class([p1,p2], [.4,.6])
samps = np.array([dist.run_sample_post(rs) for x in range(n_samps)])
print samps.shape



# # #Train VAE on samples 
model = BVAE(hyperparams)


# values, labels, valid_values, valid_labels = model.train2(train_x=samps, valid_x=[],
#             epochs=epochs, batch_size=n_batch,
#             display_step=display_step,
#             path_to_load_variables='',
#             # path_to_load_variables=param_file,
#             path_to_save_variables=param_file)



#EVAL
test_samps = np.array([dist.run_sample_post(rs) for x in range(n_test_samps)])
print test_samps.shape

test_results, train_results, test_labels, train_labels = model.eval(data=test_samps,
            batch_size=n_batch,
            display_step=display_step,
            path_to_load_variables=param_file,
            data2=samps)


model.init_params(param_file)
print 'Decoder entropy', model.decoder_entropy()


fadfas

fig1 = 1
fig2 = 1
fig3 = 1
fig4 = 1
fig5 = 1
fig6 = 1
fig7 = 1




#PLOT

prior_samp_max = 1.
prior_samp_min = -1.


#This shows relative intensity of the distributions
# plt.title('Relative intensity of distributions')

# model = BVAE(hyperparams)
# model.init_params(param_file)


nx = ny = 5
x_values = np.linspace(prior_samp_min, prior_samp_max, nx)
y_values = np.linspace(prior_samp_min, prior_samp_max, ny)

numticks = 50
# numticks = 101


var_over_n_samples = 10









if fig1:

    print 'Fig 1'


    # disrtibutions = [1]



    # posteriors = [ttp.log_posterior_0, ttp.log_posterior_1, 
    #                 ttp.log_posterior_2, ttp.log_posterior_3,
    #                     ttp.logprob_two_moons, ttp.logprob_wiggle]
    # posterior_names = ['RIWA', 'top_bot', 'vert_hori', 'top_left', 'two_moons', 'wiggles']

    # models = []

    # alpha=.2
    # rows = len(disrtibutions)
    if n_samps ==5:

        rows = 3
        columns = 5 # one for dist, 1 for samples #len(models) +1 #+1 for posteriors


        fig = plt.figure(figsize=(5+columns,3+rows), facecolor='white')

        # for p_i in range(len(disrtibutions)):

        #######################
        #ROW 1

        p_i = 0

        ######
        #COL 1
        ax = plt.subplot2grid((rows,columns), (p_i,0), frameon=False)#, colspan=3)
        #Plot distribution


        plot_isocontours(ax, dist.run_log_post, cmap='Blues')
        if p_i == 0: ax.annotate('p(X)', xytext=(.4, 1.1), xy=(0, 1), textcoords='axes fraction')


        ######
        #COL 2
        #Plot samples from distribution
        ax = plt.subplot2grid((rows,columns), (p_i,1), frameon=False)#, colspan=3)


        if p_i == 0: ax.annotate('Dataset Samples', xytext=(.2, 1.1), xy=(0, 1), textcoords='axes fraction')
        plot_isocontours(ax, dist.run_log_post, cmap='Blues')

        plot_scatter(ax, samps, color='blue')








        ######
        #COL 3
        ax = plt.subplot2grid((rows,columns), (p_i,2), frameon=False)#, colspan=3)
        if p_i == 0: ax.annotate('VAE p(x)', xytext=(.4, 1.1), xy=(0, 1), textcoords='axes fraction')

        # # #Train VAE on samples 
        # model = BVAE(hyperparams)
        # values, labels, valid_values, valid_labels = model.train2(train_x=samps, valid_x=[],
        #             epochs=epochs, batch_size=n_batch,
        #             display_step=display_step,
        #             path_to_load_variables='',
        #             # path_to_load_variables=parameter_path+to_load_parameter_file,
        #             path_to_save_variables=param_file)


        model.init_params(param_file)
        means, logvars = model.sample_prior(param_file, n_samps=n_z_samps)

        # print means
        # print 
        # print logvars




        plot_isocontours_mog(ax, ttp.G_class2, means, logvars, cmap='Reds')
        plot_scatter(ax, samps, color='blue', alpha=.5, s=5)





        ######
        #COL 4
        ax = plt.subplot2grid((rows,columns), (p_i,3), frameon=False)#, colspan=3)
        # dist = ttp.G_class()
        plot_isocontours(ax, dist.run_log_post, cmap='Blues')
        if p_i == 0: ax.annotate('VAE p(x) means', xytext=(.4, 1.1), xy=(0, 1), textcoords='axes fraction')
        plot_scatter(ax, means, color='green')





        ######
        #COL 5
        ax = plt.subplot2grid((rows,columns), (p_i,4), frameon=False)#, colspan=3)

        for i in range(len(means)):
            plot_isocontours_mog(ax, ttp.G_class2, [means[i]], [logvars[i]], cmap='Reds', alpha=.2)

        # plot_isocontours(ax, dist.run_log_post, cmap='Blues')
        if p_i == 0: ax.annotate('VAE p(x) all', xytext=(.4, 1.1), xy=(0, 1), textcoords='axes fraction')

        plot_scatter(ax, samps, color='blue')

        # plot_scatter(ax, means, color='green')



        #######################
        #ROW 2

        p_i = 1


        for i in range(len(samps)):

            ax = plt.subplot2grid((rows,columns), (p_i,i), frameon=False)#, colspan=3)
            # dist = ttp.G_class()
            plot_isocontours(ax, dist.run_log_post, cmap='Blues')

            plot_scatter(ax, np.array([samps[i]]),  color='blue')



        #######################
        #ROW 3

        p_i = 2

        # model = BVAE(hyperparams)
        means, logvars = model.reconstruct(param_file, samps=samps)
        # print means
        # print 
        # print logvars



        for i in range(len(samps)):

            ax = plt.subplot2grid((rows,columns), (p_i,i), frameon=False)#, colspan=3)
            # dist = ttp.G_class()
            plot_isocontours(ax, dist.run_log_post, cmap='Blues')

            
            plot_isocontours_mog(ax, ttp.G_class2, [means[i]], [logvars[i]], cmap='Reds')

            plot_scatter(ax, np.array([samps[i]]),  color='blue', alpha=.3, s=5)




            # for q_i in range(len(models)):

            #     print model_names[q_i]
            #     ax = plt.subplot2grid((rows,columns), (p_i,q_i+1), frameon=False)#, colspan=3)
            #     model = models[q_i](posteriors[p_i])
            #     # model.train(10000, save_to=home+'/Documents/tmp/vars.ckpt')
            #     model.train(10000, save_to='')
            #     samps = model.sample(1000)
            #     plot_kde(ax, samps, cmap='Reds')
            #     plot_isocontours(ax, posterior.run_log_post, cmap='Blues', alpha=alpha)
            #     if p_i == 0: ax.annotate(model_names[q_i], xytext=(.38, 1.1), xy=(0, 1), textcoords='axes fraction')





    else:

        rows = 1
        columns = 5 # one for dist, 1 for samples #len(models) +1 #+1 for posteriors


        fig = plt.figure(figsize=(5+columns,3+rows), facecolor='white')

        # for p_i in range(len(disrtibutions)):

        #######################
        #ROW 1

        p_i = 0

        ######
        #COL 1
        ax = plt.subplot2grid((rows,columns), (p_i,0), frameon=False)#, colspan=3)
        #Plot distribution


        plot_isocontours(ax, dist.run_log_post, cmap='Blues')
        if p_i == 0: ax.annotate('p(X)', xytext=(.4, 1.1), xy=(0, 1), textcoords='axes fraction')


        ######
        #COL 2
        #Plot samples from distribution
        ax = plt.subplot2grid((rows,columns), (p_i,1), frameon=False)#, colspan=3)


        if p_i == 0: ax.annotate('Dataset Samples', xytext=(.2, 1.1), xy=(0, 1), textcoords='axes fraction')
        plot_isocontours(ax, dist.run_log_post, cmap='Blues')

        plot_scatter(ax, samps, color='blue')








        ######
        #COL 3
        ax = plt.subplot2grid((rows,columns), (p_i,2), frameon=False)#, colspan=3)
        if p_i == 0: ax.annotate('VAE p(x)', xytext=(.4, 1.1), xy=(0, 1), textcoords='axes fraction')




        model.init_params(param_file)
        means, logvars = model.sample_prior(param_file, n_samps=n_z_samps)

        # print means
        # print 
        # print logvars




        plot_isocontours_mog(ax, ttp.G_class2, means, logvars, cmap='Reds')
        plot_scatter(ax, samps, color='blue', alpha=.5, s=5)





        ######
        #COL 4
        ax = plt.subplot2grid((rows,columns), (p_i,3), frameon=False)#, colspan=3)
        # dist = ttp.G_class()
        plot_isocontours(ax, dist.run_log_post, cmap='Blues')
        if p_i == 0: ax.annotate('VAE p(x) means', xytext=(.4, 1.1), xy=(0, 1), textcoords='axes fraction')
        plot_scatter(ax, means, color='green')





        ######
        #COL 5
        ax = plt.subplot2grid((rows,columns), (p_i,4), frameon=False)#, colspan=3)

        for i in range(len(means)):
            plot_isocontours_mog(ax, ttp.G_class2, [means[i]], [logvars[i]], cmap='Reds', alpha=.2)

        # plot_isocontours(ax, dist.run_log_post, cmap='Blues')
        if p_i == 0: ax.annotate('VAE p(x) all', xytext=(.4, 1.1), xy=(0, 1), textcoords='axes fraction')

        plot_scatter(ax, samps, color='blue')

        # plot_scatter(ax, means, color='green')
    plt.savefig(home+'/Documents/tmp/plots1'+file_type)
    print 'saved'







if fig2:

    print 'Fig 2'


    # xlimits=[-6, 6]
    # ylimits=[-6, 6]
    # x = np.linspace(*xlimits, num=numticks)
    # y = np.linspace(*ylimits, num=numticks)
    plt.figure(figsize=(8, 10))  





    # y_values2 = np.linspace(3, -3, nx)

    canvas = np.empty((numticks*ny, numticks*nx))
    for i, yi in enumerate(y_values):
        print i
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]])
            x_mean, x_logvar = model.generate(z_mu)

            px_z = ttp.G_class2(x_mean, x_logvar)
            Z = return_z_values(func=px_z.run_log_post, numticks=numticks)

            canvas[(nx-i-1)*numticks:(nx-i)*numticks, j*numticks:(j+1)*numticks] = Z

          

    # Xi, Yi = np.meshgrid(x_values, y_values)
    # plt.imshow(canvas, origin="upper", cmap="Reds")
    # plt.imshow(canvas, cmap="Reds")

    if np.sum(canvas) == 0:
        canvas[0][0] = .00001

    plt.contour(canvas, cmap='Reds')

    # plt.contour(canvas, origin="upper", cmap='Reds')
    # plt.tight_layout()


    # px_z = ttp.G_class2([0.,0.], [0.,0.])
    Z = return_z_values(func=dist.run_log_post, numticks=numticks)
    canvas = np.empty((numticks*ny, numticks*nx))
    for i, yi in enumerate(x_values):
        print i
        for j, xi in enumerate(y_values):
            # z_mu = np.array([[xi, yi]])
            # x_mean, x_logvar = model.generate(z_mu)



            canvas[(nx-i-1)*numticks:(nx-i)*numticks, j*numticks:(j+1)*numticks] = Z

    # plt.imshow(canvas, origin="upper", cmap="Reds")
    # x = np.linspace(*[-6, 6], num=numticks)
    # y = np.linspace(*[-6, 6], num=numticks)
    # X, Y = np.meshgrid(x, y)
    # plt.contour(canvas, origin="upper", cmap='Blues', alpha=.3)
    if np.sum(canvas) == 0:
        canvas[0][0] = .00001
    plt.contour(canvas, cmap='Blues', alpha=.3)




    # plt.scatter(samps.T[0], samps.T[1], color='green', marker='x', zorder=2)


        # ax.set_yticks([])
        # ax.set_xticks([])
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')

    # plt.show()
    # fasdf
    plt.savefig(home+'/Documents/tmp/plots2'+file_type)
    print 'saved'







if fig3:

    print 'Fig 3'


    #THis version allows me to plot the samples easily
    # Individual contours

    rows = 5
    columns = 5


    # px_z = ttp.G_class2([0.,0.], [0.,0.])
    Z_G = return_z_values(func=dist.run_log_post, numticks=numticks)


    fig = plt.figure(figsize=(3+columns,3+rows), facecolor='white')

    # canvas = np.empty((numticks*ny, numticks*nx))
    for i, yi in enumerate(x_values):
        print i
        for j, xi in enumerate(y_values):

            ax = plt.subplot2grid((rows,columns), (i,j), frameon=False)#, colspan=3)

            if i == 0 and j==0:
               ax.annotate('Single W', xytext=(.1, .03), xy=(0, 1), textcoords='figure fraction', fontsize=8)
     

            z_mu = np.array([[xi, yi]])
            x_mean, x_logvar = model.generate(z_mu)


            # print x_mean
            # print x_logvar
            # print

            px_z = ttp.G_class2(x_mean, x_logvar)
            Z = return_z_values(func=px_z.run_log_post, numticks=numticks)

            # canvas[(nx-i-1)*numticks:(nx-i)*numticks, j*numticks:(j+1)*numticks] = Z

            x = np.linspace(*[-6, 6], num=numticks)
            y = np.linspace(*[-6, 6], num=numticks)
            X, Y = np.meshgrid(x, y)

            if np.sum(Z) == 0:
                Z[0][0] = .00001

            plt.contour(X,Y,Z,cmap='Reds')
            # plt.imshow(Z, cmap="Reds")
            ax.set_yticks([])
            ax.set_xticks([])
            ax.annotate(str(xi)+','+str(yi), xytext=(.4, .9), xy=(0, 1), textcoords='axes fraction', fontsize=5)
            

            # plot_scatter(ax, samps, color='blue', size=5)

            plt.scatter(samps.T[0], samps.T[1], color='blue', marker='x', zorder=2, s=5)
            cs = plt.contour(X, Y, Z_G, cmap='Blues', alpha=.3)


    # plt.show()
    # plt.tight_layout()
    plt.tight_layout(pad=0.0, w_pad=-2.0, h_pad=-2.0)
    plt.savefig(home+'/Documents/tmp/plots3_'+file_type)
    print 'saved'






if fig4:


    print 'Fig 4'


    # this one shows the variance of x wrt W given z


    rows = 5
    columns = 5


    # px_z = ttp.G_class2([0.,0.], [0.,0.])
    Z_G = return_z_values(func=dist.run_log_post, numticks=numticks)


    fig = plt.figure(figsize=(3+columns,3+rows), facecolor='white')

    # canvas = np.empty((numticks*ny, numticks*nx))
    for i, yi in enumerate(x_values):
        print i
        for j, xi in enumerate(y_values):

            ax = plt.subplot2grid((rows,columns), (i,j), frameon=False)#, colspan=3)

            if i == 0 and j==0:
               ax.annotate('Variance', xytext=(.1, .03), xy=(0, 1), textcoords='figure fraction', fontsize=8)
     

            plt.scatter(samps.T[0], samps.T[1], color='blue', marker='x', zorder=2, s=5)
            cs = plt.contour(X, Y, Z_G, cmap='Blues', alpha=.3)




            z_mu = np.array([[xi, yi]])

            Zs = []

            for k in range(var_over_n_samples):

                x_mean, x_logvar = model.generate(z_mu)


                px_z = ttp.G_class2(x_mean, x_logvar)
                Z = return_z_values(func=px_z.run_log_post, numticks=numticks)

                Zs.append(Z)

            Zs = np.array(Zs)

            Z = np.var(Zs, axis=0)

            # canvas[(nx-i-1)*numticks:(nx-i)*numticks, j*numticks:(j+1)*numticks] = Z

            x = np.linspace(*[-6, 6], num=numticks)
            y = np.linspace(*[-6, 6], num=numticks)
            X, Y = np.meshgrid(x, y)

            if np.sum(Z) == 0:
                Z[0][0] = .00001

            plt.contour(X,Y,Z,cmap='Reds')
            # plt.imshow(Z, cmap="Reds")
            ax.set_yticks([])
            ax.set_xticks([])
            ax.annotate(str(xi)+','+str(yi), xytext=(.4, .9), xy=(0, 1), textcoords='axes fraction', fontsize=5)
            

            # plot_scatter(ax, samps, color='blue', size=5)


    plt.tight_layout(pad=0.0, w_pad=-2.0, h_pad=-2.0)

    # plt.show()
    plt.savefig(home+'/Documents/tmp/plots4'+file_type)
    print 'saved'







if fig5:

    print 'Fig 5'


    # this one shows the means of x wrt W given z

    rows = 5
    columns = 5

    var_over_n_samples = 10

    # px_z = ttp.G_class2([0.,0.], [0.,0.])
    Z_G = return_z_values(func=dist.run_log_post, numticks=numticks)


    fig = plt.figure(figsize=(3+columns,3+rows), facecolor='white')


    # canvas = np.empty((numticks*ny, numticks*nx))
    for i, yi in enumerate(x_values):
        print i
        for j, xi in enumerate(y_values):


            ax = plt.subplot2grid((rows,columns), (i,j), frameon=False)#, colspan=3)

            if i == 0 and j==0:
               ax.annotate('Mean', xytext=(.1, .03), xy=(0, 1), textcoords='figure fraction', fontsize=8)
     


            plt.scatter(samps.T[0], samps.T[1], color='blue', marker='x', zorder=2, s=5)
            cs = plt.contour(X, Y, Z_G, cmap='Blues', alpha=.3)




            z_mu = np.array([[xi, yi]])

            Zs = []

            for k in range(var_over_n_samples):

                x_mean, x_logvar = model.generate(z_mu)


                px_z = ttp.G_class2(x_mean, x_logvar)
                Z = return_z_values(func=px_z.run_log_post, numticks=numticks)

                Zs.append(Z)

            Zs = np.array(Zs)

            Z = np.mean(Zs, axis=0)

            # canvas[(nx-i-1)*numticks:(nx-i)*numticks, j*numticks:(j+1)*numticks] = Z


            x = np.linspace(*[-6, 6], num=numticks)
            y = np.linspace(*[-6, 6], num=numticks)
            X, Y = np.meshgrid(x, y)

            if np.sum(Z) == 0:
                Z[0][0] = .00001

            plt.contour(X,Y,Z,cmap='Reds')
            # plt.imshow(Z, cmap="Reds")
            ax.set_yticks([])
            ax.set_xticks([])
            ax.annotate(str(xi)+','+str(yi), xytext=(.4, .9), xy=(0, 1), textcoords='axes fraction', fontsize=5)
            

            # plot_scatter(ax, samps, color='blue', size=5)

    plt.tight_layout(pad=0.0, w_pad=-2.0, h_pad=-2.0)


    # plt.show()
    plt.savefig(home+'/Documents/tmp/plots5'+file_type)
    print 'saved'






if fig6:

    print 'Fig 6'

    #Variance but on the canvas
    #This shows relative intensity of the distributions


    plt.figure(figsize=(8, 10))  

    plt.text(.1,.1, 'variance')




    canvas = np.empty((numticks*ny, numticks*nx))
    for i, yi in enumerate(y_values):
        print i
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]])

            Zs = []
            for k in range(var_over_n_samples):

                x_mean, x_logvar = model.generate(z_mu)

                px_z = ttp.G_class2(x_mean, x_logvar)
                Z = return_z_values(func=px_z.run_log_post, numticks=numticks)
                Zs.append(Z)

            Zs = np.array(Zs)
            Z = np.var(Zs, axis=0)

            canvas[(nx-i-1)*numticks:(nx-i)*numticks, j*numticks:(j+1)*numticks] = Z
          
    # plt.contour(canvas, cmap='Reds')
    # print np.max(canvas)
    # canvas = (canvas- np.mean(canvas)) / np.std(canvas)
    canvas = np.flip(canvas, axis=0)
    plt.imshow(canvas, cmap="Reds", alpha=1.)





    Z = return_z_values(func=dist.run_log_post, numticks=numticks)
    canvas = np.empty((numticks*ny, numticks*nx))
    for i, yi in enumerate(x_values):
        print i
        for j, xi in enumerate(y_values):
            canvas[(nx-i-1)*numticks:(nx-i)*numticks, j*numticks:(j+1)*numticks] = Z

    canvas = np.flip(canvas, axis=0)
    plt.contour(canvas, cmap='Blues', alpha=.1)

    # plt.imshow(canvas, cmap="Blues", alpha=.9)





    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    # plt.show()
    # fasdf
    plt.savefig(home+'/Documents/tmp/plots6'+file_type)
    print 'saved'







if fig7:

    print 'Fig 7'

    #Mean but on the canvas
    #This shows relative intensity of the distributions


    plt.figure(figsize=(8, 10))  

    plt.text(.1,.1, 'mean')




    canvas = np.empty((numticks*ny, numticks*nx))
    for i, yi in enumerate(y_values):
        print i
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]])

            Zs = []
            for k in range(var_over_n_samples):

                x_mean, x_logvar = model.generate(z_mu)

                px_z = ttp.G_class2(x_mean, x_logvar)
                Z = return_z_values(func=px_z.run_log_post, numticks=numticks)
                Zs.append(Z)

            Zs = np.array(Zs)
            Z = np.mean(Zs, axis=0)

            canvas[(nx-i-1)*numticks:(nx-i)*numticks, j*numticks:(j+1)*numticks] = Z
          
    # plt.contour(canvas, cmap='Reds')
    # print np.max(canvas)
    # canvas = (canvas- np.mean(canvas)) / np.std(canvas)
    # print np.max(canvas)
    # print np.mean(canvas)
    # print np.min(canvas)

    canvas = np.flip(canvas, axis=0)



    plt.imshow(canvas, cmap="Reds", alpha=1.)





    Z = return_z_values(func=dist.run_log_post, numticks=numticks)
    canvas = np.empty((numticks*ny, numticks*nx))
    for i, yi in enumerate(x_values):
        print i
        for j, xi in enumerate(y_values):
            canvas[(nx-i-1)*numticks:(nx-i)*numticks, j*numticks:(j+1)*numticks] = Z

    canvas = np.flip(canvas, axis=0)

    plt.contour(canvas, cmap='Blues', alpha=.1)

    # plt.imshow(canvas, cmap="Blues", alpha=.9)





    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    # plt.show()
    # fasdf
    plt.savefig(home+'/Documents/tmp/plots7'+file_type)
    print 'saved'






