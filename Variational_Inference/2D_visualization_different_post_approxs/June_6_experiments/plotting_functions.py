

import numpy as np
import matplotlib.pyplot as plt 

import scipy.stats as st




def plot_isocontours(ax, func, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1.):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = np.exp(func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T))
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z, cmap=cmap, alpha=alpha)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')


def plot_kde(ax, samps, xlimits=[-6, 6], ylimits=[-6, 6],
                     numticks=101, cmap=None, alpha=1.):

    samps = np.array(samps)
    x_ = samps[:, 0]
    y_ = samps[:, 1]

    values = np.vstack([x_, y_])
    kernel = st.gaussian_kde(values)


    # Peform the kernel density estimate
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)

    positions = np.vstack([X.ravel(), Y.ravel()])

    f = np.reshape(kernel(positions).T, X.shape)

    # cfset = ax.contourf(X, Y, f, cmap=cmap)
    cfset = ax.contour(X, Y, f, cmap=cmap)


    ax.set_yticks([])
    ax.set_xticks([])
    plt.gca().set_aspect('equal', adjustable='box')



# def plot_iw_isocontours(ax, p, q, sampler, k=3, xlimits=[-6, 6], 
#                         ylimits=[-6, 6], numticks=101, 
#                         n_sample_batches=3, cmap='Blues'):

#     x = np.linspace(*xlimits, num=numticks)
#     y = np.linspace(*ylimits, num=numticks)
#     X, Y = np.meshgrid(x, y)
#     X = X.T
#     Y = Y.T

#     X_vec = np.reshape(X, [-1,1])
#     Y_vec = np.reshape(Y, [-1,1])
#     zipped = np.concatenate([X_vec, Y_vec], axis=1)

#     # log_p = np.log(p(zipped))
#     # log_q = np.log(q(zipped))
#     log_p = p(zipped)
#     log_q = q(zipped)
#     log_w = log_p - log_q
#     exp_log_p = np.exp(log_p)
#     exp_log_w = np.exp(log_w)

#     grid_values = np.zeros([numticks*numticks])

#     for b in range(n_sample_batches):
#         if k >1:
#             samples = sampler(k-1)
#             log_p_ = np.log(p(samples))
#             log_q_ = np.log(q(samples))
#             log_w_ = log_p_ - log_q_
#             sum_p_q = np.sum(np.exp(log_w_))
#         else:
#             sum_p_q = 0

#         denominator = (exp_log_w + sum_p_q) / k
#         grid_values += exp_log_p / denominator

#     grid_values = grid_values / n_sample_batches
#     # print np.sum(grid_values) / len(grid_values)
#     # print np.max(grid_values)
#     grid_values = np.reshape(grid_values, [numticks, numticks])
#     plt.contourf(X, Y, grid_values, cmap=cmap) #, vmin=10**(-10))
#     ax.set_yticks([])
#     ax.set_xticks([])
#     plt.gca().set_aspect('equal', adjustable='box')





# def plot_hamil_kde_isocontours(ax, p, q, hamil_func, k=3, xlimits=[-6, 6], 
#                         ylimits=[-6, 6], numticks=101, 
#                         n_sample_batches=3, cmap='Blues'):

#     x = np.linspace(*xlimits, num=numticks)
#     y = np.linspace(*ylimits, num=numticks)
#     X, Y = np.meshgrid(x, y)
#     X = X.T
#     Y = Y.T

#     X_vec = np.reshape(X, [-1,1])
#     Y_vec = np.reshape(Y, [-1,1])
#     zipped = np.concatenate([X_vec, Y_vec], axis=1)

#     # log_p = np.log(p(zipped))
#     # log_q = np.log(q(zipped))
#     log_p = p(zipped)
#     log_q = q(zipped)
#     log_w = log_p - log_q
#     exp_log_p = np.exp(log_p)
#     exp_log_w = np.exp(log_w)

#     grid_values = np.zeros([numticks*numticks])

    
#     for b in range(n_sample_batches):



#         if k >1:
#             zT, log_w_ = hamil_func(k-1)

#             #KDE
#             if b == 0:
#                 samps = zT
#             else:
#                 samps = np.concatenate([samps, zT], axis=0)

#             # #HI
#             # sum_p_q = np.sum(np.exp(log_w_))

#         else:
#             sum_p_q = 0

#         # # #HI
#         # denominator = (exp_log_w + sum_p_q) / k
#         # grid_values += exp_log_p / denominator


#     #KDE
#     x_ = samps[:, 0]
#     y_ = samps[:, 1]
#     positions = np.vstack([X.ravel(), Y.ravel()])
#     values = np.vstack([x_, y_])
#     kernel = st.gaussian_kde(values)
#     grid_values = np.reshape(kernel(positions).T, X.shape)

#     # #HI
#     # grid_values = np.reshape(grid_values, [numticks, numticks])
#     # grid_values = grid_values / n_sample_batches


#     plt.contourf(X, Y, grid_values, cmap=cmap) #, vmin=10**(-10))


#     ax.set_yticks([])
#     ax.set_xticks([])
#     plt.gca().set_aspect('equal', adjustable='box')

#     # print np.sum(grid_values) / len(grid_values)
#     # print np.max(grid_values)





# def plot_qT_approx_isocontours(ax, q0_z, q0_v, qT_v, sample_qT_v, reverse_hamil_func, k=3, xlimits=[-6, 6], 
#                         ylimits=[-6, 6], numticks=101, 
#                         n_sample_batches=3, cmap='Blues'):

#     x = np.linspace(*xlimits, num=numticks)
#     y = np.linspace(*ylimits, num=numticks)
#     X, Y = np.meshgrid(x, y)
#     X = X.T
#     Y = Y.T

#     X_vec = np.reshape(X, [-1,1])
#     Y_vec = np.reshape(Y, [-1,1])
#     zipped = np.concatenate([X_vec, Y_vec], axis=1)

#     grid_values = np.zeros([numticks*numticks])
#     for k_i in range(k):

#         # sample a velocites/momentums, and get their prob
#         vT = sample_qT_v(1) #[1,D]
#         log_qT_vT = qT_v(vT) #[1,1]
#         N_ones = np.ones([numticks*numticks,1])
#         vT_repeated = np.dot(N_ones, vT) #[N,D]

#         # transform zT and vT back to z0,v0
#         z0, v0 = reverse_hamil_func(numticks*numticks, zipped, vT_repeated) #[N,2]

#         # get prob of z0,v0 under proposal
#         # log_q0_z0 = np.log(np.maximum(q0_z(z0), np.exp(-40.))) #[N,1]
#         log_q0_z0 = q0_z(z0) #[N,1]


#         log_q0_v0 = q0_v(v0) #[N,1]


#         # print log_q0_z0
#         # print log_q0_v0
#         # print log_qT_vT
#         # prob of zT is average of k samples, of q(v0)q(z0)/p(vT)
#         log_qT_zT = log_q0_z0 + log_q0_v0 - log_qT_vT #broadcasted I think
#         # print log_qT_zT


#         qT_zT = np.exp(log_qT_zT) #[N,1]
#         # print qT_zT

#         grid_values += qT_zT



#     grid_values = np.reshape(grid_values, [numticks, numticks])
#     grid_values = grid_values / k



#     plt.contourf(X, Y, grid_values, cmap=cmap) #, vmin=10**(-10))

#     ax.set_yticks([])
#     ax.set_xticks([])
#     plt.gca().set_aspect('equal', adjustable='box')







# def plot_scatter_hamil_points(z0, zT):

#     plt.scatter(z0.T[0], z0.T[1], color='yellow', marker='x')
#     plt.scatter(zT.T[0], zT.T[1], color='green', marker='x')
#     ax.set_yticks([])
#     ax.set_xticks([])
#     plt.xlim(-6, 6)
#     plt.ylim(-6, 6)
#     plt.gca().set_aspect('equal', adjustable='box')


# def plot_scatter_hamil_points3(z0, zT, reverse_z0):

#     plt.scatter(z0.T[0], z0.T[1], color='yellow', marker='x')
#     plt.scatter(zT.T[0], zT.T[1], color='green', marker='x')
#     plt.scatter(reverse_z0.T[0], reverse_z0.T[1], color='orange', marker='x')
#     ax.set_yticks([])
#     ax.set_xticks([])
#     plt.xlim(-6, 6)
#     plt.ylim(-6, 6)
#     plt.gca().set_aspect('equal', adjustable='box')

