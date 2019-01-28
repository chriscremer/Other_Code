

from __future__ import division
import numpy as np
import matplotlib.pyplot as pl






# PLOTS:
ylim = [-6,6]

rows = 2
cols = 2
fig = pl.figure(figsize=(6+cols,2+rows), facecolor='white', dpi=150)   


n = 50     
x_lim = [-12.5,12.5]
X_linspace = np.linspace(x_lim[0], x_lim[1], n).reshape(-1,1)

for i in range (rows):
    for j in range (cols):

        des = 2*np.random.rand() -.5
        freq = 2*np.random.rand() 

        f = lambda x: (-x*des+np.sin(freq*x)).flatten()

        ax = pl.subplot2grid((rows,cols), (i,j), frameon=False, colspan=1, rowspan=1)
        ax.plot(X_linspace, f(X_linspace), 'b-', label='True', linewidth=1.)
        ax.tick_params(labelsize=6)




pl.tight_layout()
# pl.gca().set_aspect('equal')

pl.show()














fasdf








# This is the true unknown function we are trying to approximate
# f = lambda x: 4*np.sin(0.7*x).flatten()# + 5
f = lambda x: (-x*.4+np.sin(1.2*x)).flatten()# + 5
# f = lambda x: (x).flatten()# + 5
# f = lambda x: (0.25*(x**2)).flatten()

























# Define the kernel
def kernel(a, b):
    # print (a.shape)# [A,1]
    # print (b.shape) # [B,1]
    # # fasdf
    """ GP squared exponential kernel """
    kernelParameter = 1. #0.1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T) #this is expansion of (a-b)^2
    out= np.exp(-.5 * (sqdist/kernelParameter))
    # print (out.shape) #[A,B]
    return out 


# def kernel(a, b):
#     return np.dot(a, b.T) + kernel2(a, b) + (np.dot(a, b.T) * kernel2(a, b) )

# # Define the kernel
# def kernel2(a, b):
#     # print (a.shape)# [A,1]
#     # print (b.shape) # [B,1]
#     # # fasdf
#     """ GP squared exponential kernel """
#     kernelParameter = 1. #0.1
#     sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T) #this is expansion of (a-b)^2
#     out= s_ * np.exp(-.5 * (sqdist/kernelParameter))
#     # print (out.shape) #[A,B]
#     return out 





N = 10         # number of training points.
n = 50         # number of test points.
train_noise = 0.1    # noise variance.
# s_ = 10.
s_ = .01

x_lim = [-12.5,12.5]

# Sample some input points and noisy versions of the function evaluated at
# these points. 
X = np.random.uniform(-10., 10., size=(N,1))
y = f(X) + train_noise*np.random.randn(N)

# X_real = np.linspace(-5, 5, 50).reshape(-1,1)
# y_real = f(X_real)

# points we're going to make predictions at.
X_linspace = np.linspace(x_lim[0], x_lim[1], n).reshape(-1,1)


#VERSION 1, from Nando
# compute the mean at all points.
K = kernel(X, X)
L = np.linalg.cholesky(K + s_*np.eye(N)) # this really helps it, maybe it assumes noise
# L = np.linalg.cholesky(K ) #+ s_*np.eye(N))
solve = np.linalg.solve(L, y)
K_2 = kernel(X, X_linspace)
Lk = np.linalg.solve(L, K_2)
mu = np.dot(Lk.T, solve)
# print (mu)


# #VERSION 2, mine
# K = kernel(X, X)
# y_div_K = np.expand_dims(np.linalg.solve(K, y),1)
# K2 = kernel(X, X_linspace)
# mu = np.dot(K2.T,y_div_K).T[0]
# print (mu.shape)
# print (mu)
# #ok his makes sense becasue we need the cholesky layer on anyways and Lk..







# # compute the variance at all points.
# K_ = kernel(X_linspace, X_linspace)
# Ks = np.linalg.solve(K,K2) #[10,50]  k2/ K
# Ks = np.dot(K2.T,Ks) #[50,50]
# var = K_ - Ks
# var = np.diag(var)
# # print (var)
# # fsa
# std = np.sqrt(var)
# print(std.shape)
# # fasd
# # s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
# # linspace_std = np.sqrt(s2)


# compute the variance at our plot points.
K_ = kernel(X_linspace, X_linspace)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
std = np.sqrt(s2)


# #Compute likelihood of training data under GP
# # compute the mean and var at training points
# # K = kernel(X, X)
# # L = np.linalg.cholesky(K + s_*np.eye(N))
# Lk = np.linalg.solve(L, kernel(X, X_linspace))
# mu = np.dot(Lk.T, np.linalg.solve(L, y))

# K_ = kernel(X_linspace, X_linspace)
# s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
# linspace_std = np.sqrt(s2)
# # prob under gaussian
#training datalikelihood doenst really make sense



# #Compute likelihood of validation set data under GP
X_val = np.random.uniform(-5, 5, size=(N,1))
y_val = f(X_val) + train_noise*np.random.randn(N)
# print (y_val)
# print (y)
# fsad
# get means
K2 = kernel(X, X_val)
Lk_val = np.linalg.solve(L, K2)
mu_val = np.dot(Lk_val.T, solve)
# get vars
K_val = kernel(X_val, X_val)
s2 = np.diag(K_val) - np.sum(Lk_val**2, axis=0)
std_val = np.sqrt(s2)
# print (mu.shape, std.shape)

logprob = -.5 * (y_val-mu_val)**2 / std_val
logprob = np.mean(logprob)
# print ('val log prob:', logprob)
# fsdfa








# PLOTS:
ylim = [-6,6]

rows = 2
cols = 2
fig = pl.figure(figsize=(6+cols,2+rows), facecolor='white', dpi=150)    


# Plot GP mean and std
ax = pl.subplot2grid((rows,cols), (0,0), frameon=False, colspan=1, rowspan=1)
ax.plot(X, y, 'g+', ms=6, label='Datapoints')
ax.plot(X_linspace, f(X_linspace), 'b-', label='True', linewidth=1.)
ax.plot(X_linspace, mu, 'r--', label='GP', linewidth=1.)
pl.gca().fill_between(X_linspace.flat, mu-std, mu+std, color="#dddddd")
ax.set_title('GP',fontsize=6,family='serif')
ax.legend(fontsize=6)
# ax.title('GP')
# ax.axis([-5, 5, ylim[0], ylim[1]])
ax.tick_params(labelsize=6)




# draw samples from the posterior 
n_samps = 5
ax = pl.subplot2grid((rows,cols), (0,1), frameon=False, colspan=1, rowspan=1)
L_ = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L_, np.random.normal(size=(n,n_samps)))
ax.plot(X_linspace, f_post, linewidth=1.)
ax.plot(X_linspace, f(X_linspace), 'b-', label='True', linewidth=1.)
ax.plot(X, y, 'g+', ms=6, label='Datapoints')
ax.tick_params(labelsize=6)
ax.legend(fontsize=6)
ax.set_title('Posterior Samples',fontsize=6,family='serif')

# draw samples from the prior 
n_samps = 5
ax = pl.subplot2grid((rows,cols), (1,1), frameon=False, colspan=1, rowspan=1)
L_ = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L_, np.random.normal(size=(n,n_samps)))
ax.plot(X_linspace, f_prior, linewidth=1.)
ax.plot(X_linspace, f(X_linspace), 'b-', label='True', linewidth=1.)
ax.plot(X, y, 'g+', ms=6, label='Datapoints')
ax.tick_params(labelsize=6)
ax.legend(fontsize=6)
ax.set_title('Prior Samples',fontsize=6,family='serif')


# Plot GP mean and std
ax = pl.subplot2grid((rows,cols), (1,0), frameon=False, colspan=1, rowspan=1)
ax.plot(X_val, y_val, 'b+', ms=6, label='Datapoints')
ax.plot(X_linspace, f(X_linspace), 'b-', label='True', linewidth=1.)
ax.plot(X_linspace, mu, 'r--', label='GP', linewidth=1.)
pl.gca().fill_between(X_linspace.flat, mu-std, mu+std, color="#dddddd")
ax.set_title('Validation logprob:{:.4f}'.format(logprob),fontsize=6,family='serif')
ax.legend(fontsize=6)
# ax.title('GP')
# ax.axis([-5, 5, ylim[0], ylim[1]])
ax.tick_params(labelsize=6)







pl.tight_layout()
# pl.gca().set_aspect('equal')

pl.show()

























faasfd


pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=6, label='Datapoints')
# pl.plot(X_real, y_real)
pl.plot(Xtest, f(Xtest), 'b-', label='True')
pl.gca().fill_between(Xtest.flat, mu-test_std, mu+test_std, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2, label='GP')
# pl.savefig('predictive.png', bbox_inches='tight')
# pl.title('Mean predictions plus 3 st.deviations')
# pl.title('True function and datapoints')
pl.legend()
pl.title('GP')
pl.axis([-5, 5, ylim[0], ylim[1]])

# # draw samples from the prior at our test points.
# L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
# f_prior = np.dot(L, np.random.normal(size=(n,10)))
# pl.figure(2)
# pl.clf()
# pl.plot(Xtest, f_prior)
# pl.title('Ten samples from the GP prior')
# pl.axis([-5, 5, -20, 20])
# pl.savefig('prior.png', bbox_inches='tight')

# # draw samples from the posterior at our test points.
# L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
# f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
# pl.figure(3)
# pl.clf()
# pl.plot(Xtest, f_post)
# pl.title('Ten samples from the GP posterior')
# pl.axis([-5, 5, -20, 20])
# pl.savefig('post.png', bbox_inches='tight')

pl.show()