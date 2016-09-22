#!/usr/bin/env python
"""
Mixture model using mean-field variational inference.
Probability model
  Mixture of Gaussians
  pi ~ Dirichlet(alpha)
  for k = 1, ..., K
    mu_k ~ N(0, cI)
    sigma_k ~ Inv-Gamma(a, b)
  for n = 1, ..., N
    c_n ~ Multinomial(pi)
    x_n|c_n ~ N(mu_{c_n}, sigma_{c_n})
Variational model
  Likelihood:
    q(pi) prod_{k=1}^K q(mu_k) q(sigma_k)
    q(pi) = Dirichlet(alpha')
    q(mu_k) = N(mu'_k, Sigma'_k)
    q(sigma_k) = Inv-Gamma(a'_k, b'_k)
  (We collapse the c_n latent variables in the probability model's
  joint density.)
Data: x = {x_1, ..., x_N}, where each x_i is in R^2
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf

from edward.models import Dirichlet, Normal, InverseGamma #mean field models
from edward.stats import dirichlet, invgamma, multivariate_normal_diag, norm
from edward.util import get_dims, log_sum_exp

plt.style.use('ggplot')

def build_toy_dataset(N):
  pi = np.array([0.4, 0.6])
  mus = [[1, 1], [-1, -1]]
  stds = [[0.1, 0.1], [0.1, 0.1]]
  x = np.zeros((N, 2), dtype=np.float32)
  for n in range(N):
    k = np.argmax(np.random.multinomial(1, pi))
    x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

  return x


class MixtureGaussian:
  """
  Mixture of Gaussians
  p(x, z) = [ prod_{n=1}^N sum_{k=1}^K pi_k N(x_n; mu_k, sigma_k) ]
            [ prod_{k=1}^K N(mu_k; 0, cI) Inv-Gamma(sigma_k; a, b) ]
            Dirichlet(pi; alpha)
  where z = {pi, mu, sigma} and for known hyperparameters a, b, c, alpha.
  Parameters
  ----------
  K : int
    Number of mixture components.
  D : float, optional
    Dimension of the Gaussians.
  """
  def __init__(self, K, D):
    self.K = K
    self.D = D
    self.n_vars = (2 * D + 1) * K

    #HYPERPARAMETERS
    self.a = 1.0
    self.b = 1.0
    self.c = 3.0
    self.alpha = tf.ones([K])

  def log_prob(self, xs, zs):
    """
    Return scalar, the log joint density log p(xs, zs).

    Given n_minibatch data points, n_samples of variables
    Summing over the datapoints makes sense since the joint is the only place in the 
    estiamtion of the gradient that has the data points, and its the log, so we can sum
    over them
    BUT summing over the variables doenst make sense,its supposed to be one at a time

    """
    x = xs['x']
    pi, mus, sigmas = zs['pi'], zs['mu'], zs['sigma']

    # print(get_dims(x)) #[n_minibatch, D]
    # print(get_dims(pi)) #[K]
    # print(get_dims(mus)) #[K*D]
    # print(get_dims(sigmas)) #[K*D]

    log_prior = dirichlet.logpdf(pi, self.alpha)
    log_prior += tf.reduce_sum(norm.logpdf(mus, 0.0, self.c))
    log_prior += tf.reduce_sum(invgamma.logpdf(sigmas, self.a, self.b))

    # log-likelihood is
    # sum_{n=1}^N log sum_{k=1}^K exp( log pi_k + log N(x_n; mu_k, sigma_k) )
    # Create a K x N matrix, whose entry (k, n) is
    # log pi_k + log N(x_n; mu_k, sigma_k).
    n_minibatch = get_dims(x)[0]   #this is [n_minibatch, D], with [0] its just n_minibatch
    #OH I think they compute the matrix so that they can do log sum exp, since they need to find the max value

    matrix = []
    for k in range(self.K):

      # bbbb = tf.log(pi[k])
      # print(get_dims(bbbb))
      # aaaa= multivariate_normal_diag.logpdf(x,  mus[(k * self.D):((k + 1) * self.D)],  sigmas[(k * self.D):((k + 1) * self.D)])
      # print(get_dims(aaaa))
      # fadad

      matrix += [tf.ones(n_minibatch) * tf.log(pi[k]) +
                 multivariate_normal_diag.logpdf(x,  mus[(k * self.D):((k + 1) * self.D)],  sigmas[(k * self.D):((k + 1) * self.D)])]

    matrix = tf.pack(matrix)
    # log_sum_exp() along the rows is a vector, whose nth
    # element is the log-likelihood of data point x_n.
    vector = log_sum_exp(matrix, 0)
    # Sum over data points to get the full log-likelihood.
    log_lik = tf.reduce_sum(vector)

    return log_prior + log_lik

  def predict(self, xs, zs):
    """Calculate a K x N matrix of log-likelihoods, per-cluster and
    per-data point."""
    x = xs['x']
    pi, mus, sigmas = zs['pi'], zs['mu'], zs['sigma']

    matrix = []
    for k in range(self.K):
      matrix += [multivariate_normal_diag.logpdf(x,
                 mus[(k * self.D):((k + 1) * self.D)],
                 sigmas[(k * self.D):((k + 1) * self.D)])]

    return tf.pack(matrix)





ed.set_seed(42)
x_train = build_toy_dataset(500)
plt.scatter(x_train[:, 0], x_train[:, 1])
plt.axis([-3, 3, -3, 3])
plt.title("Simulated dataset")
plt.show()

K = 2
D = 2
model = MixtureGaussian(K, D)

qpi_alpha = tf.nn.softplus(tf.Variable(tf.random_normal([K])))
# qpi_alpha = tf.nn.softplus(tf.Variable([1.,1.])) #so that it doesnt sample a 1 and small number causing error
qmu_mu = tf.Variable(tf.random_normal([K * D]))
qmu_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([K * D])))
qsigma_alpha = tf.nn.softplus(tf.Variable(tf.random_normal([K * D])))
qsigma_beta = tf.nn.softplus(tf.Variable(tf.random_normal([K * D])))

qpi = Dirichlet(alpha=qpi_alpha)
qmu = Normal(mu=qmu_mu, sigma=qmu_sigma)
qsigma = InverseGamma(alpha=qsigma_alpha, beta=qsigma_beta)


#THIS IS ALL IN THE INFERENCE FILE
# n_iter : int, optional
#     Number of iterations for optimization.
# n_minibatch : int, optional
#     Number of samples for data subsampling. Default is to use
#     all the data. Subsampling is available only if all data
#     passed in are NumPy arrays and the model is not a Stan
#     model. For subsampling details, see
#     tf.train.slice_input_producer and tf.train.batch.
# n_print : int, optional
#     Number of iterations for each print progress. To suppress print
#     progress, then specify None.
# optimizer : str, optional
#     Whether to use TensorFlow optimizer or PrettyTensor
#     optimizer when using PrettyTensor. Defaults to TensorFlow.
# scope : str, optional
#     Scope of TensorFlow variable objects to optimize over.

# n_samples (int, optional) - Number of samples from variational model for calculating stochastic gradients.
  #found here:http://edwardlib.org/api/edward.inferences.html

#OH so n_smaples is how many parameter samples you take and n_minibatch is how many data points you take

data = {'x': x_train}
inference = ed.MFVI({'pi': qpi, 'mu': qmu, 'sigma': qsigma}, data, model)

inference.initialize(n_iter=4000, n_samples=10, n_minibatch=100)

max_iter_loop = 1
for iters in range(max_iter_loop):
  print(iters)
  # if iters == 0:
  #   inference.run_no_finalize(n_iter=100, n_samples=10, n_minibatch=20)
  # elif iters < max_iter_loop-1:
  #   inference.run_no_init_or_finalize(n_iter=100, n_samples=10, n_minibatch=20)
  # else:
  #   inference.run_no_finalize(n_iter=100, n_samples=10, n_minibatch=20)
  inference.run_no_init_or_finalize()

inference.finalize()


# Average per-cluster and per-data point likelihood over many posterior samples.
print('predicting posterior')
log_liks = []
for s in range(100):
  if s%20 == 0:
    print (s)
  zrep = {'pi': qpi.sample(()),
          'mu': qmu.sample(()),
          'sigma': qsigma.sample(())}
  log_liks += [model.predict(data, zrep)]

log_liks = tf.reduce_mean(log_liks, 0)

# Choose the cluster with the highest likelihood for each data point.
clusters = tf.argmax(log_liks, 0).eval()
plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters, cmap=cm.bwr)
plt.axis([-3, 3, -3, 3])
plt.title("Predicted cluster assignments")
plt.show()




