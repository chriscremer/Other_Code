# Implements auto-encoding variational Bayes.

from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.optimizers import adam
# from data import load_mnist, save_images

import scipy as sp
import matplotlib.pyplot as plt

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] // 2
    mean, log_std = params[:, :D], params[:, D:]
    return mean, log_std

def sample_diag_gaussian(mean, log_std, rs):
    return rs.randn(*mean.shape) * np.exp(log_std) + mean

# def bernoulli_log_density(targets, unnormalized_logprobs):
#     # unnormalized_logprobs are in R
#     # Targets must be -1 or 1
#     label_probabilities = -np.logaddexp(0, -unnormalized_logprobs*targets)
#     return np.sum(label_probabilities, axis=-1)   # Sum across pixels.

def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 0.5 * (np.tanh(x) + 1)

def init_net_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a (weights, biases) tuples for all layers."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def batch_normalize(activations):
    mbmean = np.mean(activations, axis=0, keepdims=True)
    return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""
    for W, b in params[:-1]:
        outputs = batch_normalize(np.dot(inputs, W) + b)  # linear transformation
        inputs = relu(outputs)                            # nonlinear transformation
    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb
    return outputs

def nn_predict_gaussian(params, inputs):
    # Returns means and diagonal variances
    return unpack_gaussian_params(neural_net_predict(params, inputs))

# def generate_from_prior(gen_params, num_samples, noise_dim, rs):
def generate_from_prior(gen_params, num_samples, num_latent_dims): #, rs):
    latents = npr.randn(num_samples, num_latent_dims)
    # return sigmoid(neural_net_predict(gen_params, latents))
    return neural_net_predict(gen_params, latents), latents

def p_images_given_latents(gen_params, images, latents):
    means, log_std = unpack_gaussian_params(neural_net_predict(gen_params, latents))
    # return bernoulli_log_density(images, preds)
    return diag_gaussian_log_density(images, means, log_std)


def vae_lower_bound(gen_params, rec_params, data, rs):
    # We use a simple Monte Carlo estimate of the KL
    # divergence from the prior.
    q_means, q_log_stds = nn_predict_gaussian(rec_params, data)
    latents = sample_diag_gaussian(q_means, q_log_stds, rs)
    q_latents = diag_gaussian_log_density(latents, q_means, q_log_stds)
    p_latents = diag_gaussian_log_density(latents, 0, 1)
    likelihood = p_images_given_latents(gen_params, data, latents)
    # print (np.mean(p_latents), np.mean(likelihood) , np.mean(q_latents))
    return np.mean(p_latents + likelihood - q_latents)

def vae_lower_bound_me(gen_params, rec_params, data, rs):
    q_means, q_log_stds = nn_predict_gaussian(rec_params, data)
    latents = sample_diag_gaussian(q_means, q_log_stds, rs)
    q_latents = diag_gaussian_log_density(latents, q_means, q_log_stds)
    p_latents = diag_gaussian_log_density(latents, 0, 1)
    likelihood = p_images_given_latents(gen_params, data, latents)
    return (np.mean(p_latents), np.mean(likelihood) , np.mean(q_latents))

def bimodal_data():

    N = 100

    Gauss1 = sp.stats.multivariate_normal(mean=[2,2], cov=[.01, .01])
    Gauss2 = sp.stats.multivariate_normal(mean=[1,3], cov=[.01, .01])

    dataset = []
    for i in range(N):
        if np.random.rand() > .5:
            dataset.append(Gauss1.rvs())
        else:
            dataset.append(Gauss2.rvs())

    return np.array(dataset)






if __name__ == '__main__':
    # Model hyper-parameters
    latent_dim = 200
    data_dim = 2
    gen_layer_sizes = [latent_dim, 10, 10, data_dim * 2]
    rec_layer_sizes = [data_dim, 10, 10, latent_dim * 2]

    # Training parameters
    param_scale = 0.1
    batch_size = 20
    num_epochs = 2000
    step_size = 0.001

    init_gen_params = init_net_params(param_scale, gen_layer_sizes)
    init_rec_params = init_net_params(param_scale, rec_layer_sizes)
    combined_init_params = (init_gen_params, init_rec_params)
    # print(np.array(init_gen_params).shape)
    # print(np.array(init_rec_params).shape)


    print("Loading training data...")
    X = bimodal_data()
    print(X.shape)



    num_batches = int(np.ceil(len(X) / batch_size))
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    seed = npr.RandomState(0)
    def objective(combined_params, iter):
        data_idx = batch_indices(iter)
        gen_params, rec_params = combined_params
        return -vae_lower_bound(gen_params, rec_params, X[data_idx], seed) / data_dim

    # Get gradients of objective using autograd.
    objective_grad = grad(objective)

    print("     Epoch     |    Objective  |       Fake probability | Real Probability  ")
    def print_perf(combined_params, iter, grad):
        if iter % 10 == 0:
            gen_params, rec_params = combined_params
            bound = np.mean(objective(combined_params, iter))
            print("{}|{:15}|{:20}".format(iter, iter//num_batches, bound))

            # rs = npr.RandomState(0)
            # pz, lik, qz = vae_lower_bound_me(gen_params, rec_params, X, rs)
            # print (pz, lik, qz)

            # fake_data = generate_from_prior(gen_params, 2, latent_dim, seed)
            # print(fake_data)
            # save_images(fake_data, 'vae_samples.png', vmin=0, vmax=1)
            # print (gen_params[0])

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, combined_init_params, step_size=step_size,
                            num_iters=num_epochs * num_batches, callback=print_perf)


    # q_means, q_log_stds = nn_predict_gaussian(optimized_params, [[2,2],[1,1]])
    # print(q_means, q_log_stds)

    rs = npr.RandomState(0)
    gen_params, rec_params = optimized_params
    # print ('herr')
    # print (gen_params[0])
    plt.scatter(X.T[0], X.T[1], color='green', label='Training set')
    # plt.show()

    #Plot 100 generated samples
    gen_data, latents = generate_from_prior(gen_params=gen_params, num_samples=100, num_latent_dims=latent_dim)
    # print (gen_data.shape)
    # print (latents.shape)
    # print (latents)
    # print(gen_data)
    # print (gen_data.T[0])
    # print (gen_data.T[1])
    plt.scatter(gen_data.T[0], gen_data.T[1], color='blue', label='Prior Samples')
    # plt.show()


    #Reconstruct
    data = X
    # print (data)
    q_means, q_log_stds = nn_predict_gaussian(rec_params, data)
    # print (q_means, q_log_stds)
    latents = sample_diag_gaussian(q_means, q_log_stds, rs)
    # print (latents)
    gen_data = neural_net_predict(gen_params, latents)
    # print (gen_data)
    plt.scatter(gen_data.T[0], gen_data.T[1], color='red', label='Reconstruct')


    plt.plot([],[],label=str(latent_dim)+' latent dims')

    plt.legend()
    plt.show()





