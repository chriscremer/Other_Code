

# so the idea is to move the log outside the average over smaples
# its no longer a lower bound but thats ok 
# were going to reduce batch size over time so it gets closer to weighting samples equally, ie lower bound
# so to change:
#    - objective 


from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.optimizers import adam
from data import load_mnist, save_images


def log_mean_exp(x):

    #[B]
    max_ = np.max(x)
    lme = np.log(np.mean(np.exp(x-max_))) + max_
    return lme

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] // 2
    mean, log_std = params[:, :D], params[:, D:]
    return mean, log_std

def sample_diag_gaussian(mean, log_std, rs):
    return rs.randn(*mean.shape) * np.exp(log_std) + mean

def bernoulli_log_density(targets, unnormalized_logprobs):
    # unnormalized_logprobs are in R
    # Targets must be -1 or 1
    label_probabilities = -np.logaddexp(0, -unnormalized_logprobs*targets)
    return np.sum(label_probabilities, axis=-1)   # Sum across pixels.

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

def generate_from_prior(gen_params, num_samples, noise_dim, rs):
    latents = rs.randn(num_samples, noise_dim)
    return sigmoid(neural_net_predict(gen_params, latents))

def p_images_given_latents(gen_params, images, latents):
    preds = neural_net_predict(gen_params, latents)
    return bernoulli_log_density(images, preds)

# def vae_lower_bound(gen_params, rec_params, data, rs):
#     # We use a simple Monte Carlo estimate of the KL
#     # divergence from the prior.
#     q_means, q_log_stds = nn_predict_gaussian(rec_params, data)
#     latents = sample_diag_gaussian(q_means, q_log_stds, rs)
#     q_latents = diag_gaussian_log_density(latents, q_means, q_log_stds)
#     p_latents = diag_gaussian_log_density(latents, 0, 1)
#     likelihood = p_images_given_latents(gen_params, data, latents)
#     return np.mean(p_latents + likelihood - q_latents) #average over batch


# def iw_iwae_lower_bound(gen_params, rec_params, data, rs):
#     # We use a simple Monte Carlo estimate of the KL
#     # divergence from the prior.
#     q_means, q_log_stds = nn_predict_gaussian(rec_params, data)
#     latents = sample_diag_gaussian(q_means, q_log_stds, rs)
#     q_latents = diag_gaussian_log_density(latents, q_means, q_log_stds)
#     p_latents = diag_gaussian_log_density(latents, 0, 1)
#     likelihood = p_images_given_latents(gen_params, data, latents)

#     #so here do log mean exp

#     return log_mean_exp(p_latents + likelihood - q_latents) 


def annealed_iw_iwae_lower_bound(gen_params, rec_params, data, rs, annealing):
    # We use a simple Monte Carlo estimate of the KL
    # divergence from the prior.
    q_means, q_log_stds = nn_predict_gaussian(rec_params, data)
    latents = sample_diag_gaussian(q_means, q_log_stds, rs)
    q_latents = diag_gaussian_log_density(latents, q_means, q_log_stds)
    p_latents = diag_gaussian_log_density(latents, 0, 1)
    likelihood = p_images_given_latents(gen_params, data, latents)

    #so here do log mean exp

    return ((annealing)*log_mean_exp(p_latents + likelihood - q_latents)) + ((1.-annealing)*np.mean(p_latents + likelihood - q_latents)) 


if __name__ == '__main__':
    # Model hyper-parameters
    latent_dim = 10
    data_dim = 784  # How many pixels in each image (28x28).
    gen_layer_sizes = [latent_dim, 300, 200, data_dim]
    rec_layer_sizes = [data_dim, 200, 300, latent_dim * 2]

    # Training parameters
    param_scale = 0.01
    batch_size = 200
    original_batch_size = 200
    num_epochs = 15
    step_size = 0.001

    print("Loading training data...")
    N, train_images, _, test_images, _ = load_mnist()
    on = train_images > 0.5
    train_images = train_images * 0 - 1
    train_images[on] = 1.0

    init_gen_params = init_net_params(param_scale, gen_layer_sizes)
    init_rec_params = init_net_params(param_scale, rec_layer_sizes)
    combined_init_params = (init_gen_params, init_rec_params)

    params = combined_init_params









    # annealing = 1.
    # ann_range = np.array(range(10)[::-1])
    # ann_range = ann_range / len(ann_range)
    # # print (ann_range)

    # fasdf


    # for times in range(10):

        # batch_size = int(original_batch_size * (.5**times))
        # if batch_size < 1: batch_size =1
    # annealing = ann_range[times]


    num_batches = int(np.ceil(len(train_images) / batch_size))
    def batch_indices(iter):
        idx = iter % num_batches

        # if iter % 200 == 0:
        #     print (batch_size)

        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    seed = npr.RandomState(0)
    def objective(combined_params, iter):
        data_idx = batch_indices(iter)
        gen_params, rec_params = combined_params

        annealing = (.999**iter)

        # return -vae_lower_bound(gen_params, rec_params, train_images[data_idx], seed) #/ data_dim
        # return -iw_iwae_lower_bound(gen_params, rec_params, train_images[data_idx], seed) #/ data_dim
        return -annealed_iw_iwae_lower_bound(gen_params, rec_params, train_images[data_idx], seed, annealing) #/ data_dim



    # Get gradients of objective using autograd.
    objective_grad = grad(objective)

    # epoch = 0

    print("     Epoch     |    Objective  |       Fake probability | Real Probability  ")
    def print_perf(combined_params, iter, grad):
        if iter % 10 == 0:
            gen_params, rec_params = combined_params
            # bound = np.mean(objective(combined_params, iter)) #average over what?
            bound = objective(combined_params, iter) #average over what?

            annealing = (.999**iter)
            

            # fake_data = generate_from_prior(gen_params, 20, latent_dim, seed)
            # save_images(fake_data, 'vae_samples.png', vmin=0, vmax=1)
            # epoch = iter//num_batches
            # batch_size = original_batch_size * (.5**epoch)
            # if batch_size < 1: batch_size =1
            # print ('batch size', batch_size)


            print("{}|{:15}|{:20}|{:25}".format(iter, iter//num_batches, bound, annealing))

        # if new_epoch != epoch:
        #     batch_size = batch_size / 2
        #     if batch_size < 1: batch_size =1
        #     print ('batch size', batch_size)
        #     epoch = new_epoch

    # # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    # optimized_params = adam(objective_grad, combined_init_params, step_size=step_size,
    #                         num_iters=num_epochs * num_batches, callback=print_perf)


    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, params, step_size=step_size,
                            num_iters=2000, callback=print_perf)

    params = optimized_params









