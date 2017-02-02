

import numpy as np
import tensorflow as tf
import pickle
import math 

from os.path import expanduser
home = expanduser("~")

# from VAE_mnist import VAE
from VAE_mnist2 import VAE
# from IWAE_mnist import IWAE
from IWAE_mnist2 import IWAE

from NQAE_mnist import NQAE

'''
This is trying to reproduce the results of the IWAE paper 
'''



#########################################
#Load MNIST
#########################################

# import gzip
# with gzip.open('mnist.pkl.gz', 'rb') as f:
with open(home+ '/storage/mnist.pkl', 'rb') as f:
	train_set, valid_set, test_set = pickle.load(f)

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

print 'Train'
print train_x.shape
print train_y.shape
print "Valid"
print valid_x.shape
print valid_y.shape
print 'Test'
print test_x.shape
print test_y.shape

#########################################
#Train VAE and IWAE
#########################################
timelimit = 300
f_height=28
f_width=28
batch_size = 20
n_particles = 10

network_architecture = \
    dict(n_hidden_recog_1=200, # 1st layer encoder neurons
         n_hidden_recog_2=200, # 2nd layer encoder neurons
         n_hidden_gener_1=200, # 1st layer decoder neurons
         n_hidden_gener_2=200, # 2nd layer decoder neurons
         n_input=f_height*f_width, # 784 image
         n_z=50)  # dimensionality of latent space


network_architecture_for_NQAE = \
    dict(n_hidden_recog_1=200, # 1st layer encoder neurons
         n_hidden_recog_2=200, # 2nd layer encoder neurons
         n_hidden_recog_3=200,
         n_hidden_gener_1=200, # 1st layer decoder neurons
         n_hidden_gener_2=200, # 2nd layer decoder neurons
         n_input=f_height*f_width, # 784 image
         n_z=50,
         rnn_state_size=10)  # dimensionality of latent space

print 'Training NQAE'
nqae = NQAE(network_architecture_for_NQAE, transfer_fct=tf.tanh, learning_rate=0.001, batch_size=batch_size, n_particles=n_particles)
nqae.train(train_x=train_x, train_y=train_y, timelimit=timelimit, max_steps=99999, display_step=50, path_to_load_variables='', path_to_save_variables=home+ '/VAE/quae.ckpt')

print 'Training VAE'
vae = VAE(network_architecture, transfer_fct=tf.tanh, learning_rate=0.001, batch_size=batch_size, n_particles=n_particles)
vae.train(train_x=train_x, train_y=train_y, timelimit=timelimit, max_steps=99999, display_step=50, path_to_load_variables='', path_to_save_variables=home+ '/VAE/vae.ckpt')

print 'Training IWAE'
iwae = IWAE(network_architecture, transfer_fct=tf.tanh, learning_rate=0.001, batch_size=batch_size, n_particles=n_particles)
iwae.train(train_x=train_x, train_y=train_y, timelimit=timelimit, max_steps=99999, display_step=50, path_to_load_variables='', path_to_save_variables=home+ '/VAE/iwae.ckpt')



#########################################
#Sample 5000 times from them
#Get log likelihood of the test set
#########################################
print 'Negative Log Likelihood'

print 'vae means'
x_means_VAE = []
for i in range(5000/batch_size):
	generation = vae.generate()
	for j in range(len(generation)):
		x_means_VAE.append(generation[j])

print 'iwae means'
x_means_IWAE = []
for i in range(5000/batch_size):
	generation = iwae.generate()
	for j in range(len(generation)):
		x_means_IWAE.append(generation[j])

print 'nqae means'
x_means_NQAE = []
for i in range(5000/batch_size):
	generation = nqae.generate()
	for j in range(len(generation)):
		x_means_NQAE.append(generation[j])

def neg_log_likelihood(test_data, means):

	neg_log_like = 0
	for i in range(len(test_data)):
		if i % 50 == 0:
			print i
		for j in range(len(means)):

			a = test_data[i] * np.log(means[j])
			b = (1-test_data[i]) * np.log(1- means[j])
			c = a + b
			d = np.sum(c)
			neg_log_like += d

	return (-neg_log_like / len(test_data)) / len(means)


print 'VAE', neg_log_likelihood(test_x, x_means_VAE)
print 'IWAE', neg_log_likelihood(test_x, x_means_IWAE)
print 'NQAE', neg_log_likelihood(test_x, x_means_NQAE)

#########################################
#Visualize generated data
#########################################
import matplotlib.cm as cm
import matplotlib.pyplot as plt
generation = iwae.generate()[0]
plt.imshow(generation.reshape((28, 28)), cmap=cm.Greys_r)
plt.show()











