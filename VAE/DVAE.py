#Dynamic VAE



import numpy as np
import tensorflow as tf

from os.path import expanduser
home = expanduser("~")


class DynamicVariationlAutoencoder():

	def __init__(self, batch_size=5):

		self.graph=tf.Graph()

		self.batch_size = batch_size
		self.image_height = 100
		self.image_width = 100
		self.n_channels = 1
		self.n_time_steps = 10
		
		self.filter_height = 5
		self.filter_width = 5
		self.filter_out_channels1 = 2

		self.state_size = 100

		self.learning_rate = .0001


		with self.graph.as_default():
			#PLACEHOLDERS
			self.x = tf.placeholder("float", shape=[self.batch_size, self.n_time_steps, self.image_height, self.image_width, self.n_channels])
		
			#VARIABLES



		# tf Graph input
		self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
		
		# Create autoencoder network
		self._create_network()
		# Define loss function based variational upper-bound and 
		# corresponding optimizer
		self._create_loss_optimizer()
		
		# Initializing the tensor flow variables
		init = tf.initialize_all_variables()

		# Launch the session
		self.sess = tf.InteractiveSession()
		self.sess.run(init)


	def transition_network(self, state):
		'''
		Takes in a state and outputs the means and variances of the next state
		'''


		return

	def emission_network(self, state):
		'''
		aka decoder matrix
		Input: state/z_t
		Output: observation/image/x_t distribution
		'''


		return

	def calc_loss(self):

		return


	def train_minibatch(self):

		return

	def transform(self):

	def generate(self):

	def reconstruct(self):






















