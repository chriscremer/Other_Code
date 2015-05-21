


import numpy as np


class Sigmoid_Activation:

	def function(z):
	    """The sigmoid function."""
	    return 1.0/(1.0+np.exp(-z))

	function_vec = np.vectorize(function)

	def function_prime(z):
	    """Derivative of the sigmoid function."""
	    return (1.0/(1.0+np.exp(-z))) * (1-(1.0/(1.0+np.exp(-z))))

	function_prime_vec = np.vectorize(function_prime)


class Linear_Activation:

	def function(z):
	    
	    return z

	function_vec = np.vectorize(function)

	def function_prime(z):
	    
	    return 1

	function_prime_vec = np.vectorize(function_prime)