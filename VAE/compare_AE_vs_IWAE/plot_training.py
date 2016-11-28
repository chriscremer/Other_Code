




import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import matplotlib.pyplot as plt

from scipy.stats import logistic
# import math

# def sigmoid(x):
#   return 1 / (1 + math.exp(-x))



#View training errors
with open(home+ '/Downloads/elbos_stage6.pkl', 'rb') as f:
	data = pickle.load(f)

data = np.array(data)

print data.shape



plt.plot(data[10:,0],label="Training ELBO")
# plt.plot(data[10:,1])
plt.plot(data[10:,2])
plt.plot(data[10:,3], label='Validation ELBO')

# print data[10:,0]
# print data[10:,2]
plt.legend(fontsize=6)
plt.show()


fasda


#View reconstruction samples
with open(home+ '/Downloads/reconstructed_image.pkl', 'rb') as f:
	data = pickle.load(f)
	x = np.array(data[0])
	x_mean = data[1]
	print x.shape
	print x_mean.shape

	for samp in range(20):
		# image = np.array(x)[0][samp]
		image = x[samp]

		print image.shape
		pixels = logistic.cdf(image.reshape((28, 28)))
		plt.imshow(pixels, cmap='gray')
		plt.show()

		image = np.array(x_mean)[0][samp]
		pixels = logistic.cdf(image.reshape((28, 28)))
		plt.imshow(pixels, cmap='gray')
		plt.show()




#View generate samples
with open(home+ '/Downloads/generated_image.pkl', 'rb') as f:
	data = pickle.load(f)

	for samp in range(20):
		image = np.array(data)[0][0][samp]
		# print data.shape


		pixels = logistic.cdf(image.reshape((28, 28)))

		# Plot
		# plt.title('Label is {label}'.format(label=label))
		plt.imshow(pixels, cmap='gray')
		plt.show()








