

#Make sequential data where a ball moves up and down a 1d vector
#Allows me to see the uncertainty in future predicitons 
# see fig 4 https://arxiv.org/pdf/1603.06277v3.pdf


import numpy as np

import matplotlib.pyplot as plt

import scipy.misc

from os.path import expanduser
home = expanduser("~")





def get_sequence(n_timesteps = 100, vector_height = 30, ball_speed = 1, direction = 1):

	# position = int(vector_height / 2)
	position = np.random.randint(0, vector_height)

	sequence = []
	for t in range(n_timesteps):

		position += ball_speed*direction

		if position < 0:
			position =0
			direction *= -1
		if position >= vector_height-1:
			position = vector_height-1
			direction *= -1

		state = np.zeros([vector_height])
		state[position] = 1.
		sequence.append(state)

	# [timesteps, vector_height]
	sequence = np.array(sequence)
	return sequence








if __name__ == "__main__":


	save_to = home + '/data/' #for boltz
	# save_to = home + '/Documents/tmp/' # for mac


	n_timesteps = 100
	vector_height = 30
	# ball_height = 3
	ball_speed = 1
	direction = 1
	position = int(vector_height / 2)



	#Initialize vector
	bouncy_ball = np.zeros([vector_height,1])

	#Randomly start the ball at a spot and a direction

	#Depending on speed, move the ball in the next vector

	sequence = []

	for t in range(n_timesteps):

		position += ball_speed*direction

		if position < 0:
			position =0
			direction *= -1
		if position >= vector_height-1:
			position = vector_height-1
			direction *= -1


		state = np.zeros([vector_height])
		state[position] = 1.

		sequence.append(state)



	sequence = np.array(sequence).T
	print sequence.shape

	#Continue for a number of steps

	scipy.misc.imsave(save_to +'outfile.jpg', sequence)

	#Visualize the concatenation of all timesteps. 



	#Also it would be really cool to see it learn in real time. 
	# So after each time step, show the predictions .







