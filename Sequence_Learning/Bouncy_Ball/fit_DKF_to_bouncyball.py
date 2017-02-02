


import numpy as np

from os.path import expanduser
home = expanduser("~")

from bouncing_ball_1d import get_sequence
from DKF7 import DKF

#Define the sequence
n_timesteps = 1
vector_height = 8

def get_data():

    sequence = get_sequence(n_timesteps = n_timesteps, vector_height = vector_height)

    action = [0]*n_timesteps
    action = np.array(action)
    action = np.reshape(action, [-1,1])

    return sequence, action


#Specify where to save stuff

save_to = home + '/data/' #for boltz
# save_to = home + '/Documents/tmp/' # for mac
# path_to_load_variables=save_to + 'dkf_ball_vars.ckpt'
path_to_load_variables=''
path_to_save_variables=save_to + 'bouncy_ball.ckpt'
# path_to_save_variables=''



#Define the model setup

training_steps = 8000
n_input = vector_height
n_time_steps = n_timesteps
batch_size = 2

train = 1
visualize = 1


network_architecture = \
    dict(   encoder_net=[20],
            decoder_net=[20],
            trans_net=[20],
            n_input=n_input,
            n_z=4,  
            n_actions=1) 

print 'Initializing model..'
dkf = DKF(network_architecture, batch_size=batch_size, n_time_steps=n_time_steps)

if train==1:
    print 'Training'
    dkf.train(get_data=get_data, steps=training_steps, display_step=20, 
                path_to_load_variables=path_to_load_variables, 
                path_to_save_variables=path_to_save_variables)


if visualize==1:
    import scipy.misc
    print 'Visualizing'

    # sequence = dkf.get_generated_sequence(n_timesteps, path_to_load_variables=path_to_save_variables)

    # sequence = dkf.get_generated_sequence_given_x_frames(n_timesteps, path_to_load_variables=path_to_save_variables, get_data)

    sequence = dkf.reconstruct_frame(n_timesteps=n_timesteps, path_to_load_variables=path_to_save_variables, get_data=get_data)

    print sequence.shape
    sequence = np.reshape(sequence, [-1, n_input])

    print sequence

    scipy.misc.imsave(save_to +'outfile.jpg', sequence)
    # scipy.misc.toimage(image_array, cmin=0.0, cmax=...).save('outfile.jpg')
    print 'saved sequence to ' + save_to +'outfile.jpg'














