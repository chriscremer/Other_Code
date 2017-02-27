# import sys
# sys.path.insert(0,'../..')

import numpy as np
import pickle

# import autograd_lstm as al

import my_rnn2 as my_rnn

from os.path import expanduser
home = expanduser("~")
import imageio

with open('pendulum_frames.pkl', 'r') as f:
    sequences = pickle.load(f)

n_sequences = len(sequences) #number of sequences
n_timesteps = len(sequences[0]) #timesteps
# print len(sequences[0][0]) #action frame tuple
action_size = len(sequences[0][0][0]) #action
frame_size = sequences[0][0][1].shape #frame
frame_length = frame_size[0] * frame_size[1]

print str(n_sequences)+' sequences. ' + str(n_timesteps) + ' timesteps'
print str(frame_size) + ' frame size'

frames = []
actions = []
for i in range(n_sequences):
    frame_sequence = []
    action_sequence = []
    for j in range(n_timesteps):
        frame_sequence.append(np.reshape(sequences[i][j][1], [-1]))
        action_sequence.append(sequences[i][j][0])
    frames.append(frame_sequence)
    actions.append(action_sequence)

state_size = 30
batch_size = 25

save_to = home + '/Documents/tmp/' # for mac
path_to_save_variables=save_to + 'rnn_vars.ckpt'
# path_to_save_variables=''



# now use rnn to make predictions
rnn = my_rnn.RNN(input_size=frame_length, action_size=action_size, state_size=state_size, output_size=frame_length, max_sequence_length=n_timesteps, batch_size=batch_size)

# print 'Training'
# rnn.train(frames=frames, actions=actions, steps=10000, display_step=20, path_to_load_variables='', path_to_save_variables=path_to_save_variables)

print 'Testing'
gen, real = rnn.test(frames=frames, actions=actions, path_to_load_variables=path_to_save_variables)

#take one from batch
gen = gen[:,0]
print gen.shape
real = np.array(real)[0]
print real.shape
# print real.shape



#MAKE GIFs

#selecting one sequence to view
# to_convert_to_gif = []
# for t in range(len(sequences[0])):
#     to_convert_to_gif.append(sequences[0][t][1])

def make_gif(to_convert_to_gif, name):    
    #convert to uint8
    for i in range(len(to_convert_to_gif)): 
        to_convert_to_gif[i] = to_convert_to_gif[i] * (255. / np.max(to_convert_to_gif[i]))
        to_convert_to_gif[i] = to_convert_to_gif[i].astype('uint8')

    #convert to gif
    kargs = { 'duration': .3 }
    imageio.mimsave(home+'/Downloads/' + name + '.gif', to_convert_to_gif, 'GIF', **kargs)
    print 'saved to ' + home+'/Downloads/'+ name + '.gif'

# [t,h,w]
gen = np.reshape(gen, [n_timesteps, frame_size[0], frame_size[1]])

# make_gif(to_convert_to_gif=to_convert_to_gif, name='gen_pendulum')

real = np.array(real)
real = np.reshape(real, [n_timesteps, frame_size[0], frame_size[1]])

to_gif = np.concatenate((gen, real), axis=2)

make_gif(to_convert_to_gif=to_gif, name='pendulum')












