
import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import imageio

def make_ball_gif_with_various_speeds(n_frames=3, f_height=30, f_width=30, ball_size=5, max_1=False, vector_output=False):
    
    row = random.randint(0,f_height-ball_size-1)
    # speed = random.randint(1,9)
    speed = random.randint(1,3)
    # speed= 1
    
    gif = []
    for i in range(n_frames):

        hot = np.zeros([f_height,f_width])
        if i*speed+ball_size >= f_width:
            hot[row:row+ball_size:1,f_width-ball_size:f_width+ball_size:1] = 255.
        else:
            hot[row:row+ball_size:1,i*speed:i*speed+ball_size:1] = 255.
        gif.append(hot.astype('uint8'))


    gif = np.array(gif)

    if max_1:
        for i in range(len(gif)):
            gif[i] = gif[i] / np.max(gif[i])


    if n_frames == 1:
        if vector_output:
            gif = np.reshape(gif, [f_height*f_width])
    else:
        if vector_output:
            gif = np.reshape(gif, [n_frames,f_height*f_width])

    return gif




def make_ball_with_actions(n_frames=3, f_height=30, f_width=30, ball_size=5, max_1=False, vector_output=False):

    sequence = []
    action_list = []
    blank_frame = np.zeros([f_height,f_width])
    pos = [f_height/2, f_width/2] #top_left_of_ball
    # init_frame[pos[0]:pos[0]+ball_size, pos[1]:pos[1]+ball_size] = 255.

    #For each time step
    for i in range(n_frames):
        #Actions: up, down, right, left
        action = np.random.choice(4, 1, p=[.25, .25, .25, .25])[0]

        if action == 0:
            if pos[0] + 1 < f_height-1:
                pos[0] = pos[0] + 1
        elif action == 1:
            if pos[0] - 1 >= 0:
                pos[0] = pos[0] - 1
        elif action == 2:
            if pos[1] + 1 < f_width-1:
                pos[1] = pos[1] + 1
        elif action == 3:
            if pos[1] - 1 >= 0:
                pos[1] = pos[1] - 1

        new_frame = np.zeros([f_height,f_width])
        new_frame[pos[0]:pos[0]+ball_size, pos[1]:pos[1]+ball_size] = 255.
        sequence.append(new_frame)
        action_list.append(action)

    return np.array(sequence), action




if __name__ == "__main__":

    sequence, action_list = make_ball_with_actions(n_frames=20, f_height=30, f_width=30, ball_size=5, max_1=False, vector_output=False)
    kargs = { 'duration': .6 }
    imageio.mimsave(home+"/Downloads/sequence_gif.gif", sequence, 'GIF', **kargs)
    print 'saved', home+"/Downloads/sequence_gif.gif"

    print 'DONE'









