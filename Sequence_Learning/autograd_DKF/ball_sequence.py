

import numpy as np

def make_ball_gif(n_frames=3, f_height=30, f_width=30, ball_size=5, max_1=False, vector_output=False):

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
        action_array = np.zeros([4])
        action_array[action] = 1
        action_list.append(action_array)

    sequence = np.array(sequence)

    if max_1:
        for i in range(len(sequence)):
            sequence[i] = sequence[i] / np.max(sequence[i])

    if vector_output:
        if vector_output:
            sequence = np.reshape(sequence, [n_frames,f_height*f_width])

    return np.array(sequence), action_list