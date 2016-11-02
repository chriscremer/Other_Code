
from os.path import expanduser
home = expanduser("~")
from PIL import Image
# import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import random
import imageio




def make_ball_gif(n_frames=10, f_height=10, f_width=10, ball_size=2):
    
    row = random.randint(0,f_height-ball_size-1)
    # speed = random.randint(1,9)
    speed = random.randint(1,2)

    
    #gif = np.zeros([n_frames,f_height,f_width])
    gif = []
    for i in range(n_frames):

        hot = np.zeros([f_height,f_width])
        hot[row:row+ball_size:1,i*speed:i*speed+ball_size:1] = 255
        gif.append(hot.astype('uint8'))
        
    return gif




gif = make_ball_gif()

gif = np.array(gif)
print gif.shape

kargs = { 'duration': .5 }
imageio.mimsave(home+"/Downloads/mygif2.gif", gif, 'GIF', **kargs)
print 'saved'
# imageio.imsave('pi.png', np.array(range(10)) * np.ones((10,10)))

#flatten
gif2 = []
for i in range(len(gif)):
	gif2.append(gif[i].flatten())

gif2 = np.array(gif2)
print gif2.shape

from pykalman import KalmanFilter

kf = KalmanFilter(n_dim_state=3, n_dim_obs=100)

measurements = gif2

kf = kf.em(measurements, n_iter=5)

(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)