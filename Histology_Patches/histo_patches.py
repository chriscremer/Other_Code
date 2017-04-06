

#Load image
#Convert to greyscale
#Split it into patches
#Visualize and save patches.

import os  
import numpy as np
# import cv2  #for viz
# from skimage.measure import block_reduce

from PIL import Image

from os.path import expanduser
home = expanduser("~")

import scipy.misc

import pickle


# save_to = home + '/Downloads/'
save_to = home + '/Documents/histo_patches/'

images_location = home + '/Downloads/histology/imgs/'


patches = []
for file_ in os.listdir(images_location):
    print file_

    if 'jpg' in file_ and file_ != '0.jpg':

      

        # im = Image.open(home + '/Downloads/mS14-03950_5.jpg')
        # im = Image.open(home + '/Downloads/test_histo.png')
        # im = Image.open(home + '/Downloads/infiltrating_carcinoma_stomach.png')
        # im = Image.open(home + '/Downloads/infiltrating_prostate_carcinoma.png')
        # im = Image.open(home + '/Downloads/histology/imgs/0.jpg')


        im = Image.open(images_location+ file_)



        im = im.convert('L')

        image = np.array(im)

        shape = image.shape
        width = shape[0]
        length = shape[1]

        max_width = int(width / 28)
        max_length = int(length / 28)

        # print max_length
        # print max_width
        # print max_width * max_length #n patches




        # patches = []
        for i in range(max_length):
          for j in range(max_width):

              # left_corner = i*50
              y_ = i*28
              x_ = j*28
              patch = image[x_:x_+28, y_:y_+28]

              patches.append(patch)


        # print len(patches)
        # print patches[0].shape


        #TO SEE PATCHES

        # for i in range(20):
        #     # print i

        #     ind = np.random.randint(0,len(patches))

        #     # print ind


        #     if i ==0:
        #         # print patches[ind].shape
        #         all_patches = patches[ind]
        #     else:
        #         # print patches[ind].shape
        #         all_patches = np.concatenate([all_patches, patches[ind]], axis=1)

        #     # print all_patches.shape


        # scipy.misc.imsave(save_to +'patches.png', all_patches)
        # print 'saved to ', save_to +'patches.png'

print len(patches)


with open(save_to+'prostate_carcinoma_mar27_2017.pkl', 'w') as f:

    pickle.dump(patches,f)
    print 'saved to ', save_to+'prostate_carcinoma_mar27_2017.pkl'












