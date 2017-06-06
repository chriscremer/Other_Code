

# Load problem
    # call it to get another batch of sequences. 

# Load models
# Functions that models have:
    # predict next frame given current frame.

#Since test set is the same as the training, plot the training error
# Also outupt real sequence vs predicted.
#Have baseline models, for example, predict same as last frame. 




import numpy as np

import matplotlib.pyplot as plt

from os.path import expanduser
home = expanduser("~")

import sys
sys.path.insert(0, './problems')
sys.path.insert(0, './models')

from block_moving import sequence










if __name__ == '__main__':

    #[B,T,X]
    seq_gen = sequence()
    seq_batch = seq_gen.get_batch()


    #Plot 
    # f,axarr=plt.subplots(1,2,figsize=(12,6))
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    # ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    # ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    # ax4 = plt.subplot2grid((3, 3), (2, 0))
    # ax5 = plt.subplot2grid((3, 3), (2, 1))


    #Concatenate the frames
    batch_index = 0
    for i in range(len(seq_batch[batch_index])):
        if i ==0:
            concat = seq_batch[batch_index][i]

        else:
            concat = np.concatenate([concat,np.ones([concat.shape[0],1])], axis=1)
            concat = np.concatenate([concat,seq_batch[batch_index][i]], axis=1)



    ax1.imshow(concat, cmap='gray')

    # axarr[0].hist(all_means, 100, facecolor='green', alpha=0.75)  #normed=True
    # axarr[0].set_title('Decoder Means')
    # axarr[0].set_xlim([-3.,3.])

    # plt.grid('off')
    # axarr[1,0].axis('off')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])

    plt.show()

    # plt.savefig(experiment_log_path+ m + '_k' + str(k_training) + '_z'+str(z_size) + '_epochs'+str(epochs)+'_smalldata.png')
    # print 'saved fig to' + experiment_log_path+ m + '_k' + str(k_training) + '_z'+str(z_size) + '_epochs'+str(epochs)+'_smalldata.png'













                    

