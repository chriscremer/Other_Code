


import imageio


import os
from os.path import expanduser
home = expanduser("~")


# dir_ = home+ '/Downloads/frames_a2c_dropout_6000000'
dir_ = home+ '/Documents/tmp/a2c_reg_and_dropout_pong2/PongNoFrameskip-v4/a2c_dropout/seed0/frames_a2c_dropout_PongNoFrameskip-v4_6000000'


print('making gif')
max_epoch = 0
for file_ in os.listdir(dir_):
    if 'plt' in file_:
        numb = file_.split('plt')[1].split('.')[0]
        numb = int(numb)
        if numb > max_epoch:
            max_epoch = numb
        # print (numb)
        # fadsfa

# print ('max_epoch', max_epoch)
# fadad


images = []
# for file_ in os.listdir(dir_):
for i in range(max_epoch+1):
    # print(file_)
    # fsdfa

    # images.append(imageio.imread(dir_+'/'+file_))
    images.append(imageio.imread(dir_+'/'+'plt'+str(i)+'.png'))

    
imageio.mimsave(dir_+'/movie.gif', images)
print ('made gif', dir_+'/movie.gif')



