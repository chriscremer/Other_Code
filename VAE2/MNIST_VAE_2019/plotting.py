


import numpy as np

from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def plot_curve2(data_dict, exp_dir):

    rows = len(data_dict) - 1
    cols = 1
    fig = plt.figure(figsize=(8+cols,8+rows), facecolor='white') #, dpi=150)

    steps = data_dict['steps']
    col=0
    row=0
    for k,v in data_dict.items():
        if k == 'steps':
            continue

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

        if type(v) == dict:
            for k2,v2 in v.items():
                ax.plot(steps, v2, label=k2+' '+str(v2[-1]))

        else:
            ax.plot(steps, v, label=v[-1])

        ax.legend()
        ax.grid(True, alpha=.3) 
        ax.set_ylabel(k)

        if row==0:
            ax.set_title(exp_dir)

        row+=1

    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = exp_dir+'curveplot.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()







def plot_images(real, recon, image_dir, step):

    def make_image_subplot(rows, cols, row, col, image, text):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False)
       
        # image = image.data.cpu().numpy() * 255.
        # image = np.rollaxis(image, 1, 0)
        # image = np.rollaxis(image, 2, 1)# [112,112,3]
        image = np.reshape(image, [28,28])
        # image = np.uint8(image)
        ax.imshow(image, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)


    cols = 2
    rows = 5
    fig = plt.figure(figsize=(4+cols,2+rows), facecolor='white') #, dpi=150)

    # print (np.max(recon))
    # print (np.min(recon))

    for i in range(rows):
        make_image_subplot(rows, cols, row=i, col=0, image=real[i], text='')
        make_image_subplot(rows, cols, row=i, col=1, image=recon[i], text='')


    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = image_dir+'images_'+str(step)+'.png'
    plt.savefig(plt_path)
    print ('saved images plot', plt_path)
    plt.close()






























































