


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
       


        if n_channels ==1:
            image = np.reshape(image, [28,28])
            ax.imshow(image, cmap='gray')
        else:
            # image = image.data.cpu().numpy() * 255.
            image = image * 255.
            image = np.rollaxis(image, 1, 0)
            image = np.rollaxis(image, 2, 1)# [112,112,3]
            image = np.uint8(image)
            ax.imshow(image)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)

    n_channels = real.shape[1]

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








# from attributes
def get_sentence(question_idx_to_token, list_of_word_idxs, newline_every=[]): #, answer):
    sentence =''
    list_of_word_idxs = list_of_word_idxs.cpu().numpy()#[0]
    for i in range(len(list_of_word_idxs)):
        word = question_idx_to_token[int(list_of_word_idxs[i])]
        sentence += ' ' + word
        if i in newline_every:
        # print(newline_every)
        # if (i+1) % newline_every==0 and i!=0:
            # print ('fdsas')
            sentence += '\n'
    return sentence




def plot_images_dif_zs(image1, image2, text1, text2, new_image1, new_image2, image_dir, step):


    def make_text_subplot(rows, cols, row, col, text, above_text=''):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1)
        ax.text(-0.15, .2, text, transform=ax.transAxes, family='serif', size=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.19, 1.08, above_text, transform=ax.transAxes, family='serif', size=6)


    def make_image_subplot(rows, cols, row, col, image, text):

        ax = plt.subplot2grid((rows,cols), (row,col), frameon=False)
       
        if n_channels ==1:
            image = np.reshape(image, [28,28])
            ax.imshow(image, cmap='gray')
        else:
            # image = image.data.cpu().numpy() * 255.
            image = image * 255.
            image = np.rollaxis(image, 1, 0)
            image = np.rollaxis(image, 2, 1)# [112,112,3]
            image = np.uint8(image)
            ax.imshow(image)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)




    n_channels = image1.shape[1]

    cols = 3
    rows = 2
    fig = plt.figure(figsize=(4+cols,2+rows), facecolor='white') #, dpi=150)

    # print (np.max(recon))
    # print (np.min(recon))

    # for i in range(rows):
    make_text_subplot(rows, cols, row=0, col=0, text=text1)
    make_text_subplot(rows, cols, row=1, col=0, text=text2)
    make_image_subplot(rows, cols, row=0, col=1, image=image1[0], text='')
    make_image_subplot(rows, cols, row=1, col=1, image=image2[0], text='')
    make_image_subplot(rows, cols, row=0, col=2, image=new_image1[0], text='')
    make_image_subplot(rows, cols, row=1, col=2, image=new_image2[0], text='')


    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = image_dir+'images_dif_z_'+str(step)+'.png'
    plt.savefig(plt_path)
    print ('saved images plot', plt_path)
    plt.close()


























































