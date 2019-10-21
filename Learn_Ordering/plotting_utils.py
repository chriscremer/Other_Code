





def make_contour_subplot(rows, cols, row, col, image, text, legend=False):

    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False)
   
    # print (image.shape)
    image = image.view(112,112) 
    image = image.data.cpu().numpy() 
    image = np.rot90(image)
    image = np.rot90(image)
    image = np.flip(image,1)
    # image = np.uint8(image)
    cs = ax.contourf(image, cmap='Blues')
    # ax.legend()
    ax.set_aspect('equal')

    # # if legend:
    # h1,l1 = cs.legend_elements()
    # ax.legend(h1, l1, fontsize = 'x-small', loc=5)



    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)






def make_image_subplot(rows, cols, row, col, image, text):

    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False)
   
    if image.shape[0] != 1:
        image = image.data.cpu().numpy() * 255.
        image = np.rollaxis(image, 1, 0)
        image = np.rollaxis(image, 2, 1)# [112,112,3]
        image = np.uint8(image)
        ax.imshow(image) 

    else:
        image = image.view(112,112) * 255.
        image = image.data.cpu().numpy() 
        image = np.uint8(image)
        ax.imshow(image, cmap='gray')


    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)


def plot_curve2(results_dict, exp_dir):

    rows = len(results_dict) - 1
    cols = 1
    fig = plt.figure(figsize=(8+cols,8+rows), facecolor='white') #, dpi=150)

    steps = results_dict['steps']
    col=0
    row=0
    for k,v in results_dict.items():
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

