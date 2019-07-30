


import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt






def plot_cov(zs, img_dir, step):

    z_mean = np.mean(zs, 0)
    z_sd = np.std(zs, 0)
    centered_z = zs - z_mean
    # print (centered_z.shape)
    cov = 0
    for i in range (len(centered_z)):
        aa = np.expand_dims(centered_z[i],1)
        cov += np.matmul(aa, aa.T)
        # print (cov.shape)
        # fadsf
        # cov += centered_z[i]
    cov = cov  / float(len(zs))

    #correlation
    cov = cov / np.expand_dims(z_sd,0)
    cov = cov / np.expand_dims(z_sd,1)
    # print (np.max(cov), np.min(cov))
    # fadfs

    cov = np.abs(cov)

    # max_ = np.max(cov)
    # cov = cov / max_
    
    # print (np.max(cov), np.min(cov))
    # fsda

    rows = 1
    cols = 1
    fig = plt.figure(figsize=(8+cols,8+rows), facecolor='white') #, dpi=150)

    col=0
    row=0
    ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    ax.imshow(cov)
    # ax.imshow(cov, cmap='Blues')

    # steps = data_dict['steps']
    # col=0
    # row=0
    # for k,v in data_dict.items():
    #     if k == 'steps':
    #         continue

    #     ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

    #     if type(v) == dict:
    #         for k2,v2 in v.items():
    #             ax.plot(steps, v2, label=k2+' '+str(v2[-1]))

    #     else:
    #         ax.plot(steps, v, label=v[-1])

    #     ax.legend()
    #     ax.grid(True, alpha=.3) 
    #     ax.set_ylabel(k)

    #     if row==0:
    #         ax.set_title(img_dir)

    #     row+=1

    # save_dir = home+'/Documents/Grad_Estimators/GMM/'
    plt_path = img_dir+'cov'+str(step)+'.png'
    plt.savefig(plt_path)
    print ('saved training plot', plt_path)
    plt.close()

