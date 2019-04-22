
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision

# data_path = '../' #img_align_celeba/'
data_path = 'celeb_data/'


train_dataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=torchvision.transforms.ToTensor()
)




train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=1,
    shuffle=True
)



print(len(train_dataset))
print(len(train_loader))


# fsd

# its [218,178]

# x_l = int(218/2 -32)
# x_r = int(218/2 +32)
# x_t = int(178/2 +32)
# x_b = int(178/2 -32)


downsample = torch.nn.AvgPool2d(2,2,0)
# im, x1, y1, x1 + 138, y1 + 138)

# for batch_idx, (data, target) in enumerate(train_loader):
#     # your training procedure
#     print (batch_idx)


# fasds

for batch_idx, (data, target) in enumerate(train_loader):
    # your training procedure
    print (batch_idx)
    print (data.shape)
    print (torch.max(data), torch.min(data))

    print (target)
    fsd


    x1, y1 = 48, 30


    cols = 3
    rows = 40
    fig = plt.figure(figsize=(7+cols,2+rows), facecolor='white', dpi=150)

    for i in range(rows):

        # images= data[:, :, x_l:x_r, x_b:x_t][0]

        image = data[i]
        # print (image.shape)

        cropped_image = image[:, x1:x1+128, y1:y1+128]

        # print (cropped_image.shape)

        downsampled_image = downsample(cropped_image)

        # print (downsampled_image.shape)
        # fsda


        ax = plt.subplot2grid((rows,cols), (i,0), frameon=False)
       
        image = image.data.cpu().numpy() * 255.
        image = np.rollaxis(image, 1, 0)
        image = np.rollaxis(image, 2, 1)# [112,112,3]
        image = np.uint8(image)
        ax.imshow(image) #, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)

        ax = plt.subplot2grid((rows,cols), (i,1), frameon=False)
        image = cropped_image.data.cpu().numpy() * 255.
        image = np.rollaxis(image, 1, 0)
        image = np.rollaxis(image, 2, 1)# [112,112,3]
        image = np.uint8(image)
        ax.imshow(image) #, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.text(0.1, 1.04, text, transform=ax.transAxes, family='serif', size=6)

        ax = plt.subplot2grid((rows,cols), (i,2), frameon=False)
        image = downsampled_image.data.cpu().numpy() * 255.
        image = np.rollaxis(image, 1, 0)
        image = np.rollaxis(image, 2, 1)# [112,112,3]
        image = np.uint8(image)
        ax.imshow(image) #, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])      


    # plt.tight_layout()
    plt_path = 'img4830.png'
    plt.savefig(plt_path)
    print ('saved viz',plt_path)
    plt.close(fig)







    fdfa



