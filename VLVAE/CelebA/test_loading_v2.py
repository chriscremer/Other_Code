

# this version doesnt use imageloader since I need to pair text and images
# and this one gets the associated attributes

import numpy as np
import os
from PIL import Image

import csv

import pickle


import torch


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# img = 'celeb_data/img_align_celeba/000001.jpg'
# jpgfile = Image.open(img)
# print(jpgfile.bits, jpgfile.size, jpgfile.format)









# # go through attributes, keep teh ones I want, save info to pickle .
    # make it a dict


# print (attr_indexes)
# fdsa

def get_attr_dict():

    attr_file = 'celeb_data/list_attr_celeba.txt'

    attr_to_keep = ['Bushy_Eyebrows',
    'Male',
    'Mouth_Slightly_Open',
    'Smiling',
    'Bald',
    'Bangs',
    'Black_Hair' ,
    'Blond_Hair' ,
    'Brown_Hair' ,
    'Eyeglasses' ,
    'Gray_Hair',
    'Heavy_Makeup' ,
    'Mustache' ,
    'Pale_Skin',
    'Receding_Hairline' ,
    'Straight_Hair' ,
    'Wavy_Hair',
    'Wearing_Hat']

    attr_order=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 
            'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 
            'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 
            'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 
            'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 
            'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 
            'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 
            'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 
            'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young', '']

    attr_indexes = [attr_order.index(x) for x in attr_to_keep]

    # print (attr_indexes)
    # dfasf


    attr_dict = {}


    with open(attr_file, 'r') as f:
        spamreader = csv.reader(f)#, delimiter=' ')
        count = 0
        for row in spamreader:
            # print (row)
            # print(row[0].split())
            # row.remove('')
            # print (row)
            row = row[0].split()

            

            if count in [0,1]:
                count+=1
                continue

            numb = row[0]


            

            attr_dict[numb] = {}

            # # print (row)
            # for i in range(len(row)-1):
            #     print (i, row[i+1])

            for i in range(len(attr_to_keep)):


                # if i == 22:
                #     print (numb, attr_to_keep[i], row[attr_indexes[i]])

                #     fsadfa

                # print (len (row))
                # sadf


                attr_dict[numb][attr_to_keep[i]] = row[attr_indexes[i] + 1]
                # attr_dict[numb][attr_to_keep[i]] = row[1:][attr_indexes[i]]


                # print (numb, attr_to_keep[i], row[attr_indexes[i]+1])

            # fdsfd

            count+=1

            # if count > 10:

            #     # print (attr_dict)
            #     fdsf

    return attr_dict
        

# # TO GET THE DATA , UNCOMMENT THIS STUFF
# data_dir1 = 'celeb_data/'
# attr_dict = get_attr_dict()
# with open(data_dir1+"attr_dict.pkl", "wb" ) as f:
#     pickle.dump(attr_dict, f)
# print ('dumped to pickle')


# fasfas




data_dir1 = 'celeb_data/'
with open(data_dir1 + "attr_dict.pkl", "rb" ) as f:
    attr_dict = pickle.load(f)

# print(attr_dict['000001.jpg'])
# fsdsa



def get_text(image_atts):

    text = ''
    for key, value in image_atts.items():
        # print (key,value)
        # if key in ['Brown_Hair', 'Receding_Hairline', 'Pale_Skin']:
        # if key in ['Receding_Hairline', 'Pale_Skin']:
        #     continue
        if key == 'Male':
            if value == '1':
                text += ' ' + key
            else:
                text += ' ' + 'Female'

        elif value == '1':
            text += ' ' + key

    # print (text)
    # fasd
    return text





data_dir = 'celeb_data/img_align_celeba/'


print (len(list(os.listdir('celeb_data/img_align_celeba/'))))
n_images = len(list(os.listdir('celeb_data/img_align_celeba/')))

#Make batch
start_idx = 33
batch_size = 25
batch = []
texts = []
idxs = list(range(start_idx,start_idx+batch_size))
for i in idxs:

    numb = str(i) 

    # print (len(numb))
    while len(numb) != 6:
        numb = '0' + numb


    img_file = data_dir+ numb + '.jpg'


    jpgfile = Image.open(img_file)
    img = np.array(jpgfile)

    # print (img.shape)
    img = np.rollaxis(img, 2, 1)# [112,112,3]
    
    # print (img.shape)
    img = np.rollaxis(img, 1, 0)

    img = img / 255.

    # print (np.max(img), np.min(img))
    # # print (img.shape)

    # fsdfa

    
    text = get_text(attr_dict[numb + '.jpg'])
    texts.append(text)
    # print (i, numb, attr_dict[numb + '.jpg'])
    print (i, numb, text)
    print ()


    


    batch.append(img)

    # print (img.shape)
    # fsd



batch = np.array(batch)


batch = torch.Tensor(batch)

print (batch.shape)


x1, y1 = 48, 25
downsample = torch.nn.AvgPool2d(2,2,0)


cols = 3
rows = batch_size
fig = plt.figure(figsize=(7+cols,2+rows), facecolor='white', dpi=150)

for i in range(rows):

    # images= data[:, :, x_l:x_r, x_b:x_t][0]

    image = batch[i]
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
    ax.text(0.1, 1.04, str(idxs[i]) + ' ' +str(i), transform=ax.transAxes, family='serif', size=6)

    ax = plt.subplot2grid((rows,cols), (i,1), frameon=False)
    image = cropped_image.data.cpu().numpy() * 255.
    image = np.rollaxis(image, 1, 0)
    image = np.rollaxis(image, 2, 1)# [112,112,3]
    image = np.uint8(image)
    ax.imshow(image) #, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.1, 1.04, texts[i], transform=ax.transAxes, family='serif', size=6)


    ax = plt.subplot2grid((rows,cols), (i,2), frameon=False)
    image = downsampled_image.data.cpu().numpy() * 255.
    image = np.rollaxis(image, 1, 0)
    image = np.rollaxis(image, 2, 1)# [112,112,3]
    image = np.uint8(image)
    ax.imshow(image) #, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])      


# plt.tight_layout()
plt_path = 'img4825.png'
plt.savefig(plt_path)
print ('saved viz',plt_path)
plt.close(fig)











fdsadsa