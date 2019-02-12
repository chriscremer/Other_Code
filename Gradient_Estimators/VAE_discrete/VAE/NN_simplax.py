

import torch

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Resize(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), input.size(1))

class Resize2(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), input.size(1),1,1)



class Surrogate(nn.Module):
    def __init__(self, kwargs, input_nc, image_encoding_size, n_residual_blocks=3, input_size=112, ):
        super(Surrogate, self).__init__()



        self.__dict__.update(kwargs)


        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]
        self.part1 = nn.Sequential(*model)



        model = []
        # Downsampling
        in_features = 64 + 3
        out_features = in_features*2
        # image_dim = input_size
        for i in range(2):
            # if i ==1 :
            #     out_features = out_features*2
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
            # image_dim = image_dim / 2
        self.part2 = nn.Sequential(*model)


        model = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        self.part3 = nn.Sequential(*model)


        model = []
        # [256*28*28]
        feats = 6*2 # mean and logvar
        n_channels_now = 268
        model.append(nn.Conv2d(n_channels_now, feats, 3, stride=1, padding=1)) #[20*28*28]
        # this is better alternative then reshaping to flatten
        model.append(nn.Conv2d(feats, 128, 8, stride=1, padding=0)) #[B,128,1,1] 
        model.append(Resize()) #[B,128] 
        model.append(nn.Linear(128, 1))
        self.part4 = nn.Sequential(*model)


        # model.append(nn.InstanceNorm2d(10))
        # model.append(nn.ReLU(inplace=True))


        # model.append(Flatten())
        # model.append(nn.Linear(feats*16*16, 1000))

        # # model.append(nn.Linear(10*14*14, 1000))  #56
        # model.append(nn.ReLU(inplace=True))
        # model.append(nn.Linear(1000, image_encoding_size))

        # self.model = nn.Sequential(*model)

        model = []
        model.append(nn.Linear(self.z_size, 128))
        model.append(Resize2()) #[B,128,1,1] 
        model.append(nn.ConvTranspose2d(128, 3, 32, stride=1, padding=0)) #[B,3,32,32] 
        self.part_z = nn.Sequential(*model)

    def forward(self, x, z):
        # print (self.model(x).shape)
        # fsdf
        # return self.model(x, z)

        out = self.part1(x)
        z = self.part_z(z)
        # print (out.shape, z.shape)
        out = torch.cat([out, z], dim=1) #[B,67,32,32]
        # print (out.shape)
        # fsafsd
        out = self.part2(out)
        # print (out.shape)
        out = self.part3(out)
        # print (out.shape)
        out = self.part4(out)
        # print (out.shape)

        # fasd


        return out 









