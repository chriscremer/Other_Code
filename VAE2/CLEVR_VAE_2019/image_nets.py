


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




# class Generator(nn.Module):
#     def __init__(self, input_nc, output_nc, n_residual_blocks=9):
#         super(Generator, self).__init__()

#         # Initial convolution block       
#         model = [   nn.ReflectionPad2d(3),
#                     nn.Conv2d(input_nc, 64, 7),
#                     nn.InstanceNorm2d(64),
#                     nn.ReLU(inplace=True) ]

#         # Downsampling
#         in_features = 64
#         out_features = in_features*2
#         for _ in range(2):
#             model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
#                         nn.InstanceNorm2d(out_features),
#                         nn.ReLU(inplace=True) ]
#             in_features = out_features
#             out_features = in_features*2

#         # Residual blocks
#         for _ in range(n_residual_blocks):
#             model += [ResidualBlock(in_features)]

#         # Upsampling
#         out_features = in_features//2
#         for _ in range(2):
#             model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
#                         nn.InstanceNorm2d(out_features),
#                         nn.ReLU(inplace=True) ]
#             in_features = out_features
#             out_features = in_features//2

#         # Output layer
#         model += [  nn.ReflectionPad2d(3),
#                     nn.Conv2d(64, output_nc, 7),
#                     nn.Tanh() 
#                     ]

#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         return self.model(x)








class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



class Resize_2(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 10,28,28)



class Resize_3(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 10,16,16)









class Image_encoder(nn.Module):
    def __init__(self, input_nc, image_encoding_size, n_residual_blocks=3, input_size=112):
        super(Image_encoder, self).__init__()


        hs = 100 # hidden size

        in_features = 32 #64

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, in_features, 7),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        out_features = in_features*2
        for i in range(2):
            # if i ==1 :
            #     out_features = out_features*2
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # print (in_features, out_features)
        # fasf

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # [256*28*28]
        model.append(nn.Conv2d(in_features, 10, 3, stride=1, padding=1)) #[20*28*28]
        model.append(nn.InstanceNorm2d(10))
        model.append(nn.ReLU(inplace=True))
        model.append(Flatten())

        if input_size ==112:
            model.append(nn.Linear(10*28*28, hs)) # 112
        elif input_size ==64:
            model.append(nn.Linear(10*16*16, hs))

        # model.append(nn.Linear(10*14*14, 1000))  #56
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Linear(hs, image_encoding_size))
        

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # print (self.model(x).shape)
        # fsdf
        return self.model(x)






class ResidualBlock2(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock2, self).__init__()

        conv_block = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(in_features, in_features, 7),
                        # nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(in_features, in_features, 7)]
                        # nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)





class Image_decoder(nn.Module):
    def __init__(self, input_size, output_nc, n_residual_blocks=3, output_dim=112):
        super(Image_decoder, self).__init__()

        model = []
        hs = 100 # hidden size

        model.append(nn.Linear(input_size, hs))
        model.append(nn.ReLU(inplace=True))

        if output_dim==112:
            model.append(nn.Linear(hs, 10*28*28))
            model.append(nn.ReLU(inplace=True))
            model.append(Resize_2())
        elif output_dim==64:
            model.append(nn.Linear(hs, 10*16*16))
            model.append(nn.ReLU(inplace=True))
            model.append(Resize_3())


        in_features = 128 #256 # 128


        model.append(nn.ConvTranspose2d(10, in_features, 3, stride=1, padding=1)) #, output_padding=1)) #[20*28*28]
        model.append(nn.InstanceNorm2d(in_features))
        
        # out_features = in_features*2


        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

          #[B,128,28,28]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # [32,32,112,112]
        # its half because I had to sample z.
        # print (in_features)
        # fasd


        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, output_nc, 7),
                    # nn.Tanh() 
                    ]


        # for _ in range(3):
        #     model += [ResidualBlock2(output_nc)]


        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)






