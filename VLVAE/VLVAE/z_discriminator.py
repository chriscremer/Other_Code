


# from cyclegangenerator_bottleneck import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import os


import torch.autograd as autograd









class Discriminator(nn.Module):
    def __init__(self, kwargs):
        super(Discriminator, self).__init__()

        z_size = kwargs['z_size']
        # self.x_enc_size = kwargs['x_enc_size']
        # self.y_enc_size = kwargs['y_enc_size']

        # self.vocab_size = kwargs['vocab_size']
        # self.q_max_len = kwargs['q_max_len']
        # self.embed_size = kwargs['embed_size']

        # self.linear_hidden_size = 500 #self.x_enc_size + self.y_enc_size
        self.linear_hidden_size2 = 200
        self.act_func = F.leaky_relu  #kwargs['act_func'] #F.leaky_relu  # F.relu # # F.tanh ##  F.elu F.softplus




        # Attribute classifier
        # self.image_encoder_for_classi = Generator_part1(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=3)
        self.classifier_fc1 = nn.Linear(z_size, self.linear_hidden_size2)
        self.classifier_fc2 = nn.Linear(self.linear_hidden_size2, self.linear_hidden_size2)
        self.classifier_fc3 = nn.Linear(self.linear_hidden_size2, 1)


        self.classifier_params = [list(self.classifier_fc1.parameters())  
                                + list(self.classifier_fc2.parameters()) + list(self.classifier_fc3.parameters())]

        self.bn1 = nn.BatchNorm1d(self.linear_hidden_size2)
        self.bn2 = nn.BatchNorm1d(self.linear_hidden_size2)


        self.optimizer_classi = optim.Adam(self.classifier_params[0], lr=.0002, weight_decay=.0000001)



    def update(self, x1, x2):

        self.optimizer_classi.zero_grad()  
        classi_loss, acc = self.discrim_loss(x1=x1, x2=x2)
        classi_loss.backward()
        self.optimizer_classi.step()

        return classi_loss, acc





    def predict(self, x):
        # out = self.image_encoder_for_classi(x)
        out = self.act_func(self.bn1(self.classifier_fc1(x)))
        out = self.act_func(self.bn2(self.classifier_fc2(out)))
        out = self.classifier_fc3(out)
        return torch.sigmoid(out)


    def discrim_loss(self, x1, x2):

        

        x = torch.cat([x1,x2], 0)
        B = x.shape[0]
        y = torch.ones(B, 1).cuda()
        y[int(B/2):] = 0.

        # print (y)

        y_hat = self.predict(x)

        # target_attributes = attributes.contiguous().view(B*self.q_max_len)
        # y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)

        loss =  F.binary_cross_entropy(input=y_hat, target=y) #* self.w_logpy

        acc = self.accuracy(y_hat, y)

        return loss, acc


    def accuracy(self, y_hat, y):

        # print (y_hat.shape, y.shape)

        B = y.shape[0]


        # y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)
        # y_hat = torch.max(y_hat, 1)[1]
        # y_hat = y_hat.contiguous().view(B,self.q_max_len)
        y_hat = (y_hat > .5) #.float()

        acc = torch.mean((y_hat == y.byte()).float())
        return acc


    # def classifier_attribute_accuracies(self, x, attributes):

    #     B = x.shape[0]


    #     y_hat = self.classifier(x)

    #     # target_attributes = attributes.contiguous().view(self.B*self.q_max_len)
    #     y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)
    #     y_hat = torch.max(y_hat, 1)[1]
    #     y_hat = y_hat.contiguous().view(B,self.q_max_len)

    #     acc = torch.mean((y_hat == attributes).float(), 0)
    #     # print (acc.shape)
    #     # fasffd
    #     return acc[0], acc[1], acc[2], acc[3]






    def load_params_v3(self, save_dir, step):
        save_to=os.path.join(save_dir, "classifier_params" + str(step)+".pt")
        self.load_state_dict(torch.load(save_to))
        print ('loaded params', save_to)





    def save_params_v3(self, save_dir, step):
        save_to=os.path.join(save_dir, "classifier_params" + str(step)+".pt")
        torch.save(self.state_dict(), save_to)
        print ('saved params', save_to)
        





























class Discriminator2(nn.Module):
    def __init__(self, kwargs):
        super(Discriminator2, self).__init__()

        z_size = kwargs['z_size']
        self.linear_hidden_size2 = 200
        self.act_func = F.leaky_relu  #kwargs['act_func'] #F.leaky_relu  # F.relu # # F.tanh ##  F.elu F.softplus

        # Attribute classifier
        # self.image_encoder_for_classi = Generator_part1(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=3)
        self.classifier_fc1 = nn.Linear(z_size, self.linear_hidden_size2)
        self.classifier_fc2 = nn.Linear(self.linear_hidden_size2, self.linear_hidden_size2)
        self.classifier_fc3 = nn.Linear(self.linear_hidden_size2, 1)


        self.classifier_params = [list(self.classifier_fc1.parameters())  
                                + list(self.classifier_fc2.parameters()) + list(self.classifier_fc3.parameters())]

        self.bn1 = nn.BatchNorm1d(self.linear_hidden_size2)
        self.bn2 = nn.BatchNorm1d(self.linear_hidden_size2)


        self.optimizer_classi = optim.Adam(self.classifier_params[0], lr=.0002, weight_decay=.0000001)



    def update(self, x1, x2):

        self.optimizer_classi.zero_grad()  
        classi_loss, acc = self.discrim_loss(x1=x1, x2=x2)
        classi_loss.backward()
        self.optimizer_classi.step()

        return classi_loss, acc





    def predict(self, x):
        # out = self.image_encoder_for_classi(x)
        # out = self.act_func(self.bn1(self.classifier_fc1(x)))
        # out = self.act_func(self.bn2(self.classifier_fc2(out)))
        out = self.act_func(self.classifier_fc1(x))
        out = self.act_func(self.classifier_fc2(out))
        out = self.classifier_fc3(out)
        # return torch.sigmoid(out)
        return out



    def get_gradient_penalty(self, fake_target, real_target):

        # if txt_data is None:
        #     assert source_data is None

        # batch_size = real_target.size(0)

        alpha = torch.rand(fake_target.shape[0], 1).cuda() #, 1, 1)
        # alpha = alpha.to(real_target.device)

        interpolates = alpha * real_target + ((1. - alpha) * fake_target)


        interpolates.requires_grad = True
        # if source_data is None:
        disc_interpolates = self.predict(interpolates)
        # else:
        # disc_interpolates = predict(source_data, interpolates, txt_data)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(disc_interpolates),
                                  create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        # gradient_penalty = (gradients.norm(2, dim=1) ** 2).mean()
        return gradient_penalty






    def discrim_loss(self, real, generated):

        grad_w = .1

        disc_real = self.predict(real).mean()
        disc_fake = self.predict(generated).mean()
        disc_cost = disc_fake - disc_real
        grad_penalty = self.get_gradient_penalty(generated, real)
        disc_cost += grad_w * grad_penalty

        # print 
        

        # x = torch.cat([x1,x2], 0)
        # B = x.shape[0]
        # self.B = B
        # y = torch.ones(B, 1).cuda()
        # y[int(B/2):] = 0.

        # # print (y)

        # y_hat = self.predict(x)

        # # target_attributes = attributes.contiguous().view(B*self.q_max_len)
        # # y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)

        # loss =  F.binary_cross_entropy(input=y_hat, target=y) #* self.w_logpy

        # acc = self.accuracy(y_hat, y)

        # return loss, acc

        return disc_cost, disc_fake, disc_real, grad_w * grad_penalty


    # def accuracy(self, y_hat, y):

    #     # print (y_hat.shape, y.shape)

    #     B = y.shape[0]


    #     # y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)
    #     # y_hat = torch.max(y_hat, 1)[1]
    #     # y_hat = y_hat.contiguous().view(B,self.q_max_len)
    #     y_hat = (y_hat > .5) #.float()

    #     acc = torch.mean((y_hat == y.byte()).float())
    #     return acc


    # def classifier_attribute_accuracies(self, x, attributes):

    #     B = x.shape[0]


    #     y_hat = self.classifier(x)

    #     # target_attributes = attributes.contiguous().view(self.B*self.q_max_len)
    #     y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)
    #     y_hat = torch.max(y_hat, 1)[1]
    #     y_hat = y_hat.contiguous().view(B,self.q_max_len)

    #     acc = torch.mean((y_hat == attributes).float(), 0)
    #     # print (acc.shape)
    #     # fasffd
    #     return acc[0], acc[1], acc[2], acc[3]






    def load_params_v3(self, save_dir, step):
        save_to=os.path.join(save_dir, "classifier_params" + str(step)+".pt")
        self.load_state_dict(torch.load(save_to))
        print ('loaded params', save_to)





    def save_params_v3(self, save_dir, step):
        save_to=os.path.join(save_dir, "classifier_params" + str(step)+".pt")
        torch.save(self.state_dict(), save_to)
        print ('saved params', save_to)
        






































