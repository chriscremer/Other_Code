




import sys
sys.path.insert(0, '..')


from image_encoder_decoder import Generator_part1

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import os











class attribute_classifier(nn.Module):
    def __init__(self, kwargs):
        super(attribute_classifier, self).__init__()


        self.x_enc_size = kwargs['x_enc_size']
        self.y_enc_size = kwargs['y_enc_size']

        self.vocab_size = kwargs['vocab_size']
        self.q_max_len = kwargs['q_max_len']
        self.embed_size = kwargs['embed_size']

        self.linear_hidden_size = 500 #self.x_enc_size + self.y_enc_size
        self.linear_hidden_size2 = 200
        self.act_func = F.leaky_relu #kwargs['act_func'] #F.leaky_relu  # F.relu # # F.tanh ##  F.elu F.softplus




        # Attribute classifier
        self.image_encoder_for_classi = Generator_part1(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=3, input_size=kwargs['input_size'])
        self.classifier_fc1 = nn.Linear(self.x_enc_size, self.linear_hidden_size2)
        self.classifier_fc2 = nn.Linear(self.linear_hidden_size2, self.linear_hidden_size2)
        self.classifier_fc3 = nn.Linear(self.linear_hidden_size2, 19)

        self.classifier_params = [list(self.image_encoder_for_classi.parameters())  + list(self.classifier_fc1.parameters())  
                                + list(self.classifier_fc2.parameters()) + list(self.classifier_fc3.parameters())]


        self.optimizer_classi = optim.Adam(self.classifier_params[0], lr=.0002, weight_decay=.0000001)



    def update(self, x, q):

        self.optimizer_classi.zero_grad()  
        classi_loss, acc = self.classifier_loss(x=x, attributes=q)
        classi_loss.backward()
        self.optimizer_classi.step()

        return classi_loss, acc





    def classifier(self, x):
        out = self.image_encoder_for_classi(x)
        out = self.act_func(self.classifier_fc1(out))
        out = self.act_func(self.classifier_fc2(out))
        out = self.classifier_fc3(out)
        return out


    def classifier_loss(self, x, attributes):

        B = x.shape[0]

        y_hat = self.classifier(x)
        loss =  F.binary_cross_entropy_with_logits(input=y_hat, target=attributes) #* self.w_logpy

        # print (y_hat)
        # print ()
        # print (torch.sigmoid(y_hat))
        # fads
        acc = self.accuracy(torch.sigmoid(y_hat), attributes)

        return loss, acc


    def accuracy(self, y_hat, y):

        # print (y_hat.shape, y.shape)

        B = y.shape[0]


        # y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)
        # y_hat = torch.max(y_hat, 1)[1]
        # y_hat = y_hat.contiguous().view(B,self.q_max_len)

        y_hat = (y_hat > .5).float()

        acc = torch.mean((y_hat == y).float())
        return acc


    def classifier_attribute_accuracies(self, x, attributes):

        B = x.shape[0]


        y_hat = self.classifier(x)
        y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat > .5).float()

        # # target_attributes = attributes.contiguous().view(self.B*self.q_max_len)
        # y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)
        # y_hat = torch.max(y_hat, 1)[1]
        # y_hat = y_hat.contiguous().view(B,self.q_max_len)

        acc = torch.mean((y_hat == attributes).float(), 0)

        # for 

        # print (acc)
        # fsadf
        # print (acc.shape)
        # fasffd
        return acc #acc[0], acc[1], acc[2], acc[3]



    def accuracy_sequence(self, x, attributes):

        y_hat = self.classifier(x)
        y_hat = torch.sigmoid(y_hat).data.cpu().numpy()

        accs =[]
        for i in range(len(x)):
            #convert sequence to 19 dim vec
            attributes1 = np.zeros((19))
            for j in range(len(attributes[i])):
                attributes1[attributes[i][j]-1] = 1.

            y_hat_b = (y_hat[i] > .5)
            acc = np.mean(y_hat_b == attributes1) #, 0)   
            accs.append(acc)

        return np.mean(accs)







    def load_params_v3(self, save_dir, step):
        save_to=os.path.join(save_dir, "classifier_params" + str(step)+".pt")
        self.load_state_dict(torch.load(save_to))
        print ('loaded params', save_to)





    def save_params_v3(self, save_dir, step):
        save_to=os.path.join(save_dir, "classifier_params" + str(step)+".pt")
        torch.save(self.state_dict(), save_to)
        print ('saved params', save_to)
        





















