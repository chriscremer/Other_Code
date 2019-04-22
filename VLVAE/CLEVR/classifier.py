

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

















class attribute_classifier_with_relations(nn.Module):
    def __init__(self, kwargs):
        super(attribute_classifier_with_relations, self).__init__()


        self.x_enc_size = kwargs['x_enc_size']
        self.y_enc_size = kwargs['y_enc_size']

        self.vocab_size = kwargs['vocab_size']
        self.q_max_len = kwargs['q_max_len']
        self.embed_size = kwargs['embed_size']

        self.linear_hidden_size = 500 #self.x_enc_size + self.y_enc_size
        self.linear_hidden_size2 = 200
        self.act_func = kwargs['act_func'] #F.leaky_relu  # F.relu # # F.tanh ##  F.elu F.softplus


        self.n_outputs = 8
        self.n_relations = kwargs['vocab_size']


        # Attribute classifier
        self.image_encoder_for_classi = Generator_part1(input_nc=4, image_encoding_size=self.x_enc_size, n_residual_blocks=3)

        # self.classifier_fc1 = nn.Linear(self.x_enc_size+self.n_relations, self.linear_hidden_size2)
        self.classifier_fc1 = nn.Linear(self.n_relations, 4)
        self.classifier_fc2 = nn.Linear(4, 12544)

        # self.classifier_fc3 = nn.Linear(self.linear_hidden_size2, self.linear_hidden_size2)
        # self.classifier_fc4 = nn.Linear(self.linear_hidden_size2, self.linear_hidden_size2)
        # self.classifier_fc5 = nn.Linear(self.linear_hidden_size2, self.linear_hidden_size2)
        self.classifier_fc6 = nn.Linear(self.x_enc_size, self.n_outputs*self.vocab_size)

        # self.bn1 = nn.BatchNorm1d(self.linear_hidden_size2)
        # self.bn2 = nn.BatchNorm1d(self.linear_hidden_size2)
        # self.bn3 = nn.BatchNorm1d(self.linear_hidden_size2)

        self.classifier_params = [list(self.image_encoder_for_classi.parameters())  + list(self.classifier_fc1.parameters())  
                                + list(self.classifier_fc2.parameters()) 
                                # + list(self.classifier_fc3.parameters())
                                # + list(self.classifier_fc4.parameters()) + list(self.classifier_fc5.parameters())
                                + list(self.classifier_fc6.parameters())
                                # + list(self.bn1.parameters()) + list(self.bn2.parameters()) + list(self.bn3.parameters())
                                ]

        self.optimizer_classi = optim.Adam(self.classifier_params[0], lr=.0002, weight_decay=.0000001)


        # self.encoder_embed = encoder_embed

        



    def update(self, x, q):

        self.optimizer_classi.zero_grad()  
        classi_loss, acc = self.classifier_loss(x=x, attributes=q)
        classi_loss.backward()
        self.optimizer_classi.step()

        return classi_loss, acc




    def classifier(self, x, relation):

        # print (x.shape)
        # print (relation.shape)

        relation = self.classifier_fc2(self.act_func(self.classifier_fc1(relation)))
        relation = relation.view(-1, 1, 112,112)
        x = torch.cat([x, relation], 1)

        # print(out.shape)

        # fsfasd

        out = self.image_encoder_for_classi(x)

        # out = torch.cat([out, relation], 1)
        # out = self.act_func(self.bn1(self.classifier_fc1(out)))

        # res = self.act_func(self.bn2(self.classifier_fc2(out)))
        # res = self.classifier_fc3(res)
        # out = out + res 

        # res = self.act_func(self.bn3(self.classifier_fc4(out)))
        # res = self.classifier_fc5(res)
        # out = out + res 

        out = self.classifier_fc6(out)

        return out





    # def classifier(self, x, relation):

    #     # print (x.shape)
    #     # print (relation.shape)

    #     # fsfasd

    #     out = self.image_encoder_for_classi(x)

    #     out = torch.cat([out, relation], 1)
    #     out = self.act_func(self.bn1(self.classifier_fc1(out)))

    #     res = self.act_func(self.bn2(self.classifier_fc2(out)))
    #     res = self.classifier_fc3(res)
    #     out = out + res 

    #     res = self.act_func(self.bn3(self.classifier_fc4(out)))
    #     res = self.classifier_fc5(res)
    #     out = out + res 

    #     out = self.classifier_fc6(out)

    #     return out



    # def classifier(self, x, relation):

    #     out = self.image_encoder_for_classi(x)

    #     out = torch.cat([out, relation], 1)

    #     out = self.act_func(self.bn1(self.classifier_fc1(out)))
    #     out = self.act_func(self.bn2(self.classifier_fc2(out)))
    #     out = self.classifier_fc3(out)
    #     return out






    def classifier2(self, x, attributes):

        # print(attributes.shape)

        B = x.shape[0]

        relation = attributes[:,4]

        self.y_onehot = torch.FloatTensor(B, self.vocab_size).cuda()
        self.y_onehot.zero_()
        self.y_onehot.scatter_(1, relation.view(B,1).long(), 1)
        relation = self.y_onehot

        
        attributes_less_relation = torch.cat([attributes[:,:4], attributes[:,5:]], 1)
        # print (attributes.shape)

        y_hat = self.classifier(x, relation)
        return y_hat





    def classifier_loss(self, x, attributes):

        # print(attributes.shape)

        B = x.shape[0]

        relation = attributes[:,4]

        self.y_onehot = torch.FloatTensor(B, self.vocab_size).cuda()
        self.y_onehot.zero_()
        self.y_onehot.scatter_(1, relation.view(B,1).long(), 1)
        relation = self.y_onehot

        # relation = self.encoder_embed(relation)

        # print (relation.shape)
        # fdsdafsdf


        # print (attributes.shape)
        # print (relation.shape)
        # print (relation)
        # print (embed.shape)
        # fdsf

        attributes_less_relation = torch.cat([attributes[:,:4], attributes[:,5:]], 1)
        # print (attributes.shape)

        y_hat = self.classifier(x, relation)

        target_attributes = attributes_less_relation.contiguous().view(B*self.n_outputs)
        y_hat = y_hat.contiguous().view(B*self.n_outputs, self.vocab_size)

        loss =  F.cross_entropy(input=y_hat, target=target_attributes) #* self.w_logpy

        # attributes = attributes.contiguous().view(B, self.n_outputs, self.vocab_size)
        acc = self.accuracy(y_hat, attributes)

        return loss, acc


    def accuracy(self, y_hat, y):

        B = y.shape[0]

        # print (y.shape)
        y = torch.cat([y[:,:4], y[:,5:]], 1)
        # print (y_hat.shape)

        

        if y_hat.shape[1] == self.n_outputs+1:
            y_hat = torch.cat([y_hat[:,:4], y_hat[:,5:]], 1)
        # elif:
        #     y_hat = y_hat.contiguous().view(B, self.n_outputs, self.vocab_size)

        # print (y_hat.shape)
        # print()

        y_hat = y_hat.contiguous().view(B*self.n_outputs, self.vocab_size)
        y_hat = torch.max(y_hat, 1)[1]
        y_hat = y_hat.contiguous().view(B,self.n_outputs)

        acc = torch.mean((y_hat == y).float())
        return acc


    def classifier_attribute_accuracies(self, x, attributes):

        B = x.shape[0]



        relation = attributes[:,4]

        self.y_onehot = torch.FloatTensor(B, self.vocab_size).cuda()
        self.y_onehot.zero_()
        self.y_onehot.scatter_(1, relation.view(B,1).long(), 1)
        relation = self.y_onehot


        attributes = torch.cat([attributes[:,:4], attributes[:,5:]], 1)

        y_hat = self.classifier(x, relation)

        # target_attributes = attributes.contiguous().view(self.B*self.q_max_len)
        y_hat = y_hat.contiguous().view(B*self.n_outputs, self.vocab_size)
        y_hat = torch.max(y_hat, 1)[1]
        y_hat = y_hat.contiguous().view(B,self.n_outputs)

        acc = torch.mean((y_hat == attributes).float(), 0)
        # print (acc.shape)
        # fasffd
        return acc[0], acc[1], acc[2], acc[3]






    def load_params_v3(self, save_dir, step):
        save_to=os.path.join(save_dir, "classifier_params" + str(step)+".pt")
        self.load_state_dict(torch.load(save_to))
        print ('loaded params', save_to)





    def save_params_v3(self, save_dir, step):
        save_to=os.path.join(save_dir, "classifier_params" + str(step)+".pt")
        torch.save(self.state_dict(), save_to)
        print ('saved params', save_to)
        

































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
        self.act_func = kwargs['act_func'] #F.leaky_relu  # F.relu # # F.tanh ##  F.elu F.softplus




        # Attribute classifier
        self.image_encoder_for_classi = Generator_part1(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=3)
        self.classifier_fc1 = nn.Linear(self.x_enc_size, self.linear_hidden_size2)
        self.classifier_fc2 = nn.Linear(self.linear_hidden_size2, self.linear_hidden_size2)
        self.classifier_fc3 = nn.Linear(self.linear_hidden_size2, self.q_max_len*self.vocab_size)
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

        target_attributes = attributes.contiguous().view(B*self.q_max_len)
        y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)

        loss =  F.cross_entropy(input=y_hat, target=target_attributes) #* self.w_logpy

        acc = self.accuracy(y_hat, attributes)

        return loss, acc


    def accuracy(self, y_hat, y):

        # print (y_hat.shape, y.shape)

        B = y.shape[0]


        y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)
        y_hat = torch.max(y_hat, 1)[1]
        y_hat = y_hat.contiguous().view(B,self.q_max_len)

        acc = torch.mean((y_hat == y).float())
        return acc


    def classifier_attribute_accuracies(self, x, attributes):

        B = x.shape[0]


        y_hat = self.classifier(x)

        # target_attributes = attributes.contiguous().view(self.B*self.q_max_len)
        y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)
        y_hat = torch.max(y_hat, 1)[1]
        y_hat = y_hat.contiguous().view(B,self.q_max_len)

        acc = torch.mean((y_hat == attributes).float(), 0)
        # print (acc.shape)
        # fasffd
        return acc[0], acc[1], acc[2], acc[3]






    def load_params_v3(self, save_dir, step):
        save_to=os.path.join(save_dir, "classifier_params" + str(step)+".pt")
        self.load_state_dict(torch.load(save_to))
        print ('loaded params', save_to)





    def save_params_v3(self, save_dir, step):
        save_to=os.path.join(save_dir, "classifier_params" + str(step)+".pt")
        torch.save(self.state_dict(), save_to)
        print ('saved params', save_to)
        








































class classifier_of_prior(nn.Module):
    def __init__(self, kwargs):
        super(classifier_of_prior, self).__init__()


        self.x_enc_size = kwargs['x_enc_size']
        self.y_enc_size = kwargs['y_enc_size']

        self.vocab_size = kwargs['vocab_size']
        self.q_max_len = kwargs['q_max_len']
        self.embed_size = kwargs['embed_size']

        self.linear_hidden_size = 500 #self.x_enc_size + self.y_enc_size
        self.linear_hidden_size2 = 200
        self.act_func = kwargs['act_func'] #F.leaky_relu  # F.relu # # F.tanh ##  F.elu F.softplus




        # Attribute classifier
        self.image_encoder_for_classi = Generator_part1(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=3)
        self.classifier_fc1 = nn.Linear(self.x_enc_size, self.linear_hidden_size2)
        self.classifier_fc2 = nn.Linear(self.linear_hidden_size2, self.linear_hidden_size2)
        self.classifier_fc3 = nn.Linear(self.linear_hidden_size2, 1)
        self.classifier_params = [list(self.image_encoder_for_classi.parameters())  + list(self.classifier_fc1.parameters())  
                                + list(self.classifier_fc2.parameters()) + list(self.classifier_fc3.parameters())]


        self.optimizer_classi = optim.Adam(self.classifier_params[0], lr=.0002, weight_decay=.0000001)

        self.dropout = nn.Dropout(.5)

    def update(self, x, q):

        self.optimizer_classi.zero_grad()  
        classi_loss, acc = self.classifier_loss(x=x, attributes=q)
        classi_loss.backward()
        self.optimizer_classi.step()

        return classi_loss, acc





    def classifier(self, x):
        out = self.image_encoder_for_classi(x)
        out = self.dropout(self.act_func(self.classifier_fc1(out)))
        out = self.dropout(self.act_func(self.classifier_fc2(out)))
        out = self.classifier_fc3(out)
        return out


    def classifier_loss(self, x, attributes):

        B = x.shape[0]

        y_hat = self.classifier(x)

        # target_attributes = attributes.contiguous().view(B*self.q_max_len)
        # y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)

        # print (y_hat.shape) #[30,1]
        y_hat = y_hat.view(B)
        # print (attributes.shape) #[30]

        # print (y_hat)
        # print (attributes)

        # print (F.sigmoid(y_hat))
        # print (F.sigmoid(y_hat) > .5)


        loss =  F.binary_cross_entropy_with_logits(input=y_hat, target=attributes) #* self.w_logpy

        # print ('jejej')

        acc = self.accuracy(y_hat, attributes)

        return loss, acc


    def accuracy(self, y_hat, y):

        B = y.shape[0]

        y_hat = (torch.sigmoid(y_hat) > .5).float()
        correct = (y_hat.eq(y)).float().mean()

        # print (correct)
        # fdasdf






        # print (y.shape)
        # print (y_hat.shape)
        # print (y_hat)
        # print (y)


        # y_hat = y_hat > .5
        # # acc=mezn

        # print (y.shape)
        # print (y_hat.shape)
        # print (y_hat)
        # print (y)


        # # y_hat = y_hat.contiguous().view(B*self.q_max_len, self.vocab_size)
        # # y_hat = torch.max(y_hat, 1)[1]
        # # y_hat = y_hat.contiguous().view(B,self.q_max_len)

        # acc = torch.mean((y_hat == y.byte()).float())
        return correct


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
        
















# #old version that used embeddign for relation
# class attribute_classifier_with_relations(nn.Module):
#     def __init__(self, kwargs, encoder_embed):
#         super(attribute_classifier_with_relations, self).__init__()


#         self.x_enc_size = kwargs['x_enc_size']
#         self.y_enc_size = kwargs['y_enc_size']

#         self.vocab_size = kwargs['vocab_size']
#         self.q_max_len = kwargs['q_max_len']
#         self.embed_size = kwargs['embed_size']

#         self.linear_hidden_size = 500 #self.x_enc_size + self.y_enc_size
#         self.linear_hidden_size2 = 200
#         self.act_func = kwargs['act_func'] #F.leaky_relu  # F.relu # # F.tanh ##  F.elu F.softplus


#         self.n_outputs = 8
#         self.n_relations = kwargs['embed_size']


#         # Attribute classifier
#         self.image_encoder_for_classi = Generator_part1(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=3)
#         self.classifier_fc1 = nn.Linear(self.x_enc_size+self.n_relations, self.linear_hidden_size2)
#         self.classifier_fc2 = nn.Linear(self.linear_hidden_size2, self.linear_hidden_size2)
#         self.classifier_fc3 = nn.Linear(self.linear_hidden_size2, self.n_outputs*self.vocab_size)

#         self.classifier_params = [list(self.image_encoder_for_classi.parameters())  + list(self.classifier_fc1.parameters())  
#                                 + list(self.classifier_fc2.parameters()) + list(self.classifier_fc3.parameters())]

#         self.optimizer_classi = optim.Adam(self.classifier_params[0], lr=.0002, weight_decay=.0000001)


#         self.encoder_embed = encoder_embed



#     def update(self, x, q):

#         self.optimizer_classi.zero_grad()  
#         classi_loss, acc = self.classifier_loss(x=x, attributes=q)
#         classi_loss.backward()
#         self.optimizer_classi.step()

#         return classi_loss, acc






#     def classifier(self, x, relation):

#         out = self.image_encoder_for_classi(x)

#         out = torch.cat([out, relation], 1)

#         out = self.act_func(self.classifier_fc1(out))
#         out = self.act_func(self.classifier_fc2(out))
#         out = self.classifier_fc3(out)
#         return out






#     def classifier2(self, x, attributes):

#         # print(attributes.shape)

#         B = x.shape[0]

#         relation = attributes[:,4]

#         relation = self.encoder_embed(relation)

#         attributes_less_relation = torch.cat([attributes[:,:4], attributes[:,5:]], 1)
#         # print (attributes.shape)

#         y_hat = self.classifier(x, relation)
#         return y_hat





#     def classifier_loss(self, x, attributes):

#         # print(attributes.shape)

#         B = x.shape[0]

#         relation = attributes[:,4]

#         print (relation.shape)

#         relation = self.encoder_embed(relation)

#         print (relation.shape)
#         fdsdafsdf


#         # print (attributes.shape)
#         # print (relation.shape)
#         # print (relation)
#         # print (embed.shape)
#         # fdsf

#         attributes_less_relation = torch.cat([attributes[:,:4], attributes[:,5:]], 1)
#         # print (attributes.shape)

#         y_hat = self.classifier(x, relation)

#         target_attributes = attributes_less_relation.contiguous().view(B*self.n_outputs)
#         y_hat = y_hat.contiguous().view(B*self.n_outputs, self.vocab_size)

#         loss =  F.cross_entropy(input=y_hat, target=target_attributes) #* self.w_logpy

#         # attributes = attributes.contiguous().view(B, self.n_outputs, self.vocab_size)
#         acc = self.accuracy(y_hat, attributes)

#         return loss, acc


#     def accuracy(self, y_hat, y):

#         B = y.shape[0]

#         # print (y.shape)
#         y = torch.cat([y[:,:4], y[:,5:]], 1)
#         # print (y_hat.shape)

        

#         if y_hat.shape[1] == self.n_outputs+1:
#             y_hat = torch.cat([y_hat[:,:4], y_hat[:,5:]], 1)
#         # elif:
#         #     y_hat = y_hat.contiguous().view(B, self.n_outputs, self.vocab_size)

#         # print (y_hat.shape)
#         # print()

#         y_hat = y_hat.contiguous().view(B*self.n_outputs, self.vocab_size)
#         y_hat = torch.max(y_hat, 1)[1]
#         y_hat = y_hat.contiguous().view(B,self.n_outputs)

#         acc = torch.mean((y_hat == y).float())
#         return acc


#     def classifier_attribute_accuracies(self, x, attributes):

#         B = x.shape[0]



#         relation = attributes[:,4]
#         relation = self.encoder_embed(relation)
#         attributes = torch.cat([attributes[:,:4], attributes[:,5:]], 1)

#         y_hat = self.classifier(x, relation)

#         # target_attributes = attributes.contiguous().view(self.B*self.q_max_len)
#         y_hat = y_hat.contiguous().view(B*self.n_outputs, self.vocab_size)
#         y_hat = torch.max(y_hat, 1)[1]
#         y_hat = y_hat.contiguous().view(B,self.n_outputs)

#         acc = torch.mean((y_hat == attributes).float(), 0)
#         # print (acc.shape)
#         # fasffd
#         return acc[0], acc[1], acc[2], acc[3]






#     def load_params_v3(self, save_dir, step):
#         save_to=os.path.join(save_dir, "classifier_params" + str(step)+".pt")
#         self.load_state_dict(torch.load(save_to))
#         print ('loaded params', save_to)





#     def save_params_v3(self, save_dir, step):
#         save_to=os.path.join(save_dir, "classifier_params" + str(step)+".pt")
#         torch.save(self.state_dict(), save_to)
#         print ('saved params', save_to)
        




