


import torch
import torch.nn as nn
import torch.nn.functional as F




class Text_encoder(nn.Module):
    def __init__(self, exp_dict): #vocab_size, word_embedding_size, output_size):
        super(Text_encoder, self).__init__()

        self.act_func = F.leaky_relu
        hs = 100 #hidden size
        self.hs = hs


        vocab_size = exp_dict['vocab_size'] #= vocab_size
        word_embedding_size = exp_dict['word_embedding_size'] #= 50
        y_enc_size = exp_dict['y_enc_size'] #= 50

        self.encoder_embed = nn.Embedding(vocab_size, word_embedding_size) 

        self.encoder_rnn = nn.GRU(input_size=word_embedding_size, hidden_size=hs, num_layers=1, 
                                    dropout=0, batch_first=True, bidirectional=False)



        self.linear1 = nn.Linear(hs, hs)
        # self.qzy_bn1 = nn.BatchNorm1d(self.linear_hidden_size)
        self.linear2 = nn.Linear(hs, hs)
        self.linear3 = nn.Linear(hs, y_enc_size)






    def encode(self, y):

        embed = self.encoder_embed(y)
        h0 = torch.zeros(1, embed.shape[0], self.hs).type_as(embed.data)
        out, ht = self.encoder_rnn(embed, h0)
        state = out[:,-1] 

        state = self.act_func(state)
        res = self.act_func(self.linear2(self.act_func(self.linear1(state))))
        state = state + res
        state = self.linear3(state)

        return state






































