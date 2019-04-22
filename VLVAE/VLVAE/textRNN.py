



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



class TextRNN(nn.Module):
    def __init__(self, kwargs):
        super(TextRNN, self).__init__()

        self.y_enc_size = kwargs['y_enc_size']
        self.vocab_size = kwargs['vocab_size']
        self.q_max_len = kwargs['q_max_len']
        self.embed_size = kwargs['embed_size']
        self.act_func = kwargs['act_func']

        self.linear_hidden_size2 = 200


        input_size = self.embed_size + self.y_enc_size 


        self.q_generator_rnn = nn.GRUCell(input_size=input_size, hidden_size=self.y_enc_size)

        self.q_generator_linear = nn.Linear(self.y_enc_size, self.linear_hidden_size2)
        self.q_generator_linear2 = nn.Linear(self.linear_hidden_size2, self.linear_hidden_size2)
        self.q_generator_linear3 = nn.Linear(self.linear_hidden_size2, self.vocab_size)

        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(self.linear_hidden_size2)






    def predict_word(self, rnn_hidden):

        rnn_hidden = self.dropout(rnn_hidden)
        linear_output = self.act_func(self.bn1(self.q_generator_linear(rnn_hidden)))
        linear_output = self.q_generator_linear2(linear_output)
        rnn_hidden = rnn_hidden + linear_output
        linear_output = self.q_generator_linear3(rnn_hidden)
        return linear_output


    def teacher_force(self, z, y_embed=None, y=None, generate=False, embeder=None):

        B = z.size()[0]
        L = self.q_max_len
        H = self.q_generator_rnn.hidden_size

        h = torch.zeros(B, H).cuda() #.type_as(y_embed.data)
        prev_word = torch.zeros(B, self.embed_size).cuda()

        # z = self.dropout(z)

        sampled_words = []
        logpys = []
        word_preds = []
        for i in range(L):

            input_ = torch.cat([prev_word, z], 1)
            h = self.q_generator_rnn(input_, h)

            # print (torch.max(h), torch.mean(h), torch.min(h))
            h = torch.clamp(h, min=-2., max=2.)

            linear_output = self.predict_word(h)

            if generate:
                samp_word = Categorical(logits=linear_output).sample()
                sampled_words.append(samp_word)
                prev_word = embeder(samp_word.view(B,1)).view(B, self.embed_size)

                # print()
                # # print (F.softmax(linear_output).shape)
                # print (F.softmax(linear_output)[0])
                # # print (F.softmax(linear_output)[0].shape)
                # # print (torch.max(F.softmax(linear_output)[0], 0))
                # # print (torch.max(F.softmax(linear_output)[0], 0)[1])
                # print ('real', 'pred')
                # print (y[:,i][0].data, torch.max(F.softmax(linear_output)[0], 0)[1].data)
                # # print ('logprobs', logpy)
                



            else:
                prev_word = y_embed[:,i]
                logpy = -F.cross_entropy(input=linear_output, target=y[:,i], reduction='none') #, reduce=False) #* self.w_logpy
                logpys.append(logpy)

                # print()
                # # print (F.softmax(linear_output).shape)
                # print (F.softmax(linear_output)[0])
                # # print (F.softmax(linear_output)[0].shape)
                # # print (torch.max(F.softmax(linear_output)[0], 0))
                # # print (torch.max(F.softmax(linear_output)[0], 0)[1])
                # print ('real', 'pred')
                # print (y[:,i][0].data.cpu().numpy(), torch.max(F.softmax(linear_output)[0], 0)[1].data.cpu().numpy())
                # print ('logprobs', logpy)
                

            
            word_preds.append(linear_output)

        # fasdf

        if generate:
            return torch.stack(word_preds,1), torch.stack(sampled_words,1)
        else:
            return torch.stack(word_preds,1), torch.sum(torch.stack(logpys,1),1) # sum is over words































class TextFCN(nn.Module):
    def __init__(self, kwargs):
        super(TextFCN, self).__init__()

        self.y_enc_size = kwargs['y_enc_size']
        self.vocab_size = kwargs['vocab_size']
        self.q_max_len = kwargs['q_max_len']
        # self.embed_size = kwargs['embed_size']
        self.act_func = kwargs['act_func']

        self.linear_hidden_size2 = 200


        input_size = self.y_enc_size 


        # self.q_generator_rnn = nn.GRUCell(input_size=input_size, hidden_size=self.y_enc_size)

        self.q_generator_linear = nn.Linear(self.y_enc_size, self.linear_hidden_size2)
        self.q_generator_linear2 = nn.Linear(self.linear_hidden_size2, self.linear_hidden_size2)
        self.q_generator_linear3 = nn.Linear(self.linear_hidden_size2, self.vocab_size*self.q_max_len)

        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(self.linear_hidden_size2)






    def predict_word(self, rnn_hidden):

        rnn_hidden = self.dropout(rnn_hidden)
        linear_output = self.act_func(self.bn1(self.q_generator_linear(rnn_hidden)))
        linear_output = self.q_generator_linear2(linear_output)
        rnn_hidden = rnn_hidden + linear_output
        linear_output = self.q_generator_linear3(rnn_hidden)
        return linear_output


    def teacher_force(self, z, y_embed=None, y=None, generate=False, embeder=None):

        B = z.size()[0]
        L = self.q_max_len

        linear_output = self.predict_word(z) #[B, length of text * vocab size]
        linear_output = linear_output.view(B,L,self.vocab_size)

        sampled_words = []
        logpys = []
        word_preds = []
        for i in range(L):

            if generate:
                samp_word = Categorical(logits=linear_output[:,i]).sample()
                sampled_words.append(samp_word)

            else:
                # prev_word = y_embed[:,i]
                logpy = -F.cross_entropy(input=linear_output[:,i], target=y[:,i], reduction='none') #, reduce=False) #* self.w_logpy
                logpys.append(logpy)

            word_preds.append(linear_output[:,i])


        if generate:
            return torch.stack(word_preds,1), torch.stack(sampled_words,1)
        else:
            return torch.stack(word_preds,1), torch.sum(torch.stack(logpys,1),1) # sum is over words























class TextRNN2(nn.Module):
    def __init__(self, kwargs):
        super(TextRNN2, self).__init__()

        self.y_enc_size = kwargs['y_enc_size']
        self.vocab_size = kwargs['vocab_size']
        self.q_max_len = kwargs['q_max_len']
        self.embed_size = kwargs['embed_size']
        self.act_func = kwargs['act_func']

        self.linear_hidden_size2 = 200


        input_size = self.embed_size + self.y_enc_size 


        # self.q_generator_rnn = nn.GRUCell(input_size=input_size, hidden_size=self.y_enc_size)

        self.q_generator_linear = nn.Linear(self.y_enc_size, self.linear_hidden_size2)
        self.q_generator_linear2 = nn.Linear(self.linear_hidden_size2, self.linear_hidden_size2)
        self.q_generator_linear3 = nn.Linear(self.linear_hidden_size2, self.vocab_size)

        self.q_generator_linear4 = nn.Linear(self.y_enc_size + self.y_enc_size + self.embed_size, self.linear_hidden_size2)
        self.q_generator_linear5 = nn.Linear(self.y_enc_size + self.y_enc_size + self.embed_size, self.linear_hidden_size2)
        self.q_generator_linear6 = nn.Linear(self.linear_hidden_size2, self.y_enc_size)

        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(self.linear_hidden_size2)


    def predict_word(self, rnn_hidden):

        rnn_hidden = self.dropout(rnn_hidden)
        linear_output = self.act_func(self.bn1(self.q_generator_linear(rnn_hidden)))
        linear_output = self.q_generator_linear2(linear_output)
        rnn_hidden = rnn_hidden + linear_output
        linear_output = self.q_generator_linear3(rnn_hidden)
        return linear_output



    def predict_next_state(self, input_):

        linear_output1 = self.q_generator_linear4(input_)

        # rnn_hidden = self.dropout(input_)
        linear_output = self.act_func(self.bn1(self.q_generator_linear5(input_)))
        linear_output = self.q_generator_linear6(linear_output)

        rnn_hidden = linear_output + linear_output1
        # linear_output = self.q_generator_linear7(rnn_hidden)
        return rnn_hidden




    def teacher_force(self, z, y_embed=None, y=None, generate=False, embeder=None):

        B = z.size()[0]
        L = self.q_max_len
        H = self.y_enc_size

        h = torch.zeros(B, H).cuda() #.type_as(y_embed.data)
        prev_word = torch.zeros(B, self.embed_size).cuda()

        sampled_words = []
        logpys = []
        word_preds = []
        for i in range(L):

            input_ = torch.cat([prev_word, z], 1)
            input_ = torch.cat([input_, h], 1)
            # h = self.q_generator_rnn(input_, h)
            h = self.predict_next_state(input_)

            # h = torch.clamp(h, min=-2., max=2.)

            linear_output = self.predict_word(h)

            if generate:
                samp_word = Categorical(logits=linear_output).sample()
                sampled_words.append(samp_word)
                prev_word = embeder(samp_word.view(B,1)).view(B, self.embed_size)


            else:
                prev_word = y_embed[:,i]
                logpy = -F.cross_entropy(input=linear_output, target=y[:,i], reduction='none') #, reduce=False) #* self.w_logpy
                logpys.append(logpy)
  
            word_preds.append(linear_output)

        if generate:
            return torch.stack(word_preds,1), torch.stack(sampled_words,1)
        else:
            return torch.stack(word_preds,1), torch.sum(torch.stack(logpys,1),1) # sum is over words














