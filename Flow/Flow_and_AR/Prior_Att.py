

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


# from invertible_layers import Layer

def concat_position(x, max_seq_len):
    # x: [B,L,D]

    B = x.shape[0]

    # max_seq_len = n
    pe = np.linspace(0, 1, max_seq_len)
    pe = np.tile(pe, (B,1))
    pe = np.expand_dims(pe, 2)
    pe = torch.from_numpy(pe).float().cuda()
    x = torch.cat([x, pe], 2)

    pe = np.linspace(0, 10, max_seq_len)
    pe = np.tile(pe, (B,1))
    pe = np.expand_dims(pe, 2)
    pe = torch.from_numpy(pe).float().cuda()
    x = torch.cat([x, pe], 2)

    pe = np.linspace(-1, 1, max_seq_len)
    pe = np.tile(pe, (B,1))
    pe = np.expand_dims(pe, 2)
    pe = torch.from_numpy(pe).float().cuda()
    x = torch.cat([x, pe], 2)

    return x #x: [B,L,D+1]



class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm



class Att(nn.Module):
    def __init__(self, dim_in, dim_out, L):
        super(Att, self).__init__()

        self.act_func = F.leaky_relu
        self.d_k = 400
        self.n_heads = 5
        # self.out_dim = 20
        self.L = L

        # self.nopeak_mask = torch.from_numpy(np.tril(np.ones((1, n-1, n)),k=0).astype('uint8')).cuda()
        self.nopeak_mask = torch.from_numpy(np.tril(np.ones((1, L, L)),k=0).astype('uint8')).cuda()
        # nopeak_mask = torch.from_numpy(nopeak_mask) == 0

        self.q_linear = nn.Linear(dim_in, self.d_k)
        self.v_linear = nn.Linear(dim_in, self.d_k)
        self.k_linear = nn.Linear(dim_in, self.d_k)

        self.m1 = nn.Linear(self.d_k, 20)
        self.m2 = nn.Linear(20, dim_out)

    def predict_output(self, z):

        out = self.act_func(self.m1(z))
        out = self.m2(out)
        return out



    def multi_head_attention(self, x):
        B = x.shape[0]

        n_heads = self.n_heads

        k = self.k_linear(x).view(B, self.L, n_heads, self.d_k//n_heads)
        q = self.q_linear(x).view(B, self.L, n_heads, self.d_k//n_heads) #dont query for first position
        v = self.v_linear(x).view(B, self.L, n_heads, self.d_k//n_heads)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2) #[B,H,L,D/H]

        values = self.attention(q,k,v, d_k=self.d_k//n_heads)

        # concatenate heads and put through final linear layer
        concat = values.transpose(1,2).contiguous().view(B, self.L, self.d_k)
        
        output = self.predict_output(concat)
        return output



    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        
        # scores = torch.sin(torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)  )

        # scores = torch.exp(-torch.sin(torch.abs(q - k)/  math.sqrt(d_k) )**2 )
        
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

        mask = self.nopeak_mask
        scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)  #[B,H,D,D]
        
        output = torch.matmul(scores, v)  #[B,H,D,D]
        return output
















class Att_Stack(nn.Module):
    def __init__(self, input_dim, output_dim, L):
        super(Att_Stack, self).__init__()

        dim2 = 1000

        self.att1 = Att(dim_in=input_dim, dim_out=dim2, L=L)
        self.att2 = Att(dim_in=dim2, dim_out=dim2, L=L)
        self.att3 = Att(dim_in=dim2, dim_out=dim2, L=L)

        self.norm_1 = Norm(dim2)
        self.norm_2 = Norm(dim2)
        self.norm_3 = Norm(dim2)

        self.m1 = nn.Linear(dim2, dim2)
        self.m2 = nn.Linear(dim2, output_dim)

        self.act_func = F.leaky_relu

    def forward(self,x):


        B = x.shape[0]
        D = x.shape[1]
        L = x.shape[2]*x.shape[3]
        preL = x.shape[2]

        x = x.view(B,D,L)
        x = x.permute(0,2,1) #[B,L,D]

        #shift right 
        batch_shifted = torch.cat([torch.zeros(B,1,D).cuda(),x],dim=1)
        batch_shifted = batch_shifted[:,:-1]
        x = concat_position(batch_shifted, L)

        




        x = self.att1.multi_head_attention(x)
        x2 = self.norm_1(x)
        x = x + self.att2.multi_head_attention(x2)
        x2 = self.norm_2(x)
        x = x + self.att3.multi_head_attention(x2)
        x = self.norm_3(x)
        x = self.predict_output(x)



        x = x.permute(0,2,1)
        x = x.view(B,D*2,preL,preL)

        return x

    def predict_output(self, z):

        out = self.act_func(self.m1(z))
        out = self.m2(out)
        # mean = out[:,:,0]
        # logvar = out[:,:,1]
        # logvar = torch.clamp(logvar, min=-10., max=10.)
        # # out[:,:,1:] = torch.clamp(out[:,:,1:], min=-15., max=10.)
        # # return mean, logvar
        # # print (mean.shape)
        # mean = mean.view(B,n,1)
        # logvar = logvar.view(B,n,1)
        # out = torch.cat([mean,logvar], dim=2)

        return out























# Abstract Class for bijective functions
class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    def forward_(self, x, objective):
        raise NotImplementedError

    def reverse_(self, y, objective):
        raise NotImplementedError



class Att_Prior(Layer):
    def __init__(self, input_shape, args):
        super(Att_Prior, self).__init__()
        
        self.input_shape = input_shape
        # print ('p(z)', input_shape)
        
        input_shape_2 = [input_shape[0],input_shape[1], input_shape[2]*input_shape[3]]
        print ('p(z)', input_shape_2)
        
        
        #plus 3 for positions
        self.model = Att_Stack(input_dim=input_shape_2[1] + 3, output_dim=input_shape_2[1]*2, L=input_shape_2[2])
        # self.model = PixelCNN(nr_resnet=args.AR_resnets, nr_filters=args.AR_channels, 
        #             input_channels=input_shape[1], nr_logistic_mix=2)


        # self.loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)

        # logsd_limitvalue
        self.lm = 10.

    def gauss_log_prob(self, x, mean, logsd):
        Log2PI = float(np.log(2 * np.pi))
        # return  -0.5 * (Log2PI + 2. * logsd + ((x - mean) ** 2) / torch.exp(2. * logsd))

        aaa = Log2PI + 2. * logsd 
        var = torch.exp(2. * logsd)
        bbb = ((x - mean) ** 2) / var
        return  -0.5 * (aaa + bbb)
 
    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c // 2 ** 2, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(bs, c // 2 ** 2, h * 2, w * 2)
        return x



    def forward_(self, x, objective):
        B = x.shape[0]
        # D = x.shape[1]
        # L = x.shape[2]*x.shape[3]
        # preL = x.shape[2]

        # x_mod = x.view(B,D,L)
        # x_mod = x_mod.permute(0,2,1)

        mean_and_logsd = self.model(x)


        # # CONFIRM CORRECT DEPENDENCE
        # print (x.shape)
        # print (mean_and_logsd.shape)
        # # fsad

        # # inputs = x[:,:,5,5]
        # inputs = x[:,:,0,1]
        # # outputs = torch.sum(mean_and_logsd[:,:,5,6])
        # outputs = torch.sum(mean_and_logsd[:,:,0,0])
        # grad = torch.autograd.grad(outputs, x)[0]

        # print (grad.shape)
        # print (torch.sum(grad))
        # faasdf







        # mean_and_logsd = mean_and_logsd.permute(0,2,1)
        # mean_and_logsd = mean_and_logsd.view(B,D*2,preL,preL)

        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)


        mean = torch.tanh(mean / 100.) * 100.
        logsd = (torch.tanh(logsd /self.lm ) * self.lm ) - (self.lm /2.)
        

        LL = self.gauss_log_prob(x, mean=mean, logsd=logsd)


        LL = LL.view(B,-1).sum(-1)

        objective += LL

        return x, objective







    def reverse_(self, x, objective, args=None):
        bs, c, h, w = self.input_shape

        samp = torch.zeros(bs, c, h, w).cuda()
        for i in range(h):

            if args:
                if args.special_sample:
                    if i%5==0:
                        print ('h', i, '/'+str(h))



            for j in range(w):
                mean_and_logsd   = self.model(samp)
                mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)

                mean = torch.tanh(mean / 100.) * 100.
                # logsd = (torch.tanh(logsd /4.) * 4.) -2.
                logsd = (torch.tanh(logsd /self.lm ) * self.lm ) - (self.lm /2.)

                x_ = torch.zeros(bs, c, h, w).cuda()

                if args:
                    if args.special_sample:
                        for ii in range (bs):

                            if ii < 32:
                                if ii < i:
                                    eps = torch.zeros_like(mean).normal_().cuda()
                                    x_[ii] = mean[ii] + torch.exp(logsd)[ii] * eps[ii] 
                                else:
                                    x_[ii] = mean[ii]

                                # x_[i*j:] = mean[i*j:]
                                # x_[:i*j] = mean[i*j:] + torch.exp(logsd)[i*j:] * torch.zeros_like(mean).normal_().cuda()[i*j:]
                            else:
                                if ii-32 > i:
                                    eps = torch.zeros_like(mean).normal_().cuda()
                                    x_[ii] = mean[ii] + torch.exp(logsd)[ii] * eps[ii] 
                                else:
                                    x_[ii] = mean[ii]                            


                        # if i < args.special_h:
                        #     x_ = mean 

                    else:
                        eps = torch.zeros_like(mean).normal_().cuda()
                        x_ = mean + torch.exp(logsd) * eps 
                else:
                    eps = torch.zeros_like(mean).normal_().cuda()
                    x_ = mean + torch.exp(logsd) * eps 

                samp[:, :, i, j] = x_[:, :, i, j]

        # I THINK the model cahgnes smap.. need to copy it first ..
        # B = x.shape[0]
        samp1 = samp.clone()
        B = bs
        mean_and_logsd = self.model(samp)
        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
        mean = torch.tanh(mean / 100.) * 100.
        # logsd = (torch.tanh(logsd /4.) * 4.) -2.
        logsd = (torch.tanh(logsd /self.lm ) * self.lm ) - (self.lm /2.)
        LL = self.gauss_log_prob(samp, mean=mean, logsd=logsd)
        LL = LL.view(B,-1).sum(-1)
        objective -= LL

        return samp1, objective
         






