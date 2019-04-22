


import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler



from image_encoder_decoder import Generator_part1, Generator_part2
from textRNN import TextRNN
from distributions import lognormal, Flow1 #Normal,  #, IAF_flow
from torch.distributions import Beta




class VLVAE(nn.Module):
    def __init__(self, kwargs):
        super(VLVAE, self).__init__()


        kwargs['act_func'] = F.leaky_relu

        # self.__dict__.update(kwargs.__dict__)
        self.__dict__.update(kwargs)


        self.num_dir=1
        rnn_num_layers =1
        self.NULL = 0
        self.START = 1
        self.END = 2

        self.beta_scale = 100.

        self.linear_hidden_size = 500 #self.x_enc_size + self.y_enc_size
        self.linear_hidden_size2 = 200

        lr = .0005
        



        # Decoder Parameters
        self.z_to_enc_fc1 = nn.Linear(self.z_size, self.linear_hidden_size)
        self.z_to_enc_bn1 = nn.BatchNorm1d(self.linear_hidden_size)
        self.z_to_enc_fc2 = nn.Linear(self.linear_hidden_size, self.linear_hidden_size)
        self.z_to_enc_fc3 = nn.Linear(self.linear_hidden_size, self.linear_hidden_size)
        self.z_to_enc_fc4 = nn.Linear(self.linear_hidden_size, self.linear_hidden_size)
        self.z_to_enc_bn2 = nn.BatchNorm1d(self.linear_hidden_size)
        self.z_to_enc_fc5 = nn.Linear(self.linear_hidden_size, self.y_enc_size)

        z_to_enc_params = [list(self.z_to_enc_fc1.parameters())  + list(self.z_to_enc_bn1.parameters()) 
                        + list(self.z_to_enc_fc2.parameters()) + list(self.z_to_enc_fc3.parameters()) 
                        + list(self.z_to_enc_fc4.parameters()) + list(self.z_to_enc_bn2.parameters())
                        + list(self.z_to_enc_fc5.parameters())]

        self.encoder_embed = nn.Embedding(self.vocab_size, self.embed_size) 
        self.text_generator = TextRNN(kwargs)
        self.image_decoder = Generator_part2(z_size=self.x_enc_size, output_nc=3, n_residual_blocks=3, output_size=self.input_size)

        decoder_params = [ z_to_enc_params[0] + list(self.text_generator.parameters())   
                            + list(self.image_decoder.parameters()) +  list(self.encoder_embed.parameters()) ]


        # Marginal inference net for y only
        self.encoder_rnn2 = nn.GRU(input_size=self.embed_size, hidden_size=self.y_enc_size, num_layers=1, 
                                    dropout=0, batch_first=True, bidirectional=False)
        self.qzy_fc1 = nn.Linear(self.y_enc_size, self.linear_hidden_size)
        self.qzy_bn1 = nn.BatchNorm1d(self.linear_hidden_size)
        self.qzy_fc2 = nn.Linear(self.linear_hidden_size, self.linear_hidden_size)
        self.qzy_fc3 = nn.Linear(self.linear_hidden_size, self.z_size*2)
        



        if self.flow_int:
            self.flow = Flow1(self.z_size)  
            # self.flow = IAF_flow(self.z_size)  
            # self.qRNN = qRNN(kwargs)
            y_inf_net_params = [list(self.encoder_rnn2.parameters()) + list(self.qzy_fc1.parameters()) 
                                + list(self.qzy_bn1.parameters()) + list(self.qzy_fc2.parameters())
                                    + list(self.qzy_fc3.parameters()) + list(self.flow.parameters())]
        else:
            y_inf_net_params = [list(self.encoder_rnn2.parameters()) + list(self.qzy_fc1.parameters()) 
                                    + list(self.qzy_bn1.parameters()) + list(self.qzy_fc2.parameters())
                                    + list(self.qzy_fc3.parameters()) ]


        # x
        self.image_encoder2 = Generator_part1(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=3, input_size=self.input_size)
        self.qzx_fc1 = nn.Linear(self.x_enc_size, self.linear_hidden_size)
        self.qzx_bn1 = nn.BatchNorm1d(self.linear_hidden_size)
        self.qzx_fc2 = nn.Linear(self.linear_hidden_size, self.linear_hidden_size)
        self.qzx_fc3 = nn.Linear(self.linear_hidden_size,self.z_size*2)
        x_inf_params = [list(self.image_encoder2.parameters()) + list(self.qzx_fc1.parameters()) 
                        + list(self.qzx_bn1.parameters()) + list(self.qzx_fc2.parameters())
                        + list(self.qzx_fc3.parameters()) ]



        # Decode to both modalities
        self.image_encoder = Generator_part1(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=3, input_size=self.input_size)
        self.bn3 = nn.BatchNorm1d(self.linear_hidden_size)
        self.fc4 = nn.Linear(self.linear_hidden_size, self.linear_hidden_size)
        self.fc5 = nn.Linear(self.linear_hidden_size, self.z_size*2)


        # # Inference Net q(z|x,y) 
        if self.joint_inf:
            self.encoder_rnn = nn.GRU(input_size=self.embed_size, hidden_size=self.y_enc_size, 
                                        num_layers=1, dropout=0, batch_first=True, bidirectional=False)
            self.fc3 = nn.Linear(self.y_enc_size + self.x_enc_size, self.linear_hidden_size)
            encoder_for_decode0_params = [list(self.image_encoder.parameters()) + list(self.fc3.parameters()) 
                                    + list(self.bn3.parameters()) + list(self.fc4.parameters())
                                    + list(self.fc5.parameters()) + list(self.encoder_rnn.parameters())]


        else:
            self.fc3 = nn.Linear(self.x_enc_size, self.linear_hidden_size)
            encoder_for_decode0_params = [list(self.image_encoder.parameters()) + list(self.fc3.parameters()) 
                                    + list(self.bn3.parameters()) + list(self.fc4.parameters())
                                    + list(self.fc5.parameters())]


        decode0_params = [encoder_for_decode0_params[0] + decoder_params[0] + y_inf_net_params[0] ]  

        SSL_params = [x_inf_params[0] + decoder_params[0] ]

        self.optimizer_decode1 = optim.Adam(x_inf_params[0], lr=lr, weight_decay=.0000001)
        self.scheduler_decode1 = lr_scheduler.StepLR(self.optimizer_decode1, step_size=1, gamma=0.999995)

        self.optimizer = optim.Adam(decode0_params[0], lr=lr, weight_decay=.0000001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.999995)

        self.optimizer_SSL = optim.Adam(SSL_params[0], lr=lr, weight_decay=.0000001)
        self.scheduler_SSL = lr_scheduler.StepLR(self.optimizer_SSL, step_size=1, gamma=0.999995)





    def encode_attributes(self, embed):
        h0 = torch.zeros(1, self.B, self.y_enc_size).type_as(embed.data)
        out, ht = self.encoder_rnn(embed, h0)
        return out[:,-1] 


    def encode_attributes2(self, embed):
        h0 = torch.zeros(1, self.B, self.y_enc_size).type_as(embed.data)
        out, ht = self.encoder_rnn2(embed, h0)
        return out[:,-1] 



    def inference_net(self, x_enc, y_enc=None):
        # print (x.shape, y.shape)
        if self.joint_inf:
            input_ = torch.cat([x_enc, y_enc], 1)
            padded_input = torch.cat([input_, torch.zeros(self.B, self.linear_hidden_size - self.x_enc_size - self.y_enc_size).cuda()], 1)
        else:
            input_ = x_enc
            padded_input = torch.cat([input_, torch.zeros(self.B, self.linear_hidden_size - self.x_enc_size).cuda()], 1)

        out = self.act_func(self.bn3(self.fc3(input_)))
        out = self.fc4(out)
        out = out + padded_input 
        out = self.fc5(out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]
        return mean, logvar


    def inference_net_x(self, x_enc):

        input_ = x_enc
        padded_input = torch.cat([x_enc, torch.zeros(self.B, self.linear_hidden_size-self.x_enc_size).cuda()], 1)

        out = self.act_func(self.qzx_bn1(self.qzx_fc1(input_)))
        out = self.qzx_fc2(out)
        out = out + padded_input 
        out = self.qzx_fc3(out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]

        return mean, logvar



    def inference_net_y(self, y_enc):

        input_ = y_enc
        padded_input = torch.cat([y_enc, torch.zeros(self.B, self.linear_hidden_size-self.y_enc_size).cuda()], 1)

        out = self.act_func(self.qzy_bn1(self.qzy_fc1(input_)))
        out = self.qzy_fc2(out)
        res = out + padded_input 

        # out = self.act_func(self.qzy_bn2(self.qzy_fc3(res)))
        # out = self.qzy_fc4(out)
        # res = out + res 

        out = self.qzy_fc3(res)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]

        return mean, logvar


    def z_to_enc(self, z):

        b = z.shape[0]
        padded_input = torch.cat([z,torch.zeros(b, self.linear_hidden_size - self.z_size).cuda()], 1)

        out = self.act_func(self.z_to_enc_bn1(self.z_to_enc_fc1(z)))
        out = self.z_to_enc_fc2(out)
        padded_input = out + padded_input

        out = self.act_func(self.z_to_enc_bn2(self.z_to_enc_fc3(padded_input)))
        out = self.z_to_enc_fc4(out)
        padded_input = out + padded_input

        z_dec = self.z_to_enc_fc5(padded_input)
        return z_dec




    def sample(self, mu, logvar, k=1):

        eps = torch.FloatTensor(self.B, self.z_size).normal_().cuda() #[B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mu  #[B,Z]

        logpz = lognormal(z, torch.zeros(self.B, self.z_size).cuda(), 
                            torch.zeros(self.B, self.z_size).cuda()) #[B]
        logqz = lognormal(z, mu.detach(), logvar.detach())
        # [P,B,Z], [P,B]
        return z, logpz, logqz




    def forward(self, x=None, q=None, warmup=1., generate=False, inf_type=1, dec_type=0): #, k=1): #, marginf_type=0):
        # x: [B,3,112,112]
        # q: [B,L] 
        # inf type: 0 is both, 1 is only x, 2 is only y
        # dec type: 0 is both, 1 is only x, 2 is only y

        outputs = {}

        if inf_type in [0,2] or dec_type in [0,2]:
            embed = self.encoder_embed(q)

        if inf_type == 0:
            x_enc = self.image_encoder(x)
            y_enc = self.encode_attributes(embed)
            mu, logvar = self.inference_net(x_enc, y_enc)
            z, logpz, logqz = self.sample(mu, logvar) 

        elif inf_type == 1:
            if self.joint_inf:
                x_enc = self.image_encoder2(x)
                mu, logvar = self.inference_net_x(x_enc)
            else:
                if dec_type ==0:
                    x_enc = self.image_encoder(x)
                    mu, logvar = self.inference_net(x_enc)
                else:
                    x_enc = self.image_encoder2(x)
                    mu, logvar = self.inference_net_x(x_enc)                
            z, logpz, logqz = self.sample(mu, logvar) 

        elif inf_type == 2:
            y_enc = self.encode_attributes2(embed)
            mu, logvar = self.inference_net_y(y_enc)
            if self.flow_int:
                z, logpz, logqz = self.flow.sample(mu, logvar) 
            else:
                z, logpz, logqz = self.sample(mu, logvar) 



        z_dec = self.z_to_enc(z)


        if dec_type == 0: 
            # Decode Image
            x_hat = self.image_decoder(z_dec)
            alpha = torch.sigmoid(x_hat)

            beta = Beta(alpha*self.beta_scale, (1.-alpha)*self.beta_scale)
            logpx = beta.log_prob(x) #[120,3,112,112]
            logpx = torch.sum(logpx.view(self.B, -1),1) # [B]  

            word_preds, logpy = self.text_generator.teacher_force(z_dec, embed, q)

            logpx = logpx * self.w_logpx
            logpy = logpy * self.w_logpy

            #CE of q(z|y)
            if inf_type == 1:
                embed = self.encoder_embed(q)
                y_enc = self.encode_attributes2(embed)
            mu_y, logvar_y = self.inference_net_y(y_enc)
            if self.flow_int:
                # logqzy = self.flow.logprob(z.detach(), mu_y, logvar_y)
                logqzy = self.flow.logprob(z, mu_y, logvar_y)
            else:
                logqzy = lognormal(z, mu_y, logvar_y) 
            logqzy = logqzy * self.w_logqy

            log_ws = logpx + logpy + logpz - logqz #+ logqzy
            elbo = torch.mean(log_ws)
            # warmed_elbo = torch.mean(logpx + logpy + logqzy - logqz + warmup*( logpz - logqz))
            warmed_elbo = torch.mean(logpx + logpy + logqzy + warmup*( logpz - logqz))


            outputs['logpx'] = torch.mean(logpx)
            outputs['x_recon'] = alpha
            outputs['logpy'] = torch.mean(logpy)
            outputs['logqzy'] = torch.mean(logqzy)


        elif dec_type == 1:
            # Decode Image
            x_hat = self.image_decoder(z_dec)
            alpha = torch.sigmoid(x_hat)

            beta = Beta(alpha*self.beta_scale, (1.-alpha)*self.beta_scale)
            logpx = beta.log_prob(x) #[120,3,112,112]

            logpx = torch.sum(logpx.view(self.B, -1),1) # [PB]  * self.w_logpx
            logpx = logpx * self.w_logpx

            log_ws = logpx + logpz - logqz

            elbo = torch.mean(log_ws)
            warmed_elbo = torch.mean(logpx + warmup*( logpz - logqz))

            outputs['logpx'] = torch.mean(logpx)
            outputs['x_recon'] = alpha
        

        elif dec_type == 2:
            #Decode Text
            word_preds, logpy = self.text_generator.teacher_force(z_dec, embed, q)
            logpy = logpy * self.w_logpy

            log_ws = logpy + logpz - logqz 
            elbo = torch.mean(log_ws)
            warmed_elbo = torch.mean(logpy + warmup*( logpz - logqz))

            outputs['logpy'] = torch.mean(logpy)

     
        outputs['welbo'] = warmed_elbo
        outputs['elbo'] = elbo
        outputs['logws'] = log_ws
        outputs['logpz'] = torch.mean(logpz)
        outputs['logqz'] = torch.mean(logqz)
        outputs['logvar'] = logvar


        if generate:

            word_preds, sampled_words = self.text_generator.teacher_force(z_dec, generate=generate, embeder=self.encoder_embed)
            if dec_type == 2:
                alpha = torch.sigmoid(self.image_decoder(z_dec))

            return outputs, alpha, word_preds, sampled_words

        return outputs





    def sample_prior(self, std=1.):
        z = torch.FloatTensor(self.B, self.z_size).normal_().cuda() * std
        z_dec = self.z_to_enc(z)
        x_hat = self.image_decoder(z_dec)
        x_samp = torch.sigmoid(x_hat)
        y_hat, sampled_words = self.text_generator.teacher_force(z_dec, generate=True, embeder=self.encoder_embed)
        return x_samp, y_hat, sampled_words




    def load_params_v3(self, save_dir, step):
        save_to=os.path.join(save_dir, "model_params" + str(step)+".pt")
        state_dict = torch.load(save_to)
        # # # print (state_dict)
        # for key, val in state_dict.items():
        #     print (key)
        # fddsf
        self.load_state_dict(state_dict)
        print ('loaded params', save_to)


    def save_params_v3(self, save_dir, step):
        save_to=os.path.join(save_dir, "model_params" + str(step)+".pt")
        torch.save(self.state_dict(), save_to)
        print ('saved params', save_to)
        
























