







import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler



# from encoder_decoder import Encoder, Decoder
from encoder_decoder_32 import Encoder, Decoder
from distributions import lognormal, Flow1 #Normal,  #, IAF_flow
from torch.distributions import Beta




class VAE(nn.Module):
    def __init__(self, kwargs):
        super(VAE, self).__init__()


        kwargs['act_func'] = F.leaky_relu

        self.__dict__.update(kwargs)

        self.beta_scale = 100.

        self.linear_hidden_size = 500 
        # self.linear_hidden_size2 = 200

        lr = .0004
        

        # q(z|x)
        self.image_encoder2 = Encoder(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=3, input_size=self.input_size)
        self.qzx_fc1 = nn.Linear(self.x_enc_size, self.x_enc_size)
        self.qzx_bn1 = nn.BatchNorm1d(self.x_enc_size)
        self.qzx_fc2 = nn.Linear(self.x_enc_size, self.x_enc_size)
        self.qzx_fc3 = nn.Linear(self.x_enc_size,self.z_size*2)

        x_inf_params = [list(self.image_encoder2.parameters()) + list(self.qzx_fc1.parameters()) 
                        + list(self.qzx_bn1.parameters()) + list(self.qzx_fc2.parameters())
                        + list(self.qzx_fc3.parameters()) ]




        # p(x|z) p(y|z)
        self.z_to_enc_fc1 = nn.Linear(self.z_size, self.x_enc_size)
        self.z_to_enc_bn1 = nn.BatchNorm1d(self.x_enc_size)
        self.z_to_enc_fc2 = nn.Linear(self.x_enc_size, self.x_enc_size)
        self.z_to_enc_fc3 = nn.Linear(self.x_enc_size, self.x_enc_size)
        self.z_to_enc_fc4 = nn.Linear(self.x_enc_size, self.x_enc_size)
        self.z_to_enc_bn2 = nn.BatchNorm1d(self.x_enc_size)
        self.z_to_enc_fc5 = nn.Linear(self.x_enc_size, self.x_enc_size)

        z_to_enc_params = [list(self.z_to_enc_fc1.parameters())  + list(self.z_to_enc_bn1.parameters()) 
                        + list(self.z_to_enc_fc2.parameters()) + list(self.z_to_enc_fc3.parameters()) 
                        + list(self.z_to_enc_fc4.parameters()) + list(self.z_to_enc_bn2.parameters())
                        + list(self.z_to_enc_fc5.parameters())]

        self.image_decoder = Decoder(z_size=self.x_enc_size, output_nc=3, n_residual_blocks=3, output_size=self.input_size)

        decoder_params = [ z_to_enc_params[0] + list(self.image_decoder.parameters()) ]



        decode0_params = [x_inf_params[0] + decoder_params[0]]  
        self.optimizer_x = optim.Adam(decode0_params[0], lr=lr, weight_decay=.0000001)
        self.scheduler_x = lr_scheduler.StepLR(self.optimizer_x, step_size=1, gamma=0.999995)







    # def inference_net(self, x_enc, y_enc=None):
    #     # print (x.shape, y.shape)
    #     # if self.joint_inf:
    #     b = x_enc.shape[0]
    #     input_ = torch.cat([x_enc, y_enc], 1)
    #     padded_input = torch.cat([input_, torch.zeros(b, self.linear_hidden_size - self.x_enc_size - self.linear_hidden_size).cuda()], 1)
    #     # else:
    #     #     input_ = x_enc
    #     #     padded_input = torch.cat([input_, torch.zeros(self.B, self.linear_hidden_size - self.x_enc_size).cuda()], 1)

    #     out = self.act_func(self.bn3(self.fc3(input_)))
    #     out = self.fc4(out)
    #     out = out + padded_input 
    #     out = self.fc5(out)
    #     mean = out[:,:self.z_size]
    #     logvar = out[:,self.z_size:]
    #     logvar = torch.clamp(logvar, min=-15., max=10.)
    #     return mean, logvar


    def inference_net(self, x):
        # b = x_enc.shape[0]

        x_enc = self.image_encoder2(x)
        input_ = x_enc
        # padded_input = torch.cat([x_enc, torch.zeros(b, self.linear_hidden_size-self.x_enc_size).cuda()], 1)

        out = self.act_func(self.qzx_bn1(self.qzx_fc1(input_)))
        out = self.qzx_fc2(out)
        out = out + input_ 
        out = self.qzx_fc3(out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]
        logvar = torch.clamp(logvar, min=-15., max=10.)
        return mean, logvar





    def z_to_dec(self, z):

        # b = z.shape[0]
        # padded_input = torch.cat([z,torch.zeros(b, self.linear_hidden_size - self.z_size).cuda()], 1)

        out = self.act_func(self.z_to_enc_bn1(self.z_to_enc_fc1(z)))
        out = self.z_to_enc_fc2(out)
        z = out + z

        out = self.act_func(self.z_to_enc_bn2(self.z_to_enc_fc3(z)))
        out = self.z_to_enc_fc4(out)
        z = out + z

        z_dec = self.z_to_enc_fc5(z)
        return z_dec




    def sample(self, mu, logvar, k=1):

        B = mu.shape[0]

        eps = torch.FloatTensor(B, self.z_size).normal_().cuda() #[B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mu  #[B,Z]

        logpz = lognormal(z, torch.zeros(B, self.z_size).cuda(), 
                            torch.zeros(B, self.z_size).cuda()) #[B]
        logqz = lognormal(z, mu.detach(), logvar.detach())
        # [P,B,Z], [P,B]
        return z, logpz, logqz




    def forward(self, x=None, warmup=1., inf_net=None): #, k=1): #, marginf_type=0):
        # x: [B,3,112,112]
        # q: [B,L] 
        # inf type: 0 is both, 1 is only x, 2 is only y
        # dec type: 0 is both, 1 is only x, 2 is only y

        outputs = {}

        if inf_net is None:
        	mu, logvar = self.inference_net(x)
        else:
        	mu, logvar = inf_net.inference_net(x)   



        z, logpz, logqz = self.sample(mu, logvar) 

        z_dec = self.z_to_dec(z)

        B = z_dec.shape[0]

        # Decode Image
        x_hat = self.image_decoder(z_dec)
        alpha = torch.sigmoid(x_hat)

        beta = Beta(alpha*self.beta_scale, (1.-alpha)*self.beta_scale)
        x_noise = torch.clamp(x + torch.FloatTensor(x.shape).uniform_(0., 1./256.).cuda(), min=1e-5, max=1-1e-5)
        # logpx = beta.log_prob(x + torch.FloatTensor(x.shape).uniform_(0., 1./256.).cuda()) #[120,3,112,112]  # add uniform noise here
        logpx = beta.log_prob(x_noise) #[120,3,112,112]  # add uniform noise here

        logpx = torch.sum(logpx.view(B, -1),1) # [PB]  * self.w_logpx
        # logpx = logpx * self.w_logpx

        log_ws = logpx + logpz - logqz

        outputs['logpx'] = torch.mean(logpx)
        outputs['x_recon'] = alpha
        outputs['welbo'] = torch.mean(logpx + warmup*( logpz - logqz))
        outputs['elbo'] = torch.mean(log_ws)
        outputs['logws'] = log_ws
        outputs['z'] = z
        outputs['logpz'] = torch.mean(logpz)
        outputs['logqz'] = torch.mean(logqz)
        outputs['logvar'] = logvar

        # print (outputs['elbo'], outputs['welbo'], outputs['logpz'], outputs['logqz'])
        # fafs


        # if generate:
        #     # word_preds, sampled_words = self.text_generator.teacher_force(z_dec, generate=generate, embeder=self.encoder_embed)
        #     # if dec_type == 2:
        #     alpha = torch.sigmoid(self.image_decoder(z_dec))
        #     return outputs, alpha #, word_preds, sampled_words

        return outputs





    def sample_prior(self, std=1., B=20, z=None):

        # B  = 20
        if z is None:
        	z = torch.FloatTensor(B, self.z_size).normal_().cuda() * std

        z_dec = self.z_to_dec(z)
        x_hat = self.image_decoder(z_dec)
        x_samp = torch.sigmoid(x_hat)
        # y_hat, sampled_words = self.text_generator.teacher_force(z_dec, generate=True, embeder=self.encoder_embed)
        return x_samp#, y_hat, sampled_words




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
        







    #Auxiliary Functions



    # def get_entropy_and_samples(self, q=None, inf_type=2, dec_type=1, k_entropy=1, k_recons=1):
    #     # x: [B,3,112,112]
    #     # q: [B,L] 
    #     # inf type: 0 is both, 1 is only x, 2 is only y
    #     # dec type: 0 is both, 1 is only x, 2 is only y

    #     # outputs = {}

    #     self.eval()

    #     if inf_type in [0,2] or dec_type in [0,2]:
    #         embed = self.encoder_embed(q)

    #     # elif inf_type == 2:
    #     y_enc = self.encode_attributes2(embed)
    #     mu, logvar = self.inference_net_y(y_enc)

    #     # if self.flow_int:
    #     #     z, logpz, logqz = self.flow.sample(mu, logvar) 
    #     # else:
    #     #     z, logpz, logqz = self.sample(mu, logvar) 


    #     # get entropy of qz
    #     logqs = []
    #     for i in range(k_entropy):

    #         z, logpz, logqz = self.flow.sample(mu, logvar) 
    #         logqs.append(logqz)

    #     logqs = torch.stack(logqs)
    #     entropy = - torch.mean(logqs)

    #     # print (entropy)

    #     image_recons = []
    #     for i in range(k_entropy):

    #         z, logpz, logqz = self.flow.sample(mu, logvar) 
    #         # logqs.append(logqz)

    #         # print(z.shape)

    #         z_dec = self.z_to_enc(z)

    #         x_hat = self.image_decoder(z_dec)
    #         # print (x_hat.shape)
    #         # fs
    #         alpha = torch.sigmoid(x_hat)

    #         image_recons.append(alpha[0])

    #     return entropy, image_recons










    def get_z_samples(self, q=None, inf_type=2):

        self.eval()

        if inf_type in [0,2] or dec_type in [0,2]:
            embed = self.encoder_embed(q)

        y_enc = self.encode_attributes2(embed)
        mu, logvar = self.inference_net_y(y_enc)

        if self.flow_int:
            z, logpz, logqz = self.flow.sample(mu, logvar) 
        else:
            z, logpz, logqz = self.sample(mu, logvar) 

        return z





    def get_z_samples2(self, x=None, inf_type=1):

        self.eval()

        x_enc = self.image_encoder2(x)
        mu, logvar = self.inference_net_x(x_enc)
            
        z, logpz, logqz = self.sample(mu, logvar) 

        return z




















































