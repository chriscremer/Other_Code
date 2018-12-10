







import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler



# from encoder_decoder import Encoder, Decoder
# from encoder_decoder_32 import Encoder, Decoder
from enc_dec_32_full import Encoder, Decoder
from distributions import lognormal, Flow1, Flow1_grid #Normal,  #, IAF_flow
from torch.distributions import Beta

from inference_net_grid import 


class VAE(nn.Module):
    def __init__(self, kwargs):
        super(VAE, self).__init__()


        kwargs['act_func'] = F.leaky_relu

        self.__dict__.update(kwargs)

        self.beta_scale = 100.
        lr = .0004
        

        # q(z|x)
        # self.image_encoder2 = Encoder(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=self.enc_res_blocks, input_size=self.input_size)
        self.q = Encoder(input_nc=3, image_encoding_size=self.x_enc_size, n_residual_blocks=self.enc_res_blocks, input_size=self.input_size)

        # p(z)
        self.prior = Flow1_grid(z_shape=[6,8,8], n_flows=self.n_prior_flows)

        # p(x|z)
        self.image_decoder = Decoder(z_size=self.x_enc_size, output_nc=3, n_residual_blocks=self.dec_res_blocks, output_size=self.input_size)


        decode0_params = [list(self.image_encoder2.parameters()) 
                            + list(self.image_decoder.parameters()) + list(self.prior.parameters())]  
        self.optimizer_x = optim.Adam(decode0_params[0], lr=lr, weight_decay=.0000001)
        self.scheduler_x = lr_scheduler.StepLR(self.optimizer_x, step_size=1, gamma=0.999995)







    # def inference_net(self, x):
    #     mean_logvar = self.image_encoder2(x)
    #     mean = mean_logvar[:,:6]
    #     logvar = mean_logvar[:,6:]
    #     logvar = torch.clamp(logvar, min=-15., max=10.)
    #     return mean, logvar






    def sample(self, mu, logvar, k=1):

        B = mu.shape[0]
        eps = torch.FloatTensor(B,6,8,8).normal_().cuda() #[B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mu  #[B,Z]
        
        # logpz = lognormal(flat_z, torch.zeros(B, self.z_size).cuda(), 
        #                     torch.zeros(B, self.z_size).cuda()) #[B]
        # logpz = self.prior.logprob(z)
        
        flat_z = z.view(B, -1)
        logqz = lognormal(flat_z, mu.view(B, -1).detach(), logvar.view(B, -1).detach())
        return z, logqz




    def forward(self, x=None, warmup=1., inf_net=None): #, k=1): #, marginf_type=0):

        outputs = {}

        if inf_net is None:
            mu, logvar = self.inference_net(x)
        else:
            mu, logvar = inf_net.inference_net(x)   

        z, logqz = self.sample(mu, logvar)

         

        z, logqz = self.q.sample(mu, logvar) 



        logpz = self.prior.logprob(z)


        # Decode Image
        x_hat = self.image_decoder(z)
        alpha = torch.sigmoid(x_hat)
        beta = Beta(alpha*self.beta_scale, (1.-alpha)*self.beta_scale)
        x_noise = torch.clamp(x + torch.FloatTensor(x.shape).uniform_(0., 1./256.).cuda(), min=1e-5, max=1-1e-5)
        # logpx = beta.log_prob(x + torch.FloatTensor(x.shape).uniform_(0., 1./256.).cuda()) #[120,3,112,112]  # add uniform noise here
        logpx = beta.log_prob(x_noise) #[120,3,112,112]  # add uniform noise here
        B = z.shape[0]
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

        # # B  = 20
        # if z is None:
        #   z = torch.FloatTensor(B, self.z_size).normal_().cuda() * std

        # eps = torch.FloatTensor(B,6,8,8).normal_().cuda() #[B,Z]
        # z = eps
        # print (z.shape)
        # fafa
        B = z.shape[0]
        z=z.view(B,6,8,8)
        z = self.prior.sample(shape=z.shape, eps=z)

        # z_dec = self.z_to_dec(z)
        x_hat = self.image_decoder(z)
        x_samp = torch.sigmoid(x_hat)
        # y_hat, sampled_words = self.text_generator.teacher_force(z_dec, generate=True, embeder=self.encoder_embed)
        return x_samp#, y_hat, sampled_words

        # return torch.FloatTensor(10, 3, 32, 32).normal_().cuda()




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
        

















































