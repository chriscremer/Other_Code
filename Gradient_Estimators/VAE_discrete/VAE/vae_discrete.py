







import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler



# from encoder_decoder import Encoder, Decoder
from enc_dec_discrete import Decoder
# from distributions import lognormal, Flow1 #Normal,  #, IAF_flow


from inference_net_discrete import Inference_Q_Bernoulli

from torch.distributions import Beta
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


def harden(x):
    return (x>.5).float()

class VAE(nn.Module):
    def __init__(self, kwargs):
        super(VAE, self).__init__()


        kwargs['act_func'] = F.leaky_relu

        self.__dict__.update(kwargs)

        self.beta_scale = 100.

        lr = .0004
        
        self.q = Inference_Q_Bernoulli(kwargs)

        # self.prior = Bernoulli(logits=torch.zeros(self.z_size).cuda())
        # self.prior = Bernoulli(probs=.1*torch.ones(self.z_size).cuda())

        self.generator = Decoder(kwargs=kwargs, z_size=self.z_size, output_nc=3, n_residual_blocks=self.dec_res_blocks, output_size=self.input_size)

        params = [list(self.q.parameters()) + list(self.generator.parameters()) ] #+ list(self.prior.parameters())]  
        self.optimizer_x = optim.Adam(params[0], lr=lr, weight_decay=.0000001)
        self.scheduler_x = lr_scheduler.StepLR(self.optimizer_x, step_size=1, gamma=0.999995)








    def forward(self, x=None, warmup=1., inf_net=None): #, k=1): #, marginf_type=0):

        outputs = {}
        B = x.shape[0]

        if inf_net is None:
            # mu, logvar = self.inference_net(x)
            z, logits = self.q.sample(x) 
        else:
            # mu, logvar = inf_net.inference_net(x)   
            z, logqz = inf_net.sample(x) 

        # print (z[0])
        # b = harden(z)
        # print (b[0])
        
        # logpz = torch.sum( self.prior.log_prob(b), dim=1)

        # print (logpz[0])
        # print (logpz.shape)
        # fdasf

        probs_q = torch.sigmoid(logits)
        probs_q = torch.clamp(probs_q, min=.00000001, max=.9999999)
        probs_p = torch.ones(B, self.z_size).cuda() *.5
        KL = probs_q*torch.log(probs_q/probs_p) + (1-probs_q)*torch.log((1-probs_q)/(1-probs_p))
        KL = torch.sum(KL, dim=1)

        # print (z.shape)
        # Decode Image
        x_hat = self.generator.forward(z)
        alpha = torch.sigmoid(x_hat)
        beta = Beta(alpha*self.beta_scale, (1.-alpha)*self.beta_scale)
        x_noise = torch.clamp(x + torch.FloatTensor(x.shape).uniform_(0., 1./256.).cuda(), min=1e-5, max=1-1e-5)
        logpx = beta.log_prob(x_noise) #[120,3,112,112]  # add uniform noise here

        logpx = torch.sum(logpx.view(B, -1),1) # [PB]  * self.w_logpx

        # print (logpx.shape,logpz.shape,logqz.shape)
        # fsdfda

        log_ws = logpx - KL #+ logpz - logqz

        outputs['logpx'] = torch.mean(logpx)
        outputs['x_recon'] = alpha
        # outputs['welbo'] = torch.mean(logpx + warmup*( logpz - logqz))
        outputs['welbo'] = torch.mean(logpx + warmup*(KL))
        outputs['elbo'] = torch.mean(log_ws)
        outputs['logws'] = log_ws
        outputs['z'] = z
        outputs['logpz'] = torch.zeros(1) #torch.mean(logpz)
        outputs['logqz'] = torch.mean(KL)
        # outputs['logvar'] = logvar

        return outputs











    def sample_prior(self, std=1., B=20, z=None):

        # bern = Bernoulli(logits=torch.zeros(self.z_size).cuda())

        # bern.sample()


        # # B  = 20
        # if z is None:
        #   z = torch.FloatTensor(B, self.z_size).normal_().cuda() * std

        # z_dec = self.z_to_dec(z)
        x_hat = self.generator.forward(z)
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










    # def get_z_samples(self, q=None, inf_type=2):

    #     self.eval()

    #     if inf_type in [0,2] or dec_type in [0,2]:
    #         embed = self.encoder_embed(q)

    #     y_enc = self.encode_attributes2(embed)
    #     mu, logvar = self.inference_net_y(y_enc)

    #     if self.flow_int:
    #         z, logpz, logqz = self.flow.sample(mu, logvar) 
    #     else:
    #         z, logpz, logqz = self.sample(mu, logvar) 

    #     return z





    # def get_z_samples2(self, x=None, inf_type=1):

    #     self.eval()

    #     x_enc = self.image_encoder2(x)
    #     mu, logvar = self.inference_net_x(x_enc)
            
    #     z, logpz, logqz = self.sample(mu, logvar) 

    #     return z




















































