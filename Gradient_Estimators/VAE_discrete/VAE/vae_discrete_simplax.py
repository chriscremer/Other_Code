







import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler



# from encoder_decoder import Encoder, Decoder
from enc_dec_discrete import Decoder
from inference_net_discrete import Inference_Q_Bernoulli
from NN_simplax import Surrogate

from torch.distributions import Beta
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

def isnan(x):
    return x != x

def prob_to_logit(prob):
    return torch.log(prob) - torch.log(1-prob)

def logit_to_prob(logit):
    return torch.sigmoid(logit)

def harden(x):
    return (x>.5).float()

class VAE(nn.Module):
    def __init__(self, kwargs):
        super(VAE, self).__init__()


        kwargs['act_func'] = F.leaky_relu

        self.__dict__.update(kwargs)

        self.beta_scale = 100.

        # lr = .0004
        
        self.q = Inference_Q_Bernoulli(kwargs)

        self.surr = Surrogate(kwargs=kwargs, input_nc=3, image_encoding_size=self.x_enc_size, 
                                    n_residual_blocks=self.enc_res_blocks, input_size=self.input_size)

        # self.prior = Bernoulli(logits=torch.zeros(self.z_size).cuda())
        # self.prior = Bernoulli(probs=.1*torch.ones(self.z_size).cuda())

        self.generator = Decoder(kwargs=kwargs, z_size=self.z_size, output_nc=3, 
                                    n_residual_blocks=self.dec_res_blocks, output_size=self.input_size)

        params = [list(self.q.parameters())] # + list(self.generator.parameters()) ] #+ list(self.prior.parameters())]  
        self.optimizer_x = optim.Adam(params[0], lr=.00004, weight_decay=.0000001)
        self.scheduler_x = lr_scheduler.StepLR(self.optimizer_x, step_size=1, gamma=0.999995)

        self.optimizer_surr = optim.Adam(self.surr.parameters(), lr=.0004, weight_decay=.0000001)
        self.scheduler_surr = lr_scheduler.StepLR(self.optimizer_surr, step_size=1, gamma=0.999995)





    def f(self, x, z, logits, hard=False):

        B = x.shape[0]

        # image likelihood given b
        # b = harden(z).detach()
        x_hat = self.generator.forward(z)
        alpha = torch.sigmoid(x_hat)
        beta = Beta(alpha*self.beta_scale, (1.-alpha)*self.beta_scale)
        x_noise = torch.clamp(x + torch.FloatTensor(x.shape).uniform_(0., 1./256.).cuda(), min=1e-5, max=1-1e-5)
        logpx = beta.log_prob(x_noise) #[120,3,112,112]  # add uniform noise here
        logpx = torch.sum(logpx.view(B, -1),1) # [PB]  * self.w_logpx

        # prior is constant I think 
        # for q(b|x), we just want to increase its entropy 
        if hard:
            dist = Bernoulli(logits=logits)
        else:
            dist = RelaxedBernoulli(torch.Tensor([1.]).cuda(), logits=logits)
            
        logqb = dist.log_prob(z.detach())
        logqb = torch.sum(logqb,1)

        return logpx, logqb, alpha






    def forward(self, grad_est_type, x=None, warmup=1., inf_net=None): #, k=1): #, marginf_type=0):

        outputs = {}
        B = x.shape[0]

        #Samples from relaxed bernoulli 
        z, logits, logqz = self.q.sample(x) 

        if isnan(logqz).any():
            print(torch.sum(isnan(logqz).float()).data.item())
            print(torch.mean(logits).data.item())
            print(torch.max(logits).data.item())
            print(torch.min(logits).data.item())
            print(torch.max(z).data.item())
            print(torch.min(z).data.item())
            fdsfad

        
        # Compute discrete ELBO
        b = harden(z).detach()
        logpx_b, logq_b, alpha1 = self.f(x, b, logits, hard=True)
        fhard = (logpx_b - logq_b).detach()
        

        if grad_est_type == 'SimpLAX':
            # Control Variate
            logpx_z, logq_z, alpha2 = self.f(x, z, logits, hard=False)
            fsoft = logpx_z.detach() #- logq_z
            c = self.surr(x, z).view(B)

            # REINFORCE with Control Variate
            Adv = (fhard - fsoft - c).detach()
            cost1 = Adv * logqz

            # Unbiased gradient of fhard/elbo
            cost_all = cost1 + c + fsoft # + logpx_b

            # Surrogate loss
            surr_cost = torch.abs(fhard - fsoft - c)#**2



        elif grad_est_type == 'RELAX':

            #p(z|b)
            theta = logit_to_prob(logits)
            v = torch.rand(z.shape[0], z.shape[1]).cuda()
            v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
            z_tilde = logits.detach() + torch.log(v_prime) - torch.log1p(-v_prime)
            z_tilde = torch.sigmoid(z_tilde)

            # Control Variate
            logpx_z, logq_z, alpha2 = self.f(x, z, logits, hard=False)
            fsoft = logpx_z.detach() #- logq_z
            c_ztilde = self.surr(x, z_tilde).view(B)
            c_z = self.surr(x, z).view(B)

            # REINFORCE with Control Variate
            dist_bern = Bernoulli(logits=logits)
            logqb = dist_bern.log_prob(b.detach())
            logqb = torch.sum(logqb,1)

            Adv = (fhard - fsoft - c_ztilde).detach()
            cost1 = Adv * logqb

            # Unbiased gradient of fhard/elbo
            cost_all = cost1 + fsoft + c_z - c_ztilde#+ logpx_b

            # Surrogate loss
            surr_cost = torch.abs(fhard - fsoft - c_ztilde)#**2




        elif grad_est_type == 'SimpLAX_nosoft':
            # Control Variate
            logpx_z, logq_z, alpha2 = self.f(x, z, logits, hard=False)
            # fsoft = logpx_z.detach() #- logq_z
            c = self.surr(x, z).view(B)

            # REINFORCE with Control Variate
            Adv = (fhard - c).detach()
            cost1 = Adv * logqz

            # Unbiased gradient of fhard/elbo
            cost_all = cost1 + c  # + logpx_b

            # Surrogate loss
            surr_cost = torch.abs(fhard - c)#**2



        elif grad_est_type == 'RELAX_nosoft':

            #p(z|b)
            theta = logit_to_prob(logits)
            v = torch.rand(z.shape[0], z.shape[1]).cuda()
            v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
            z_tilde = logits.detach() + torch.log(v_prime) - torch.log1p(-v_prime)
            z_tilde = torch.sigmoid(z_tilde)

            # Control Variate
            logpx_z, logq_z, alpha2 = self.f(x, z, logits, hard=False)
            # fsoft = logpx_z.detach() #- logq_z
            c_ztilde = self.surr(x, z_tilde).view(B)
            c_z = self.surr(x, z).view(B)

            # REINFORCE with Control Variate
            dist_bern = Bernoulli(logits=logits)
            logqb = dist_bern.log_prob(b.detach())
            logqb = torch.sum(logqb,1)

            Adv = (fhard - c_ztilde).detach()
            cost1 = Adv * logqb

            # Unbiased gradient of fhard/elbo
            cost_all = cost1 + c_z - c_ztilde#+ logpx_b

            # Surrogate loss
            surr_cost = torch.abs(fhard - c_ztilde)#**2






        # Confirm generator grad isnt in encoder grad
        # logprobgrad = torch.autograd.grad(outputs=torch.mean(fhard), inputs=(logits), retain_graph=True)[0]
        # print (logprobgrad.shape, torch.max(logprobgrad), torch.min(logprobgrad))

        # logprobgrad = torch.autograd.grad(outputs=torch.mean(fsoft), inputs=(logits), retain_graph=True)[0]
        # print (logprobgrad.shape, torch.max(logprobgrad), torch.min(logprobgrad))
        # fsdfads


        outputs['logpx'] = torch.mean(logpx_b)
        outputs['x_recon'] = alpha1
        # outputs['welbo'] = torch.mean(logpx + warmup*( logpz - logqz))
        outputs['welbo'] = torch.mean(cost_all) #torch.mean(logpx_b + warmup*(KL))
        outputs['elbo'] = torch.mean(logpx_b - logq_b - 138.63)
        # outputs['logws'] = log_ws
        outputs['z'] = z
        outputs['logpz'] = torch.zeros(1) #torch.mean(logpz)
        outputs['logqz'] = torch.mean(logq_b)
        outputs['surr_cost'] = torch.mean(surr_cost)

        outputs['fhard'] = torch.mean(fhard)
        # outputs['fsoft'] = torch.mean(fsoft)
        # outputs['c'] = torch.mean(c)
        outputs['logq_z'] = torch.mean(logq_z)
        outputs['logits'] = torch.mean(logits)

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
        pretrained_dict = torch.load(save_to)
        # # # # print (state_dict)
        # for key, val in pretrained_dict.items():
        #     print (key, val.shape)
        #     if 'generator' in key:
        #         print (key)

        load_only_generator = 1

        #load only generator 
        if load_only_generator:
            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'generator' in k}
            # for key, val in pretrained_dict.items():
            #     print (key, val.shape)
            #     # if 'generator' in key:
            #     #     print (key)
            # fdsa
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            self.load_state_dict(model_dict)
            print ('loaded GENERATOR params', save_to)
        else:
            self.load_state_dict(pretrained_dict)
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




















































