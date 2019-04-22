

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import math

# from IAF import *

# Gaussian

def lognormal(x, mean, logvar):
    # return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).mean(1))

    assert len(x.shape) == len(mean.shape)
    
    return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(1))

    # return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(2))







# def lognormal2(x, mean, logvar):
#     '''
#     x: [P,B,Z]
#     mean,logvar: [B,Z]
#     output: [P,B]
#     '''

#     assert len(x.size()) == 3
#     assert len(mean.size()) == 2
#     assert len(logvar.size()) == 2
#     assert x.size()[1] == mean.size()[0]

#     D = x.size()[2]

#     if torch.cuda.is_available():
#         term1 = D * torch.log(torch.cuda.FloatTensor([2.*math.pi])) #[1]
#     else:
#         term1 = D * torch.log(torch.FloatTensor([2.*math.pi])) #[1]


#     return -.5 * (Variable(term1) + logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(2))













# class Normal(nn.Module):

#     def __init__(self, z_size, mu=None, logvar=None): #, mean, logvar):
#         #mean,logvar: [B,Z]
#         super(Normal, self).__init__()

#         # if torch.cuda.is_available():
#         #     self.dtype = torch.cuda.FloatTensor
#         # else:
#         #     self.dtype = torch.FloatTensor

        

#         # self.B = mean.size()[0]
#         # # self.z_size = mean.size()[1]
#         self.z_size = z_size # hyper_config['z_size']
#         # self.x_size = hyper_config['x_size']
#         # # dfas

#         # self.mean = mean
#         # self.logvar = logvar
#         if mu is not None:
#             self.mu = mu
#             self.logvar = logvar



#     def sample(self, mean=None, logvar=None, k=1):
#         #return [P,B,Z]

#         if mean is None:
#             mean = self.mu
#             logvar = self.logvar   

#         B = mean.size()[0]         

#         # eps = Variable(torch.FloatTensor(self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
#         # eps = torch.FloatTensor(self.B, self.z_size).normal_().type(self.dtype) #[P,B,Z]

#         # print (k, )

#         eps = torch.FloatTensor(k, B, self.z_size).normal_().cuda()

#         z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
#         # logqz = lognormal(z, mean, logvar) #[P,B]

#         return z#, logqz



#     def log_prob(self, z, mean=None, logvar=None):
#         #z [P,B,Z]  [B,Z]
#         # return [P,B]


#         # self.B = mean.size()[0]

#         # eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
#         # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
#         # logqz = lognormal(z, mean, logvar) #[P,B]

#         if mean is None:
#             mean = self.mu
#             logvar = self.logvar


#         assert len(z.shape) == 3
#         assert len(mean.shape) == 2
#         # logqz = -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).mean(1))

#         var = torch.exp(logvar)
#         log_scale = .5 * logvar
#         logprob = -((z - mean) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))


#         # sffdasf


#         return logprob





# class Gaussian(nn.Module):

#     def __init__(self, z_size): #, mean, logvar):
#         #mean,logvar: [B,Z]
#         super(Gaussian, self).__init__()

#         if torch.cuda.is_available():
#             self.dtype = torch.cuda.FloatTensor
#         else:
#             self.dtype = torch.FloatTensor

        

#         # self.B = mean.size()[0]
#         # # self.z_size = mean.size()[1]
#         self.z_size = z_size # hyper_config['z_size']
#         # self.x_size = hyper_config['x_size']
#         # # dfas

#         # self.mean = mean
#         # self.logvar = logvar


#     def sample(self, mean, logvar, k=1):

#         self.B = mean.size()[0]

#         # eps = Variable(torch.FloatTensor(self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
#         eps = torch.FloatTensor(self.B, self.z_size).normal_().type(self.dtype) #[P,B,Z]
#         z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
#         logqz = lognormal(z, mean, logvar) #[P,B]

#         return z, logqz



#     def logprob(self, z, mean, logvar):

#         # self.B = mean.size()[0]

#         # eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
#         # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
#         logqz = lognormal(z, mean, logvar) #[P,B]

#         return logqz






class Flow1(nn.Module):

    def __init__(self, z_size, n_flows=2):#, mean, logvar):
        #mean,logvar: [B,Z]
        super(Flow1, self).__init__()

        # if torch.cuda.is_available():
        #     self.dtype = torch.cuda.FloatTensor
        # else:
        #     self.dtype = torch.FloatTensor

        # self.hyper_config = hyper_config
        # self.B = mean.size()[0]
        self.z_size = z_size #hyper_config['z_size']
        # self.x_size = hyper_config['x_size']

        # self.act_func = hyper_config['act_func']
        

        count =1


        # n_flows = n_flows
        self.n_flows = n_flows
        h_s = 50

        self.z_half_size = int(self.z_size / 2)

        
        self.flow_params = []
        for i in range(n_flows):
            #first is for v, second is for z
            self.flow_params.append([
                                [nn.Linear(self.z_half_size, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)],
                                [nn.Linear(self.z_half_size, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)]
                                ])
        
        for i in range(n_flows):

            self.add_module(str(count), self.flow_params[i][0][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][2])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][2])
            count+=1
    



    def norm_flow(self, params, z1, z2):
        # print (z.size())
        h = torch.tanh(params[0][0](z1))
        mew_ = params[0][1](h)
        # sig_ = F.sigmoid(params[0][2](h)+5.) #[PB,Z]
        sig_ = torch.sigmoid(params[0][2](h)) #[PB,Z]

        z2 = z2*sig_ + mew_
        logdet = torch.sum(torch.log(sig_), 1)


        h = torch.tanh(params[1][0](z2))
        mew_ = params[1][1](h)
        # sig_ = F.sigmoid(params[1][2](h)+5.) #[PB,Z]
        sig_ = torch.sigmoid(params[1][2](h)) #[PB,Z]
        z1 = z1*sig_ + mew_
        logdet2 = torch.sum(torch.log(sig_), 1)

        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet





    def sample(self, mean, logvar, k=1):

        self.B = mean.size()[0]
        # gaus = Gaussian(self.z_size)

        # q(z0)
        # z, logqz0 = gaus.sample(mean, logvar, k)

        eps = torch.FloatTensor(self.B, self.z_size).normal_().cuda() #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz0 = lognormal(z, mean, logvar) #[P,B]



        #[PB,Z]
        # z = z.view(-1,self.z_size)
        # v = v.view(-1,self.z_size)

        #Split z  [PB,Z/2]
        z1 = z.narrow(1, 0, self.z_half_size)
        z2 = z.narrow(1, self.z_half_size, self.z_half_size) 

        #Transform
        logdetsum = 0.
        for i in range(self.n_flows):

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z1, z2, logdet = self.norm_flow(params,z1,z2)
            logdetsum += logdet

        logdetsum = logdetsum.view(self.B)

        #Put z back together  [PB,Z]
        z = torch.cat([z1,z2],1)

        # z = z.view(self.B, self.z_size)


        logqz = logqz0-logdetsum

        logpz = lognormal(z, torch.zeros(self.B, self.z_size).cuda(), 
                            torch.zeros(self.B, self.z_size).cuda())

        return z, logpz, logqz





    def norm_flow_reverse(self, params, z1, z2):

        h = torch.tanh(params[1][0](z2))
        mew_ = params[1][1](h)
        sig_ = torch.sigmoid(params[1][2](h)) #[PB,Z]
        z1 = (z1 - mew_) / sig_
        logdet2 = torch.sum(torch.log(sig_), 1)

        h = torch.tanh(params[0][0](z1))
        mew_ = params[0][1](h)
        sig_ = torch.sigmoid(params[0][2](h)) #[PB,Z]
        z2 = (z2 - mew_) / sig_
        logdet = torch.sum(torch.log(sig_), 1)
        
        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet




    def logprob(self, z, mu, logvar):

        #reverse z_T to z_0 to get q0(z0)

        B = z.shape[0]

        #Split z  [B,Z/2]
        z1 = z.narrow(1, 0, self.z_half_size)
        z2 = z.narrow(1, self.z_half_size, self.z_half_size) 


        #Reverse Transform
        logdetsum = 0.
        reverse_ = list(range(self.n_flows))[::-1]
        for i in reverse_:

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z1, z2, logdet = self.norm_flow_reverse(params,z1,z2)
            logdetsum += logdet

        logdetsum = logdetsum.view(B)


        z0 = torch.cat([z1,z2],1)
        logqz0 = lognormal(z0, mu, logvar)

        logqz = logqz0-logdetsum

        return logqz 


















# class IAF_flow(nn.Module):

#     def __init__(self, z_size):#, mean, logvar):
#         #mean,logvar: [B,Z]
#         super(IAF_flow, self).__init__()

#         self.z_size = z_size 

#         self.n_flows = 2

#         self.gaus = Gaussian(self.z_size)

#         self.iaf1 = IAF(self.z_size, 0)
#         # self.iaf2 = IAF(self.z_size, 0)




#     def sample(self, mean, logvar, k=1):

#         self.B = mean.size()[0]
        

#         # q(z0)
#         z, logqz0 = self.gaus.sample(mean, logvar)

#         z, log_det1 = self.iaf1(z)
#         # z, log_det2 = self.iaf2(z)

#         logdetsum = log_det1 #+ log_det2

#         logqz = logqz0 - logdetsum

#         logpz = lognormal(z, torch.zeros(self.B, self.z_size).cuda(), 
#                             torch.zeros(self.B, self.z_size).cuda())

#         return z, logpz, logqz

















# # Laplace

# def laplace_logprob(x,l):

#     # print (torch.max(x), torch.min(x))
#     # dsfsda

#     means = F.tanh(l[:,:3])
#     log_scales = l[:,3:]




#     centered_x = -torch.abs(x - means)
#     # inv_stdv = torch.exp(-log_scales)

#     # out = centered_x * inv_stdv
#     # out = out - log_scales -.3

#     log_probs = torch.mean(centered_x,0)
#     log_probs = torch.sum(log_probs)

#     return log_probs


# def laplace_sample(l):

#     means = F.tanh(l[:,:3])
#     log_scales = l[:,3:]

#     u = torch.FloatTensor(means.size()).cuda()
#     # u.uniform_(1e-5, 1. - 1e-5)
#     u.uniform_(-.5+1e-5, .5-1e-5)
#     x = means - torch.exp(log_scales) * (torch.sign(u) * torch.log(1. - 2*torch.abs(u)))


#     print (torch.max(x), torch.min(x))

#     x = torch.clamp(x, min=-1, max=1.)


#     print (torch.max(x), torch.min(x))
#     print (torch.max(means), torch.min(means))
#     print (torch.max(log_scales), torch.min(log_scales))
#     print ()


#     return means







# # Discretized Logistic

# def discretized_logistic_logprob(x, l):
#     # x: [B,H,W,C] in range [-1,1]
#     # l: [B,H,W,C*2]

#     # print (x.shape)
#     # print (l.shape)


#     means = l[:,:3]
#     log_scales = l[:,3:]

#     # print (torch.mean(log_scales))

#     centered_x = x - means
#     inv_stdv = torch.exp(-log_scales)

#     # cdf_max = F.sigmoid(inv_stdv * (centered_x + 1. / 255.))
#     # cdf_min = F.sigmoid(inv_stdv * (centered_x - 1. / 255.))




#     inv_stdv = torch.exp(-log_scales)
#     plus_in = inv_stdv * (centered_x + 1. / 255.)
#     cdf_plus = F.sigmoid(plus_in)
#     min_in = inv_stdv * (centered_x - 1. / 255.)
#     cdf_min = F.sigmoid(min_in)


#     # I dont get the stuff below


#     log_cdf_plus = plus_in - F.softplus(plus_in)
#     # log probability for edge case of 255 (before scaling)
#     log_one_minus_cdf_min = -F.softplus(min_in)
#     cdf_delta = cdf_plus - cdf_min  # probability for all other cases
#     mid_in = inv_stdv * centered_x
#     # log probability in the center of the bin, to be used in extreme cases
#     # (not actually used in our code)
#     log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

#     # now select the right output: left edge case, right edge case, normal
#     # case, extremely low prob case (doesn't actually happen for us)

#     # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
#     # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

#     # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
#     # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
#     # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
#     # if the probability on a sub-pixel is below 1e-5, we use an approximation
#     # based on the assumption that the log-density is constant in the bin of
#     # the observed sub-pixel value
    
#     inner_inner_cond = (cdf_delta > 1e-5).float()
#     inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
#     inner_cond       = (x > 0.999).float()
#     inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
#     cond             = (x < -0.999).float()
#     log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out


#     # print (log_probs.shape)

#     log_probs = torch.mean(log_probs,0)
#     log_probs = torch.sum(log_probs)

#     # print (log_probs)
#     # fdsf

#     # logprob = torch.log(cdf_max - cdf_min)

    

#     return log_probs



# def sample_from_discretized_logistic(l):

#     means = l[:,:3]
#     log_scales = l[:,3:]

#     u = torch.FloatTensor(means.size()).cuda()
#     u.uniform_(1e-5, 1. - 1e-5)
#     x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

#     print (torch.max(x), torch.min(x))
#     print (torch.max(means), torch.min(means))
#     print ()

#     x = torch.clamp(x, min=-1, max=1.)

#     return x




















class Flow1_Cond(nn.Module):

    def __init__(self, z_size, h_size, n_flows=2): ##, mean, logvar):
        #mean,logvar: [B,Z]
        super(Flow1_Cond, self).__init__()

        self.z_size = z_size 
        self.h_size = h_size   #this is the datapoint encoding
        h_size_enc = 50


        self.n_flows = n_flows
        h_s = 50

        self.z_half_size = int(self.z_size / 2)


        self.h_encoder = nn.Linear(h_size, h_size_enc)

        
        self.flow_params = []
        for i in range(n_flows):
            #first is for v, second is for z
            self.flow_params.append([
                                [nn.Linear(self.z_half_size + h_size_enc, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)],
                                [nn.Linear(self.z_half_size + h_size_enc, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)]
                                ])
        
        count =1
        for i in range(n_flows):

            self.add_module(str(count), self.flow_params[i][0][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][2])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][2])
            count+=1
    



    def norm_flow(self, params, z1, z2, x):


        # print (z.size())
        h = torch.tanh(params[0][0](torch.cat([z1,x],1)))
        mew_ = params[0][1](h)
        # sig_ = F.sigmoid(params[0][2](h)+5.) #[PB,Z]
        sig_ = torch.sigmoid(params[0][2](h)) #[PB,Z]

        z2 = z2*sig_ + mew_
        logdet = torch.sum(torch.log(sig_), 1)


        h = torch.tanh(params[1][0](torch.cat([z2,x],1)))
        mew_ = params[1][1](h)
        # sig_ = F.sigmoid(params[1][2](h)+5.) #[PB,Z]
        sig_ = torch.sigmoid(params[1][2](h)) #[PB,Z]
        z1 = z1*sig_ + mew_
        logdet2 = torch.sum(torch.log(sig_), 1)

        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet





    def sample(self, mean, logvar, h):

        self.B = mean.size()[0]
        # gaus = Gaussian(self.z_size)

        # q(z0)
        # z, logqz0 = gaus.sample(mean, logvar, k)

        eps = torch.FloatTensor(self.B, self.z_size).normal_().cuda() #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz0 = lognormal(z, mean, logvar) #[P,B]



        #[PB,Z]
        # z = z.view(-1,self.z_size)
        # v = v.view(-1,self.z_size)

        #Split z  [PB,Z/2]
        z1 = z.narrow(1, 0, self.z_half_size)
        z2 = z.narrow(1, self.z_half_size, self.z_half_size) 


        h = self.h_encoder(h)

        #Transform
        logdetsum = 0.
        for i in range(self.n_flows):

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z1, z2, logdet = self.norm_flow(params,z1,z2,h)
            logdetsum += logdet

        logdetsum = logdetsum.view(self.B)

        #Put z back together  [PB,Z]
        z = torch.cat([z1,z2],1)

        # z = z.view(self.B, self.z_size)


        logqz = logqz0-logdetsum

        logpz = lognormal(z, torch.zeros(self.B, self.z_size).cuda(), 
                            torch.zeros(self.B, self.z_size).cuda())

        return z, logpz, logqz





    def norm_flow_reverse(self, params, z1, z2, x):

        # print (torch.cat([z2,x],1).shape)


        h = torch.tanh(params[1][0](torch.cat([z2,x],1)))
        mew_ = params[1][1](h)
        sig_ = torch.sigmoid(params[1][2](h)) #[PB,Z]
        z1 = (z1 - mew_) / sig_
        logdet2 = torch.sum(torch.log(sig_), 1)

        h = torch.tanh(params[0][0](torch.cat([z1,x],1)))
        mew_ = params[0][1](h)
        sig_ = torch.sigmoid(params[0][2](h)) #[PB,Z]
        z2 = (z2 - mew_) / sig_
        logdet = torch.sum(torch.log(sig_), 1)
        
        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet




    def logprob(self, z, mu, logvar, h):

        #reverse z_T to z_0 to get q0(z0)

        B = z.shape[0]

        #Split z  [B,Z/2]
        z1 = z.narrow(1, 0, self.z_half_size)
        z2 = z.narrow(1, self.z_half_size, self.z_half_size) 

        h = self.h_encoder(h)

        #Reverse Transform
        logdetsum = 0.
        reverse_ = list(range(self.n_flows))[::-1]
        for i in reverse_:

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z1, z2, logdet = self.norm_flow_reverse(params,z1,z2,h)
            logdetsum += logdet

        logdetsum = logdetsum.view(B)


        z0 = torch.cat([z1,z2],1)
        logqz0 = lognormal(z0, mu, logvar)

        logqz = logqz0-logdetsum

        return logqz 





