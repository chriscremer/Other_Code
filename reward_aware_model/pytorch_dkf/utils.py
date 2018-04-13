


import math
import torch
from torch.autograd import Variable




# def lognormal(x, mean, logvar):
#     '''
#     x: [B,Z]
#     mean,logvar: [B,Z]
#     output: [B]
#     '''

#     # # D = x.size()[1]
#     # # term1 = D * torch.log(torch.FloatTensor([2.*math.pi])) #[1]
#     # term2 = logvar.sum(1) #sum over D, [B]
#     # dif_cov = (x - mean).pow(2)
#     # # dif_cov.div(torch.exp(logvar)) #exp_()) #[P,B,D]
#     # term3 = (dif_cov/torch.exp(logvar)).sum(1) #sum over D, [P,B]
#     # # all_ = Variable(term1) + term2 + term3  #[P,B]
#     # all_ = term2 + term3  #[P,B]
#     # log_N = -.5 * all_
#     # return log_N

#     # term2 = logvar.sum(1) #sum over D, [B]
#     # dif_cov = (x - mean).pow(2)
#     # term3 = (dif_cov/torch.exp(logvar)).sum(1) #sum over D, [P,B]
#     # all_ = term2 + term3  #[P,B]
#     # log_N = -.5 * all_
#     # return log_N

#     # one line 
#     return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(1))


    



def lognormal2(x, mean, logvar):
    '''
    x: [P,B,Z]
    mean,logvar: [B,Z]
    output: [P,B]
    '''

    return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(2))



def lognormal3(x, mean, logvar):
    '''
    x: [P,B,Z]
    mean,logvar: [P,B,Z]
    output: [P,B]
    '''

    return -.5 * (logvar.sum(2) + ((x - mean).pow(2)/torch.exp(logvar)).sum(2))


    
def log_bernoulli(pred_no_sig, target):
    '''
    pred_no_sig is [P, B, X] 
    t is [B, X]
    output is [P, B]
    '''

    return -(torch.clamp(pred_no_sig, min=0)
                        - pred_no_sig * target
                        + torch.log(1. + torch.exp(-torch.abs(pred_no_sig)))).sum(2) #sum over dimensions


























