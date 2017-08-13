import math
import torch
from torch.autograd import Variable



# chagne this to have no divisions

def lognormal(x, mean, logvar):
    '''
    x: [B,Z]
    mean,logvar: [B,Z]
    output: [B]
    '''

    D = x.size()[1]
    term1 = D * torch.log(torch.FloatTensor([2.*math.pi])) #[1]
    term2 = logvar.sum(1) #sum over D, [B]
    dif_cov = (x - mean).pow(2)
    dif_cov.div(torch.exp(logvar)) #exp_()) #[P,B,D]
    term3 = dif_cov.sum(1) #sum over D, [P,B]

    all_ = Variable(term1) + term2 + term3  #[P,B]
    log_N = -.5 * all_
    return log_N

