


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



import sys
sys.path.insert(0, 'utils')
from utils import lognormal2 as lognormal






class standard(nn.Module):

    def __init__(self, model, hyper_config):
        super(standard, self).__init__()

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.z_size = model.z_size
        self.x_size = model.x_size
        self.act_func = model.act_func


        #Encoder
        self.fc1 = nn.Linear(self.x_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, self.z_size*2)



    def forward(self, k, x, logposterior):
        '''
        k: number of samples
        x: [B,X]
        logposterior(z) -> [P,B]
        '''

        self.B = x.size()[0]

        #Encode
        out = self.act_func(self.fc1(x))
        out = self.act_func(self.fc2(out))
        out = self.fc3(out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]

        #Sample
        eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz = lognormal(z, mean, logvar) #[P,B]

        return z, logqz


























class flow1(nn.Module):

    def __init__(self, model, hyper_config):
        super(flow1, self).__init__()

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.n_flows = hyper_config['n_flows']

        self.z_size = model.z_size
        self.x_size = model.x_size
        self.act_func = model.act_func


        #Encoder
        self.fc1 = nn.Linear(self.x_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, self.z_size*2)


        self.params = []
        for i in range(self.n_flows):

            self.params.append([nn.Linear(self.z_size, 100), nn.Linear(100, self.z_size), nn.Linear(100, self.z_size)])

        # self.param_list = nn.ParameterList(self.params)

        count =1
        for i in range(self.n_flows):

            self.add_module(str(count), self.params[i][0])
            count+=1
            self.add_module(str(count), self.params[i][1])
            count+=1
            self.add_module(str(count), self.params[i][2])
            count+=1

    

    def forward(self, k, x, logposterior):
        '''
        k: number of samples
        x: [B,X]
        logposterior(z) -> [P,B]
        '''

        self.B = x.size()[0]
        self.P = k

        #Encode
        out = self.act_func(self.fc1(x))
        out = self.act_func(self.fc2(out))
        out = self.fc3(out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]

        #Sample
        eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz = lognormal(z, mean, logvar) #[P,B]

        logdetsum = 0.
        for i in range(self.n_flows):

            z, logdet = self.norm_flow(self.params[i],z)
            logdetsum += logdet


        return z, logqz-logdetsum




    def norm_flow(self, params, z):

        # [Z]
        mask = Variable(torch.zeros(self.z_size)).type(self.dtype)
        mask[:int(self.z_size/2)] = 1.
        mask = mask.view(1,1,-1)

        # [P,B,Z]
        z1 = z*mask
        # [PB,Z]
        z1 = z1.view(-1, self.z_size)

        h = F.tanh(params[0](z1))
        mew_ = params[1](h)
        sig_ = F.sigmoid(params[2](h)+5.) #[PB,Z]

        z = z.view(-1, self.z_size)
        mask = mask.view(1, -1)

        z2 = (z*sig_ +mew_)*(1-mask)
        z = z1 + z2
        # [PB]
        logdet = torch.sum((1-mask)*torch.log(sig_), 1)
        # [P,B]
        logdet = logdet.view(self.P,self.B)
        #[P,B,Z]
        z = z.view(self.P,self.B,self.z_size)


        #Other half

        # [Z]
        mask2 = Variable(torch.zeros(self.z_size)).type(self.dtype)
        mask2[int(self.z_size/2):] = 1.
        mask = mask2.view(1,1,-1)

        # [P,B,Z]
        z1 = z*mask
        # [PB,Z]
        z1 = z1.view(-1, self.z_size)

        h = F.tanh(params[0](z1))
        mew_ = params[1](h)
        sig_ = F.sigmoid(params[2](h)+5.) #[PB,Z]

        z = z.view(-1, self.z_size)
        mask = mask.view(1, -1)

        z2 = (z*sig_ +mew_)*(1-mask)
        z = z1 + z2
        # [PB]
        logdet2 = torch.sum((1-mask)*torch.log(sig_), 1)
        # [P,B]
        logdet2 = logdet2.view(self.P,self.B)
        #[P,B,Z]
        z = z.view(self.P,self.B,self.z_size)

        logdet = logdet + logdet2
        

        return z, logdet


























# class auxvar(nn.Module):

#     def __init__(self, model, hyper_config):
#         super(flow1, self).__init__()

#         if torch.cuda.is_available():
#             self.dtype = torch.cuda.FloatTensor
#         else:
#             self.dtype = torch.FloatTensor

#         # self.n_flows = hyper_config['n_flows']

#         self.z_size = model.z_size
#         self.x_size = model.x_size
#         self.act_func = model.act_func

#         # self.params = []
#         # for i in range(self.n_flows):

#         #     # self.params.append([nn.Linear(int(self.z_size/2), 100), nn.Linear(100, int(self.z_size/2)), nn.Linear(100, int(self.z_size/2))])
#         #     self.params.append([nn.Linear(self.z_size, 100), nn.Linear(100, self.z_size), nn.Linear(100, self.z_size)])

#         # self.param_list = nn.ParameterList(self.params)

#         # count =1
#         # for i in range(self.n_flows):

#         #     self.add_module(str(count), self.params[i][0])
#         #     count+=1
#         #     self.add_module(str(count), self.params[i][1])
#         #     count+=1
#         #     self.add_module(str(count), self.params[i][2])
#         #     count+=1

    
#         #q(v|x)
#         self.fc1 = nn.Linear(self.x_size, 200)
#         self.fc2 = nn.Linear(200, 200)
#         self.fc3 = nn.Linear(200, self.z_size*2)

#         #q(z|x,v)




#     def forward(self, k, x, logposterior):
#         '''
#         z: [P,B,Z]
#         '''
#         self.P = k
#         self.B = mean.size()[0]

#         #q(v|x)


#         # q(z|x,v)
#         eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
#         z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
#         logqz = lognormal(z, mean, logvar) #[P,B]

#         #q(v|x,z0)
#         out = self.act_func(self.fc1(x))
#         out = self.act_func(self.fc2(out))
#         out = self.fc3(out)
#         mean = out[:,:self.z_size]
#         logvar = out[:,self.z_size:]
#         return mean, logvar



#         logdetsum = 0.
#         for i in range(self.n_flows):

#             z, logdet = self.norm_flow(self.params[i],z)
#             logdetsum += logdet

#         return z, logqz0+logqv0-logdetsum-logrvT


#     def norm_flow(self, params, z):

#         # [Z]
#         mask = Variable(torch.zeros(self.z_size)).type(self.dtype)
#         mask[:int(self.z_size/2)] = 1.
#         mask = mask.view(1,1,-1)

#         # [P,B,Z]
#         z1 = z*mask
#         # [PB,Z]
#         z1 = z1.view(-1, self.z_size)

#         h = F.tanh(params[0](z1))
#         mew_ = params[1](h)
#         sig_ = F.sigmoid(params[2](h)+5.) #[PB,Z]

#         z = z.view(-1, self.z_size)
#         mask = mask.view(1, -1)

#         z2 = (z*sig_ +mew_)*(1-mask)
#         z = z1 + z2
#         # [PB]
#         logdet = torch.sum((1-mask)*torch.log(sig_), 1)
#         # [P,B]
#         logdet = logdet.view(self.P,self.B)
#         #[P,B,Z]
#         z = z.view(self.P,self.B,self.z_size)


#         #Other half

#         # [Z]
#         mask2 = Variable(torch.zeros(self.z_size)).type(self.dtype)
#         mask2[int(self.z_size/2):] = 1.
#         mask = mask2.view(1,1,-1)

#         # [P,B,Z]
#         z1 = z*mask
#         # [PB,Z]
#         z1 = z1.view(-1, self.z_size)

#         h = F.tanh(params[0](z1))
#         mew_ = params[1](h)
#         sig_ = F.sigmoid(params[2](h)+5.) #[PB,Z]

#         z = z.view(-1, self.z_size)
#         mask = mask.view(1, -1)

#         z2 = (z*sig_ +mew_)*(1-mask)
#         z = z1 + z2
#         # [PB]
#         logdet2 = torch.sum((1-mask)*torch.log(sig_), 1)
#         # [P,B]
#         logdet2 = logdet2.view(self.P,self.B)
#         #[P,B,Z]
#         z = z.view(self.P,self.B,self.z_size)

#         logdet = logdet + logdet2
        

#         return z, logdet








class aux_nf(nn.Module):

    def __init__(self, model, hyper_config):
        super(aux_nf, self).__init__()

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.n_flows = hyper_config['n_flows']

        self.z_size = model.z_size
        self.x_size = model.x_size
        self.act_func = model.act_func


        #q(v|x)
        self.fc1 = nn.Linear(self.x_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, self.z_size*2)
        #q(z|x,v)
        self.fc4 = nn.Linear(self.x_size+self.z_size, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, self.z_size*2)
        #r(v|x,z)
        self.fc7 = nn.Linear(self.x_size+self.z_size, 200)
        self.fc8 = nn.Linear(200, 200)
        self.fc9 = nn.Linear(200, self.z_size*2)


        self.params = []
        for i in range(self.n_flows):

            #first is for v, second is for z
            self.params.append([
                                [nn.Linear(self.z_size, 100), nn.Linear(100, self.z_size), nn.Linear(100, self.z_size)],
                                [nn.Linear(self.z_size, 100), nn.Linear(100, self.z_size), nn.Linear(100, self.z_size)]
                                ])

        # self.param_list = nn.ParameterList(self.params)

        count =1
        for i in range(self.n_flows):

            self.add_module(str(count), self.params[i][0][0])
            count+=1
            self.add_module(str(count), self.params[i][1][0])
            count+=1
            self.add_module(str(count), self.params[i][0][1])
            count+=1
            self.add_module(str(count), self.params[i][1][1])
            count+=1
            self.add_module(str(count), self.params[i][0][2])
            count+=1
            self.add_module(str(count), self.params[i][1][2])
            count+=1
    

    def forward(self, k, x, logposterior):
        '''
        k: number of samples
        x: [B,X]
        logposterior(z) -> [P,B]
        '''

        self.B = x.size()[0]
        self.P = k

        #q(v|x)
        out = self.act_func(self.fc1(x))
        out = self.act_func(self.fc2(out))
        out = self.fc3(out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]

        #Sample v0
        eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
        v = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqv0 = lognormal(v, mean, logvar) #[P,B]

        #[PB,Z]
        v = v.view(-1,self.z_size)
        #[PB,X]
        x_tiled = x.repeat(k,1)
        #[PB,X+Z]
        # print (x_tiled.size())
        # print (v.size())

        xv = torch.cat((x_tiled, v),1)

        #q(z|x,v)
        out = self.act_func(self.fc4(xv))
        out = self.act_func(self.fc5(out))
        out = self.fc6(out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]

        #Sample z0
        eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz0 = lognormal(z, mean, logvar) #[P,B]

        #[PB,Z]
        z = z.view(-1,self.z_size)


        logdetsum = 0.
        for i in range(self.n_flows):

            z, v, logdet = self.norm_flow(self.params[i],z,v)
            logdetsum += logdet


        xz = torch.cat((x_tiled,z),1)
        #r(vT|x,zT)
        out = self.act_func(self.fc7(xz))
        out = self.act_func(self.fc8(out))
        out = self.fc9(out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]

        v = v.view(k,self.B,self.z_size)
        logrvT = lognormal(v, mean, logvar) #[P,B]

        z = z.view(k,self.B,self.z_size)

        return z, logqz0+logqv0-logdetsum-logrvT



 
    def norm_flow(self, params, z, v):

        # # [Z]
        # mask = Variable(torch.zeros(self.z_size)).type(self.dtype)
        # mask[:int(self.z_size/2)] = 1.
        # mask = mask.view(1,1,-1)

        # [P,B,Z]
        # z1 = z*mask
        # [PB,Z]
        # z1 = z1.view(-1, self.z_size)

        h = F.tanh(params[0][0](z))
        mew_ = params[0][1](h)
        sig_ = F.sigmoid(params[0][2](h)+5.) #[PB,Z]

        # z = z.view(-1, self.z_size)
        # mask = mask.view(1, -1)

        v = v*sig_ + mew_

        # z2 = (z*sig_ +mew_)*(1-mask)
        # z = z1 + z2
        # [PB]
        # logdet = torch.sum((1-mask)*torch.log(sig_), 1)
        logdet = torch.sum(torch.log(sig_), 1)

        # [P,B]
        # logdet = logdet.view(self.P,self.B)
        #[P,B,Z]
        # z = z.view(self.P,self.B,self.z_size)


        #Other half

        # [Z]
        # mask2 = Variable(torch.zeros(self.z_size)).type(self.dtype)
        # mask2[int(self.z_size/2):] = 1.
        # mask = mask2.view(1,1,-1)

        # [P,B,Z]
        # z1 = z*mask
        # [PB,Z]
        # z1 = z1.view(-1, self.z_size)

        h = F.tanh(params[1][0](v))
        mew_ = params[1][1](h)
        sig_ = F.sigmoid(params[1][2](h)+5.) #[PB,Z]

        # z = z.view(-1, self.z_size)
        # mask = mask.view(1, -1)

        z = z*sig_ + mew_

        # z2 = (z*sig_ +mew_)*(1-mask)
        # z = z1 + z2
        # # [PB]
        # logdet2 = torch.sum((1-mask)*torch.log(sig_), 1)
        logdet2 = torch.sum(torch.log(sig_), 1)

        # [P,B]
        # logdet2 = logdet2.view(self.P,self.B)
        #[P,B,Z]
        # z = z.view(self.P,self.B,self.z_size)

        #[PB]
        logdet = logdet + logdet2
        
        #[PB,Z], [PB]
        return z, v, logdet






















class hnf(nn.Module):

    def __init__(self, model, hyper_config):
        super(hnf, self).__init__()

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.n_flows = hyper_config['n_flows']

        self.z_size = model.z_size
        self.x_size = model.x_size
        self.act_func = model.act_func


        #q(v|x)
        self.fc1 = nn.Linear(self.x_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, self.z_size*2)
        #q(z|x,v)
        self.fc4 = nn.Linear(self.x_size+self.z_size, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, self.z_size*2)
        #r(v|x,z)
        self.fc7 = nn.Linear(self.x_size+self.z_size, 200)
        self.fc8 = nn.Linear(200, 200)
        self.fc9 = nn.Linear(200, self.z_size*2)


        self.params = []
        for i in range(self.n_flows):

            #first is for v, second is for z
            self.params.append([
                                [nn.Linear(self.z_size, 100), nn.Linear(100, self.z_size), nn.Linear(100, self.z_size)],
                                [nn.Linear(self.z_size, 100), nn.Linear(100, self.z_size), nn.Linear(100, self.z_size)]
                                ])

        # self.param_list = nn.ParameterList(self.params)

        count =1
        for i in range(self.n_flows):

            self.add_module(str(count), self.params[i][0][0])
            count+=1
            self.add_module(str(count), self.params[i][1][0])
            count+=1
            self.add_module(str(count), self.params[i][0][1])
            count+=1
            self.add_module(str(count), self.params[i][1][1])
            count+=1
            self.add_module(str(count), self.params[i][0][2])
            count+=1
            self.add_module(str(count), self.params[i][1][2])
            count+=1
    

    def forward(self, k, x, logposterior):
        '''
        k: number of samples
        x: [B,X]
        logposterior(z) -> [P,B]
        '''

        self.B = x.size()[0]
        self.P = k


        if torch.cuda.is_available():
            self.grad_outputs = torch.ones(k, self.B).cuda()
        else:
            self.grad_outputs = torch.ones(k, self.B)

        #q(v|x)
        out = self.act_func(self.fc1(x))
        out = self.act_func(self.fc2(out))
        out = self.fc3(out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]

        #Sample v0
        eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
        v = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqv0 = lognormal(v, mean, logvar) #[P,B]

        #[PB,Z]
        v = v.view(-1,self.z_size)
        #[PB,X]
        x_tiled = x.repeat(k,1)
        #[PB,X+Z]
        # print (x_tiled.size())
        # print (v.size())
        xv = torch.cat((x_tiled, v),1)

        #q(z|x,v)
        out = self.act_func(self.fc4(xv))
        out = self.act_func(self.fc5(out))
        out = self.fc6(out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]

        #Sample z0
        eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz0 = lognormal(z, mean, logvar) #[P,B]

        #[PB,Z]
        z = z.view(-1,self.z_size)


        logdetsum = 0.
        for i in range(self.n_flows):

            z, v, logdet = self.norm_flow(self.params[i],z,v,logposterior)
            logdetsum += logdet


        xz = torch.cat((x_tiled,z),1)
        #r(vT|x,zT)
        out = self.act_func(self.fc7(xz))
        out = self.act_func(self.fc8(out))
        out = self.fc9(out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]

        v = v.view(k,self.B,self.z_size)
        logrvT = lognormal(v, mean, logvar) #[P,B]

        z = z.view(k,self.B,self.z_size)

        return z, logqz0+logqv0-logdetsum-logrvT



 
    def norm_flow(self, params, z, v, logposterior):

        h = F.tanh(params[0][0](z))
        mew_ = params[0][1](h)
        sig_ = F.sigmoid(params[0][2](h)+5.) #[PB,Z]


        z_reshaped = z.view(self.P, self.B, self.z_size)

        gradients = torch.autograd.grad(outputs=logposterior(z_reshaped), inputs=z_reshaped,
                          grad_outputs=self.grad_outputs,
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.detach()

        gradients = gradients.view(-1,self.z_size)


        v = v*sig_ + mew_*gradients

        logdet = torch.sum(torch.log(sig_), 1)


        h = F.tanh(params[1][0](v))
        mew_ = params[1][1](h)
        sig_ = F.sigmoid(params[1][2](h)+5.) #[PB,Z]

        z = z*sig_ + mew_*v

        logdet2 = torch.sum(torch.log(sig_), 1)

        #[PB]
        logdet = logdet + logdet2
        
        #[PB,Z], [PB]
        return z, v, logdet




























