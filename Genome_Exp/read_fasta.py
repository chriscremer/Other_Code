
import numpy as np
import re

import torch

# header=""
# counts=0
# newline=""


# print (lines[25])
# print (lines[25:27])
# fadssd




# # f2=open('motifs.xls','w')
# count =0
# with open('Mus_musculus.GRCm38.dna.chromosome.1.fa','r') as f1:
#     for line in f1:
#         # print (count, line)
#         # print (len(line))
#         # fasa
#         if 'NN' not in line and count != 0:
#             print (count)
#             fadsa

#         count+=1
#         # if count > 120000:
#         #     fdsfafsd

        
#         # if line.startswith('>'):
#         #     header=line
#         #     #print header
#         #     nextline=line.next()
#         #     for i in nextline:
#         #         motif="ML[A-Z][A-Z][IV]R"
#         #         if re.findall(motif,nextline):
#         #             counts+=1
#         #             #print (header+'\t'+counts+'\t'+motif+'\n')
#         #     fout.write(header+'\t'+counts+'\t'+motif+'\n')
#     print (count)
# # f1.close()
# # f2.close()

# I think after 50001 are good lines

# hey = {'A': 0,
#         'G': 1, 
#         'C': 2,
#         'T': 3,}

# hey = {'A': [1,0,],
#         'G': 1, 
#         'C': 2,
#         'T': 3,}

# print (hey['A','C'])

# dfsad


def to_print_mean(x):
    return torch.mean(x).data.cpu().numpy()
def to_print(x):
    return x.data.cpu().numpy()


def logsumexp(x):

    max_ = torch.max(x, dim=1, keepdim=True)[0]
    # print (max_.shape)
    # fdsf
    # print (x[0,:,0])
    # print (torch.exp(x - max_)[0,:,0])
    # fasdf
    lse = torch.log(torch.sum(torch.exp(x - max_), dim=1, keepdim=True)) + max_
    return lse

def preprocess(line):
    # if 'N' in line:
    #     print ('theres N in here.')
    #     print (line)
    #     fdsfa

    #convert to one hot, I think pytorch might have a some encodign stuff
    # for loop would be slow..
    # print (line)
    # fasdfa

    onehotline = []
    for i in range(len(line)):
        # print (line[i])
        if line[i] == 'A':
            onehotline.append([1,0,0,0])
            # onehotline.append(np.array([1,0,0,0]))
        elif line[i] == 'T':
            # onehotline.append(np.array([0,1,0,0]))
            onehotline.append([0,1,0,0])
        elif line[i] == 'C':
            onehotline.append([0,0,1,0])
        elif line[i] == 'G':
            onehotline.append([0,0,0,1])

    # print(len(line))
    if len(onehotline) !=60:
        print ('length issue')
        print (line)
        fsdaf
    onehotline = np.array(onehotline)
    # print (onehotline.shape)
    # fdsfa
    return onehotline




f=open('Mus_musculus.GRCm38.dna.chromosome.1.fa','r')
lines=f.readlines()

# for i in range(50037, 3257868):
good_indexes = []
for i in range(len(lines)):
    if 'N' not in lines[i] and len(lines[i])==61:
        good_indexes.append(i)
        # print (i, lines[i])
        # fasdf
# print (len(good_indexes))
# fsadfa

# print (len(lines))
# fds



# print (batch.shape)




conv1 = torch.nn.Conv1d(in_channels=4, out_channels=30, kernel_size=5, stride=1, padding=0, dilation=1, groups=1, bias=True)
conv2 = torch.nn.Conv1d(in_channels=30, out_channels=30, kernel_size=5, stride=2, padding=0, dilation=1, groups=1, bias=True)
conv3 = torch.nn.Conv1d(in_channels=30, out_channels=30, kernel_size=5, stride=2, padding=0, dilation=1, groups=1, bias=True)
conv4 = torch.nn.Conv1d(in_channels=30, out_channels=2, kernel_size=11, stride=1, padding=0, dilation=1, groups=1, bias=True)

deconv1 = torch.nn.ConvTranspose1d(in_channels=2, out_channels=30, kernel_size=11, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
deconv2 = torch.nn.ConvTranspose1d(in_channels=30, out_channels=30, kernel_size=5, stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)
deconv3 = torch.nn.ConvTranspose1d(in_channels=30, out_channels=30, kernel_size=5, stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)
deconv4 = torch.nn.ConvTranspose1d(in_channels=30, out_channels=4, kernel_size=5, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

params = [list(conv1.parameters()) + list(conv2.parameters()) + list(conv3.parameters()) + list(conv4.parameters()) +
            list(deconv1.parameters()) + list(deconv2.parameters()) + list(deconv3.parameters()) + list(deconv4.parameters()) ]
optim = torch.optim.Adam(params[0], lr=1e-4, weight_decay=1e-7)
# fasdf
batch_size = 100

for step in range(10000):

    batch = []
    for i in range(batch_size):
        ind = np.random.randint(len(good_indexes))
        seg = preprocess(lines[good_indexes[ind]])
        batch.append(seg)
    batch = np.array(batch)
    batch = torch.tensor(batch).float()
    batch = batch.permute(0, 2, 1)

    # print(batch.shape)
    out = conv1(batch)
    # print('1', out.shape)
    out = conv2(out)
    # print('2', out.shape)
    out = conv3(out)
    # print('3', out.shape)
    z = conv4(out)
    # print('4', out.shape)
    out = deconv1(z)
    # print('1', out.shape)
    out = deconv2(out)
    # print('2', out.shape)
    out = deconv3(out)
    # print('3', out.shape)
    out = deconv4(out)
    # print('4', out.shape)

    #CE LOSS
    CE = torch.mean(  -torch.sum(out * batch, dim=1, keepdim=True) + logsumexp(out))  
    pzloss = .001*torch.mean(z**2)
    loss = CE + pzloss


    optim.zero_grad()
    loss.backward()  
    optim.step()

    if step % 50==0:
        print (step, 'loss:', to_print(loss), 'CE:', to_print(CE), 'pzloss:', to_print(pzloss), )



# aa = logsumexp(out)
# print (out[0,:,0])
# print (aa[0,:,0])
# print (aa.shape)
# print (torch.sum(out * batch, dim=1, keepdim=True).shape)
# fad


# print (batch[0,:,0])
# print (out[0,:,0])
# print (loss[0,:,0])

# print (loss.shape)




















