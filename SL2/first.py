

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt 


digits = load_digits()
print(digits.data.shape)
print(digits.target.shape)

import numpy as np


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



# from RNN import RNN
from RNN_LSTM import RNN

# plt.gray() 
# plt.matshow(digits.images[0]) 
# plt.show()




# Task: read in some digits and repeat the labels of those digits 
# could begin with single digit and no time delay . aka standard class prediction
# then increase difficulty 


# print (digits.target)


n_data = digits.data.shape[0]
print(n_data)

#Scale data 
digits.data = digits.data / 16.
# print (np.max(digits.data))
# print (np.min(digits.data))



# data_index = np.random.randint(n_data)

# print (digits.target[data_index])

# plt.gray() 
# plt.matshow(digits.images[data_index]) 
# plt.show()


#Batch [T,B,X]


#How to tell the RNN what task to do? 
# Specific train function for each task?? ya I guess

input_size = 64
output_size = 10
z_size = 10
    
specs = { 'input_size': input_size,
            'output_size': output_size,
            'z_size': z_size,
            # 'update_net': [input_size+z_size, 30, z_size],
            # 'update_net': [input_size+z_size, 100, 100, z_size],
            # 'output_net': [z_size, 30, output_size]
        }


# specs = { 'input_size': input_size,
#             'output_size': output_size,
#             'z_size': z_size,
#             'input_net': [input_size+z_size, 30, z_size],
#             # 'input_net': [input_size+z_size, 100, 100, z_size],
#             'output_net': [z_size, 30, output_size]
#         }   






#LSTM

# lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
# inputs = [Variable(torch.randn((1, 3))) for _ in range(5)]  # make a sequence of length 5
# # initialize the hidden state and cell state
# hidden = (Variable(torch.randn(1, 1, 3)),Variable(torch.randn((1, 1, 3)))) 

# for i in inputs:
#     # Step through the sequence one element at a time.
#     # after each step, hidden contains the hidden state.
#     #out contains the hidden state
#     #hidden contains (hidden, cell)  states
#     #so this is pretty confusing, no reason to have out there
#     out, hidden = lstm(i.view(1, 1, -1), hidden)  #[1,B,X]
#     print ('out')
#     print (out)
#     print ('hidden')
#     print (hidden)
#     # print(out.data.numpy())
#     # print(hidden.data.numpy())


# fadad

model = RNN(specs)
print (model)
batch_size = 5



# try:
model.train(data_x=digits.data, data_y=digits.target, batch_size=batch_size)
# except:
#     pass


# model.init_state(batch_size)
# pred = model.output_prediction()

# print (pred)




print ('Done.')


















































