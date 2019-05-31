    



from os.path import expanduser
home = expanduser("~")

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import _pickle as pickle
import argparse
import time
import subprocess
import json

from collections import deque 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn


from vae3 import VAE

from distributions import Gauss
from inference_net import Inf_Net
from generator import Generator



file_ = home+ "/Documents/Inf_Sub/vae_encoder_only_gauss/gaps/results.pkl"

with open(file_, "rb" ) as f:
    gaps_dict = pickle.load(f)


L_vae_q= gaps_dict['L_vae_q']
L_iwae_q =gaps_dict['L_iwae_q'] 
L_vae_qstar = gaps_dict['L_vae_qstar'] 
L_iwae_qstar = gaps_dict['L_iwae_qstar'] 


print (file_)
print ('\nq bounds')
print ('L_vae_q', np.mean(L_vae_q), np.std(L_vae_q))
print ('L_iwae_q', np.mean(L_iwae_q), np.std(L_iwae_q))

print ('\nq* bounds')
print ('L_vae_qstar', np.mean(L_vae_qstar), np.std(L_vae_qstar))
print ('L_iwae_qstar', np.mean(L_iwae_qstar), np.std(L_iwae_qstar))
print ('amort', np.mean(L_vae_qstar) - np.mean(L_vae_q))
print ('approx', np.mean(L_iwae_qstar) - np.mean(L_vae_qstar))






file_ = home+ "/Documents/Inf_Sub/vae_encoder_only_flow2/gaps/results.pkl"

with open(file_, "rb" ) as f:
    gaps_dict = pickle.load(f)


L_vae_q= gaps_dict['L_vae_q']
L_iwae_q =gaps_dict['L_iwae_q'] 
L_vae_qstar = gaps_dict['L_vae_qstar'] 
L_iwae_qstar = gaps_dict['L_iwae_qstar'] 


print()
print()
print (file_)
print ('\nq bounds')
print ('L_vae_q', np.mean(L_vae_q), np.std(L_vae_q))
print ('L_iwae_q', np.mean(L_iwae_q), np.std(L_iwae_q))

print ('\nq* bounds')
print ('L_vae_qstar', np.mean(L_vae_qstar), np.std(L_vae_qstar))
print ('L_iwae_qstar', np.mean(L_iwae_qstar), np.std(L_iwae_qstar))
print ('amort', np.mean(L_vae_qstar) - np.mean(L_vae_q))
print ('approx', np.mean(L_iwae_qstar) - np.mean(L_vae_qstar))






file_ = home+ "/Documents/Inf_Sub/vae_encoder_only_flow4/gaps/results.pkl"

with open(file_, "rb" ) as f:
    gaps_dict = pickle.load(f)


L_vae_q= gaps_dict['L_vae_q']
L_iwae_q =gaps_dict['L_iwae_q'] 
L_vae_qstar = gaps_dict['L_vae_qstar'] 
L_iwae_qstar = gaps_dict['L_iwae_qstar'] 


print()
print()
print (file_)
print ('\nq bounds')
print ('L_vae_q', np.mean(L_vae_q), np.std(L_vae_q))
print ('L_iwae_q', np.mean(L_iwae_q), np.std(L_iwae_q))

print ('\nq* bounds')
print ('L_vae_qstar', np.mean(L_vae_qstar), np.std(L_vae_qstar))
print ('L_iwae_qstar', np.mean(L_iwae_qstar), np.std(L_iwae_qstar))
print ('amort', np.mean(L_vae_qstar) - np.mean(L_vae_q))
print ('approx', np.mean(L_iwae_qstar) - np.mean(L_vae_qstar))







