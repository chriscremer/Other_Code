



import numpy as np
import tensorflow as tf
import math
import json
import random
import pickle
import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# sys.path.insert(0,'../tools')
# import protein_model_tools
from os.path import expanduser
home = expanduser("~")



experts = pickle.load(open(home + '/Documents/Protein_data/RASH/experts/expert_0.pkl', "rb" ))

experts = np.array(experts)

print experts.shape

# print experts[0][0]
# print experts[0][1]
# print
# print experts[0][2]
# print experts[1][0]