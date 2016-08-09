

import numpy as np
import os
import numpy as np
import tensorflow as tf
import math
import random
import pickle
import json

import phase_1_model
import protein_model_tools

from multiprocessing import Pool


#RASH
L = 166
msa_file = 'RASH_HUMAN2_RASH_HUMAN2_jackhmmer_e-10_m30_complete_run.fa'
msa, n_aa = protein_model_tools.convert_msa(L, msa_file)
print len(msa), len(msa[0])


def learn_expert(ij):
	
	expert_weights = np.zeros((n_aa,n_aa))
	
	#ij2 = ij[0]
	i = ij[0]
	j = ij[1]

	if i == j:
		print 'start ' + str(i) + ' ' + str(j)
		print 'done ' + str(i) + ' ' + str(j)
		return expert_weights
		
	else:
		print 'start ' + str(i) + ' ' + str(j)
		#print str((i*L)+j) + '/' + str(total) + '  '+ str(i) + ' ' + str(j)
		#Make datasets
		X_train = []
		y_train = []
		X_valid = []
		y_valid = []
		for samp in range(len(msa)):
			if samp < 500:
				vec = np.zeros((n_aa))
				vec[msa[samp][i]] = 1
				X_valid.append(vec)
				vec = np.zeros((n_aa))
				vec[msa[samp][j]] = 1
				y_valid.append(vec)	
			else:		
				vec = np.zeros((n_aa))
				vec[msa[samp][i]] = 1
				X_train.append(vec)
				vec = np.zeros((n_aa))
				vec[msa[samp][j]] = 1
				y_train.append(vec)
		X_train = np.array(X_train)
		y_train = np.array(y_train)
		X_valid = np.array(X_valid)
		y_valid = np.array(y_valid)

		#Train model
		model1 = phase_1_model.Expert_Learner(n_aa=22, tol=.01, batch_size=5, lr=.0001, mom=.9, lmbd=.0001)
		model1.fit([X_train,y_train], [X_valid,y_valid])

		experts_weights = model1.w
		# experts_biases[i,j] = model1.b

		print 'done ' + str(i) + ' ' + str(j)
		
		return expert_weights


#for each position, use pool to learn the L experts, save it to experts folder
todo_list = range(25,165)
todo_list = todo_list[::-1]
for i in todo_list:
    print i
    
    work = [[i,j] for j in range(L)]
    #print work
    
    
    p = Pool(8)
    experts = p.map(learn_expert, work)
    
    #save experts
    pickle.dump(experts, open('Experts/expert_' + str(i) + '.pkl', "wb" ))
    print 'saved expert ' + str(i)


