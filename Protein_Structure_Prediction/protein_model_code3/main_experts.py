


import numpy as np
import json

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,'../tools')
import protein_model_tools

from os.path import expanduser
home = expanduser("~")

import phase_1_model

import pickle


if __name__ == "__main__":


	#Get data

	#MAKE MY OWN DATA
	# L=166
	# n_aa=22
	# msa = protein_model_tools.make_data(n_samps=20000, L=166, n_aa=22)

	#RASH
	L = 166
	msa_file = home + '/Documents/Protein_data/RASH/RASH_HUMAN2_833a6535-26d0-4c47-8463-7970dae27a32_evfold_result/alignment/RASH_HUMAN2_RASH_HUMAN2_jackhmmer_e-10_m30_complete_run.fa'
	msa, n_aa = protein_model_tools.convert_msa(L, msa_file)
	print len(msa), len(msa[0])


	#Learn the experts
	experts_weights = np.zeros((L,L,n_aa,n_aa))
	# experts_biases = np.zeros((L,L,n_aa))

	total = L*L

	for i in range(L):
		for j in range(L):

			if i == j:
				continue

			print
			print str((i*L)+j) + '/' + str(total) + '  '+ str(i) + ' ' + str(j)
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

			experts_weights[i,j] = model1.w
			# experts_biases[i,j] = model1.b

			# if j ==3:
			# 	fjds;kf

			
		pickle.dump(experts_weights, open(home + '/Documents/Protein_data/RASH/experts.pkl', "wb" ))
		print 'saved'
	








