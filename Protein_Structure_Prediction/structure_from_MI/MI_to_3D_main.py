



import numpy as np
import json

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,'../tools')
import protein_model_tools

from os.path import expanduser
home = expanduser("~")

import pickle
from os import listdir

import MI_to_3D_model


if __name__ == "__main__":


	#Get data
	#Load MI scores
	#Do stuff to them 


	with open(home + '/Documents/Protein_data/RASH/MIs/MI_bw_1.pkl', 'rb' ) as f:
		MI = pickle.load(f)

	print MI.shape

	print np.max(MI)
	print np.mean(MI)
	print np.min(MI)

	#push them up by the min - so that there all positive
	#divide all by max - so that max is 1
	# should all be between 0 and 1

	#make diagonal all very low, ie should be zero in the end
	#make both sides symetrical


	min_ = np.min(MI)

	for i in range(len(MI)):
		for j in range(len(MI)):
			if i == j:
				MI[i,j] = min_

	for i in range(len(MI)):
		for j in range(i+1, len(MI)):
			
				MI[j,i] = MI[i,j]

	print
	print np.max(MI)
	print np.mean(MI)
	print np.min(MI)

	# print MI[:5,:5]

	MI = MI - np.min(MI)
	MI = MI / np.max(MI)

	print np.max(MI)
	print np.mean(MI)
	print np.min(MI)

	print MI[:5,:5]



	# sdfasda

	#RASH
	L = 166
	# msa_file = home + '/Documents/Protein_data/RASH/RASH_HUMAN2_833a6535-26d0-4c47-8463-7970dae27a32_evfold_result/alignment/RASH_HUMAN2_RASH_HUMAN2_jackhmmer_e-10_m30_complete_run.fa'
	# msa, n_aa = protein_model_tools.convert_msa(L, msa_file)
	# print len(msa), len(msa[0])	


	#Learn hyperparameters - Grid search
	nu_list = [.001, .01,]
	rho_list = [.0001, .001]
	sigma_adj_list = [.001, .01]
	sigma_repel_list = [.001, .01]
	
	best_error = -1
	best_hypers = []
	best_coords = []

	count =0
	
	for iter1 in range(1):
		print 'Iteration ' + str(iter1)

		# X_train, X_valid = protein_model_tools.split_train_valid(msa, 500)
		# print 'Train: ' + str(X_train.shape)
		# print 'Valid ' + str(X_valid.shape)
		# print 'Protein Length ' + str(L)
		# print 'Number of AAs ' + str(n_aa)

		for nu in nu_list:

			for rho in rho_list:

				for sigma_repel in sigma_repel_list:

					for sigma_adj in sigma_adj_list:

						print
						print 'Repel ' + str(rho) + ' Attract ' + str(nu) + ' sigma_repel ' + str(sigma_repel) + ' sigma_adj ' + str(sigma_adj) 

						#Train model
						model1 = MI_to_3D_model.Structure_Learner(L=L, tol=.001, B=20, lr=.0001, mom=.9, nu=nu, rho=rho, sigma_repel=sigma_repel, sigma_adj=sigma_adj, coordinates=[])
						model1.fit(MI)
						coordinates = np.reshape(model1.coordinates, [model1.coordinates.shape[0], model1.coordinates.shape[1]]).T
						# print 'Predicted coords ' + str(coordinates.shape)


						# fdsafasfsad

						#CHECK OUT THE RESULT


						# rme = protein_model_tools.compare_stuctures3(crysto_coords, coordinates)
						# print 'Error: ' + str(rme)

						#SCALE COORDINATES
						#MULTIPLY coordinates by the ratio of the average distance between adjacent amino acids
						# scaled_coordinates = protein_model_tools.scale_coordinates(coordinates, crysto_coords)

						# rme = protein_model_tools.compare_stuctures3(crysto_coords, scaled_coordinates)
						# print 'Error: ' + str(rme)


						# if rme < best_error or best_error == -1:
						# 	best_error = rme
						# 	best_hypers = [nu, lmbd, rho, sigma_adj, sigma_repel]
						# 	best_coords = coordinates
						# # record_hypers.append([nu, lmbd, rho, sigma_adj_list, sigma_repel_list])
						# # record_error.append(rme)
						# protein_model_tools.make_contact_map(scaled_coordinates, evfold_scaled_coordinates, crysto_coords, n_closest=100, score_ev=error_ev, score_pred=rme)


						with open(home + '/Documents/Protein_data/RASH/coordinates/MI_bw_1' + str(count) + '.pkl', 'wb') as f:

							hypers = [nu, rho, sigma_repel, sigma_adj]
							pickle.dump([coordinates, hypers], f)

						print 'saved coordinates to file'

						count += 1



	# #Save coordinates to JSON for viewing in 3D
	# coor_list = []
	# for i in range(len(best_coords)):
	# 	temp=[]
	# 	for j in range(len(best_coords[i])):
	# 		temp.append(float(best_coords[i][j]))
	# 	coor_list.append(temp)

	# with open(home + '/Documents/Protein_data/RASH/MI_coordinates.json', 'w') as outfile:
	# 	json.dump(coor_list, outfile)
	# print 'saved'


	# print 'Best error= ' + str(best_error)
	# print 'Best hypers= ' + str(best_hypers)


	# scaled_coordinates = protein_model_tools.scale_coordinates(best_coords, crysto_coords)
	# # protein_model_tools.make_contact_map(scaled_coordinates, evfold_scaled_coordinates, crysto_coords, n_closest=100)
	# protein_model_tools.make_contact_map(scaled_coordinates, evfold_scaled_coordinates, crysto_coords, n_closest=100, score_ev=error_ev, score_pred=best_error)






	# print 'made plot'










