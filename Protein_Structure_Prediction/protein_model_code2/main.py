


import numpy as np
import json

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,'../protein_model_code2')
sys.path.insert(0,'../tools')
# sys.path.insert(0, '../convert_msa')
# sys.path.insert(0, '../score_predictions')

from os.path import expanduser
home = expanduser("~")

# from model import Position_Encoder
import model
import protein_model_tools



if __name__ == "__main__":


	#Import data

	# #RASH
	# L = 166
	# msa_file = home + '/Documents/Protein_data/RASH/RASH_HUMAN2_833a6535-26d0-4c47-8463-7970dae27a32_evfold_result/alignment/RASH_HUMAN2_RASH_HUMAN2_jackhmmer_e-10_m30_complete_run.fa'
	
	# #PCBP1 
	# # L = 83
	# # msa_file = home + '/Documents/Protein_data/PCBP1/PCBP1_HUMAN_09573638-996d-4c87-afd8-3288c15cd244_evfold_result/alignment/PCBP1_HUMAN_PCBP1_HUMAN_jackhmmer_e-20_m30_complete_run.fa'

	# msa, n_aa = protein_model_tools.convert_msa(L, msa_file)

	# crysto_pdb_file = home + '/Documents/Protein_data/RASH/5p21.pdb'
	# # crysto_pdb_file = home + '/Documents/Protein_data/PCBP1/1wvn.pdb'
	# crysto_coords = protein_model_tools.get_coords_from_pbd(crysto_pdb_file)

	# # #Get EVfold prediction
	# evfold_structure_file = home + '/Documents/Protein_data/RASH/RASH_HUMAN2_833a6535-26d0-4c47-8463-7970dae27a32_evfold_result/structure_outputs/rank_sortable/RASH_HUMAN2_rank01_80_3_hMIN.pdb'
	# evofold_prediction = protein_model_tools.get_coords_from_pbd_for_ev(evfold_structure_file)
	# evofold_prediction = evofold_prediction[:L]
	# # error = protein_model_tools.compare_stuctures3(crysto_coords, evofold_prediction)
	# # print 'EVfold error ' + str(error) 
	# # Calculate error wrt crystollagraphy
	# evfold_scaled_coordinates = protein_model_tools.scale_coordinates(evofold_prediction, crysto_coords)
	# error_ev = protein_model_tools.compare_stuctures3(crysto_coords, evfold_scaled_coordinates)
	# print 'EVfold error ' + str(error_ev) 

	# init = open(home + '/Documents/Protein_data/RASH/predicted_RASH_hypers4.json', 'r')
	# initialiation = json.load(init)
	# initialiation = np.array(initialiation)
	initialiation = []



	#MAKE MY OWN DATA
	msa = protein_model_tools.make_data(n_samps=2000, L=166, n_aa=22)
	n_aa = 22
	L = 166

	# scaled_coordinates = protein_model_tools.scale_coordinates(initialiation, crysto_coords)
	# error_pred = protein_model_tools.compare_stuctures3(crysto_coords, scaled_coordinates)
	# print 'Pred error ' + str(error_pred) 

	# protein_model_tools.make_contact_map(scaled_coordinates, evfold_scaled_coordinates, crysto_coords, n_closest=100, score_ev=error_ev, score_pred=error_pred)
	# vsfadfa

	#Learn hyperparameters
	#Grid search
	nu_list = [.00001]
	lmbd_list = [.01]
	rho_list = [.01]
	sigma_adj_list = [.01]
	sigma_repel_list = [.1]
	
	
	

	# record_hypers = []
	# record_error = []
	best_error = -1
	best_hypers = []
	best_coords = []
	
	for iter1 in range(1):
		print 'Iteration ' + str(iter1)

		X_train, X_valid = protein_model_tools.split_train_valid(msa, 300)
		print 'Train: ' + str(X_train.shape)
		print 'Valid ' + str(X_valid.shape)
		print 'Protein Length ' + str(L)
		print 'Number of AAs ' + str(n_aa)

		for nu in nu_list:

			for lmbd in lmbd_list:

				for rho in rho_list:

					for sigma_repel in sigma_repel_list:

						for sigma_adj in sigma_adj_list:

							print
							print 'Repel/Nu ' + str(nu) + ' Attract/Lmbd ' + str(lmbd) + ' Decay/Rho ' + str(rho) + ' sigma_repel ' + str(sigma_repel) + ' sigma_adj ' + str(sigma_adj) 

							#Train model
							model1 = model.Position_Encoder(L=L, n_aa=n_aa, tol=.000001, B=100, K=4, lr=.00005, mom=.9, nu=nu, lmbd=lmbd, rho=rho, sigma_repel=sigma_repel, sigma_adj=sigma_adj, coordinates=initialiation)
							model1.fit(X_train, X_valid)
							coordinates = np.reshape(model1.coordinates, [model1.coordinates.shape[0], model1.coordinates.shape[1]]).T
							# print 'Predicted coords ' + str(coordinates.shape)


							fdsafasfsad

							#CHECK OUT THE RESULT


							# rme = protein_model_tools.compare_stuctures3(crysto_coords, coordinates)
							# print 'Error: ' + str(rme)

							#SCALE COORDINATES
							#MULTIPLY coordinates by the ratio of the average distance between adjacent amino acids
							scaled_coordinates = protein_model_tools.scale_coordinates(coordinates, crysto_coords)

							rme = protein_model_tools.compare_stuctures3(crysto_coords, scaled_coordinates)
							print 'Error: ' + str(rme)


							if rme < best_error or best_error == -1:
								best_error = rme
								best_hypers = [nu, lmbd, rho, sigma_adj, sigma_repel]
								best_coords = coordinates
							# record_hypers.append([nu, lmbd, rho, sigma_adj_list, sigma_repel_list])
							# record_error.append(rme)
							protein_model_tools.make_contact_map(scaled_coordinates, evfold_scaled_coordinates, crysto_coords, n_closest=100, score_ev=error_ev, score_pred=rme)



	#Save coordinates to JSON for viewing in 3D
	coor_list = []
	for i in range(len(best_coords)):
		temp=[]
		for j in range(len(best_coords[i])):
			temp.append(float(best_coords[i][j]))
		coor_list.append(temp)

	with open(home + '/Documents/Protein_data/RASH/predicted_RASH_hypers7.json', 'w') as outfile:
		json.dump(coor_list, outfile)
	print 'saved'


	print 'Best error= ' + str(best_error)
	print 'Best hypers= ' + str(best_hypers)


	scaled_coordinates = protein_model_tools.scale_coordinates(best_coords, crysto_coords)
	# protein_model_tools.make_contact_map(scaled_coordinates, evfold_scaled_coordinates, crysto_coords, n_closest=100)
	protein_model_tools.make_contact_map(scaled_coordinates, evfold_scaled_coordinates, crysto_coords, n_closest=100, score_ev=error_ev, score_pred=best_error)






	print 'made plot'













