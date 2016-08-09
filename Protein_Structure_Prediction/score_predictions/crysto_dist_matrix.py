



import numpy as np
import csv
from os.path import expanduser
home = expanduser("~")
import json







def get_coords_from_pbd(crysto_pdb_file):

	#Calculate coordinates from the crysto
	coordinates = []
	pos = 0
	with open(crysto_pdb_file, 'rb') as f:
		temp_coor = []
		for row in f:
			if row.split()[0] == 'ATOM':
				aaa = row.split()
				if int(aaa[5]) == pos or pos == 0:
					pos = int(aaa[5])
					temp_coor.append([float(aaa[6]), float(aaa[7]), float(aaa[8])])
				else:
					pos = int(aaa[5]) 
					coordinates.append(np.mean(np.array(temp_coor), axis=0))
					temp_coor = []
					temp_coor.append([float(aaa[6]), float(aaa[7]), float(aaa[8])])
	coordinates.append(np.mean(np.array(temp_coor), axis=0))
	coordinates = np.array(coordinates)
	print 'Crysto shape ' + str(coordinates.shape)

	return coordinates



def compare_stuctures(crysto_pdb_file, predicted_coords):

	crysto_coords = get_coords_from_pbd(crysto_pdb_file)

	#maybe I should scale here

	#calculate distance matrices 
	crysto_dist_matrix = np.zeros((len(crysto_coords), len(crysto_coords)))
	for i in range(len(crysto_coords)):
		for j in range(len(crysto_coords)):
			crysto_dist_matrix[i,j] = np.sum((crysto_coords[i] - crysto_coords[j])**2)

	predicted_dist_matrix = np.zeros((len(predicted_coords), len(predicted_coords)))
	for i in range(len(predicted_coords)):
		for j in range(len(predicted_coords)):
			predicted_dist_matrix[i,j] = np.sum((predicted_coords[i] - predicted_coords[j])**2)


	#negative dist
	crysto_dist_matrix = -crysto_dist_matrix
	predicted_dist_matrix = -predicted_dist_matrix

	#this measure of similarity will softmax all distances then just take rmse
	for i in range(len(crysto_dist_matrix)):
		# print crysto_dist_matrix[i]
		sum_row = np.sum(np.exp(crysto_dist_matrix[i]))
		# print sum_row
		# fadfas
		crysto_dist_matrix[i] = np.exp(crysto_dist_matrix[i]) / sum_row

	for i in range(len(predicted_dist_matrix)):
		sum_row = np.sum(np.exp(predicted_dist_matrix[i]))
		predicted_dist_matrix[i] = np.exp(predicted_dist_matrix[i]) / sum_row


	rmse = np.sum((crysto_dist_matrix - predicted_dist_matrix)**2) / len(crysto_dist_matrix)

	return rmse



# print dist_matrix[:10,:10]

# for i in range(len(dist_matrix-1)):
# 	print dist_matrix[i][i+1]

# lowest = 99
# for i in range(len(dist_matrix)):
# 	for j in range(i,len(dist_matrix)):
# 		# if dist_matrix[i,j] < lowest and dist_matrix[i,j] > 0:
# 		# 	lowest = dist_matrix[i,j]
# 		# 	print lowest
# 		# 	print i, j
# 		if dist_matrix[i,j] < 30 and dist_matrix[i,j] > 0 and j-i > 1:
# 			print i,j 
# 			print dist_matrix[i,j] 

# dist_matrix = list(dist_matrix)


# mat_list = []
# for i in range(len(dist_matrix)):
# 	mat_list.append(list(dist_matrix[i]))


def something_about_zisualizing():

	coor_list = []
	for i in range(len(coordinates)):
		coor_list.append(list(coordinates[i]))

	#save coordinates
	with open(home + '/Downloads/rash_human.json', 'w') as outfile:
	    json.dump(coor_list, outfile)
	print 'saved'

	#make ball-line model


