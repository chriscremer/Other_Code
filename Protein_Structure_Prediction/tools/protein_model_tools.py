

import numpy as np
import csv
from os.path import expanduser
home = expanduser("~")
import json
import math


# import plotly.plotly as py
import plotly
import plotly.graph_objs as go

def make_data(n_samps, L, n_aa):

	X = np.random.randint(1,n_aa,size=(n_samps,L))
	for i in range(len(X)):
		if X[i][10] > n_aa/2:
			X[i][0] = 4
			X[i][2] = 3
		else:
			X[i][0] = 6
			X[i][2] = 2		

		if X[i][10] > n_aa/2:
			X[i][144] = 2
		else:
			X[i][144] = 1
		if X[i][3] > 10:
			X[i][18] = 5
		X[i][12] = X[i][60]

		#Conserved ones
		X[i][34] = 17
		X[i][22] = 8
		X[i][3] = 11

		if X[i][20] > n_aa/2:
			X[i][1] = 20 #so with 50% chance, pos 1 will be 20, so semi conserved


	#90->0
	#3->18
	#12->60

	return X


def compute_dist_matrix(coords):
	#coords should be Lx3 matrix

	l = len(coords)
	#calculate distance matrices 
	dist_matrix = np.zeros((l, l))
	for i in range(l):
		for j in range(l):
			dist_matrix[i,j] = math.sqrt(np.sum((coords[i] - coords[j])**2))

	return dist_matrix



def split_train_valid(X, n_valid):

	X = list(X)
	X_valid = []
	#randomly take n_valid from X
	while len(X_valid) < n_valid:
		samp = X.pop(np.random.randint(len(X)))
		X_valid.append(samp)

	X_train = np.array(X)
	X_valid = np.array(X_valid)

	return X_train, X_valid

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
	# print 'Crysto shape ' + str(coordinates.shape)

	return coordinates


def get_coords_from_pbd_for_ev(crysto_pdb_file):

	#Calculate coordinates from the crysto
	coordinates = []
	pos = 0
	with open(crysto_pdb_file, 'rb') as f:
		temp_coor = []
		for row in f:
			if row.split()[0] == 'ATOM':
				aaa = row.split()
				if int(aaa[4]) == pos or pos == 0:
					pos = int(aaa[4])
					temp_coor.append([float(aaa[5]), float(aaa[6]), float(aaa[7])])
				else:
					pos = int(aaa[4]) 
					coordinates.append(np.mean(np.array(temp_coor), axis=0))
					temp_coor = []
					temp_coor.append([float(aaa[5]), float(aaa[6]), float(aaa[7])])
	coordinates.append(np.mean(np.array(temp_coor), axis=0))
	coordinates = np.array(coordinates)
	# print 'Crysto shape ' + str(coordinates.shape)

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
		sum_row = np.sum(np.exp(crysto_dist_matrix[i]))
		crysto_dist_matrix[i] = np.exp(crysto_dist_matrix[i]) / sum_row

	for i in range(len(predicted_dist_matrix)):
		sum_row = np.sum(np.exp(predicted_dist_matrix[i]))
		predicted_dist_matrix[i] = np.exp(predicted_dist_matrix[i]) / sum_row


	mse = np.sum((crysto_dist_matrix - predicted_dist_matrix)**2) / len(crysto_dist_matrix)

	return mse


def compare_stuctures2(crysto_pdb_file, predicted_coords):

	crysto_coords = get_coords_from_pbd(crysto_pdb_file)

	#calculate distance matrices 
	crysto_dist_matrix = np.zeros((len(crysto_coords), len(crysto_coords)))
	for i in range(len(crysto_coords)):
		for j in range(len(crysto_coords)):
			crysto_dist_matrix[i,j] = np.sum((crysto_coords[i] - crysto_coords[j])**2)

	predicted_dist_matrix = np.zeros((len(predicted_coords), len(predicted_coords)))
	for i in range(len(predicted_coords)):
		for j in range(len(predicted_coords)):
			predicted_dist_matrix[i,j] = np.sum((predicted_coords[i] - predicted_coords[j])**2)


	for i in range(len(crysto_dist_matrix)):
		for j in range(i+2,len(crysto_dist_matrix)):
			if crysto_dist_matrix[i,j] < 20:
				print i,j,crysto_dist_matrix[i,j]
	print 
	for i in range(len(predicted_dist_matrix)):
		for j in range(i+2,len(predicted_dist_matrix)):
			if predicted_dist_matrix[i,j] < 20:
				print i,j,predicted_dist_matrix[i,j]
	fasadf


	#negative dist
	crysto_dist_matrix = -crysto_dist_matrix
	predicted_dist_matrix = -predicted_dist_matrix

	#this measure of similarity will softmax all distances 
	for i in range(len(crysto_dist_matrix)):
		sum_row = np.sum(np.exp(crysto_dist_matrix[i]))
		crysto_dist_matrix[i] = np.exp(crysto_dist_matrix[i]) / sum_row

	for i in range(len(predicted_dist_matrix)):
		sum_row = np.sum(np.exp(predicted_dist_matrix[i]))
		predicted_dist_matrix[i] = np.exp(predicted_dist_matrix[i]) / sum_row

	closest = [-1]*10
	closest_ind = [0]*10
	for i in range(len(crysto_dist_matrix)):
		for j in range(i+2,len(crysto_dist_matrix)):
			if crysto_dist_matrix[i,j] > min(closest):
				index = closest.index(min(closest))
				closest[index] = crysto_dist_matrix[i,j]
				closest_ind[index] = [i,j]

	print np.max(crysto_dist_matrix)


	for i in range(len(closest)):
		print str(closest[i]) + ' ' + str(closest_ind[i])





def convert_msa(protein_length, msa_file):

	aa_letters = ['-']
	# get all possible AAs
	with open(msa_file, 'rb') as f:
		for row in f:
			if row[0] == '>':
				continue
			else:
				for i in range(len(row)):
					if row[i].lower() not in aa_letters and row[i] != '\n' and row[i] != '.':

						aa_letters.append(row[i].lower())
	# print aa_letters
	n_aa = len(aa_letters)


	msa = []
	# convert letters to numbers 
	with open(msa_file, 'rb') as f:
		
		for row in f:
			temp_samp = []
			if row[0] == '>':
				continue
			else:
				for i in range(protein_length):
					aa = row[i].lower()
					if aa == '.':
						aa = '-'
					temp_samp.append(aa_letters.index(aa))
			msa.append(temp_samp)

	# print len(msa)
	# print len(msa[0])

	return msa, n_aa



def compare_stuctures3(crysto_coords, predicted_coords):
	'''
	Takes in coordinated of crysto not the file
	'''

	# predicted_coords = scale_coordinates(predicted_coords, crysto_coords)

	#calculate distance matrices 
	crysto_dist_matrix = np.zeros((len(crysto_coords), len(crysto_coords)))
	for i in range(len(crysto_coords)):
		for j in range(len(crysto_coords)):
			crysto_dist_matrix[i,j] = math.sqrt(np.sum((crysto_coords[i] - crysto_coords[j])**2))
			# crysto_dist_matrix[i,j] = np.sum((crysto_coords[i] - crysto_coords[j])**2)

	predicted_dist_matrix = np.zeros((len(predicted_coords), len(predicted_coords)))
	for i in range(len(predicted_coords)):
		for j in range(len(predicted_coords)):
			predicted_dist_matrix[i,j] = math.sqrt(np.sum((predicted_coords[i] - predicted_coords[j])**2))
			# predicted_dist_matrix[i,j] = np.sum((predicted_coords[i] - predicted_coords[j])**2)



	#negative dist
	crysto_dist_matrix = -crysto_dist_matrix
	predicted_dist_matrix = -predicted_dist_matrix

	#this measure of similarity will softmax all distances then just take rmse
	for i in range(len(crysto_dist_matrix)):
		sum_row = np.sum(np.exp(crysto_dist_matrix[i]))
		crysto_dist_matrix[i] = np.exp(crysto_dist_matrix[i]) / sum_row

	for i in range(len(predicted_dist_matrix)):
		sum_row = np.sum(np.exp(predicted_dist_matrix[i]))
		predicted_dist_matrix[i] = np.exp(predicted_dist_matrix[i]) / sum_row


	mse = np.sum((crysto_dist_matrix - predicted_dist_matrix)**2) / len(crysto_dist_matrix)

	return mse




def scale_coordinates(predicted_coords, crysto_coords):

	#calculate distance matrices 
	crysto_dist_matrix = np.zeros((len(crysto_coords), len(crysto_coords)))
	for i in range(len(crysto_coords)):
		for j in range(len(crysto_coords)):
			crysto_dist_matrix[i,j] = math.sqrt(np.sum((crysto_coords[i] - crysto_coords[j])**2))

	predicted_dist_matrix = np.zeros((len(predicted_coords), len(predicted_coords)))
	for i in range(len(predicted_coords)):
		for j in range(len(predicted_coords)):
			predicted_dist_matrix[i,j] = math.sqrt(np.sum((predicted_coords[i] - predicted_coords[j])**2))

	#averge dist betw adj 
	dists = []
	for i in range(len(crysto_coords)-1):
		j = i + 1
		dists.append(crysto_dist_matrix[i,j])
	avg_dist = np.mean(dists)

	dists = []
	for i in range(len(predicted_coords)-1):
		j = i + 1
		dists.append(predicted_dist_matrix[i,j])
	avg_dist2 = np.mean(dists)

	ratio = avg_dist / avg_dist2
	print 'scale ratio = ' + str(ratio)

	scaled_coordinates  = ratio * predicted_coords

	return scaled_coordinates




def make_contact_map(pred_coords, evfold_coords, crysto_coords, n_closest, score_ev=None, score_pred=None):

	l = len(pred_coords)

	min_columns_away = 4

	#calculate distance matrices 
	crysto_dist_matrix = np.zeros((l, l))
	for i in range(len(crysto_coords)):
		for j in range(len(crysto_coords)):
			crysto_dist_matrix[i,j] = math.sqrt(np.sum((crysto_coords[i] - crysto_coords[j])**2))

	predicted_dist_matrix = np.zeros((l, l))
	for i in range(l):
		for j in range(l):
			predicted_dist_matrix[i,j] = math.sqrt(np.sum((pred_coords[i] - pred_coords[j])**2))

	evfold_dist_matrix = np.zeros((l, l))
	for i in range(len(evfold_coords)):
		for j in range(len(evfold_coords)):
			evfold_dist_matrix[i,j] = math.sqrt(np.sum((evfold_coords[i] - evfold_coords[j])**2))

	#get n_closest , excluding adjacent 
	crysto_closest_indexes = []
	for n in range(n_closest):
		current_closest_dist = -1
		current_closest_index = []
		for i in range(l):
			for j in range(i+min_columns_away,l):
				if (crysto_dist_matrix[i,j] < current_closest_dist or current_closest_dist == -1):
					#Check to see if its alrady in there
					already_there =0
					for thing in crysto_closest_indexes:
						if thing[0] == i and thing[1] == j:
							already_there =1
							break
					if already_there == 0:
						current_closest_dist = crysto_dist_matrix[i,j]
						current_closest_index = [i,j]
		crysto_closest_indexes.append(np.array(current_closest_index))
	crysto_closest_indexes = np.array(crysto_closest_indexes)


	pred_closest_indexes = []
	for n in range(n_closest):
		current_closest_dist = -1
		current_closest_index = []
		for i in range(l):
			for j in range(i+min_columns_away,l):
				if (predicted_dist_matrix[i,j] < current_closest_dist or current_closest_dist == -1):
					#Check to see if its alrady in there
					already_there =0
					for thing in pred_closest_indexes:
						if thing[0] == i and thing[1] == j:
							already_there =1
							break
					if already_there == 0:					
						current_closest_dist = predicted_dist_matrix[i,j]
						current_closest_index = [i,j]
		pred_closest_indexes.append(np.array(current_closest_index))
	pred_closest_indexes = np.array(pred_closest_indexes)


	evfold_closest_indexes = []
	for n in range(n_closest):
		current_closest_dist = -1
		current_closest_index = []
		for i in range(l):
			for j in range(i+min_columns_away,l):
				if (evfold_dist_matrix[i,j] < current_closest_dist or current_closest_dist == -1):
					#Check to see if its alrady in there
					already_there =0
					for thing in evfold_closest_indexes:
						if thing[0] == i and thing[1] == j:
							already_there =1
							break
					if already_there == 0:
						current_closest_dist = evfold_dist_matrix[i,j]
						current_closest_index = [i,j]
		evfold_closest_indexes.append(np.array(current_closest_index))
	evfold_closest_indexes = np.array(evfold_closest_indexes)



	#use plotly to make scater plot. evfold will be below diagonal, my preds will be above


	# plotly.offline.plot({
	# 	"data": [plotly.graph_objs.Scatter(x=crysto_closest_indexes.T[0], y=crysto_closest_indexes.T[1])],
	# 	"layout": plotly.graph_objs.Layout( title="no title")})

	# print crysto_closest_indexes

	trace0 = go.Scatter(
		x = [0,l],
		y = [0,l],
		mode = 'lines',
		name = 'Diagonal'
	)

	trace1 = go.Scatter(
		x = crysto_closest_indexes.T[0],
		y = crysto_closest_indexes.T[1],
		mode = 'markers',
		name = 'Crystallography',
		marker = dict(size = 10, color = 'rgba(255, 182, 193, .9)')
	)

	trace2 = go.Scatter(
		x = crysto_closest_indexes.T[1],
		y = crysto_closest_indexes.T[0],
		mode = 'markers',
		name = 'Crystallography',
		marker = dict(size = 10, color = 'rgba(255, 182, 193, .9)')
	)

	trace3 = go.Scatter(
		x = pred_closest_indexes.T[0],
		y = pred_closest_indexes.T[1],
		mode = 'markers',
		# name = 'Prediction '+str(round(score_pred*10000)),
		name = 'Prediction '+str(score_pred),
		marker = dict(symbol='x', color='green')
	)

	trace4 = go.Scatter(
		x = evfold_closest_indexes.T[1],
		y = evfold_closest_indexes.T[0],
		mode = 'markers',
		name = 'EVfold '+str(score_ev),
		# name = 'EVfold '+str(round(score_ev*10000)),
		marker = dict(symbol='x')
	)

	data = [trace0,trace1,trace2,trace3,trace4]

	layout = go.Layout(
		xaxis=dict(
			range=[0,l],
			type='linear',
			dtick=20,
			# autorange=False,
			# showgrid=True,
			# zeroline=False,
			# showline=False,
			# autotick=True,
			# ticks='',
			# showticklabels=False
		),
		yaxis=dict(
			range=[0,l],
			type='linear',
			dtick=20,
			# autorange=False,
			# showgrid=True,
			# zeroline=False,
			# showline=False,
			# autotick=True,
			# ticks='',
			# showticklabels=False
		),

		width=950,
		height=800
	)

	
	fig = go.Figure(data=data, layout=layout)
	plotly.offline.plot(fig, filename=home + '/Documents/Protein_data/RASH/contact_map.html')









def convert_to_one_hot(aa, n_aa):
	vec = np.zeros((n_aa,1))
	vec[aa] = 1
	return vec

def convert_samp_to_one_hot(samp, n_aa):

	one_hot_samp = []
	for i in range(len(samp)):
		vec = np.zeros((n_aa,1))
		vec[samp[i]] = 1
		one_hot_samp.append(vec)
	return np.array(one_hot_samp)

def convert_samp_to_L_by_L(samp, n_aa):

	L_by_L = []
	for i in range(len(samp)):
		this_samp = []
		for j in range(len(samp)):
			if j == i:
				this_samp.append(convert_to_one_hot(0, n_aa))
			else:
				this_samp.append(convert_to_one_hot(samp[j], n_aa))
		L_by_L.append(this_samp)

	return np.array(L_by_L)





