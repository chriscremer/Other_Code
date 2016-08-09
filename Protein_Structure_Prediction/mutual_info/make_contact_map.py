


import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,'../tools')
import protein_model_tools as tools

from os.path import expanduser
home = expanduser("~")

import pickle

import math

import plotly
import plotly.graph_objs as go



L = 166
l =L

min_columns_away = 4
n_closest = 100


def get_n_closest(coords, n_closest, min_columns_away):

	#get distance matrix
	dist_matrix = np.zeros((l, l))
	for i in range(len(coords)):
		for j in range(len(coords)):
			dist_matrix[i,j] = math.sqrt(np.sum((coords[i] - coords[j])**2))

	#get n_closest , excluding adjacent 
	closest_indexes = []
	for n in range(n_closest):
		current_closest_dist = -1
		current_closest_index = []
		for i in range(l):
			for j in range(i+min_columns_away,l):
				if (dist_matrix[i,j] < current_closest_dist or current_closest_dist == -1):
					#Check to see if its alrady in there
					already_there =0
					for thing in closest_indexes:
						if thing[0] == i and thing[1] == j:
							already_there =1
							break
					if already_there == 0:
						current_closest_dist = dist_matrix[i,j]
						current_closest_index = [i,j]
		closest_indexes.append(np.array(current_closest_index))
	closest_indexes = np.array(closest_indexes)

	return closest_indexes


#Get crysto coords
crysto_pdb_file = home + '/Documents/Protein_data/RASH/5p21.pdb'
crysto_coords = tools.get_coords_from_pbd(crysto_pdb_file)
print crysto_coords.shape

#Get EVfold prediction
evfold_structure_file = home + '/Documents/Protein_data/RASH/RASH_HUMAN2_833a6535-26d0-4c47-8463-7970dae27a32_evfold_result/structure_outputs/rank_sortable/RASH_HUMAN2_rank01_80_3_hMIN.pdb'
evofold_prediction = tools.get_coords_from_pbd_for_ev(evfold_structure_file)
evofold_prediction = evofold_prediction[:L]
print evofold_prediction.shape


crysto_closest_indexes = get_n_closest(crysto_coords, n_closest, min_columns_away)
evfold_closest_indexes = get_n_closest(evofold_prediction, n_closest, min_columns_away)


ev_score = 0.
for i in range(len(evfold_closest_indexes)):
	for j in range(len(crysto_closest_indexes)):
		if np.all(evfold_closest_indexes[i] == crysto_closest_indexes[j]):
			ev_score += 1.
ev_score = ev_score / n_closest
print ev_score


#Load MI coordinates
files = os.listdir(home + '/Documents/Protein_data/RASH/coordinates')

best_score =-1
best_indexes = []

for file_ in files:

	print file_


	with open(home + '/Documents/Protein_data/RASH/coordinates/' + file_ , 'rb' ) as f:
	# with open('MI.pkl', 'rb' ) as f:
		MI_stuff = pickle.load(f)
	# MI = pickle.load(open(home + "/Documents/Protein_data/RASH/MI.pkl", "rb"))

	MI_coords = MI_stuff[0]
	hypers = MI_stuff[1]

	# print MI_coords.shape
	# print np.max(MI_coords)
	# print np.min(MI_coords)
	# print MI_coords[:7,:7]

	pred_closest_indexes = get_n_closest(MI_coords, n_closest, min_columns_away)

	#Give it a score: % of top 100 that match
	pred_score = 0.

	#for every index pair 
	for i in range(len(pred_closest_indexes)):
		for j in range(len(crysto_closest_indexes)):
			if np.all(pred_closest_indexes[i] == crysto_closest_indexes[j]):
				pred_score += 1.

	pred_score = pred_score / n_closest

	print pred_score

	if pred_score > best_score:
		best_score = pred_score
		best_indexes = pred_closest_indexes
		print 'best ' + str(hypers)



pred_closest_indexes = best_indexes


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
	name = 'Prediction ',#+str(score_pred),
	marker = dict(symbol='x', color='green')
)

trace4 = go.Scatter(
	x = evfold_closest_indexes.T[1],
	y = evfold_closest_indexes.T[0],
	mode = 'markers',
	name = 'EVfold ',#+str(score_ev),
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
plotly.offline.plot(fig, filename=home + '/Documents/Protein_data/RASH/contact_map_MI2.html')

