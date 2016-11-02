


import numpy as np
from scipy.stats import norm
import csv
from os.path import expanduser
home = expanduser("~")

import math

# with open(home+'/Downloads/NHL-2015-16_csv.csv', 'rb') as csvfile:
# 	spamreader = csv.reader(csvfile)
# 	for row in spamreader:
# 		for i in range(len(row)):
# 			print i, row[i]
# 		fasdf


# with open(home+'/Downloads/NHL-2015-16_csv_pp.csv', 'rb') as csvfile:
# 	spamreader = csv.reader(csvfile)
# 	for row in spamreader:
# 		for i in range(len(row)):
# 			print i, row[i]
# 		fasdf


# columns
# first name : 14
# last name : 15
# GP : 22
# G : 23
# A : 24
# +/- : 27
# blocked : 30
# hits : 53
# SOG: 73


#other file
# ppp : 7

# powerplay = np.genfromtxt(home+'/Downloads/NHL-2015-16_csv_pp.csv')
# print powerplay.shape
# fsdfa

ppp_name = []
ppp = []
with open(home+'/Downloads/NHL-2015-16_csv_pp.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)
	for row in reader:
		ppp_name.append(row[0]+' '+row[1])
		ppp.append(int(row[7]))


# Make a list of all that info for each player
names = []
cats = ['GP', 'G', 'A', '+/-', 'blk', 'hit', 'sog', 'ppp']
data = []

with open(home+'/Downloads/NHL-2015-16_csv.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)
	for row in reader:
		#if GP > 20
		if int(row[22]) > 20:
			# if age > 34
			if int(row[13]) < 32:
				player = []
				names.append(row[14]+' '+row[15])
				player.append(float(row[22]))
				player.append(float(row[23])) #G
				player.append(float(row[24])) #A
				player.append(float(row[27])) #+/-
				if row[30] == '':
					player.append(0.0)
				else:
					player.append(float(row[54])) #blk
				player.append(float(row[53])) #hit
				player.append(float(row[28])) #sog

				ppp_index = ppp_name.index(row[14]+' '+row[15])
				player.append(ppp[ppp_index]) #ppp

				data.append(player)

data = np.array(data)
print data.shape

#Divide by games played
for i in range(len(data)):
	data[i] = data[i] / float(data[i][0])

means = np.mean(data, axis=0)
# print means
variances = np.var(data, axis=0)
# print variances


player_scores = []
player_score_matrix = []
for player_i in range(len(data)):
	player_score = []
	for cat_i in range(len(data[player_i])):
		if cat_i != 0:
			#cdf
			# player_score.append(norm.cdf(data[player_i][cat_i], loc=means[cat_i], scale=variances[cat_i]))
			#z-score
			player_score.append((data[player_i][cat_i] - means[cat_i]) / math.sqrt(variances[cat_i]))




	player_score = np.array(player_score)
	#reweight
	player_score = player_score * np.array([1.1,1,.4,1,1.3,1,1])


	player_score_matrix.append(player_score)
	player_scores.append(np.mean(player_score))



player_score_matrix = np.array(player_score_matrix)
print np.var(player_score_matrix, axis=0)

top = np.argsort(player_scores)[::-1]

# print cats
# for i in range(50):
# 	# print top[i]
# 	print i, names[top[i]], player_scores[top[i]]
# 	print player_score_matrix[top[i]]
# 	print


with open(home+'/Downloads/player_scores.txt', 'wb') as f:
	for i in range(len(names)):
		# print top[i]
		f.write(str(cats[1:]))
		f.write('\n')
		f.write(str(i) +' '+ str(names[top[i]]) +' '+ str(player_scores[top[i]]) +' '+ str(list(player_score_matrix[top[i]])) +'\n\n')
		



print 'saved'






