




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
with open(home+'/Documents/nhl_data/NHL-2016-17-ppp.csv', 'rU') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)
	next(reader, None)
	next(reader, None)
	for row in reader:
		# print row
		ppp_name.append(row[0]+' '+row[1])
		ppp.append(int(row[8]))
		# print ppp_name[-1]
		# print ppp[-1]

# fads


# Make a list of all that info for each player
names = []
# cats = ['GP', 'G', 'A', '+/-', 'blk', 'hit', 'sog', 'ppp']
# cats = ['GP', 'G', 'A', 'hit', 'tka', '+/-',  'sog', 'blk', 'ppp']
cats = ['GP', 'G', 'A', 'hit', '+/-',  'sog', 'blk', 'ppp']


positions = []

data = []

with open(home+'/Documents/nhl_data/NHL-2016-17.csv', 'rU') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)
	next(reader, None)
	next(reader, None)

	for row in reader:

		if row[16] == '':
			break
		#if GP > 25 and toi/gp > 13
		elif int(row[16]) > 40 and float(row[28]) > 13.:

			yob = int(row[0].split('-')[0])
			age = 2017 - yob
			# if age > 34
			if age < 32:
				player = []
				names.append(row[12]+' '+row[13])
				player.append(float(row[16])) #GP
				player.append(float(row[17])) #G
				player.append(float(row[18])) #A
				player.append(float(row[54])) #hit
				# player.append(float(row[60])) #takeaway
				player.append(float(row[22])) #plus minus


				if row[43] == 'NA':
					player.append(0.0)
				else:
					player.append(float(row[43])) #sog
				if row[30] == '':
					player.append(0.0)
				else:
					player.append(float(row[61])) #blk

				ppp_index = ppp_name.index(names[-1])
				player.append(ppp[ppp_index]) #ppp

				# player.append(row[14]) #Pos
				positions.append(row[14]) #Pos

				# print names[-1]
				# # print cats
				# print player
				# faf


				# aaa = [(i,row[i]) for i in range(len(row))]
				# for i in aaa:
				# 	print i
				# 	print
				# fsaf

				# player.append(float(row[27])) #+/-


				data.append(player)


data = np.array(data)
print 'Player Data'
print cats
print data.shape
print len(positions)





#Get goalie data

names_goalies = []
cats = ['GP', 'W', 'SV%']#, '-GAA']#, 'SO']
data_goalies = []

with open(home+'/Documents/nhl_data/NHL_Goalies_2016-17.csv', 'rU') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)

	for row in reader:
		#if GP > 25
		if int(row[14]) > 30:

			yob = int(row[3].split('-')[0])
			age = 2017 - yob
			# if age > 34
			if age < 32:
				player = []
				names_goalies.append(row[0]+' '+row[1])
				player.append(float(row[14])) #GP
				player.append(float(row[17])) #W
				player.append(float(row[24])) #SV%
				# player.append(-float(row[25])) #GAA
				# player.append(float(row[28])) #SO

				# print names_goalies[-1]
				# print player
				data_goalies.append(player)


data_goalies = np.array(data_goalies)
print 'Goalie stats'
print cats
print data_goalies.shape
print 



#Divide by games played
for i in range(len(data)):
	# data[i][1] = data[i][1] / float(data[i][0]) #G
	# data[i][2] = data[i][2] / float(data[i][0]) #A
	# data[i][3] = data[i][3] / float(data[i][0]) #hit
	# data[i][4] = data[i][4] / float(data[i][0]) #tka
	# data[i][5] = data[i][5] / float(data[i][0]) #plue minus
	# data[i][6] = data[i][6] / float(data[i][0]) #shots
	# data[i][7] = data[i][7] / float(data[i][0]) #blks
	# data[i][8] = data[i][8] / float(data[i][0]) #ppp
	data[i] = data[i] / float(data[i][0])


for i in range(len(data_goalies)):
	data_goalies[i][1] = data_goalies[i][1] / float(data_goalies[i][0]) #W
	# data_goalies[i][4] = data_goalies[i][4] / float(data_goalies[i][0]) #SO



#Put players in position bins
C = []
RW = []
LW = []
D = []
C_names = []
RW_names = []
LW_names = []
D_names = []

for i in range(len(data)):
	pos = positions[i]
	if 'C' in pos:
		C.append(data[i][1:])
		C_names.append(names[i])
	if 'RW' in pos:
		RW.append(data[i][1:])
		RW_names.append(names[i])
	if 'LW' in pos:
		LW.append(data[i][1:])
		LW_names.append(names[i])
	if 'D' in pos:
		D.append(data[i][1:])
		D_names.append(names[i])

C = np.array(C)
LW = np.array(LW)
RW = np.array(RW)
D = np.array(D)
print C.shape
print RW.shape
print LW.shape
print D.shape
print 





def make_team():

	player_names = []

	L1 = np.random.randint(len(LW_names))
	L2 = np.random.randint(len(LW_names))
	C1 = np.random.randint(len(C_names))
	C2 = np.random.randint(len(C_names))
	R1 = np.random.randint(len(RW_names))
	R2 = np.random.randint(len(RW_names))
	D1 = np.random.randint(len(D_names))
	D2 = np.random.randint(len(D_names))
	D3 = np.random.randint(len(D_names))
	D4 = np.random.randint(len(D_names))
	G1 = np.random.randint(len(names_goalies))
	G2 = np.random.randint(len(names_goalies))

	player_names.append(LW_names[L1])
	player_names.append(LW_names[L2])
	player_names.append(C_names[C1])
	player_names.append(C_names[C2])
	player_names.append(RW_names[R1])
	player_names.append(RW_names[R2])
	player_names.append(D_names[D1])
	player_names.append(D_names[D2])
	player_names.append(D_names[D3])
	player_names.append(D_names[D4])
	player_names.append(names_goalies[G1])
	player_names.append(names_goalies[G2])

	team_points = np.copy(LW[L1])
	# print team_points
	team_points += LW[L2]
	team_points += RW[R1]
	team_points += RW[R2]
	team_points += C[C1]
	team_points += C[C2]
	team_points += D[D1]
	team_points += D[D2]
	team_points += D[D3]
	team_points += D[D4]

	goalie_points = np.copy(data_goalies[G1][1:])
	goalie_points += data_goalies[G2][1:]

	# print goalie_points
	team_points = np.concatenate([team_points, goalie_points])

	return player_names, team_points



def update_player_scores(all_names, all_scores, all_count, team_names, team_score):

	for name in team_names:
		player_index = all_names.index(name)
		all_count[player_index] += 1
		# mean = prev_mean + (prev_mean + new_score) / count
		all_scores[player_index] = all_scores[player_index] + (team_score-all_scores[player_index])/ all_count[player_index]





all_names = names + names_goalies
all_scores = [0.]*len(all_names)
all_count = [0]*len(all_names)



for iter_ in range(30000):
	if iter_ % 1000 == 1:
		print iter_
		# print T1_points
		# print C[0]
		# print data_goalies[0]

	
	#Make two teams
	T1_names, T1_points = make_team()
	T2_names, T2_points = make_team()

	#Get team scores
	points = np.stack([T1_points, T2_points], axis=1)
	# print points
	T2_score = np.sum(np.argmax(points, axis=1))
	T1_score = len(points) - T2_score
	# print T2_score
	# print T1_score
	# fad

	#Update player scores
	update_player_scores(all_names, all_scores, all_count, T1_names, T1_score)
	update_player_scores(all_names, all_scores, all_count, T2_names, T2_score)





order = np.argsort(all_scores)
# scores_sorted = all_scores[order]
# names_sorted = all_names[order]
# count_sorted = all_count[order]
scores_sorted = [all_scores[i] for i in order]
names_sorted = [all_names[i] for i in order]
count_sorted = [all_count[i] for i in order]


for i in range(len(order)):
	print scores_sorted[i], count_sorted[i], names_sorted[i]

print  'Out of', str(len(T2_points)), 'cats'






#Problem: since theres no max to games played, it'll be super advantageous to have a team that can play a lot
# so playing multiple positons is very useful
# ill come back to this later

# mayeb look if I can get schedule data, see which teams play on different days the most

# so need to select bench players
# then need to go through calendar and see if team plays and if position available. 
#Problem: if the positions dont match up with the yahoo positions...
	# its not perfect but go with it. atleast you get the schedule included





fasd
#THis is old version

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






