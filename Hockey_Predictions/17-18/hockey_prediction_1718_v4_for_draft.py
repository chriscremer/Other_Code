


# new plan: just similate drafts to make teams, itll be slow but its necessary for being able to accurately construst at team
# doing it random makes bad teams, doing draft without other teams makes it too good
# so I need to sim draft every time
# also be able to pass selected players






import numpy as np
from scipy.stats import norm
import csv
from os.path import expanduser
home = expanduser("~")

import math
from random import shuffle
import time

np.set_printoptions(suppress=True)


# player data
teams = ['CHI', 'DET', 'NSH','BOS',
		'STL','CAR','WSH', 'L.A',
		'MTL','PIT', 'WPG', 'BUF',
		'CBJ','N.J','CGY', 'VAN',
		'OTT', 'NYI','COL', 'FLA', 
		'L.V', 'MIN', 'S.J', 'EDM',
		'NYR', 'PHI', 'DAL', 'ANA', 
		'TOR', 'T.B', 'ARI']

#goalie data
teams_g = ['CHI', 'DET', 'NSH','BOS',
		'STL','CAR','WSH', 'LAK',
		'MTL','PIT', 'WPG', 'BUF',
		'CBJ','NJD','CGY', 'VAN',
		'OTT', 'NYI','COL', 'FLA', 
		'L.V', 'MIN', 'SJS', 'EDM',
		'NYR', 'PHI', 'DAL', 'ANA', 
		'TOR', 'TBL', 'ARI']

# schedule data
full_teams = ['Chicago Blackhawks', 'Detroit Red Wings', 'Nashville Predators', 'Boston Bruins',
				'St. Louis Blues', 'Carolina Hurricanes', 'Washington Capitals', 'Los Angeles Kings',
				'Montreal Canadiens', 'Pittsburgh Penguins', 'Winnipeg Jets', 'Buffalo Sabres',
				'Columbus Blue Jackets', 'New Jersey Devils', 'Calgary Flames', 'Vancouver Canucks',
				'Ottawa Senators', 'New York Islanders', 'Colorado Avalanche', 'Florida Panthers', 
				'Vegas Golden Knights', 'Minnesota Wild', 'San Jose Sharks', 'Edmonton Oilers', 
				'New York Rangers', 'Philadelphia Flyers', 'Dallas Stars', 'Anaheim Ducks',
				'Toronto Maple Leafs', 'Tampa Bay Lightning', 'Arizona Coyotes']


# print len(teams)
# print len(set(teams))

# print len(full_teams)
# print len(set(full_teams))

# #Get yahoo positions
# doing it by hand becasue scrape was taking too long
pos_names = ['Brent Burns', 'Alex Ovechkin', 'Dustin Byfuglien', 'Evgeni Malkin', 'Patrick Kane',
				'David Pastrnak', 'T.J. Oshie', 'Brad Marchand', 'Jeff Carter', 'Kyle Palmieri',
				'Rickard Rakell', 'Jeff Skinner', 'Ryan O\'Reilly', 'Chris Kreider', 'Logan Couture',
				'Taylor Hall', 'Phil Kessel', 'Anders Lee', 'Filip Forsberg', 'Conor Sheary',
				'Claude Giroux', 'Jakob Silfverberg', 'Ryan Johansen', 'Vincent Trocheck', 'Brandon Saad',
				'Jaden Schwartz', 'Wayne Simmonds', 'Mats Zuccarello', 'Mika Zibanejad', 'Jordan Staal',
				'Artem Anisimov', 'Bryan Little', 'Marcus Johansson', 'Brandon Dubinsky', 'Evgeny Kuznetsov',
				'Michael Frolik', 'Adam Lowry', 'Tyler Toffoli', 'Josh Bailey', 'Nathan MacKinnon',
				'Gustav Nyquist', 'Paul Byron', 'Andrew Shaw', 'Tomas Tatar', 'Max Domi', 'Jordan Eberle',
				'Andre Burakovsky', 'Tanner Kero', 'Christian Dvorak', 'Teuvo Teravainen', 'Mark Letestu',
				'Tomas Hertl', 'Reilly Smith', 'Leon Draisaitl']
pos_yahoo = ['D', 'LW', 'D', 'C', 'RW', 'RW', 'RW', 'LW', 'C', 'RW', 'C/LW', 'LW', 'C', 'LW', 'C',
				'LW', 'RW', 'LW', 'LW', 'LW/RW', 'C', 'RW', 'C', 'C', 'LW', 'LW', 'RW', 'RW', 'C', 'C',
				'C', 'C', 'LW', 'C', 'C', 'RW', 'C', 'C/RW', 'LW/RW', 'C', 'LW/RW', 'LW/RW', 'C/RW', 'LW/RW',
				'LW', 'RW', 'LW/C', 'C', 'C', 'C/LW', 'C', 'C/LW', 'RW', 'C/RW']

print len(pos_names)
print len(pos_yahoo)
print 



#Vegas roster
vgk = ['James Neal', 'Vadim Shipachyov', 'Reilly Smith', 'Shea Theodore', 'David Perron', 'Jonathan Marchessault',
		'Erik Haula', 'Deryk Engelland', 'Oscar Lindberg', 'William Karlsson', 'Colin Miller', 'Cody Eakin', 
		'Nate Schmidt', 'Marc-Andre Fleury', 'Calvin Pickard']




days_and_which_teams_plays = []

with open(home+'/Documents/nhl_data/2017_2018_NHL_Schedule.csv', 'rU') as csvfile:
	reader = csv.reader(csvfile)
	# next(reader, None)
	prev_day = 0
	
	for row in reader:
		day = row[0]
		if day != prev_day:
			if prev_day != 0:
				days_and_which_teams_plays.append(current_day_teams)
			prev_day = day
			current_day_teams = []
		current_day_teams.append(row[2])
		current_day_teams.append(row[3])
	days_and_which_teams_plays.append(current_day_teams)

print 'Number NHL game days', str(len(days_and_which_teams_plays))
# print days_and_which_teams_plays
# fsda


# #for each team print their avg number of teams playing on teh same day
# for i in range(len(full_teams)):
# 	sum_ = 0.
# 	count = 0
# 	for d in range(len(days_and_which_teams_plays)):

# 		if full_teams[i] in days_and_which_teams_plays[d]:
# 			sum_ += len(days_and_which_teams_plays[d])
# 			count+=1
# 	print full_teams[i], sum_ / count, count


player_rankings = []
player_rankings_numbervalue = []


with open(home+'/Documents/nhl_data/player_rankings_for_sim_draft.txt', 'rU') as csvfile:
	reader = csv.reader(csvfile)
	cur_team = []
	for row in reader:

		aaa =  row[0].split(' ')
		# print aaa
		if aaa[4] in ['Riemsdyk', 'Haan', 'Zotto']:
			player_rankings.append(aaa[2]+' '+aaa[3]+' '+aaa[4])
		else:
			player_rankings.append(aaa[2]+' '+aaa[3])

		player_rankings_numbervalue.append(float(aaa[1]))

# print player_rankings[::-1]
# print





# team_points_distribution = []

# with open(home+'/Documents/nhl_data/team_points_distribution.txt', 'rU') as csvfile:
# 	reader = csv.reader(csvfile)#, delimiter=']')
# 	# next(reader, None)
# 	cur_team = []
# 	for row in reader:
# 		aaa =  row[0].split(' ')
# 		for aa in aaa:
# 			if aa not in ['[', ']', '', ' ']:
# 				# print aa
# 				# if ']' not in aa:
# 				cur_team.append(float(aa))

# 				if len(cur_team)== 10:
# 					team_points_distribution.append(cur_team)
# 					cur_team = []

# # for i in range(len(team_points_distribution)):
# # 	print team_points_distribution[i]
# # 	print len(team_points_distribution[i])
# # print len(team_points_distribution), 'teams'
# team_points_distribution = np.array(team_points_distribution)
# print team_points_distribution.shape
# print
# # fasdf






# get powerplay data
ppp_name = []
ppp = []
with open(home+'/Documents/nhl_data/NHL-2016-17-ppp.csv', 'rU') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)
	next(reader, None)
	next(reader, None)
	for row in reader:
		# ppp_name.append(row[0]+' '+row[1])
		ppp_name.append(row[1]+' '+row[0])

		ppp.append(int(row[8]))





# Make a list of all that info for each player
# cats = ['GP', 'G', 'A', '+/-', 'blk', 'hit', 'sog', 'ppp']
# cats = ['GP', 'G', 'A', 'hit', 'tka', '+/-',  'sog', 'blk', 'ppp']
cats = ['GP', 'G', 'A', 'hit', '+/-','sog', 'blk', 'ppp']
# cats = ['GP', 'G', 'A', 'hit', 'sog', 'blk', 'ppp']

data = []
names = []
positions = []
teams_of_players = []

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
			if age < 33:
				player = []
				# names.append(row[12]+' '+row[13])
				names.append(row[13]+' '+row[12])
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

				positions.append(row[14]) #Pos
				teams_of_players.append(row[15]) #Team

				# aaa = [(i,row[i]) for i in range(len(row))]
				# for i in aaa:
				# 	print i
				# 	print
				# fsaf

				data.append(player)


data = np.array(data)
print 'Player Data'
print cats
print data.shape
print len(positions)

#Update to yahoo positions
for aa in range(len(pos_names)):
	player_index = names.index(pos_names[aa])
	# print names[player_index], positions[player_index], '->', pos_yahoo[aa]
	positions[player_index] = pos_yahoo[aa]





#Get goalie data

names_goalies = []
cats_goalies = ['GP', 'W', 'SV%', 'SV']#, '-GAA']#, 'SO']
data_goalies = []

team_goalies = []

with open(home+'/Documents/nhl_data/NHL_Goalies_2016-17.csv', 'rU') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)

	for row in reader:
		#if GP > 25
		if int(row[14]) > 30:

			yob = int(row[3].split('-')[0])
			age = 2017 - yob
			# if age > 34
			if age < 34:
				player = []
				# names_goalies.append(row[0]+' '+row[1])
				names_goalies.append(row[1]+' '+row[0])
				player.append(float(row[14])) #GP
				player.append(float(row[17])) #W
				player.append(float(row[24])) #SV%
				player.append(float(row[22])) #Saves

				team_goalies.append(row[2]) #Team

				# player.append(-float(row[25])) #GAA
				# player.append(float(row[28])) #SO

				# print names_goalies[-1]
				# print player
				data_goalies.append(player)


data_goalies = np.array(data_goalies)
print 'Goalie stats'
print cats_goalies
print data_goalies.shape
positions_goalies = ['G']*len(data_goalies)
print 



#Divide by games played
for i in range(len(data)):
	data[i] = data[i] / float(data[i][0])

# for i in range(len(data_goalies)):
	# data_goalies[i][1] = data_goalies[i][1] / float(data_goalies[i][0]) #W
	# data_goalies[i][3] = data_goalies[i][3] / float(data_goalies[i][0]) #saves



# Concat zeros to data matrices to make them all have same number of cats
print 'Reshaped data'
#get rid of games played
data_goalies = data_goalies[:, 1:]
data_goalies = np.concatenate([np.zeros((len(data_goalies), len(cats)-1)), data_goalies], axis=1)
print data_goalies.shape

data = data[:, 1:]
data = np.concatenate([data, np.zeros((len(data), len(cats_goalies)-1))], axis=1)
print data.shape

all_data = np.concatenate([data, data_goalies], axis=0)
print all_data.shape
# fdsa







#Convert team names to schedule format
for i in range(len(teams_of_players)):
	t = teams_of_players[i]
	if '/' in t:
		t = t.split('/')[-1]
	t_index = teams.index(t)
	teams_of_players[i] = full_teams[t_index]

# print team_goalies
for i in range(len(team_goalies)):
	t = team_goalies[i]
	if ',' in t:
		t = t.split(', ')[-1]
	t_index = teams_g.index(t)
	team_goalies[i] = full_teams[t_index]
# print team_goalies
# fadsf






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
		C.append(data[i])
		C_names.append(names[i])
	if 'RW' in pos:
		RW.append(data[i])
		RW_names.append(names[i])
	if 'LW' in pos:
		LW.append(data[i])
		LW_names.append(names[i])
	if 'D' in pos:
		D.append(data[i])
		D_names.append(names[i])

C = np.array(C)
LW = np.array(LW)
RW = np.array(RW)
D = np.array(D)
print 'Position matrices'
print 'C', C.shape
print 'RW', RW.shape
print 'LW', LW.shape
print 'D', D.shape
print 


all_names = names + names_goalies
all_positions = positions + positions_goalies
all_teams = teams_of_players + team_goalies
all_scores = [0.]*len(all_names)
all_count = [0]*len(all_names)

print len(all_names)
print len(all_positions)
print len(all_teams)
print len(all_scores)
print len(all_count)
print
# for a in all_positions:
# 	print a.split('/')
# fdsaf


#Update Vegas roster
for aa in range(len(vgk)):

	if vgk[aa] in all_names:
		p_index = all_names.index(vgk[aa])
		# print all_names[p_index], all_teams[p_index], 
		all_teams[p_index] = 'Vegas Golden Knights'




# Look at distribution of ranking vs position
player_rankings__ = list(player_rankings[::-1])
player_rankings_numbervalue = list(player_rankings_numbervalue[::-1])

poses = ['C', 'LW', 'RW', 'D', 'G']
for p in poses:
	score_ = 0.
	spores = []
	count =0.
	for pl_i in range(len(player_rankings__)):
		#get position
		all_index = all_names.index(player_rankings__[pl_i])
		pos = all_positions[all_index]
		if p in pos:
			score_+= pl_i
			count+=1 
			# spores.append(pl_i)
			spores.append(player_rankings_numbervalue[pl_i])
	# print p, score_/count
	print p, 'mean', np.mean(spores)
	print p, 'median', np.median(spores)
	print p, 'std', np.std(spores)
	print 
#this shows that RW is different than the other positoins
# it might be saying to pick RW early or late. 
# its saying theres a lot of decent RW and few shitty ones. 
# so maybe pick late becaue there will be some good ones left.
# but there are also less RW so maybe it balances out. 
# it also has more variance, so maybe its best to get it first
# goalies have similar pattern, so maybe pick RW and goalies first
# fsdf





#Compute team score based on rankings
def compute_team_scores(team_points):

	team_points = np.array(team_points)
	team_scores = np.zeros(len(team_points))

	for c in range(len(team_points.T)):

		# print team_points.T[c]
		standing_inthiscat = np.argsort(team_points.T[c])
		for i in range(len(standing_inthiscat)):
			team_scores[standing_inthiscat[i]] += i

	return team_scores



def calc_season_total(player_names):

	player_names = list(player_names)
	# shuffle(player_names)


	players_all_index = []
	for p in player_names:
		players_all_index.append(all_names.index(p))

	season_points = np.zeros(len(C[0]))
	for d in range(len(days_and_which_teams_plays)):
		day_points = np.zeros(len(C[0]))
		L_open = 2
		R_open = 2
		C_open = 2
		D_open = 4
		G_open = 2

		# print d
		# print season_points

		for p in range(len(player_names)):

			# print L_open, R_open, C_open, D_open, G_open
			# print player_names[p]
			
			#see if team plays
			p_team = all_teams[players_all_index[p]]
			if p_team in days_and_which_teams_plays[d]:

				#see if position open
				p_pos = all_positions[players_all_index[p]].split('/') #this is a list of its positions
				#shuffle list so that order random, so that im not always doing L or R before C
				shuffle(p_pos)
				for pos_ in p_pos:
					if pos_ == 'C' and C_open > 0:
						C_open -= 1
						day_points += all_data[players_all_index[p]]
						break
					elif pos_ == 'LW' and L_open > 0:
						L_open -= 1
						day_points += all_data[players_all_index[p]]
						break
					elif pos_ == 'RW' and R_open > 0:
						R_open -= 1
						day_points += all_data[players_all_index[p]]
						break
					elif pos_ == 'D' and D_open > 0:
						D_open -= 1
						day_points += all_data[players_all_index[p]]
						break
					elif pos_ == 'G' and G_open > 0:
						G_open -= 1
						day_points += all_data[players_all_index[p]]
						break
					# else:
					# 	print d, 'this guy didnt make it ', all_names[players_all_index[p]], all_positions[players_all_index[p]], pos_


		season_points += day_points

		# print
	# fdsfas

	return season_points






def sim_draft(selected_players, new_open_positions):

	# team 0 is my team

	# n_teams = 12
	positions_static = ['C', 'LW', 'RW', 'D', 'G']
	positions_dynamic = ['C', 'LW', 'RW', 'D', 'G']
	positions_open = [3,3,3,5,2]  #C, LW, RW, D, G
	teams_statuses = [list(positions_open) for i in range(n_teams)]

	team_comps = [[] for i in range(n_teams)]

	players_to_select = list(player_rankings[::-1])

	team_order= range(n_teams)

	for round_ in range(16):

		shuffle(team_order)
		for t in team_order:

			if t ==0 and len(selected_players) > round_:
				team_comps[0].append(selected_players[round_])

			shuffle(positions_dynamic)
			pl_found =0

			#which position to look for
			for p in range(len(positions_dynamic)):

				pos_index = positions_static.index(positions_dynamic[p])
				#if position still available for this team
				if teams_statuses[t][pos_index] > 0:
					
					#find the best player with this position
					for pl in range(len(players_to_select)):

						pl_name = players_to_select[pl]
						pl_all_index = all_names.index(pl_name)
						pl_pos = all_positions[pl_all_index]

						#check their position
						if positions_dynamic[p] in pl_pos:

							#add to team
							team_comps[t].append(pl_name)
							#remove pl from list and remove pos from team
							players_to_select.pop(pl)
							teams_statuses[t][pos_index] -= 1
							pl_found =1
							break
				if pl_found:
					break

	return team_comps


def update_player_scores(all_names, all_scores, all_count, team_names, team_score):

	for name in team_names:
		player_index = all_names.index(name)
		all_count[player_index] += 1
		# mean = prev_mean + (prev_mean + new_score) / count
		all_scores[player_index] = all_scores[player_index] + (team_score-all_scores[player_index])/ all_count[player_index]















all_cats = ['G', 'A', 'hit', '+/-','sog', 'blk', 'ppp','W', 'SV%', 'SV']

taken_players=[] #used for printing clarity

my_team = [] #used for team comp
my_cur_positions_open = [3,3,3,5,2]

n_teams = 12

iter_total = 500
start = time.time()
for iter_ in range(iter_total):
	if iter_ % 500 == 1:
		print iter_
	

	#Make teams
	team_comps = sim_draft(my_team, my_cur_positions_open)

	#Compute team season points
	team_points = np.array([calc_season_total(team_comps[t]) for t in range(n_teams)])

	#Compute rank points
	team_overall_score = compute_team_scores(team_points)
	# T1_names = team_comps[0]
	# T1_score = team_overall_score[0]


	# team_score_order = np.argsort(team_overall_score)[::-1]


	# print all_cats
	# for t in team_score_order:
	# 	print team_overall_score[t], [round(x,2) for x in team_points[t]]
	# print

	# rankign_order_2 = []
	# for c in range(len(team_points.T)):
	# 	team_scores = np.zeros(len(team_points))
	# 	standing_inthiscat = np.argsort(team_points.T[c])
	# 	for i in range(len(standing_inthiscat)):
	# 		team_scores[standing_inthiscat[i]] = i
	# 	rankign_order_2.append(team_scores)
	# rankign_order_2 =  np.array(rankign_order_2).T
	# print rankign_order_2.shape
	# # print np.sum(rankign_order_2, axis=1)

	# print 

	# print all_cats
	# for t in team_score_order:
	# 	print team_overall_score[t], rankign_order_2[t]
	# print

	# for t in team_score_order:
	# 	print 'Team', t, team_overall_score[t] #team_points[t], 
	# 	print team_comps[t]
	# 	print 
	# fds


	#Update player scores
	# update_player_scores(all_names, all_scores, all_count, T1_names, T1_score)

	#or do for all teams, if noone is selected
	for t in range(n_teams):
		update_player_scores(all_names, all_scores, all_count, team_comps[t], team_overall_score[t])



	# for t in range(n_teams):
	# 	print 'Team', t, team_overall_score[t] #team_points[t], 
	# 	print team_comps[t]
	# 	print 


	# #Make two teams
	# T1_names, T1_points = make_team()


	# replace T2 with just avg points
	# T2_names, T2_points = make_team_standard()
	# if iter_ ==0:
	# 	t2_avg = T2_points
	# else:
	# 	t2_avg = t2_avg + (T2_points - t2_avg)/(iter_+1)


	# T2_points = [  166.568222,   284.718408,   1182.32981,   4.05747044,
 #   1785.74204,   882.177615,  107.064040,   4741.75770, 162.992531,   243950.702]
   #increase asssists and goals
  	# T2_points = [  180.568222,   300.718408,   1182.32981,   4.05747044,
   # 1785.74204,   882.177615,  107.064040,   4741.75770, 162.992531,   243950.702]

	# #Get team scores
	# points = np.stack([T1_points, T2_points], axis=1)
	# # print points
	# T2_score = np.sum(np.argmax(points, axis=1))
	# T1_score = len(points) - T2_score
	# # print T2_score
	# # print T1_score
	# fad
	# T1_points = np.reshape(np.array(T1_points), [1, len(T1_points)])
	# prev_team_points_and_this_team = np.concatenate([team_points_distribution,T1_points], axis=0)
	# # print prev_team_points_and_this_team.shape
	# # fsdf

	# team_scores = compute_team_scores(prev_team_points_and_this_team)
	# # print team_scores
	# # fasd
	# T1_score = team_scores[-1]

	#Update player scores
	# update_player_scores(all_names, all_scores, all_count, T1_names, T1_score)
	# update_player_scores(all_names, all_scores, all_count, T2_names, T2_score)















# print t2_avg
# print 'thats avg team'

#just so their removed from the printed list
# taken_players=['Letang Kris']
# taken_players=['Devan Dubnyk', 'Sergei Bobrovsky', 'Cam Talbot', 'Braden Holtby']





order = np.argsort(all_scores) #[::-1]
scores_sorted = [all_scores[i] for i in order]
names_sorted = [all_names[i] for i in order]
count_sorted = [all_count[i] for i in order]

countdown = range(len(order))[::-1]

for i in range(len(order)):
	all_index = all_names.index(names_sorted[i]) 

	# print "Location: {0:1} Revision {7:8}".format(scores_sorted[i],count_sorted[i])
	if names_sorted[i] in ['Connor McDavid', 'Sidney Crosby', 'Braden Holtby', 'Patrick Kane']:
		print

	# if 'G' in all_positions[all_index]:
	# 	print
	
	# print i, scores_sorted[i], count_sorted[i], names_sorted[i], all_teams[all_index], all_positions[all_index]#, all_data[all_index]
	if names_sorted[i] not in taken_players:
		print countdown[i], scores_sorted[i], count_sorted[i], names_sorted[i], all_teams[all_index], all_positions[all_index], all_data[all_index]
	else:
		print countdown[i], names_sorted[i]

# print 'Out of', str(len(T2_points)), 'cats'
print 'total iters', str(iter_total)
print 'time elapsed', (time.time() - start)




# def make_team():

# 	player_names = []

# 	L1 = np.random.randint(len(LW_names))
# 	L2 = np.random.randint(len(LW_names))
# 	C1 = np.random.randint(len(C_names))
# 	C2 = np.random.randint(len(C_names))
# 	R1 = np.random.randint(len(RW_names))
# 	R2 = np.random.randint(len(RW_names))
# 	D1 = np.random.randint(len(D_names))
# 	D2 = np.random.randint(len(D_names))
# 	D3 = np.random.randint(len(D_names))
# 	D4 = np.random.randint(len(D_names))
# 	G1 = np.random.randint(len(names_goalies))
# 	G2 = np.random.randint(len(names_goalies))
# 	S1 = np.random.randint(len(all_names))
# 	S2 = np.random.randint(len(all_names))
# 	S3 = np.random.randint(len(all_names))
# 	S4 = np.random.randint(len(all_names))

# 	player_names.append(LW_names[L1])
# 	player_names.append(LW_names[L2])
# 	player_names.append(C_names[C1])
# 	player_names.append(C_names[C2])
# 	player_names.append(RW_names[R1])
# 	player_names.append(RW_names[R2])

# 	# player_names.append('Kessel Phil')

# 	player_names.append(D_names[D1])
# 	player_names.append(D_names[D2])
# 	player_names.append(D_names[D3])
# 	player_names.append(D_names[D4])
# 	player_names.append(names_goalies[G1])
# 	player_names.append(names_goalies[G2])
# 	player_names.append(all_names[S1])
# 	player_names.append(all_names[S2])
# 	player_names.append(all_names[S3])
# 	player_names.append(all_names[S4])

# 	team_points = calc_season_total(player_names)


# 	return player_names, team_points




















#Problem: since theres no max to games played, it'll be super advantageous to have a team that can play a lot
# so playing multiple positons is very useful
# ill come back to this later

# mayeb look if I can get schedule data, see which teams play on different days the most

# so need to select bench players
# then need to go through calendar and see if team plays and if position available. 
#Problem: if the positions dont match up with the yahoo positions...
	# its not perfect but go with it. atleast you get the schedule included




# so atm:
# - add subs
# - check schedule and position when making team for each day
# - make it draft day adjusted / dependent on current squad, since current players will have large influence on who can play
	# also it will help balance the stas, like if I already have a lot of one cat, itll get me more of another
	# maybe a problem could be I do decent in all cats and lose, vs just do well in a few, but its ranked so I dont need to come first, just do well in all cats.







#SHit, need to link player to team...actually, not a problem its in the player data, but the team name isnt writtent the same way..

# ignore Las Vegas for now, theyll suck anyway


# confirm that if I take 2 ottawa d, then karlsson wont be 1st anymore, becasue they play on the same night
	# i did it with hornqvist and it works, he went from top to middle when I added two pitts RW


# must match yahoo postions, changes outcome quite a bit
# must include las vegas roster
#done



#now Ive made a team with 3goalies and it sitill says I should pick up another goalie, so something must be wrong
	#ooohh its becasue every team its compared to, also has the same number of goalies
	# so maek second team maker, which has the usual team comp
	# but also cant count the standard team stats, its only for comparison
		#can be used when no players have been picked, but after it cant



#thres a lot of variace. consider making it faster and runnig longer 
# team 2 can be replaced with averge values


# so I think the values I use for the avgs is very important, because the best team will just win in each cat
# right now the avg was made from random selection, so its probably lower in most cats, so it could be downwieghting some cats.
# so I think I should make myteam frmo lst yaer, see what their season points is, then compare to that.

# actully optimal would be to score it based on teh ranking, so have a few temas, then see where it places.. 
	# but finding the proper ranking is hard, I'd need past data. 
	# what I could do is remake all of last years teams, then get their team points, then use it as my ranking..

# becasue Im thining mcdavid got low ranking because my T2 points has a low goals adn assists so he doesnt help much
 # since winnign those cats might be easy.. but if I made it higher, he would help.. 
 # ill test this by increasing assists

 #ya mcdavid went from 18th to 6th... so this is really important to get that season poitns corect



 # one idea could be to use my current values to make teams , so that it ignores the shitty guys and I get 
 	# better estimates of the season totals and ranking distributinos 
 	# ya Id have to make 12 teams, to reflect the real situtaiton 

 	#one problem is that it would slow things down a lot, making the teams 
 	# but atleast ill be calculating the right thing, right now it can be arbitralily bad. 
 	# OR I can do this, get the season totals, fix them to that, then its fast again. just need to do it once
 			# the coding is the only standing in my way.
 	#so what needs to be done. 	
 			# read in my player rankings
 			# assign players to teams, doesnt need to be perfect but decent
 			# compute each team's total
 			# make a new point counter that does it based on ranks
 			# run counter on competed totals




# #ok so make teams
# # do it like a draft, but pick the top C, LW, RW, D, then G. for subs, pick top C, LW, RW, D
# 	# or randomly pick a position thats open, then pick the best for that position
# n_teams = 12
# positions_open = [3,3,3,5,2]  #C, LW, RW, D, G
# positions_static = ['C', 'LW', 'RW', 'D', 'G']
# teams_statuses = [list(positions_open) for i in range(n_teams)]
# positions_dynamic = ['C', 'LW', 'RW', 'D', 'G']

# team_comps = [[] for i in range(n_teams)]

# players_to_select = list(names_sorted[::-1])

# for round_ in range(16):

# 	for t in range(n_teams):

# 		shuffle(positions_dynamic)
# 		pl_found =0

# 		#which position to look for
# 		for p in range(len(positions_dynamic)):

# 			pos_index = positions_static.index(positions_dynamic[p])
# 			#if position still available for this team
# 			if teams_statuses[t][pos_index] > 0:
				
# 				#find the best player with this position
# 				for pl in range(len(players_to_select)):

# 					pl_name = players_to_select[pl]
# 					pl_all_index = all_names.index(pl_name)
# 					pl_pos = all_positions[pl_all_index]

# 					#check their position
# 					if positions_dynamic[p] in pl_pos:

# 						#add to team
# 						team_comps[t].append(pl_name)
# 						#remove pl from list and remove pos from team
# 						players_to_select.pop(pl)
# 						teams_statuses[t][pos_index] -= 1
# 						pl_found =1
# 						break
# 			if pl_found:
# 				break


# #Compute team season points

# team_points = []
# for t in range(n_teams):
# 	team_points.append(calc_season_total(team_comps[t]))





# team_overall_score = compute_team_scores(team_points)

# for t in range(n_teams):
# 	print 'Team', t, team_overall_score[t] #team_points[t], 
# 	print team_comps[t]
# 	print 



# print 'Use this to approx the distributinos of teams'
# for t in range(n_teams):
# 	print team_points[t]


print 'Done.'

