

import os
import csv
import numpy as np
import scipy.stats
#from scipy.stats.stats import pearsonr
import math
import network2




output = 'output16.csv'
f2 = open(output, 'w')
f2.write('driver_trip,prob')

files = []

driver_count = 0

#speed distribution
speed_distribution = range(-1, 30)
#accceleraton distribution
acc_distribution_buckets = range(-5, 6)

#for every driver
for driver_numb in os.listdir("drivers"):

	print 'Working on driver ' + driver_numb + ' --- ' + str(driver_count) + ' out of 2735'

	driver_count += 1

	#indexes
	#0 - total time
	#1 - avg speed
	#2 - max speed
	#3 - total dist
	#4-24 - speed distributions
	driver_variables = []


	trip_numbs = []
	#features
	#trip_total_time = []
	#trip_avg_speeds = []
	#trip_max_speeds = []
	#trip_total_dist = []




	#for every trip
	for file2 in os.listdir('drivers/' + driver_numb):
		#print file2

		trip_numb = file2.split('.')[0]
		trip_speeds = []

		trip_speed_distribution = []
		for bucket in speed_distribution:
			trip_speed_distribution.append(0)

		trip_acceleration_distribution = []
		for bucket in acc_distribution_buckets:
			trip_acceleration_distribution.append(0)

		numb_of_stops = 0
		stop_lengths = []
		current_stop_duration = 0

		with open('drivers/' + driver_numb + '/' + file2, 'r') as f:
			reader = csv.reader(f)
			row_count = 0
			previous_row = ['0.0', '0.0']
			previous_speed = 0.0
			for row in reader:
				
				if row_count < 2:
					row_count += 1
					continue
				row_count += 1

				speed = math.sqrt((float(row[0]) - float(previous_row[0]))**2 + (float(row[1]) - float(previous_row[1]))**2)
				if speed > 50:
					speed = 50.0

				if speed == 0.0:
					if previous_speed > 0.0:
						numb_of_stops += 1
					elif previous_speed == 0.0:
						current_stop_duration += 1
				elif speed > 0.0 and previous_speed == 0.0:
					stop_lengths.append(current_stop_duration)
					current_stop_duration = 0

				for speed_bucket in range(len(speed_distribution)-1, -1, -1):
					if speed > speed_distribution[speed_bucket]:
						trip_speed_distribution[speed_bucket] += 1
						break


				trip_speeds.append(speed)
				previous_row = row
				previous_speed = speed


		avg_stop_duration = np.mean(stop_lengths)

		#print trip_speeds
		#print
		#print


		#print trip_speed_distribution
		numb_of_speeds = sum(trip_speed_distribution)
		#print row_count
		fraction_speed_distribution = []
		for amount_in_bucket in trip_speed_distribution:
			fraction_speed_distribution.append(amount_in_bucket/float(numb_of_speeds))
		#print fraction_speed_distribution
		#print



		accelerations = []
		acc_dist = []
		last_speed = -1
		for speed1 in trip_speeds:
			if last_speed == -1:
				last_speed = speed1
				continue
			else:
				acc = speed1 - last_speed
				#print 'acc: ' + str(acc)
				accelerations.append(acc)
				assigned = 0
				for acc_bucket in range(len(acc_distribution_buckets)-1, -1, -1):
					#print acc_distribution_buckets[acc_bucket]
					if acc > acc_distribution_buckets[acc_bucket]:
						trip_acceleration_distribution[acc_bucket] += 1
						assigned = 1
						break
				if assigned == 0:
					trip_acceleration_distribution[0] += 1
				#print acc_distribution_buckets
				#print trip_acceleration_distribution
				last_speed = speed1




		numb_of_accs = sum(trip_acceleration_distribution)
		#print row_count
		fraction_acc_distribution = []
		for amount_in_bucket in trip_acceleration_distribution:
			fraction_acc_distribution.append(amount_in_bucket/float(numb_of_accs))

		max_acceleration = max(accelerations)
		min_acceleration = min(accelerations)

		avg_speed = np.mean(trip_speeds)
		max_speed = max(trip_speeds)
		total_distance = sum(trip_speeds)

		trip_numbs.append(trip_numb)
		#trip_total_time.append(row_count)
		#trip_avg_speeds.append(avg_speed)
		#trip_max_speeds.append(max_speed)
		#trip_total_dist.append(total_distance)
		trip_variables = [row_count, avg_speed, max_speed, total_distance, max_acceleration, min_acceleration, numb_of_stops, avg_stop_duration]
		trip_variables.extend(fraction_speed_distribution)
		trip_variables.extend(fraction_acc_distribution)

		driver_variables.append(trip_variables)




	#########################################################################
	#This was using multiple single variable gaussians
	'''	
	#trip total time
	trip_time_scores = []
	avg_trip_time = np.mean(trip_total_time)
	std_trip_time = np.std(trip_total_time)
	norm_pdf = scipy.stats.norm(avg_trip_time, std_trip_time)
	for i in range(len(trip_total_time)):
		score = norm_pdf.pdf(trip_total_time[i])
		trip_time_scores.append(score)
	#trip avg speeds
	avg_speed_scores = []
	avg_avg_speed = np.mean(trip_avg_speeds)
	std_avg_speed = np.std(trip_avg_speeds)
	norm_pdf = scipy.stats.norm(avg_avg_speed, std_avg_speed)
	for i in range(len(trip_avg_speeds)):
		score = norm_pdf.pdf(trip_avg_speeds[i])
		avg_speed_scores.append(score)
	#trip max speeds
	max_speed_scores = []
	avg_max_speed = np.mean(trip_max_speeds)
	std_max_speed = np.std(trip_max_speeds)
	norm_pdf = scipy.stats.norm(avg_max_speed, std_max_speed)
	for i in range(len(trip_max_speeds)):
		score = norm_pdf.pdf(trip_max_speeds[i])
		max_speed_scores.append(score)
	#trip total distances
	distance_scores = []
	avg_distance = np.mean(trip_total_dist)
	std_distance = np.std(trip_total_dist)
	norm_pdf = scipy.stats.norm(avg_distance, std_distance)
	for i in range(len(trip_total_dist)):
		score = norm_pdf.pdf(trip_total_dist[i])
		distance_scores.append(score)



	#combined scores
	combined_scores = []
	for i in range(len(trip_numbs)):
		combined_scores.append(trip_time_scores[i] * avg_speed_scores[i] * max_speed_scores[i] * distance_scores[i])
	'''
	#########################################################################


	#########################################################################
	#multivariate gaussian
	'''
	#array_of_variables = np.array([trip_total_time, trip_avg_speeds, trip_max_speeds, trip_total_dist])
	array_of_variables = np.array(driver_variables).T
	#print array_of_variables.shape

	#var_means = np.array([np.mean(trip_total_time), np.mean(trip_avg_speeds), np.mean(trip_max_speeds), np.mean(trip_total_dist)])
	var_means = np.mean(array_of_variables, 1)
	#print var_means.shape

	cov_matrix = np.cov(array_of_variables)

	multi_var_norm = scipy.stats.multivariate_normal(mean= var_means, cov= cov_matrix)

	multi_var_scores = []
	for i in range(len(trip_numbs)):
		samp = np.array(driver_variables[i])
		prob = multi_var_norm.pdf(samp)
		multi_var_scores.append(prob)
	'''
	#########################################################################


	#########################################################################
	#GRI algorithm

	#split samples into two groups
	group1 = driver_variables[:100]
	group2 = driver_variables[100:200]

	gri_scores = []
	for trip in group1:
		this_trip_PCCs = []
		for trip2 in group2:
			PCC = scipy.stats.stats.pearsonr(trip, trip2)
			this_trip_PCCs.append(PCC)
		gri_score = np.mean(this_trip_PCCs)
		gri_scores.append(gri_score)

	for trip in group2:
		this_trip_PCCs = []
		for trip2 in group1:
			PCC = scipy.stats.stats.pearsonr(trip, trip2)
			this_trip_PCCs.append(PCC)
		gri_score = np.mean(this_trip_PCCs)
		gri_scores.append(gri_score)

	#count = 0
	#for score in gri_scores:
	#	print str(count) + ' ' + str(score)
	#	count+=1
	#print gri_scores
	#print
	#########################################################################



	#rescale
	max_score = max(gri_scores)
	#print max_score
	min_score = min(gri_scores)
	#print min_score
	#print gri_scores.index(min_score)
	range_of_scores = max_score - min_score
	#print range_of_scores

	final_scores = []
	for score in gri_scores:
		final_score = (score-min_score)/range_of_scores
		#if final_score > 0.7:
		#	final_score = 1.0
		final_scores.append(final_score)

	#print final_scores
	#print

	for i in range(len(trip_numbs)):

		f2.write('\n' + driver_numb + '_' + trip_numbs[i] + ',' + str(final_scores[i]))
	


	

print "DONE."






