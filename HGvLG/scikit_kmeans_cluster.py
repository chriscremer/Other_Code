


from sklearn.decomposition import PCA



import scipy
import numpy

from sklearn import cluster

#to read csv
import csv
#for printing results
import pprint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_file = '/home/morrislab/ccremer/recurrence_predictor/ProCoding_all_RPMs_Tx_cleaned_ID.txt'
#data_file = 'ProCoding_all_RPMs_Tx_cleaned_ID.txt'


HG_set = ['BC010', 'BC013', 'BC035', 'BC009', 'BC027',
			 'BC036', 'BC008', 'BC031', 'BC064', 'BC034',
			 'BC042', 'BC044', 'BC002', 'BC045', 'BC017', 
			 'BC022', 'BC046', 'BC012', 'BC061']


LG_set = ['BC060', 'BC005', 'BC030', 'BC026', 'BC050',
			'BC006', 'BC019', 'BC052', 'BC040', 'BC037',
			'BC032', 'BC039', 'BC041', 'BC029', 'BC024',
			'BC020', 'BC038', 'BC054', 'BC053', 'BC014',
			'BC007', 'BC059']

boundary_grade = ['BC049', 'BC055', 'BC047', 'BC051', 'BC063', 'BC001', 'BC048', 'BC033']

recurred_and_prog = ['Sample_3_1_775_1', 'Sample_1_1_298_1', 'Sample_1_129_1', 'Sample_02_713_1', 'Sample_00_994_01',
			'Sample_00_905_01', 'Sample_01_711_1', 'Sample_01_779_1', 'Sample_2_747_1']

no_recur = ['Sample_3_1_230_1', 'Sample_2_1_146_1',  'Sample_2_1_140_1', 'Sample_6_1_524_1',
			'Sample_1_1_095_1', 'Sample_2_1_034_1','Sample_1_1_335_1', 'Sample_03_359_01', 
			'Sample_1_165_1', 'Sample_01_148_01', 'Sample_02_459_01', 
 			'Sample_02_089_1', 'Sample_00_419_01',  'Sample_00_131_1', 'Sample_01_036_1', 
 			'Sample_3_512_1', 'Sample_02_133_1']

recur_no_prog = ['Sample_3_1_129_1', 'Sample_8_1_135_1', 'Sample_2_3_025_1', 'Sample_2_1_143_1', 
				'Sample_04_937_1', 'Sample_570_01', 'Sample_00_801_1', 'Sample_4_565_1', 
				'Sample_07_953_1', 'Sample_00_3943_1']

MSH_T1HG_cohort = ['Sample_1_1_298_1', 'Sample_3_1_230_1', 'Sample_1_129_1',
				'Sample_2_1_146_1', 'Sample_3_1_129_1', 'Sample_2_1_140_1', 'Sample_8_1_135_1',
				'Sample_6_1_524_1', 'Sample_1_1_095_1', 'Sample_2_1_034_1', 'Sample_2_3_025_1',
				'Sample_2_1_143_1', 'Sample_1_1_335_1']

MSH_T1HG_cohort_outlier = 'Sample_3_1_775_1'

French_T1HG_cohort = ['Sample_00_994_01', 'Sample_00_131_1', 'Sample_00_905_01',
						'Sample_01_036_1', 'Sample_570_01', 'Sample_01_711_1', 'Sample_00_801_1',
						'Sample_01_779_1','Sample_4_565_1', 'Sample_2_747_1', 'Sample_3_512_1',
						'Sample_02_133_1', 'Sample_03_359_01', 'Sample_1_165_1', 'Sample_00_3943_1',
						'Sample_01_148_01', 'Sample_02_459_01', 'Sample_07_953_1', 'Sample_02_089_1',
						'Sample_00_419_01', 'Sample_02_713_1',] 

French_T1HG_cohort_outlier = 'Sample_04_937_1'

not_in_data = [ 'Sample_5_953_1', 'Sample_2246_01', 'Sample_4362_01']




samples_to_analyze =  ['BC060', 'BC005', 'BC030', 'BC026', 'BC050',
			'BC006', 'BC019', 'BC052', 'BC040', 'BC037',
			'BC032', 'BC039', 'BC041', 'BC029', 'BC024',
			'BC020', 'BC038', 'BC054', 'BC053', 'BC014',
			'BC007', 'BC059']




base47_gene_list = ['AHNAK2', 'CDK6', 'ALOX5AP', 'CHST15',
			 'EMP3', 'MT1X', 'MT2A', 'GLIPR1',
			 'MSN', 'TUBB6', 'PDGFC', 'PALLD',
			 'PRKCDBP', 'PRRX1', 'GAREM', 'GDPD3',
			 'BHMT', 'SLC9A2', 'GPD1L', 'CYP4B1',
			 'FBP1', 'GATA3', 'CYP2J2', 'TOX3',
			 'CAPN5', 'FAM174B', 'PPFIBP2', 'PPARG',
			 'RAB15', 'HMGCS2', 'RNF128', 'ADIRF',
			 'SCNN1B', 'SCNN1G', 'TRAK1', 'UPK2', 
			 'UPK1A','TMEM97', 'SPINK1', 'TMPRSS2',
			 'PLEKHG6', 'VGLL1', 'SEMA5A', 'TBX2', 
			 'SLC27A2']

base47_ensg_list = ['ENSG00000185567', 'ENSG00000105810', 'ENSG00000132965', 'ENSG00000182022',
			 'ENSG00000142227', 'ENSG00000187193', 'ENSG00000125148', 'ENSG00000139278',
			 'ENSG00000147065', 'ENSG00000176014', 'ENSG00000145431', 'ENSG00000129116',
			 'ENSG00000170955', 'ENSG00000116132', 'ENSG00000141441', 'ENSG00000102886',
			 'ENSG00000145692', 'ENSG00000115616', 'ENSG00000152642', 'ENSG00000142973',
			 'ENSG00000165140', 'ENSG00000107485', 'ENSG00000134716', 'ENSG00000103460',
			 'ENSG00000149260', 'ENSG00000185442', 'ENSG00000166387', 'ENSG00000132170',
			 'ENSG00000139998', 'ENSG00000134240', 'ENSG00000133135', 'ENSG00000148671',
			 'ENSG00000168447', 'ENSG00000166828', 'ENSG00000182606', 'ENSG00000110375',
			 'ENSG00000105668', 'ENSG00000109084', 'ENSG00000164266', 'ENSG00000184012',
			 'ENSG00000008323', 'ENSG00000102243', 'ENSG00000112902', 'ENSG00000121068',
			 'ENSG00000140284']

##############################################################
###
###
### Assemble Data
###
###
##############################################################

#columns corresponding to samples
column_tracker_dict = {}
column_tracker_list = []
with open(data_file, 'rU') as f:
	reader = csv.reader(f, delimiter='\t')
	for row in reader:
		column = 0
		for samp in row:
			if samp in samples_to_analyze:
				column_tracker_dict[column] = samp
				column_tracker_list.append(column)
			column +=1
		break
f.close()

#initialize the sample expressions lists so I can add to them
#its a dict not a list because list is indexed from 0-n, but the columns in the data file are 0-all other samples
data = {}
for samp in samples_to_analyze:
	data[samp] = []

#gather the corresponding data
with open(data_file, 'rU') as f:
	reader = csv.reader(f, delimiter='\t')
	b = 0 #used to skip the first row
	for row in reader:
		if b == 0:
			b = 1
			continue
		i = 0 #column that Im at
		for expr in row:
			#if not in gene list, skip

			# since im doing all genes, im removing this line
			if i == 0 and not(expr in base47_ensg_list):
				break

			if i in column_tracker_list:
				data[column_tracker_dict[i]].append(float(expr))
			i +=1
		b += 1
		#for testing
		#if b == 50:
			#break
f.close()

#pprint.pprint(data)

#change from dict to list of samples
data_list = []
samp_order_list = []
for samp in data:
	data_list.append(data[samp])
	samp_order_list.append(samp)

#print samp_order_list

#because outlier

#data_list.pop(0)

print 'length of data list: ' + str(len(data_list))
print 'length of data list list: ' + str(len(data_list[1]))

##############################################################
###
###
### Cluster
###
###
##############################################################

from sklearn.cluster import KMeans


max_numb_of_clusters = 9
scores = []
method = 1


for numb_of_clusters in range(1,max_numb_of_clusters):
	print "Clustering..." + str(numb_of_clusters) + " clusters"

	if method == 0:
		b = cluster.k_means(data_list, numb_of_clusters)
		#b = cluster.AgglomerativeClustering(data_list, numb_of_clusters)
		scores.append(b[2])

	if method == 1:
		ohhh = KMeans(n_clusters=numb_of_clusters)
		ohhh.fit(data_list)
		scores.append(ohhh.inertia_)
		print ohhh.labels_

#print ohhh.labels_
#print ohhh.inertia_
#print ohhh.cluster_centers_


##############################################################
###
###
### Elbow Calc
###
###
##############################################################

def elbow_calc(cluster_scores):

  slopes = []

  for cluster_numb in range(0,max_numb_of_clusters-3):
    slope = (cluster_scores[cluster_numb] - cluster_scores[cluster_numb+1])
    slopes.append(slope)

  slope_change = []

  for slope in range(len(slopes)-1):
    slope_change.append(slopes[slope] - slopes[slope+1])

  largest_change_in_slope = max(slope_change)

  elbow_location = slope_change.index(largest_change_in_slope) + 2

  '''
  print slopes
  print ''
  print 'slope change'
  print slope_change
  print ''
  '''

  return [elbow_location, slopes[elbow_location-2]/slopes[elbow_location-1]]
  

##############################################################
###
###
### Plot
###
###
##############################################################


#elbow = elbow_calc(scores)
elbow = 2 #for this one, Im going to set it to 2 because I want to compare

plt.figure(1)


plt.plot(range(1,max_numb_of_clusters), scores)
#need to change
plt.title('Samples: LG\nGenes: BASE47')

plt.ylabel("Inertia")
plt.xlabel("# of clusters")
'''
plt.annotate('Elbow= ' + str(elbow[0]) + '\nSlope change= ' + str(elbow[1]),
			 xy=(0.15,0.9),
			 xycoords='axes fraction', 
			 size='small', 	
			 bbox=dict(boxstyle="round", fc="0.8"), 
			 color='green')
'''


annot_x_position = [.6, .8, .4, .2, 0]



#best = KMeans(n_clusters=elbow[0])
best = KMeans(n_clusters=2)
best.fit(data_list)
labels = best.labels_



samp_order_index = 0
annot_y_position = [0.85, 0.85, 0.85, 0.85, 0.85]

for samp in best.labels_:
	'''
	if samp_order_list[samp_order_index] in recurred_and_prog:
		c = 'red'
	elif samp_order_list[samp_order_index] in no_recur:
		c = 'blue'
	elif samp_order_list[samp_order_index] in recur_no_prog:
		c = 'purple'

	elif samp_order_list[samp_order_index] in HG_set:
		c = 'red'
	elif samp_order_list[samp_order_index] in LG_set:
		c = 'blue'
	elif samp_order_list[samp_order_index] in boundary_grade:
		c = 'purple'

	else:
		c = 'brown'

	'''
	if samp == 0:
		plt.annotate(samp_order_list[samp_order_index],
			 xy=(annot_x_position[0],annot_y_position[0]),
			 xycoords='axes fraction', 
			 size='xx-small', 	
			 bbox=dict(boxstyle="round", fc="0.8"), 
			 color='blue')
		samp_order_index += 1
		annot_y_position[0] -= .03
	if samp == 1:
		plt.annotate(samp_order_list[samp_order_index],
			 xy=(annot_x_position[1],annot_y_position[1]),
			 xycoords='axes fraction', 
			 size='xx-small', 	
			 bbox=dict(boxstyle="round", fc="0.8"), 
			 color='blue')
		samp_order_index += 1
		annot_y_position[1] -= .03
	if samp == 2:
		plt.annotate(samp_order_list[samp_order_index],
			 xy=(annot_x_position[2],annot_y_position[2]),
			 xycoords='axes fraction', 
			 size='xx-small', 	
			 bbox=dict(boxstyle="round", fc="0.8"), 
			 color='blue')
		samp_order_index += 1
		annot_y_position[2] -= .03
	if samp == 3:
		plt.annotate(samp_order_list[samp_order_index],
			 xy=(annot_x_position[3],annot_y_position[3]),
			 xycoords='axes fraction', 
			 size='xx-small', 	
			 bbox=dict(boxstyle="round", fc="0.8"), 
			 color='blue')
		samp_order_index += 1
		annot_y_position[3] -= .03

'''

#may need to change
version = 2
#legend
#RECURENCE
if version == 1:
	plt.annotate('No Recurrence',
				 xy=(0.54,0.92),
				 xycoords='axes fraction', 
				 size='x-small', 	
				 bbox=dict(boxstyle="round", fc="0.8"), 
				 color='blue')
	plt.annotate('Recur No Prog',
				 xy=(0.7,0.92),
				 xycoords='axes fraction', 
				 size='x-small', 	
				 bbox=dict(boxstyle="round", fc="0.8"), 
				 color='purple')
	plt.annotate('Recur and Prog',
				 xy=(0.85,0.92),
				 xycoords='axes fraction', 
				 size='x-small', 	
				 bbox=dict(boxstyle="round", fc="0.8"), 
				 color='red')
#GRADE
else:
	plt.annotate('LG',
				 xy=(0.57,0.92),
				 xycoords='axes fraction', 
				 size='x-small', 	
				 bbox=dict(boxstyle="round", fc="0.8"), 
				 color='blue')
	plt.annotate('Boundary',
				 xy=(0.66,0.92),
				 xycoords='axes fraction', 
				 size='x-small', 	
				 bbox=dict(boxstyle="round", fc="0.8"), 
				 color='purple')
	plt.annotate('HG',
				 xy=(0.8,0.92),
				 xycoords='axes fraction', 
				 size='x-small', 	
				 bbox=dict(boxstyle="round", fc="0.8"), 
				 color='red')

'''
#to sace the graph of the distance from the cluster center for each number of clusters
plt.savefig('scikit_LG_base47_2clusters.pdf')




##############################################################
###
###
### PCA
###
###
##############################################################


pca = PCA(n_components=2)
data_with_only_two_components = pca.fit_transform(data_list)

print data_with_only_two_components


##############################################################
###
###
### Plot
###
###
##############################################################



#x = [b[0] for b in data_with_only_two_components]
#y = [b[1] for b in data_with_only_two_components]

plt.figure(2)


index = 0
for samp in data_with_only_two_components:
	if labels[index] == 0:
		plt.scatter(samp[0], samp[1], c='blue', s=40)
	elif labels[index] == 1:
		plt.scatter(samp[0], samp[1], c='yellow', s=40)
	else:
		plt.scatter(samp[0], samp[1], c='red', s=40)
	index += 1


plt.title('Samples: LG\nGenes: BASE47')
plt.ylabel("PC 1")
plt.xlabel("PC 2")

plt.savefig('pca_clusters_lg_2clusters.pdf')