

import gs_functions
import math
import numpy
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


data_file = '/data1/morrislab/ccremer/wrana_data1/ProCoding_all_RPMs_Tx_cleaned_ID.txt'


#this one was found to be basal

HG_cluster_1 = ['BC002', 'BC044', 'BC045', 'BC042', 'BC064',
				 'BC017', 'BC022', 'BC036', 'BC035', 'BC008', 
				 'BC027']

#this one was found to be luminal
HG_cluster_2 = ['BC034', 'BC046', 'BC010', 'BC013', 'BC012',
				 'BC061', 'BC009', 'BC031']


boundary_cluster_1 = ['BC049', 'BC055', 'BC051', 'BC063', 'BC033']

boundary_cluster_2 = ['BC048', 'BC047', 'BC001']


LG_cluster_1 = ['BC038', 'BC030', 'BC050', 'BC029', 'BC006', 'BC005', 'BC020', 'BC024', 'BC026']
LG_cluster_2 = ['BC019', 'BC014', 'BC039', 'BC037', 'BC032', 'BC059', 'BC054', 'BC053', 'BC052', 'BC007', 'BC040', 'BC041', 'BC060']
#LG_cluster_3 = ['BC014', 'BC039', 'BC037', 'BC032', 'BC053', 'BC007', 'BC040', 'BC041', 'BC060']

#krt14, krt6b
basal_genes = ['ENSG00000186847', 'ENSG00000185479'] #not included in data 'ENSG00000186081', 'ENSG00000026508'

#upk1b, upk2, upk3a, krt20
luminal_genes = ['ENSG00000114638', 'ENSG00000110375', 'ENSG00000100373', 'ENSG00000171431']

#first find the avg of each specified gene in each cluster


firstRow = gs_functions.firstRowList(data_file)

print 'LG clusters'

cluster_1_avg_std = gs_functions.getAvgStd(LG_cluster_1, firstRow, data_file)
cluster_1_avg = cluster_1_avg_std[0]
cluster_1_std = cluster_1_avg_std[1]

cluster_2_avg_std = gs_functions.getAvgStd(LG_cluster_2, firstRow, data_file)
cluster_2_avg = cluster_2_avg_std[0]
cluster_2_std = cluster_2_avg_std[1]

#cluster_3_avg_std = gs_functions.getAvgStd(LG_cluster_3, firstRow, data_file)
#cluster_3_avg = cluster_3_avg_std[0]
#cluster_3_std = cluster_3_avg_std[1]

print 'basal genes'
for gene in basal_genes:
	print gene
	print 'cluster_1 avg: ' + str(cluster_1_avg[gene])
	print 'cluster_2 avg: ' + str(cluster_2_avg[gene])
	#print 'cluster_3 avg: ' + str(cluster_3_avg[gene])

print

print 'luminal genes'
for gene in luminal_genes:
	print gene
	print 'cluster_1 avg: ' + str(cluster_1_avg[gene])
	print 'cluster_2 avg: ' + str(cluster_2_avg[gene])
	#print 'cluster_3 avg: ' + str(cluster_3_avg[gene])


'''
print 'HG clusters'

cluster_1_avg_std = gs_functions.getAvgStd(HG_cluster_1, firstRow, data_file)
cluster_1_avg = cluster_1_avg_std[0]
cluster_1_std = cluster_1_avg_std[1]

cluster_2_avg_std = gs_functions.getAvgStd(HG_cluster_2, firstRow, data_file)
cluster_2_avg = cluster_2_avg_std[0]
cluster_2_std = cluster_2_avg_std[1]

print 'basal genes'
for gene in basal_genes:
	print 'cluster_1 avg: ' + str(cluster_1_avg[gene])
	print 'cluster_2 avg: ' + str(cluster_2_avg[gene])

print 'luminal genes'
for gene in luminal_genes:
	print 'cluster_1 avg: ' + str(cluster_1_avg[gene])
	print 'cluster_2 avg: ' + str(cluster_2_avg[gene])
'''


#THEREFORE  the clustering on BASE47 did seperate into basal and luminal