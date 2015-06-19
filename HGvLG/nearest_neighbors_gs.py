

from sklearn.neighbors import KNeighborsRegressor

import gs_functions
import math
import numpy
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



data_file = '/data1/morrislab/ccremer/wrana_data1/ProCoding_all_RPMs_Tx_cleaned_ID.txt'

LG_set = ['BC060', 'BC005', 'BC030', 'BC026', 'BC050',
			 'BC006', 'BC019', 'BC052', 'BC040', 'BC037', 
			 'BC032', 'BC039', 'BC041', 'BC029', 'BC024', 
			 'BC020', 'BC038', 'BC054', 'BC053', 'BC014', 
			 'BC007', 'BC059', 'BC049', 'BC055', 'BC047',
			 'BC051', 'BC063']

HG_set = ['BC010', 'BC013', 'BC035', 'BC009', 'BC027', 
			'BC036', 'BC008', 'BC031', 'BC064', 'BC034', 
			'BC042', 'BC044', 'BC002', 'BC045', 'BC017', 
			'BC022', 'BC046', 'BC012', 'BC061', 'BC001',
			'BC048', 'BC033']

#one for testing, the other for training
set_1 = ['BC060', 'BC030', 'BC050',
		'BC019', 'BC040', 'BC032',
		'BC041', 'BC024', 'BC038',  
		'BC053', 'BC007', 'BC049', 
		'BC001', 'BC047', 'BC033',	
		'BC046', 'BC061', 'BC013',
		'BC009', 'BC036', 'BC031', 
		'BC034', 'BC044', 'BC045', 
		'BC022']

set_2 = [ 'BC005',  'BC026', 'BC006', 
		'BC052', 'BC037', 'BC039',
		'BC029', 'BC020', 'BC054',
		'BC014', 'BC059', 'BC055', 
		'BC048', 'BC051', 'BC063',
		'BC012', 'BC010', 'BC035', 
		'BC027', 'BC008', 'BC064',
		'BC042', 'BC002', 'BC017', ]

samples_to_analyze = set_2
samples_to_train = set_1

##############################################################
###
###
### Select significant genes / Dimension reduction
###
###
##############################################################

#First step is to select the genes/features/variables

#to keep track which column each sample is in
firstRow = gs_functions.firstRowList(data_file)
'''
# Get the average and std dev of the LG list
#make the list of LG samples in the learning set
LG_learn_set = []
for samp in samples_to_train:
	if samp in LG_set:
		LG_learn_set.append(samp)

LG_gene_avg_std = gs_functions.getAvgStd(LG_learn_set, firstRow, data_file)
LG_gene_avg = LG_gene_avg_std[0]
LG_gene_std = LG_gene_avg_std[1]

#Get the average and std dev of the HG list
HG_learn_set = []
for samp in samples_to_train:
	if samp in HG_set:
		HG_learn_set.append(samp)

HG_gene_avg_std = gs_functions.getAvgStd(HG_learn_set, firstRow, data_file)
HG_gene_avg = HG_gene_avg_std[0]
HG_gene_std = HG_gene_avg_std[1]

#Calculate significant difference, get absolute value, sort
#makes a list of tuples (geneID, sig_dif) ordered from largest sig_dif to smallest
sig_dif_abs_sorted = gs_functions.sigAvgDif(LG_gene_avg, HG_gene_avg)

#Overlapping std devs
sig_genes = []
for gene in sig_dif_abs_sorted[:1000]:
	#LG top end
	LG_avg_plus_std = (LG_gene_avg[gene[0]] + LG_gene_std[gene[0]])
	#LG bottom end
	LG_avg_minus_std = (LG_gene_avg[gene[0]] - LG_gene_std[gene[0]])
	#HG top end
	HG_avg_plus_std = (HG_gene_avg[gene[0]] + HG_gene_std[gene[0]])
	#HG bottom end
	HG_avg_minus_std = (HG_gene_avg[gene[0]] - HG_gene_std[gene[0]])
	#LG top end minus HG bottom end
	LG_aPs_minus_HG_aMs = LG_avg_plus_std - HG_avg_minus_std
	#HG top end minus LG bottom end
	HG_aPs_minus_LG_aMs = HG_avg_plus_std - LG_avg_minus_std

	if LG_aPs_minus_HG_aMs < 0.9 or HG_aPs_minus_LG_aMs < 0.9:
		sig_genes.append(gene[0])


#if len(sig_genes) < numbOfSigGenes:
#	print "You want more genes than are available"
#	print len(sig_genes)
#	print numbOfSigGenes


#sig_genes_names = sig_genes[:numbOfSigGenes]
print 'numb of sig genes= ' + str(len(sig_genes))
'''

##############################################################
###
###
### Nearest Neighbors
###
###
##############################################################


#make a list of the columns that are part of the samples_to_train
samples_columns = []
for sample in samples_to_train:
	samples_columns.append(firstRow.index(sample))

#data -> samples are the rows, not the genes
X = []
y = []

#initialize all the lists for each sample
for samp in samples_to_train:
	X.append([])
	if samp in LG_set:
		y.append(0)
	else:
		y.append(1)

with open(data_file, 'rU') as f:
		reader = csv.reader(f, delimiter='\t')
		b = 0
		for row in reader:
			#skip header
			if b == 0:
				b+=1
				continue
			#if row[0] in sig_genes:
			c = 0
			for expr in row:
				if c in samples_columns:
					X[samples_columns.index(c)].append(expr)
				c += 1
f.close()

neigh = KNeighborsRegressor(n_neighbors=9, weights='distance')
neigh.fit(X, y) 



##############################################################
###
###
### Test
###
###
##############################################################

#make a list of the columns that are part of the samples_to_train
samples_columns = []
for sample in samples_to_analyze:
	samples_columns.append(firstRow.index(sample))

#data -> samples are the rows, not the genes
X = []
y = []

#initialize all the lists for each sample
for samp in samples_to_analyze:
	X.append([])
	if samp in LG_set:
		y.append(0)
	else:
		y.append(1)

with open(data_file, 'rU') as f:
		reader = csv.reader(f, delimiter='\t')
		b = 0
		for row in reader:
			#skip header
			if b == 0:
				b+=1
				continue
			#if row[0] in sig_genes:
			c = 0
			for expr in row:
				if c in samples_columns:
					X[samples_columns.index(c)].append(expr)
				c += 1
f.close()


results = neigh.predict(X)

correct = 0.0
for i in range(len(results)):

	print 'predict: ' + str(results[i]) + ' actual: ' + str(y[i])

	if results[i] < 0.5 and y[i] == 0:
		correct += 1.0
	elif results[i] > 0.5 and y[i] == 1:
		correct += 1.0


print "Accuracy= " + str(correct/len(results)) 


##############################################################
###
###
### Plot
###
###
##############################################################

ploted = []

for point in range(len(results)):

	x_pos = 1.005
	for old_point in ploted:
		#if they are close together
		if numpy.abs(results[point] - old_point) < 0.025:
			x_pos += 0.006

	if y[point] == 0:
		plt.annotate(firstRow[samples_columns[point]], xy=(x_pos,results[point]), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"),  color='blue')
	elif y[point] == 1:
		plt.annotate(firstRow[samples_columns[point]], xy=(x_pos,results[point]), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"), color='red')
	else:
		plt.annotate('what', xy=(x_pos,results[point]), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"), color='yellow')

	ploted.append(results[point])


y = numpy.ones(numpy.shape(results))   # Make all y values the same
plt.plot(y,results,'_',ms = 20)  # Plot a line at each location specified in a

plt.xticks([])
plt.xlim(1,1.07)
plt.ylim(-0.02,1.03)
plt.title("Nearest Neighbors Grade Score")
plt.ylabel("Scale")
plt.annotate('HG', xy=(0.8,0.9), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='red')
plt.annotate('LG', xy=(0.8,0.85), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='blue')

plt.annotate('Testing Set = set_2', xy=(0.7,0.7), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='maroon')
plt.annotate('Training Set = set_1', xy=(0.7,0.65), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='maroon')



plt.savefig('Gs_nearest_neighbors_k9_set2_allgenes.pdf')

print '\a'