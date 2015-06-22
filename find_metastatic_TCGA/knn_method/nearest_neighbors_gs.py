

from sklearn.neighbors import KNeighborsRegressor

import gs_functions
import math
import numpy
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


data_metastatic = '/data1/morrislab/ccremer/TCGA_data/find_metastatic/output_metastatic_exps.txt'
data_other = '/data1/morrislab/ccremer/TCGA_data/find_metastatic/output_other_exps.txt'

set_1 = range(1, 75)
#set_2 = range(75, 147)
set_2 = range(75, 100)
#set_2 = 75-147

numbOfSigGenes = 13

##############################################################
###
###
### Select significant genes / Dimension reduction
###
###
##############################################################

##############################################################
###
### Get the average and std dev of the LG list
###
##############################################################

LG_gene_avg_std = gs_functions.getAvgStd(set_1, data_metastatic)
LG_gene_avg = LG_gene_avg_std[0]
LG_gene_std = LG_gene_avg_std[1]

##############################################################
###
### Get the average and std dev of the HG list
###
#################################################################

HG_gene_avg_std = gs_functions.getAvgStd(set_1, data_other)
HG_gene_avg = HG_gene_avg_std[0]
HG_gene_std = HG_gene_avg_std[1]

##############################################################
###
### Calculate significant difference, get absolute value, sort
###
##############################################################

#makes a list of tuples (geneID, sig_dif) ordered from largest sig_dif to smallest
sig_dif_abs_sorted = gs_functions.sigAvgDif(LG_gene_avg, HG_gene_avg)

sig_genes_names = []

for i in range(0, numbOfSigGenes):
	print sig_dif_abs_sorted[i][0] + ' ' + str(sig_dif_abs_sorted[i][1]) + ' ' + str(LG_gene_avg[sig_dif_abs_sorted[i][0]]) + ' ' + str(HG_gene_avg[sig_dif_abs_sorted[i][0]])
	sig_genes_names.append(sig_dif_abs_sorted[i][0])

##############################################################
###
###
### Nearest Neighbors
###
###
##############################################################

'''
#make a list of the columns that are part of the samples_to_train
samples_columns = []
for sample in samples_to_train:
	samples_columns.append(firstRow.index(sample))
'''

#data -> samples are the rows, not the genes
X = []
y = []

#initialize all the lists for each sample
'''
for samp in samples_to_train:
	X.append([])
	if samp in LG_set:
		y.append(0)
	else:
		y.append(1)
'''

for i in set_1:
	X.append([])
	y.append(1)
for i in set_1:
	X.append([])
	y.append(0)

with open(data_metastatic, 'rU') as f:
		reader = csv.reader(f, delimiter=' ')
		for row in reader:
			if row[0] in sig_genes_names:
				c = 0
				for expr in row:
					if c in set_1:
						X[c-1].append(expr)
					c += 1


with open(data_other, 'rU') as f:
		reader = csv.reader(f, delimiter=' ')
		for row in reader:
			if row[0] in sig_genes_names:
				c = 0
				for expr in row:
					if c in set_1:
						X[c-1+74].append(expr)
					c += 1

f.close()

neigh = KNeighborsRegressor(n_neighbors=4, weights='distance')
neigh.fit(X, y) 



##############################################################
###
###
### Test
###
###
##############################################################

#make a list of the columns that are part of the samples_to_train
#samples_columns = []
#for sample in samples_to_analyze:
#	samples_columns.append(firstRow.index(sample))

#data -> samples are the rows, not the genes
X = []
y = []

for i in set_2:
	X.append([])
	y.append(1)

print len(X)

for i in set_2:
	X.append([])
	y.append(0)

print len(X)


with open(data_metastatic, 'rU') as f:
		reader = csv.reader(f, delimiter=' ')
		for row in reader:
			if row[0] in sig_genes_names:
				c = 0
				for expr in row:
					if c in set_2:

						X[c-75].append(expr)


					c += 1


with open(data_other, 'rU') as f:
		reader = csv.reader(f, delimiter=' ')
		for row in reader:
			if row[0] in sig_genes_names:
				c = 0
				for expr in row:
					if c in set_2:

						X[c-75+25].append(expr)

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
		plt.annotate('o', xy=(x_pos,results[point]), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"),  color='blue')
	elif y[point] == 1:
		plt.annotate('x', xy=(x_pos,results[point]), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"), color='red')
	else:
		plt.annotate('what', xy=(x_pos,results[point]), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"), color='yellow')

	ploted.append(results[point])


y = numpy.ones(numpy.shape(results))   # Make all y values the same
plt.plot(y,results,'_',ms = 20)  # Plot a line at each location specified in a

plt.xticks([])
plt.xlim(1,1.07)
plt.ylim(-0.02,1.03)
plt.title("Nearest Neighbors Metastasis")
plt.ylabel("Scale")
#plt.annotate('HG', xy=(0.8,0.9), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='red')
#plt.annotate('LG', xy=(0.8,0.85), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='blue')

#plt.annotate('Testing Set = set_2', xy=(0.7,0.7), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='maroon')
#plt.annotate('Training Set = set_1', xy=(0.7,0.65), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='maroon')



plt.savefig('k4nn_metastasis.pdf')

print '\a'