
#because
import gs_functions
#to calc mean
import numpy
#to plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


print '\a'

#data_file = 'ProCoding_all_RPMs_Tx_cleaned_ID.txt'
data_metastatic = '/data1/morrislab/ccremer/TCGA_data/find_metastatic/output_metastatic_exps.txt'
data_other = '/data1/morrislab/ccremer/TCGA_data/find_metastatic/output_other_exps.txt'


# 2 fold cross validation
# Need to split the samples into two sets then run the same models
# To split, since I want relatively equal LG/HG, I will altenate between sample assignments to each set

#here ill be using set 2 to make the model, and set 1 to test, ie plotting set 1


set_1 = range(1, 75)
#set_2 = range(75, 147)
set_2 = range(75, 100)
#set_2 = 75-147



#samples_to_analyze = set_2
#samples_to_learn = set_1

numbOfSigGenes = 13

results = []

plt.vlines(1,0,1) 



##############################################################
###
###
### Learn Model
###
###
##############################################################


#to keep track which column each sample is in
#firstRow = gs_functions.firstRowList(data_file)

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

##############################################################
###
### Overlapping std devs
###
##############################################################

'''
sig_genes = []
for gene in sig_dif_abs_sorted[:1000]:
	LG_avg_plus_std = (LG_gene_avg[gene[0]] + LG_gene_std[gene[0]])
	LG_avg_minus_std = (LG_gene_avg[gene[0]] - LG_gene_std[gene[0]])
	HG_avg_plus_std = (HG_gene_avg[gene[0]] + HG_gene_std[gene[0]])
	HG_avg_minus_std = (HG_gene_avg[gene[0]] - HG_gene_std[gene[0]])
	LG_aPs_minus_HG_aMs = LG_avg_plus_std - HG_avg_minus_std
	HG_aPs_minus_LG_aMs = HG_avg_plus_std - LG_avg_minus_std

	if LG_aPs_minus_HG_aMs < 0.9 or HG_aPs_minus_LG_aMs < 0.9:
		sig_genes.append(gene[0])
'''


#print "numb of possible sig genes = " + str(len(sig_genes))
sig_genes_names = []

for i in range(0, numbOfSigGenes):
	print sig_dif_abs_sorted[i][0] + ' ' + str(sig_dif_abs_sorted[i][1]) + ' ' + str(LG_gene_avg[sig_dif_abs_sorted[i][0]]) + ' ' + str(HG_gene_avg[sig_dif_abs_sorted[i][0]])
	sig_genes_names.append(sig_dif_abs_sorted[i][0])

#sig_genes_names = sig_dif_abs_sorted[:numbOfSigGenes]

print 'done\n'


##############################################################
###
###
### Test Model
###
###
##############################################################

for sample in set_2:

	data1 = gs_functions.getData(sample, data_metastatic, LG_gene_avg, HG_gene_avg, sig_genes_names)

	#parameters = []

	#start all the parameters as equal
	#for i in range(numbOfSigGenes):
	#	parameters.append(1.0/numbOfSigGenes)


	#pretend_list = [sample]

	Score = gs_functions.score(sig_genes_names, data1)

	x_pos = 1.005
	for old_point in results:
		#if they are close together
		if numpy.abs(Score - old_point) < 0.001:
			x_pos += 0.006

	plt.annotate(sample, xy=(x_pos,Score), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"), color='black')

	results.append(Score)


print 'here2'

for sample in set_2:

	data1 = gs_functions.getData(sample, data_other, LG_gene_avg, HG_gene_avg, sig_genes_names)

	#parameters = []

	#start all the parameters as equal
	#for i in range(numbOfSigGenes):
	#	parameters.append(1.0/numbOfSigGenes)


	#pretend_list = [sample]

	Score = gs_functions.score(sig_genes_names, data1)

	x_pos = 1.005
	for old_point in results:
		#if they are close together
		if numpy.abs(Score - old_point) < 0.001:
			x_pos += 0.006

	plt.annotate(sample, xy=(x_pos,Score), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"), color='yellow')

	results.append(Score)



print "here3"


y = numpy.ones(numpy.shape(results))   # Make all y values the same
plt.plot(y,results,'_',ms = 20)  # Plot a line at each location specified in a

plt.xticks([])
plt.xlim(1,1.07)
plt.ylim(0.65,0.75)
plt.title("Metastasis")
plt.ylabel("Scale")
plt.annotate('meta', xy=(0.8,0.9), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='black')
plt.annotate('other', xy=(0.8,0.85), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='yellow')

plt.annotate('Number of Genes = ' + str(numbOfSigGenes), xy=(0.7,0.75), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='maroon')

#plt.annotate('Testing Set = set_2', xy=(0.7,0.7), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='maroon')
#plt.annotate('Training Set = set_1', xy=(0.7,0.65), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='maroon')



plt.savefig('metastatis.pdf')

print '\a'



