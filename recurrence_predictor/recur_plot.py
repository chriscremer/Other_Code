
#because
import gs_functions
#to calc mean
import numpy
#to plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


print '\a'

data_file = 'ProCoding_all_RPMs_Tx_cleaned_ID.txt'


recur_with_prog = ['Sample_3_1_775_1', 'Sample_1_1_298_1', 'Sample_1_129_1', 'Sample_02_713_1', 'Sample_00_994_01',
			'Sample_00_905_01', 'Sample_01_711_1', 'Sample_01_779_1', 'Sample_2_747_1']

no_recur = ['Sample_3_1_230_1', 'Sample_2_1_146_1',  'Sample_2_1_140_1', 'Sample_6_1_524_1', 'Sample_1_1_095_1', 'Sample_2_1_034_1',
 			'Sample_1_1_335_1', 'Sample_03_359_01', 'Sample_1_165_1', 'Sample_00_3943_1', 'Sample_01_148_01', 'Sample_02_459_01', 
 			'Sample_07_953_1', 'Sample_02_089_1', 'Sample_00_419_01',  'Sample_00_131_1', 'Sample_01_036_1', 
 			'Sample_3_512_1', 'Sample_02_133_1']

recur_no_prog = ['Sample_3_1_129_1', 'Sample_8_1_135_1', 'Sample_2_3_025_1', 'Sample_2_1_143_1', 
				'Sample_04_937_1', 'Sample_570_01', 'Sample_00_801_1', 'Sample_4_565_1']


samples_to_analyze = recur_with_prog + no_recur + recur_no_prog

numbOfSigGenes = 13

results = []

plt.vlines(1,0,1) 


for sample in samples_to_analyze:


	print sample
	print len(recur_with_prog)

	LG_set = ['Sample_3_1_230_1', 'Sample_2_1_146_1',  'Sample_2_1_140_1', 'Sample_6_1_524_1', 'Sample_1_1_095_1', 'Sample_2_1_034_1',
 			'Sample_1_1_335_1', 'Sample_03_359_01', 'Sample_1_165_1', 'Sample_00_3943_1', 'Sample_01_148_01', 'Sample_02_459_01', 
 			'Sample_07_953_1', 'Sample_02_089_1', 'Sample_00_419_01',  'Sample_00_131_1', 'Sample_01_036_1', 
 			'Sample_3_512_1', 'Sample_02_133_1']

	HG_set = ['Sample_3_1_775_1', 'Sample_1_1_298_1', 'Sample_1_129_1', 'Sample_02_713_1', 'Sample_00_994_01',
			'Sample_00_905_01', 'Sample_01_711_1', 'Sample_01_779_1', 'Sample_2_747_1']

	c = 0

	if sample in LG_set:
		LG_set.remove(sample)
		c = 1

	if sample in HG_set:
		HG_set.remove(sample)
		c = 2

	#to keep track which column each sample is in
	firstRow = gs_functions.firstRowList(data_file)

	##############################################################
	###
	### Get the average and std dev of the LG list
	###
	##############################################################

	LG_gene_avg_std = gs_functions.getAvgStd(LG_set, firstRow, data_file)
	LG_gene_avg = LG_gene_avg_std[0]
	LG_gene_std = LG_gene_avg_std[1]

	##############################################################
	###
	### Get the average and std dev of the HG list
	###
	#################################################################

	HG_gene_avg_std = gs_functions.getAvgStd(HG_set, firstRow, data_file)
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

	sig_genes_names = sig_genes[:numbOfSigGenes]

	##############################################################
	###
	###
	### Score
	###
	###
	##############################################################

	samples = LG_set + HG_set
	samples.append(sample)
	data = gs_functions.getData(samples, data_file, LG_gene_avg, HG_gene_avg, firstRow, sig_genes_names)

	parameters = []

	#start all the parameters as equal
	for i in range(numbOfSigGenes):
		parameters.append(1.0/numbOfSigGenes)


	pretend_list = [sample]

	Score = gs_functions.score(pretend_list, parameters, sig_genes_names, data)

	x_pos = 1.005
	for old_point in results:
		#if they are close together
		if numpy.abs(Score[sample] - old_point) < 0.025:
			x_pos += 0.006

	if c == 1:
		plt.annotate(sample, xy=(x_pos,Score[sample]), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"),  color='blue')
	elif c == 2:
		plt.annotate(sample, xy=(x_pos,Score[sample]), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"), color='red')
	else:
		plt.annotate(sample, xy=(x_pos,Score[sample]), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"), color='yellow')

	results.append(Score[sample])

	print Score[sample]


y = numpy.ones(numpy.shape(results))   # Make all y values the same
plt.plot(y,results,'_',ms = 20)  # Plot a line at each location specified in a

plt.xticks([])
plt.xlim(1,1.07)
plt.ylim(0,1)
plt.title("Recurrence Plot")
plt.ylabel("Score (1 = Recur, 0 = No Recur)")
plt.annotate('Recur + Prog', xy=(0.8,0.9), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='red')
plt.annotate('Recur no Prog', xy=(0.8,0.85), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='yellow')
plt.annotate('No Recur', xy=(0.8,0.8), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='blue')

plt.savefig('recur_plot.pdf')

print '\a'



