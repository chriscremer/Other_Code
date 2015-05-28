
#because
import gs_functions.py


print '\a'

data_file = '~/wrana_data1/ProCoding_all_RPMs_Tx_cleaned_ID.txt'


samples_to_analyze = ['BC060', 'BC005', 'BC030', 'BC026', 'BC050',
						'BC006', 'BC019', 'BC052', 'BC040', 'BC037',
						'BC032', 'BC039', 'BC041', 'BC029',	'BC024', 
						'BC020', 'BC038', 'BC054', 'BC053', 'BC014',
						'BC007', 'BC059', 'BC049', 'BC055', 'BC001',
						'BC048', 'BC047', 'BC051', 'BC033',	'BC063',
						'BC046', 'BC012', 'BC061', 'BC010',	'BC013',
						'BC035', 'BC009', 'BC027', 'BC036',	'BC008',
						'BC031', 'BC064', 'BC034', 'BC042',	'BC044',
						'BC002', 'BC045', 'BC017', 'BC022']


LG_set = ['BC060', 'BC005', 'BC030', 'BC026', 'BC050',
			 'BC006', 'BC019', 'BC052', 'BC040', 'BC037', 
			 'BC032', 'BC039', 'BC041', 'BC029', 'BC024', 
			 'BC020', 'BC038', 'BC054', 'BC053', 'BC014', 
			 'BC007', 'BC059']

HG_set = ['BC010', 'BC013', 'BC035', 'BC009', 'BC027', 
			'BC036', 'BC008', 'BC031', 'BC064', 'BC034', 
			'BC042', 'BC044', 'BC002', 'BC045', 'BC017', 
			'BC022', 'BC046', 'BC012', 'BC061']

numbOfSigGenes = 13

results = []

plt.vlines(1,0.05,1) 


for sample in samples_to_analyze:

	if sample in LG_training_set:
		LG_training_set = LG_training_set.remove(sample)
	if sample in HG_training_set:
		HG_training_set = HG_training_set.remove(sample)


	#to keep track which column each sample is in
	firstRow = gs_functions.firstRowList(data_file)

	##############################################################
	###
	### Get the average and std dev of the LG list
	###
	##############################################################

	LG_gene_avg_std = gs_functions.getAvgStd(LG_training_set, firstRow, data_file)
	LG_gene_avg = LG_gene_avg_std[0]
	LG_gene_std = LG_gene_avg_std[1]

	##############################################################
	###
	### Get the average and std dev of the HG list
	###
	#################################################################

	HG_gene_avg_std = gs_functions.getAvgStd(HG_training_set, firstRow, data_file)
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

	samples = LG_training_set + HG_training_set + sample
	data = gs_functions.getData(samples, data_file, LG_gene_avg, HG_gene_avg, firstRow, sig_genes_names)

	parameters = []

	#start all the parameters as equal
	for i in range(numbOfSigGenes):
		parameters.append(1.0/numbOfSigGenes)

	Score = gs_functions.score(sample, parameters, sig_genes_names, data)

	x_pos = 1.005
	for old_point in a:
		#if they are close together
		if numpy.abs(Score - old_point) < 0.025:
			x_pos += 0.006

	plt.annotate(sample, xy=(x_pos,Score), size='xx-small')

	results.append(Score)


y = numpy.ones(numpy.shape(a))   # Make all y values the same
plt.plot(y,a,'_',ms = 20)  # Plot a line at each location specified in a

plt.xlim(1,1.07)
plt.ylim(0.05,0.4)

plt.savefig('GRI_plot.pdf')

print '\a'



