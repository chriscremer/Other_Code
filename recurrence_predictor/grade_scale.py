#to read csv
import csv
#to split strings
import re
#to use mean and stdev
import numpy
#to sort dict
import operator
#for my grade scale functions
import gs_functions
#for printing results
import pprint


def getScoreWithTheseSpecificSets(LG_training_set, HG_training_set, LG_test_set, HG_test_set, numbOfSigGenes, data_file):

	#LG_set = ['BC060', 'BC005', 'BC030', 'BC026', 'BC050', 'BC006', 'BC019', 'BC052', 'BC040', 'BC037', 'BC032', 'BC039', 'BC041', 'BC029', 'BC024', 'BC020', 'BC038', 'BC054', 'BC053', 'BC014', 'BC007', 'BC059']

	#HG_set = ['BC010', 'BC013', 'BC035', 'BC009', 'BC027', 'BC036', 'BC008', 'BC031', 'BC064', 'BC034', 'BC042', 'BC044', 'BC002', 'BC045', 'BC017', 'BC022', 'BC046', 'BC012', 'BC061']


	"""
	#list of LG samples that will be included in the training data
	LG_training_set = ['BC060', 'BC005', 'BC030', 'BC026', 'BC050', 'BC006', 'BC019', 'BC052', 'BC040', 'BC037', 'BC032', 'BC039', 'BC041', 'BC029', 'BC024', 'BC020']

	#list of HG samples that will be included in the training set
	HG_training_set = ['BC010', 'BC013', 'BC035', 'BC009', 'BC027', 'BC036', 'BC008', 'BC031', 'BC064', 'BC034', 'BC042', 'BC044', 'BC002', 'BC045', 'BC017', 'BC022']

	#list of LG samples used to test the accuracy of my score
	LG_test_set = ['BC038', 'BC054', 'BC053', 'BC014', 'BC007', 'BC059']

	#list of HG samples used to test the accuracy of my score
	HG_test_set = ['BC046', 'BC012', 'BC061']

	#list of the boudary grade samples
	boundary_grade = ['BC049', 'BC055', 'BC047', 'BC051', 'BC063', 'BC001', 'BC048', 'BC033']
	"""

	#the csv file with all the data, the first column are the gene names, the first row is the sample names
	#csvFile = 'RPMs_cluster_ProCoding_Tx.csv'
	csvFile = data_file

	#to keep track which column each sample is in
	firstRow = gs_functions.firstRowList(csvFile)


	##############################################################
	###
	###
	### Get the average and std dev of the LG list
	###
	###
	##############################################################

	LG_gene_avg_std = gs_functions.getAvgStd(LG_training_set, firstRow, csvFile)
	LG_gene_avg = LG_gene_avg_std[0]
	LG_gene_std = LG_gene_avg_std[1]


	##############################################################
	###
	###
	### Get the average and std dev of the HG list
	###
	###
	##############################################################


	HG_gene_avg_std = gs_functions.getAvgStd(HG_training_set, firstRow, csvFile)
	HG_gene_avg = HG_gene_avg_std[0]
	HG_gene_std = HG_gene_avg_std[1]



	##############################################################
	###
	###
	### Calculate significant difference, get absolute value, sort
	###
	###
	##############################################################


	#makes a list of tuples (geneID, sig_dif) ordered from largest sig_dif to smallest
	sig_dif_abs_sorted = gs_functions.sigAvgDif(LG_gene_avg, HG_gene_avg)

	#print sig_dif_abs_sorted[:10]

	##############################################################
	###
	###
	### Overlapping std devs, 
	###
	###	One thing that I changed is that I merged the lists, sorted based on std devs, but first time I went in order of sig_dif, so if it doesnt work, i can try changing that
	##############################################################

	method = 0
	if method == 1:

		LG_avg_plus_std = []
		for gene in sig_dif_abs_sorted[:1000]:
			LG_avg_plus_std.append((gene[0], (LG_gene_avg[gene[0]] + LG_gene_std[gene[0]])))

		LG_avg_minus_std = []
		for gene in sig_dif_abs_sorted[:1000]:
			LG_avg_minus_std.append((gene[0], (LG_gene_avg[gene[0]] - LG_gene_std[gene[0]])))

		HG_avg_plus_std = []
		for gene in sig_dif_abs_sorted[:1000]:
			HG_avg_plus_std.append((gene[0], (HG_gene_avg[gene[0]] + HG_gene_std[gene[0]])))

		HG_avg_minus_std = []
		for gene in sig_dif_abs_sorted[:1000]:
			HG_avg_minus_std.append((gene[0], (HG_gene_avg[gene[0]] - HG_gene_std[gene[0]])))


		#LG avg+std minus HG avg-std
		LG_aPs_minus_HG_aMs = []
		for i in range(len(LG_avg_plus_std)):
			LG_aPs_minus_HG_aMs.append((LG_avg_plus_std[i][0], LG_avg_plus_std[i][1] - HG_avg_minus_std[i][1]))

		#HG avg+std minus LG avg-std
		HG_aPs_minus_LG_aMs = []
		for i in range(len(LG_avg_plus_std)):
			HG_aPs_minus_LG_aMs.append((LG_avg_plus_std[i][0], HG_avg_plus_std[i][1] - LG_avg_minus_std[i][1]))


		# I think that when I add these too lists, 
		#the genes are duplicated which leads to the risk of having the same gene twice in sig_genes
		#so I could fix this by checking to make sure its not already in.
		#but for now ill just use the other method.

		comb = LG_aPs_minus_HG_aMs + HG_aPs_minus_LG_aMs
		comb_sorted = sorted(comb, key=operator.itemgetter(1))


		#these are the genes ill be using
		sig_genes = comb_sorted[:numbOfSigGenes]

		#just the names
		sig_genes_names = []
		for name_tuple in sig_genes:
			sig_genes_names.append(name_tuple[0])

	else:

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
				"""
				print LG_avg_plus_std
				print LG_avg_minus_std
				print HG_avg_plus_std
				print HG_avg_minus_std
				print LG_aPs_minus_HG_aMs
				print HG_aPs_minus_LG_aMs
				print ''
				"""
		print ' '
		print len(sig_genes)
		print ' '

		sig_genes_names = sig_genes[:numbOfSigGenes]



	#print sig_genes_names

	##############################################################
	###
	###
	### Gradient Descent to determine value of weights/parameters
	###
	###
	##############################################################
	
	samples = LG_training_set + HG_training_set + LG_test_set + HG_test_set
	data = gs_functions.getData(samples, csvFile, LG_gene_avg, HG_gene_avg, firstRow, sig_genes_names)

	parameters = []

	#start all the parameters as equal
	for i in range(numbOfSigGenes):
		parameters.append(1.0/numbOfSigGenes)
	"""
	# m = numb of samples 
	m = len(LG_training_set) + len(HG_training_set)
	
	#iterations of gadient descent
	for i in range(100):

		tempParameters = []
		for parameter in range(len(parameters)):

			tempParameter = parameters[parameter] - ((1.0/m)*gs_functions.costDerivative(csvFile, parameters, sig_genes_names, LG_training_set, HG_training_set, data, parameter))
			tempParameters.append(tempParameter)

		parameters = tempParameters
	
	#print "1.0/m: " + str(1.0/m)
	#print 'parameters'
	#print parameters
	"""


	##############################################################
	###
	###
	### Score 
	###
	###
	##############################################################


	Score1 = gs_functions.score(LG_training_set, parameters, sig_genes_names, data)

	Score2 = gs_functions.score(HG_training_set, parameters, sig_genes_names, data)

	#score the LG samples 
	Score3 = gs_functions.score(LG_test_set, parameters, sig_genes_names, data)

	#score the HG samples
	Score4 = gs_functions.score(HG_test_set, parameters, sig_genes_names, data)

	#Score5 = gs_functions.score(boundary_grade, csvFile, LG_gene_avg, HG_gene_avg, sig_genes_names, firstRow)

	#testNetScore = gs_functions.testNetScore(Score3, Score4)

	#see how many HG are identified as HG and same for LG
	accuracy = gs_functions.accuracy(Score3, Score4)

	'''
	print 'LG_training_set'
	pprint.pprint(Score1)

	print 'HG_training_set'
	pprint.pprint(Score2)

	print 'LG_test_set'
	pprint.pprint(Score3)

	print 'HG_test_set'
	pprint.pprint(Score4)

	#print 'boundary_grade'
	#pprint.pprint(Score5)
	
'''


	print '%d genes, accuracy= %f' % (numbOfSigGenes, accuracy) 

	return accuracy

##############################################################
###
###
### End
###
###
##############################################################




