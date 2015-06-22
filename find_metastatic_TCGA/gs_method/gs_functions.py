
#to read csv
import csv
#to use mean and stdev
import numpy
#to sort dict
import operator
#for printing results
import pprint

##############################################################
###
### To keep track which column each sample is in
###	Input: csv file with all the data
###	Output: List of the first row of the file
###
##############################################################

def firstRowList(csvFile):

	column_tracker = []
	with open(csvFile, 'rU') as f:
		reader = csv.reader(f, delimiter=' ')
		#just need to look at the first row
		for row in reader:
			for i in row:
				column_tracker.append(i)
			break
	f.close()

	#print "columns"
	#print column_tracker

	return column_tracker


##############################################################
###
### Get the average and std dev of a list of samples
### Input: List of the names of the samples, column ordering(first row)
### Output: Tuple(dict(key = geneID, value = gene avg) dict2(key = geneID, value = gene std dev))
###
##############################################################

def getAvgStd(samples_columns, csvFile):


	#this will be a dict where key is GeneID and values are lists of the expressions, used to calc the avg and std dev
	geneExp_dict = {}

	#go though the rows and put in the expressions that are part of the LG_learn
	with open(csvFile, 'rU') as f:
		reader = csv.reader(f, delimiter=' ')
		for row in reader:

			geneExp_dict[row[0]] = []
			for sample in range(len(row)):
				if sample in samples_columns:
					geneExp_dict[row[0]].append(float(row[sample]))
	f.close()

	#now get avg of each gene and put in the dict
	geneAvg_dict = {}
	for key in geneExp_dict:
		geneAvg_dict[key] = numpy.mean(geneExp_dict[key])

	#now get std dev of each LG gene and put in the dict
	geneStd_dict = {}
	for key in geneExp_dict:
		geneStd_dict[key] = numpy.std(geneExp_dict[key])


	return (geneAvg_dict, geneStd_dict)




##############################################################
###
###	Calculate significant difference, get absolute value, sort
### Input: LG gene avg dict, HG gene avg dict
###	Output: list of tuples (geneID, sig_dif) ordered from largest sig_dif to smallest
###
##############################################################

def sigAvgDif(LGavg, HGavg):

	#dict to store the geneID and its sig dif value -> (LG_avg - HG_avg) / (LG_avg + HG_avg)/2
	sig_dif = {}

	for key in LGavg:

		if (LGavg[key] < 0.9 or HGavg[key] < 0.9):
			sig_dif[key] = 0
		else:
			sig_dif[key] = (LGavg[key] - HGavg[key]) / ((LGavg[key] + HGavg[key])/2)

	#take absolute value
	sig_dif_abs = {}
	for key in sig_dif:
		sig_dif_abs[key] = numpy.abs(sig_dif[key])

	#makes a list of tuples (geneID, sig_dif) ordered from largest sig_dif to smallest
	sig_dif_abs_sorted = sorted(sig_dif_abs.items(), key=operator.itemgetter(1), reverse=True)

	return sig_dif_abs_sorted


##############################################################
###
### getData
### get gene data for the sig genes for each sample in the list
###	the data is modified so that it is between 0-1 
###	also, normalized to the low and high grade averages
##############################################################

def getData(sample_column, csvFile, LG_gene_avg, HG_gene_avg, sig_genes_names):

	data = {}
	
	#for sample in sample_columns:
	with open(csvFile, 'rU') as f:
	 	#CHANGE from grade score: I added delimeter
		reader = csv.reader(f, delimiter=' ')

		foundCount = 0
		#data[sample] = {}
		for row in reader:

			if row[0] in sig_genes_names:	
				data[row[0]] = row[sample_column]
				foundCount+=1
				if foundCount == len(sig_genes_names):
					break
		f.close()
	

	#(LG - gene) / (LG - HG)
	#geneScore = {}
	#for sample in samples:
		#geneScore[sample] = []

		#print "		getData data[sample] " + str(len(data[sample]))
	geneScore = {}
	for gene in data:
		geneScore[gene] = (float(LG_gene_avg[gene]) - float(data[gene])) / (float(LG_gene_avg[gene]) - float(HG_gene_avg[gene]))
	#gene score is a dict with key as sample name, and value a list of tuples with gene name and its value



	for gene in geneScore:
		if geneScore[gene] < 0:
			geneScore[gene] = 0
		if geneScore[gene] > 1:
			geneScore[gene] = 1


	return geneScore


##############################################################
###
### Score 
### Input: list of samples to score, csv file, LG_gene_avg, HG_gene_avg
###	Output: dict of sample id and score
###
##############################################################

def score(sig_genes_names, data):

	#geneScore = data

	#initialize the scores to 0, so I can add to them
	#Score = {}
	#for sample in samples:
	#	Score[sample] = 0


	#print "sig_genes_names " + str(len(sig_genes_names))
	#print "parameters " + str(len(parameters))

	#for sample in samples:

		#print "geneScore[sample] " + str(len(geneScore[sample]))

		#pprint.pprint(geneScore[sample])

	Score = 0

	for gene in sig_genes_names:

		

		#Score[sample] = Score[sample] + ((1.0/len(sig_genes_names))* geneScore[sample][i][1])

		#x = Score[sample]
		#y = parameters[i]
		z = data[gene]
		Score += (1.0/len(sig_genes_names))*(z)

		#Score[sample] = Score[sample] + (parameters[i] * geneScore[sample][i][1])


	return Score


##############################################################
###
### Test Net Score (LG score + (1-HG score)/ #samples)
### Input: LG scores, HG scores
###	Output: test net score
### THIS IS NOT USED BECAUSE ITS NOT VERY INFORMATIVE
##############################################################

def testNetScore(LGscores, HGscores):
	testNetScore = 0
	for sample in LGscores:
		testNetScore = testNetScore + LGscores[sample]

	for sample in HGscores:
		testNetScore = testNetScore + (1 - HGscores[sample])

	return testNetScore/(len(LGscores)+len(HGscores))


##############################################################
###
### True positive + True negative / Number of samples = Accuracy
### Input: LG scores, HG scores
###	Output: Accuracy
###
##############################################################

def accuracy(LGscores, HGscores):

	numbTP = 0
	numbTN = 0
	numbSamples = len(LGscores) + len(HGscores)

	for sample in LGscores:
		if LGscores[sample] < 0.5:
			numbTP +=1

	for sample in HGscores:
		if HGscores[sample] > 0.5:
			numbTN +=1

	return (numbTP + numbTN) / float(numbSamples)



##############################################################
###
### 
### Hypothesis
###	same as score I think but only for one sample
###
##############################################################
"""
def hypothesis(parameters, sampleExps):


	hypo = 0
	for i in range(len(parameters)):
		hypo += parameters[i]*sampleExps[i][1]

	return hypo
"""

##############################################################
###
### 
### Cost Derivative
###	
###
##############################################################
"""
def costDerivative(csvFile, parameters, sig_genes_names, LG_training_set, HG_training_set, data, parameter):

	LG_sum = 0
	for sample in LG_training_set:
		sampleExps = data[sample]
		LG_sum += (hypothesis(parameters, sampleExps) - 0)*data[sample][parameter][1]


	HG_sum = 0
	for sample in HG_training_set:
		sampleExps = data[sample]
		HG_sum += (hypothesis(parameters, sampleExps) - 1)*data[sample][parameter][1]

	#print 'costDerivative: ' + str(LG_sum + HG_sum)

	return LG_sum + HG_sum
"""
##############################################################
###
### 
### End
###	
###
##############################################################