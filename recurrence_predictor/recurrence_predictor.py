import grade_scale
import random
import numpy



data_file = 'ProCoding_all_RPMs_Tx_cleaned_ID.txt'





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







#recurred_and_prog
set_1 = ['BC010', 'BC013', 'BC035', 'BC009', 'BC027',
			 'BC036', 'BC008', 'BC031', 'BC064', 'BC034',
			 'BC042', 'BC044', 'BC002', 'BC045', 'BC017', 
			 'BC022', 'BC046', 'BC012', 'BC061']

#recur_no_prog
set_2 = ['BC060', 'BC005', 'BC030', 'BC026', 'BC050',
			'BC006', 'BC019', 'BC052', 'BC040', 'BC037',
			'BC032', 'BC039', 'BC041', 'BC029', 'BC024',
			'BC020', 'BC038', 'BC054', 'BC053', 'BC014',
			'BC007', 'BC059']


##############################################################
###
###
### Change which samples are in the training and test sets
###
###
##############################################################


numbOfIterations = 4

numb_of_set1_test_samples = 2
numb_of_set2_test_samples = 2

#numbOfGenes is the number of genes that will be used in determining if the sample is low or high grade

for numbOfGenes in range(6,30):

	print "gene size: %d" %(numbOfGenes)

	accList = []

	for i in range(numbOfIterations):


		test_set_type1 = []
		training_set_type1 = []
		test_set_type2 = []
		training_set_type2 = []

		used = []
		j = 0
		while j < numb_of_set1_test_samples:
			x = random.randint(0, len(set_1)-1)
			if not(x in used):
				test_set_type1.append(set_1[x])
				j += 1
				used.append(x)

		used = []
		j = 0
		while j < numb_of_set2_test_samples:
			x = random.randint(0, len(set_2)-1)
			if not(x in used):
				test_set_type2.append(set_2[x])
				j += 1
				used.append(x)

		for j in range(len(set_1)):
			if not(set_1[j] in test_set_type1):
				training_set_type1.append(set_1[j])

		for j in range(len(set_2)):
			if not(set_2[j] in test_set_type2):
				training_set_type2.append(set_2[j])

		print "test %d" %(i)

		#print "len of LG train " + str(len(training_set_type1))
		#print "len of HG train " + str(len(training_set_type2))
		#print "len of LG test " + str(len(test_set_type1))
		#print "len of HG test " + str(len(test_set_type2))


		acc = grade_scale.getScoreWithTheseSpecificSets(training_set_type1, training_set_type2, test_set_type1, test_set_type2, numbOfGenes, data_file)
		accList.append(acc)

	print "				%d Genes, Avg Accuracy: %f" %(numbOfGenes, numpy.mean(accList))