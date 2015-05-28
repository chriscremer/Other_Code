#to read csv
import csv
#to calc PCC
from scipy.stats.stats import pearsonr  
#to calc mean
import numpy
#to plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print '\a'

data_file = 'ProCoding_all_RPMs_Tx_cleaned_ID.txt'

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


boundary_grade = ['BC049', 'BC055', 'BC047', 'BC051', 'BC063', 'BC001', 'BC048', 'BC033']


HG_set = ['BC010', 'BC013', 'BC035', 'BC009', 'BC027', 
			'BC036', 'BC008', 'BC031', 'BC064', 'BC034', 
			'BC042', 'BC044', 'BC002', 'BC045', 'BC017', 
			'BC022', 'BC046', 'BC012', 'BC061']


a = []

plt.vlines(1,0.05,0.4)  



for sample in samples_to_analyze:

	sample_1_name = sample
	sample_1_PCCs = []

	LG_set = ['BC060', 'BC005', 'BC030', 'BC026', 'BC050', 
			'BC006', 'BC019', 'BC052', 'BC040', 'BC037', 
			'BC032', 'BC039', 'BC041', 'BC029', 'BC024', 
			'BC020', 'BC038', 'BC054', 'BC053', 'BC014', 
			'BC007', 'BC059']

	#for color
	c = 0

	if sample in LG_set:
		LG_set.remove(sample)
		c = 1

	for LG_sample in LG_set:

		sample_2_name = LG_sample
		sample_1_list = []
		sample_2_list = []

		with open(data_file, 'rU') as f:
			reader = csv.reader(f, delimiter='\t')
			b = 0
			for row in reader:
				#header
				if b == 0:
					sample_1_column = row.index(sample_1_name)
					sample_2_column = row.index(sample_2_name)
					b = b+1
					continue
				else:
					sample_1_list.append(float(row[sample_1_column]))
					sample_2_list.append(float(row[sample_2_column]))

			f.close()

		PCC = pearsonr(sample_1_list, sample_2_list)
		sample_1_PCCs.append(PCC[0])

	point = 1 - numpy.mean(sample_1_PCCs)
	
	x_pos = 1.005
	for old_point in a:
		#if they are close together
		if numpy.abs(point - old_point) < 0.025:
			x_pos += 0.0055


	if c == 1:
		plt.annotate(sample, xy=(x_pos, point), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"), color='blue')
	elif sample in HG_set:
		plt.annotate(sample, xy=(x_pos, point), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"), color='red')
	else:
		plt.annotate(sample, xy=(x_pos, point), size='xx-small', bbox=dict(boxstyle="round", fc="0.8"), color='yellow')


	#plt.annotate(sample, xy=(x_pos,point), size='xx-small')
	#plt.arrow(x_pos,point, 1-x_pos, 0, width=0.0005)
	#maybe try box
	#bbox=dict(boxstyle="round", fc="0.8"), 

	a.append(point)


y = numpy.ones(numpy.shape(a))   # Make all y values the same
plt.plot(y,a,'_',ms = 20)  # Plot a line at each location specified in a

plt.xticks([])
plt.xlim(1,1.07)
plt.ylim(0.05,0.4)
plt.title("Grade Risk Index")
plt.ylabel("1-average of PCC (against each LG)")
plt.annotate('HG', xy=(0.9,0.9), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='red')
plt.annotate('Boundary', xy=(0.9,0.85), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='yellow')
plt.annotate('LG', xy=(0.9,0.8), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='blue')

plt.savefig('GRI_plot.pdf')

print '\a'