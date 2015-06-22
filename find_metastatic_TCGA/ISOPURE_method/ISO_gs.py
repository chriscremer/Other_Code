

#to import matlab file
import scipy.io
#to plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy

'''
#for names
names = ['BC060', 'BC005', 'BC030', 'BC026', 'BC050',
		 'BC006', 'BC019', 'BC052', 'BC040', 'BC037',
		 'BC032', 'BC039', 'BC041',	'BC029', 'BC024',
		 'BC020', 'BC038', 'BC054',	'BC053', 'BC014',
		 'BC007', 'BC059', 'BC049', 'BC055', 'BC001',
		 'BC048', 'BC047', 'BC051', 'BC033', 'BC063',
		 'BC046', 'BC012', 'BC061', 'BC010', 'BC013',
		 'BC035', 'BC009', 'BC027', 'BC036', 'BC008',
		 'BC031', 'BC064', 'BC034', 'BC042', 'BC044',
		 'BC002', 'BC045', 'BC017', 'BC022']


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

set_1_columns = range(0,49,2)
set_2_columns = range(1,49,2)

tumordata = set_1_columns
normaldata= set_2_columns
'''


##############################################################
###
### Read matlab outputs and calculate scores
###
##############################################################


ISO_gs = []

theta_data = scipy.io.loadmat('output_files/test_1.mat')


#these are the columns of the testing set
#HGLG_indexes_file = scipy.io.loadmat('output_files/set_1_columns.mat')
#HGLG_indexes = HGLG_indexes_file['set_1_columns']

meta_test = range(1,74)
other_test = range(74, 147)

for row in theta_data['theta']:

	score = 0.0
	i = 0
	for proportion in row[:len(meta_test)]:
		#this is referring to the panel/normaldata columns
		#if names[normaldata[i]] in HG_set:

		score += proportion
		i += 1
	ISO_gs.append(score)



##############################################################
###
### PLot
###
##############################################################

plt.vlines(1,0,1) 

ploted_points = []

i = 0
for sample in ISO_gs:

	x_pos = 1.005
	for old_point in ploted_points:
		#if they are close together
		if numpy.abs(sample - old_point) < 0.025:
			x_pos += 0.006

	ploted_points.append(sample)

	if i < 74:
		plt.annotate('m', 
						xy=(x_pos,sample), 
						size='xx-small', 
						bbox=dict(boxstyle="round", fc="0.8"),  
						color='blue')
	elif i > 73:
		plt.annotate('o', 
						xy=(x_pos,sample), 
						size='xx-small', 
						bbox=dict(boxstyle="round", fc="0.8"),  
						color='red')
	else:
		plt.annotate('what', 
						xy=(x_pos,sample), 
						size='xx-small', 
						bbox=dict(boxstyle="round", fc="0.8"),  
						color='yellow')	
	i += 1




correct = 0.0
for i in range(len(ISO_gs)):

	#print 'predict: ' + str(ISO_gs[i]) + ' actual: ' + str(y[i])

	if ISO_gs[i] > 0.5 and i < 74:
		correct += 1.0
	elif ISO_gs[i] < 0.5 and i > 73:
		correct += 1.0


print "Accuracy= " + str(correct/len(ISO_gs)) 


y = numpy.ones(numpy.shape(ISO_gs))   # Make all y values the same
plt.plot(y,ISO_gs,'_',ms = 40)  # Plot a line at each location specified in a

plt.xticks([])
plt.xlim(1,1.07)
plt.ylim(-0.01,1.01)
plt.title("ISOPURE Metastasis")
plt.ylabel("Scale")
#plt.annotate('HG', xy=(0.8,0.9), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='red')
#plt.annotate('LG', xy=(0.8,0.85), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='blue')

#plt.annotate('Testing Set = set_1', xy=(0.7,0.7), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='maroon')
#plt.annotate('Training Set = set_2', xy=(0.7,0.65), xycoords='axes fraction', size='small', bbox=dict(boxstyle="round", fc="0.8"), color='maroon')


plt.savefig('ISO_metastasis.pdf')