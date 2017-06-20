


import numpy as np

from os.path import expanduser
home = expanduser("~")
import matplotlib.pyplot as plt


ks = []
zs = []
results_test = []
results_train = []


#Parse file

# file_ = home+ '/Documents/tmp/'+ 'experiment_2017-06-14_131123.txt'
file_ = home+ '/Downloads/'+ 'experiment_2017-06-18_190607.txt'

with open(file_,'r') as f:
	for line in f:
		if 'IWAE' in line:
			aa = line.split('.')[0]
			#which k?
			k = aa.split('_')[1]
			if k not in ks:
				ks.append(k)
				results_test.append([])
				results_train.append([])

			print k
			index_ = ks.index(k)

			z = int(aa.split('_')[2].replace('z',''))
			if z not in zs:
				zs.append(z)

			#get result
			count = 0
			for line in f:
				if count==2:
					results_test[index_].append(-float(line.split(' ')[3]))
					results_train[index_].append(-float(line.split(' ')[8].replace('\n','')))
				count+=1
				if count>4:
					break


				# add result to list
			
				# add restult ot list
		# a = f.readline()
		# print a


print ks
print results_test
print results_train
print zs












#dont do vae, its the same as iwae 1
# and run longer maybe.
#would making smaller z have larger dif?
# scale values? percent difference? or nat difference.

#o next exp:
# only iwae
# 200 epochs
# k=1,2,10,50
# z=2,5,50,100

# x = [2,5,50,100]
x = np.array(zs)
# x = np.log(x)

# k1 = [146.385,119.5,96.755,96.5477]
# k2 = [143.474,119.15,95.8045,96.0346]
# k10 = [135.803,117.728,94.9128,95.3057]
# k50 = [133.429,116.303,94.6325,95.0019]

color_list = ['blue', 'red', 'green', 'purple']

fig = plt.figure(facecolor='white')

for i in range(len(ks)):
	plt.plot(x,results_test[i],label=ks[i]+'_test', c=color_list[i],linewidth=2)
	plt.plot(x,results_train[i],'--',label=ks[i]+'_train',c=color_list[i])



# plt.plot(x,k2,label='k2')
# plt.plot(x,k10,label='k10')
# plt.plot(x,k50,label='k50')


# plt.plot(x,v1,label='v1')
# plt.plot(x,v5,label='v5')
# plt.plot(x,v10,label='v10')



plt.legend()
plt.grid('off')


plt.show()


















