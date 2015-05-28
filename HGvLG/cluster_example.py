

from sklearn.cluster import KMeans

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''
points = [[10,11], [12,11], [13,14], [15,7], [5,5], [10,18], [20,12], 
			[50,41], [55,44], [55,55], [60,51], [50,51], [60,55], [46,59], 
			[100,110], [103,101], [99,111], [98,106], [105,94], [108,82], [104,116]]
'''
points = [[70,71], [72,79], [83,74], [75,87], [85,75], [70,88], [70,82], 
			[50,41], [55,44], [55,55], [60,51], [50,51], [60,55], [46,59], 
			[100,110], [103,101], [99,111], [98,106], [105,94], [108,82], [104,116],
			[60,100], [50,100], [65,80], [65,90]]


scores = []



numb_of_clusters = 7

for i in range(1,numb_of_clusters):


	cluster1 = KMeans(n_clusters=i)
	cluster1.fit(points)
	scores.append(cluster1.inertia_)
	#print cluster1.labels_
	#print cluster1.cluster_centers_



	plt.figure(i)


	index = 0
	for samp in points:
		if cluster1.labels_[index] == 0:
			plt.scatter(samp[0], samp[1], c='blue', s=40)
		elif cluster1.labels_[index] == 1:
			plt.scatter(samp[0], samp[1], c='yellow', s=40)
		elif cluster1.labels_[index] == 2:
			plt.scatter(samp[0], samp[1], c='red', s=40)
		elif cluster1.labels_[index] == 3:
			plt.scatter(samp[0], samp[1], c='grey', s=40)
		elif cluster1.labels_[index] == 4:
			plt.scatter(samp[0], samp[1], c='orange', s=40)
		else:
			plt.scatter(samp[0], samp[1], c='pink', s=40)
		index += 1

	for center in cluster1.cluster_centers_:
		plt.scatter(center[0], center[1], c='green', marker='p', s=50)


	plt.title('K-Means Example\nClusters= ' + str(i))
	plt.ylabel("y")
	plt.xlabel("x")

	plt.savefig('kmeans_example_scatter_' + str(i) + '.pdf')





plt.figure(numb_of_clusters+1)

plt.plot(range(1,numb_of_clusters), scores)
#need to change
plt.title('Cluster Example Plot')

plt.ylabel("Inertia")
plt.xlabel("# of clusters")


plt.savefig('kmeans_example_plot.pdf')

slopes = []
previous = scores[0]
for inertia in scores[1:]:
	slope = inertia - previous
	slopes.append(slope)
	previous = inertia

previous = slopes[0]
for slope in slopes[1:]:
	print previous/slope
	previous = slope
