
import numpy as np

def selection_sort(unsorted):
	#go through, find min, swap it with first
	# go though starting at second spot, find min, put it in second spot
	# continue

	# time: O(n^2)
	# space: O(1)



	for i in range(len(unsorted)):

		#find min value
		current_min = unsorted[i]
		current_min_index = i
		for j in range(i,len(unsorted)):

			if current_min > unsorted[j]:

				current_min = unsorted[j]
				current_min_index = j

		# swap	
		tmp = unsorted[i]
		unsorted[i] = unsorted[current_min_index]
		unsorted[current_min_index] = tmp


	return unsorted



