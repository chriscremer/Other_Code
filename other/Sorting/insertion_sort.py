
import numpy as np

def insertion_sort(unsorted):
	#start sorting the list then insert the next index into that list

	# time: O(n^2)
	# space: O(1)

	#how to deal if goes through whole list
			#fixed setting value after break, so it always does it
	# and if larger than all
			#fixed by having the open_index pointer 

	for i in range(1,len(unsorted)):

		i_value = unsorted[i]

		# print (list(range(i,-1,-1)))
		# print (range(i))
		# print (range(i)[::-1])
		# if i ==20:
		# 	break
		# for j in range(i)[::-1]:
		open_index = i
		for j in range(i-1,-1,-1):

			# print (j)
			if i_value < unsorted[j]:

				#move it to the right
				unsorted[open_index] = unsorted[j]
				open_index = j

				# if j ==0:
				# 	unsorted[j] = i_value

			else:
				break

		unsorted[open_index] = i_value
			# else:
		
				# break
		# print('next')
		# unsorted[j] = i_value

		# print (unsorted[:10])

	return unsorted







