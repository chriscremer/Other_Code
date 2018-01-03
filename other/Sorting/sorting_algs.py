


import numpy as np
import time 




from selection_sort import selection_sort

from insertion_sort import insertion_sort

from merge_sort import merge_sort

from quick_sort import quick_sort




if __name__ == "__main__":


	# Make list
	unsorted = np.random.rand(10000)
	print ('List length', unsorted.shape)


	# Test python sort
	print ('\nTesting python sorting algo')
	start = time.time()
	sorted_ = sorted(unsorted)
	# print (sorted_)
	print('Elapsed time:', time.time() - start)


	#Test merge_sort
	print ('\nTesting quick sort algo')
	start = time.time()
	sorted_1 = quick_sort(unsorted)
	print('Elapsed time:', time.time() - start)
	#Check to see if correct
	print (np.all(sorted_ == sorted_1))



	#Test merge_sort
	print ('\nTesting merge sort algo')
	start = time.time()
	sorted_1 = merge_sort(unsorted)
	print('Elapsed time:', time.time() - start)
	#Check to see if correct
	print (np.all(sorted_ == sorted_1))



	# #Test selection_sort
	# print ('\nTesting selection sort algo')
	# start = time.time()
	# sorted_1 = selection_sort(unsorted)
	# print('Elapsed time:', time.time() - start)
	# #Check to see if correct
	# print (np.all(sorted_ == sorted_1))



	# #Test insetion_sort
	# print ('\nTesting insertion sort algo')
	# start = time.time()
	# sorted_1 = insertion_sort(unsorted)
	# print('Elapsed time:', time.time() - start)
	# #Check to see if correct
	# print (np.all(sorted_ == sorted_1))
















