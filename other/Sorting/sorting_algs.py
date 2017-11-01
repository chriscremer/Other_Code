


import numpy as np
import time 





def insertion_sort(unsorted):

	sorted_ = unsorted
	return sorted_



def quick_sort(unsorted):

	return sorted_

def merge_sort(unsorted):

	return sorted_

def selection_sort(unsorted):

	return sorted_

def heap_sort(unsorted):

	return sorted_







if __name__ == "__main__":


	# Make list
	unsorted = np.random.rand(100000)
	print (unsorted.shape)


	# Test python sort
	start = time.time()
	sorted_ = sorted(unsorted)
	# print (sorted_)
	print('Elapsed time:', time.time() - start)


	#Test insetion_sort
	start = time.time()
	sorted_1 = insertion_sort(unsorted)
	print('Elapsed time:', time.time() - start)
	#Check to see if correct
	print (np.all(sorted_ == sorted_1))
