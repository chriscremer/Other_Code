
import numpy as np

def quick_sort(unsorted):

	if len(unsorted)>1:
		# get pivot point
		pivot = unsorted[0]
		leftmark = 1
		rightmark = len(unsorted)-1

		
		# print (unsorted, 'start')

		done = False
		while not done:
			# print (leftmark,rightmark)
			while leftmark <= rightmark and unsorted[leftmark] < pivot:
				leftmark+=1
			while leftmark <= rightmark and unsorted[rightmark] > pivot:
				rightmark-=1

			# print (leftmark,rightmark)
			# print 
			if leftmark > rightmark:
				done =1
			else:
				#swap 
				tmp = unsorted[leftmark]
				unsorted[leftmark] = unsorted[rightmark]
				unsorted[rightmark] = tmp
			# print (unsorted)
			# print (leftmark)
			# print (rightmark)

		#put pivot at split point
		# unsorted[0] = unsorted[rightmark]
		# unsorted[rightmark] = pivot
		# print (rightmark)

		# print (unsorted)
		# # print (len(unsorted))
		# print (unsorted[1:rightmark+1])
		# print (unsorted[rightmark+1:])
		# print ()
		# print (unsorted[1:rightmark+1], 'into left')
		left = quick_sort(unsorted[1:rightmark+1])
		# print (left, 'left')
		# fasds
		# print (unsorted[rightmark+1:], 'into right')
		right = quick_sort(unsorted[rightmark+1:])

		#I think the problem is that the lists are pointers,
			#but not really, because Im printing the values. 
			#Im not changing anything
			#so thers a problem with concat of lists
			#oh it seems python 3 needs so assign the + result to var..
			#try list.extend(list)
			#ah Im working with numpy array

		# print (right, 'right')
		# print (left, 'left')
		# # print (left+right)
		# # print (left.extend(right))
		# print (np.concatenate((left, right), 0))
		# print ([pivot])
		# print (left+[pivot])
		# unsorted = left+[pivot]+right 
		unsorted = np.concatenate((np.concatenate((left, [pivot]), 0), right), 0)
		# print(unsorted, 'end')
		# print ()

	return unsorted



# #test

print ([.6]+[.5]+[])

# # a = [1,2,3] + []
# a = []+[2] +[1,2,3] 
# print (a)

# a = 2
# b = a

# b = 3
# print (a)