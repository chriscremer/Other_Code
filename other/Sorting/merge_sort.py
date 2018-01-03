





def merge_sort(unsorted):

	# time: O(nlogn)  logn splits and O(n) for merging 
	# space: O(n) for storing the merged list

	if len(unsorted)> 1:

		# print (len(unsorted))
		# if len(unsorted) < 4:
		# 	print (unsorted)

		#split
		mid = int(len(unsorted)/2)
		lefthalf = unsorted[:mid]
		righthalf = unsorted[mid:]

		# print (len(righthalf))
		# fsdf

		lefthalf = merge_sort(lefthalf)
		righthalf = merge_sort(righthalf)
		# merge_sort(lefthalf)
		# merge_sort(righthalf)

		# print (unsorted)
		# print (lefthalf)
		# print (righthalf)

		#merge sorted lists
		# print ('merge')
		new_list = []
		i,j,k = 0,0,0  #pointers

		while j<len(lefthalf) and k < len(righthalf):
			if lefthalf[j] > righthalf[k]:
				# unsorted[i] = righthalf[k]
				new_list.append(righthalf[k])
				k+=1
			else:
				# unsorted[i] = lefthalf[j]
				new_list.append(lefthalf[j])
				j+=1	
			i+=1
		# print (unsorted)
		# print (lefthalf)
		# print (righthalf)
		while j<len(lefthalf):
			# unsorted[i] = lefthalf[j]
			new_list.append(lefthalf[j])
			j+=1	
			i+=1
		# print (unsorted)
		# print (lefthalf)
		# print (righthalf)
		while k<len(righthalf):
			# unsorted[i] = righthalf[k]
			new_list.append(righthalf[k])
			k+=1	
			i+=1

		# print (unsorted)
		# fasd
		return new_list

	return unsorted

# test = [1,2,3]
# a =test

# def funct(thing):

# 	thing[2] = 99


# print (test)
# funct(a)
# print (test)


