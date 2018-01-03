


# a =1
# b=a
# a=3
# print (a,b)

    
def LCS(s1,s2):
    
    lcs_matrix = [[0 for y in s2] for x in s1]
    
    l1 = len(s1)
    l2 = len(s2)
    
    for i in range(l1):
        for j in range(l2):
            
            if s1[i] == s2[j]:
                
                if i == 0 or j == 0:
                    lcs_matrix[i][j] = 1
                else:
                    lcs_matrix[i][j] = 1 + lcs_matrix[i-1][j-1] 
    
    #find largest index
    largest_value = 0
    best_i = 0
    best_j = 0
    for i in range(l1):
        for j in range(l2):
        	if lcs_matrix[i][j] > largest_value:
        		best_j=j
        		best_i=i
        		largest_value=lcs_matrix[i][j]

    lcs = ''
    while(best_i >= 0 and best_j >= 0 and lcs_matrix[best_i][best_j] > 0):
    	lcs = s1[best_i]+lcs
    	best_i -=1
    	best_j -=1

    return lcs











s1 = 'abcdeff'
s2 = 'bcd'

print(LCS(s1,s2))