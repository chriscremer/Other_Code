



def changer(number):

	number = 3

a = 6

print a

changer(a)

print a

#OUTPUT IS 6 6
#THEREFORE ITS BY VALUE RIGHT?



def list_changer(list1):

	list1 = [1,1,1]

b = [3,4,5]

print b

list_changer(b)

print b



def list_appender(list1):

	list1.append(66)

c = [33,44,55]

print c

list_appender(c)

print c



#SOOOOOO
#IF I redefine the argument/parameter then it creates a new one
#but if im modifying it then its like its passed by reference