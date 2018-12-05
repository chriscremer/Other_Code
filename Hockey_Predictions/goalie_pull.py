

import numpy as np
import matplotlib.pyplot as plt


# data from: http://people.math.sfu.ca/~tim/papers/goalie.pdf



#reult after 20k: 
aaa = [0.1455, 0.149, 0.1485, 0.1505, 0.1502, 
0.1523, 0.1517, 0.155, 0.1544, 0.1598, 0.1561, 0.1609, 
0.1577, 0.1611, 0.1577, 0.1618, 0.1608, 0.1612, 0.1604, 
0.1612, 0.1678, 0.1632, 0.1637, 0.1649, 0.1679, 0.1638, 
0.1672, 0.1695, 0.1646, 0.172, 0.1685, 0.1681, 0.1681, 
0.1678, 0.1682, 0.1651, 0.1656, 0.1684, 0.1667, 0.1676, 
0.1697, 0.1627, 0.1602, 0.1681, 0.163, 0.1619, 0.1603, 
0.1635, 0.1606, 0.1613, 0.1581, 0.1611, 0.1507, 0.1548, 
0.1527, 0.1497, 0.1433, 0.1495, 0.141, 0.1402, 0.1412]


aaa_smooth = []
# smooth_over = 3
for i in range(len(aaa)):
	if i ==0 or i == len(aaa)-1 or i ==1 or i == len(aaa)-2:
		aaa_smooth.append(aaa[i])
	else:
		aaa_smooth.append((aaa[i-1] + aaa[i] + aaa[i+1] + aaa[i-2] + aaa[i+2])/5.)

print (np.argmax(aaa), np.max(aaa), aaa[np.argmax(aaa)])

aaa = aaa_smooth

# from scipy.interpolate import spline
# xnew = np.linspace(np.arange(0,601,10).min(),np.arange(0,601,10).max(),300)




plt.plot(np.arange(0,601,10), aaa)
plt.xlabel('Seconds')
plt.ylabel('Win Percentage')
plt.show()

fasdf



total_time = 600 # ten minutes

# times = np.linspace(0,1,11)
times = np.arange(0,601,10)

print (times)

#win percentage for different times of goalie pulling
#nah but what about ties. hmmm it works, just put in .5
results = [[] for x in times]

print (results)
print (len(results))
print (len(results[0]))

#scoring rates, so its min per goal 
r1 = 27.4 # 5v5
r2 = 2.85 # against pulled goalie
r3 = 8.5 # with pulled goalie

#number of 10 secs / goal
r1_ = r1*6.
r2_ = r2*6.
r3_ = r3*6.

# goals per 10 secs
p1 = 1. / r1_
p2 = 1. / r2_
p3 = 1. / r3_

print (p1,p2,p3)



n_samples = 20000

for i in range(n_samples):

	# what min to pull goalie
	for j in range(len(times)):

		home_score = 0
		opp_score = 1
		#playing the game 
		for t in range(len(times)):

			if home_score < opp_score and t >= j:
				pulled = 1
			else:
				pulled = 0

			if not pulled:
				home_goal = np.random.binomial(n=1, p=p1, size=1)[0]
				opp_goal = np.random.binomial(n=1, p=p1, size=1)[0]
			else:
				home_goal = np.random.binomial(n=1, p=p3, size=1)[0]
				opp_goal = np.random.binomial(n=1, p=p2, size=1)[0]				

			home_score += home_goal
			opp_score += opp_goal

			# print (home_goal, opp_goal)

		if home_score > opp_score:
			result = 1.
		elif opp_score > home_score:
			result = 0.
		else:
			result = .5

		results[j].append(result)


	# print (results)
	# print (len(results))
	# print (len(results[0]))

	if i % 200 ==0:
		print (i, [np.around(np.mean(x), 4) for x in results])

	# if i % 400 ==0:
	# 	plt.plot([np.around(np.mean(x), 4) for x in results])
	# 	plt.show()




plt.plot([np.around(np.mean(x), 4) for x in results])
plt.show()











