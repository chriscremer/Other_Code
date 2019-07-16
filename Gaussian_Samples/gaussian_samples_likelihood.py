
from os.path import expanduser
home = expanduser("~")


# I think the point of this is to show that you dont sample high likelihood samples
# most samples are from the 'bubble'
# thats because in high dimensios, most of the mass is not in the center


import numpy as np

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def logprob(z):

	return np.sum(-.5*(z**2 + np.log(2*np.pi)))

def dist(z):

	return np.sqrt(np.sum(z**2))

D = 2
N = 1000
dists2 = []
LLs2 = []
for i in range(N):
	z = np.random.normal(loc=0, scale=1, size=D)
	d = dist(z)
	LL = logprob(z)
	dists2.append(d)
	LLs2.append(np.exp(LL))

print (np.mean(dists2))

max_d = np.max(dists2)
incs = np.linspace(0.,2.,num=50)
dists = []
LLs = []
for inc in incs:
	vec = np.zeros(D) + inc
	dists.append(dist(vec))
	LLs.append(np.exp(logprob(vec)))



rows = 1
cols = 1
fig = plt.figure(figsize=(6+cols,4+rows), facecolor='white', dpi=150)	
ax = plt.subplot2grid((rows,cols), (0,0), frameon=False, colspan=1, rowspan=1)
max_LL = np.max(LLs)
plt.hist(dists2, weights=max_LL*np.ones(N)/N)
plt.plot(dists, LLs)
# plt.scatter(dists2, LLs2)



ax.set_title('D='+str(D))
ax.set_ylabel('p(z)', size=6, family='serif')
ax.set_xlabel('Distance', size=6, family='serif')
ax.tick_params(labelsize=6)

plt.show()
fsadssa


plt_path = home+'/Downloads/plt.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()






fasfdsfsaf









D = 10
N = 100
for i in range(N):
	z = np.random.normal(loc=0, scale=1, size=D)
	d = dist(z)
	LL = logprob(z)

print(np.random.normal(loc=0, scale=1, size=D).shape)
print (logprob(np.random.normal(loc=0, scale=1, size=1000)).shape)
fass







