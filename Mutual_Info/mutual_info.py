

from os.path import expanduser
home = expanduser("~")

import numpy as np

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

# import scipy
# print (scipy.__version__)



def get_MI(p, D):
	MI = (-D/2) * np.log(1-p**2)
	return MI

def get_p(MI, D):
	p = np.sqrt(-(np.exp((-2*MI)/D)-1))
	return p

print(get_MI(p=.9, D=2))
print(get_p(MI=10, D=20))
fsdf








rows = 1
cols = 1
fig = plt.figure(figsize=(6+cols,4+rows), facecolor='white', dpi=150)	
ax = plt.subplot2grid((rows,cols), (0,0), frameon=False, colspan=1, rowspan=1)


# x = np.linspace(0, 5, 10, endpoint=False)
# y = multivariate_normal.pdf(x, mean=2.5, cov=0.5)
# plt.plot(x, y)

x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y

mean = [0., 0.]
cov = [[1.0, .9], [.9, 1.0]]
rv = multivariate_normal(mean, cov)
plt.contour(x, y, rv.pdf(pos), cmap='Blues')

samp = rv.rvs(size=1000)
print (samp.shape)
plt.scatter(samp[:,0], samp[:,1], alpha=.3, s=5)






ax.axis('equal')
plt.show()
fsadssa


plt_path = home+'/Downloads/plt.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()










