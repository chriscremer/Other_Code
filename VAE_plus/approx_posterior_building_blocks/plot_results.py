


import numpy as np

from os.path import expanduser
home = expanduser("~")
import matplotlib.pyplot as plt


#dont do vae, its the same as iwae 1
# and run longer maybe.
#would making smaller z have larger dif?
# scale values? percent difference? or nat difference.

#o next exp:
# only iwae
# 200 epochs
# k=1,2,10,50
# z=2,5,50,100

x = [2,5,50,100]
x = np.array(x)
x = np.log(x)

k1 = [146.385,119.5,96.755,96.5477]
k2 = [143.474,119.15,95.8045,96.0346]
k10 = [135.803,117.728,94.9128,95.3057]
k50 = [133.429,116.303,94.6325,95.0019]


fig = plt.figure(facecolor='white')

plt.plot(x,k1,label='k1')
plt.plot(x,k2,label='k2')
plt.plot(x,k10,label='k10')
plt.plot(x,k50,label='k50')


# plt.plot(x,v1,label='v1')
# plt.plot(x,v5,label='v5')
# plt.plot(x,v10,label='v10')



plt.legend()
plt.grid('off')


plt.show()


















