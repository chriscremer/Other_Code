




from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import pickle


save_dir = home+'/Documents/Grad_Estimators/GMM/'



#Load data 

#Marginal 
data_file = save_dir+ 'data_marginal.p'
with open(data_file, "rb" ) as f:
    all_dict = pickle.load(f)
print ('loaded results from ', data_file)
# for k,v in all_dict.items():
#   print (k)
y = all_dict['losses']
x = all_dict['steps']



#REINFORCE
data_file = save_dir+ 'data_reinforce.p'
with open(data_file, "rb" ) as f:
    all_dict = pickle.load(f)
print ('loaded results from ', data_file)
# for k,v in all_dict.items():
#   print (k)
y2 = all_dict['losses']
# x = all_dict['steps']



#SIMPLAX
data_file = save_dir+ 'data_simplax.p'
with open(data_file, "rb" ) as f:
    all_dict = pickle.load(f)
print ('loaded results from ', data_file)
# for k,v in all_dict.items():
#   print (k)
y3 = all_dict['losses']
# x = all_dict['steps']




rows = 1
cols = 1
fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

col =0
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(x, y, label=r'$MARGINAL$')
ax.plot(x, y2, label=r'$REINFORCE$')
ax.plot(x, y3, label=r'$SIMPLAX$')
# ax.plot(thetas, pz_grad_means, label='reinforce p(z)')
# ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')

ax.legend()
ax.set_xlabel('Steps', size=8, family='serif')
ax.set_ylabel('Loss', size=8, family='serif')

ax.grid(True, alpha=.3)
ax.tick_params(labelsize=7)

save_dir = home+'/Documents/Grad_Estimators/GMM/'
plt_path = save_dir+'gmm_results_plot.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()









