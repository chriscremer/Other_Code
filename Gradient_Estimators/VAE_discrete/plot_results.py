




from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import pickle






#Load data 

#RELAX
data_file = home+'/Documents/VAE2_exps/vae_discrete_relax_trainjustencoder2_fixed2/results.pkl'
with open(data_file, "rb" ) as f:
    all_dict = pickle.load(f)
print ('loaded results from ', data_file)
# for k,v in all_dict.items():
# 	print (k)
y = all_dict['all_train_bpd']
x = all_dict['all_steps']

#SIMPLAX
data_file = home+'/Documents/VAE2_exps/vae_discrete_simplax_trainjustencoder2/results.pkl'
with open(data_file, "rb" ) as f:
    all_dict = pickle.load(f)
print ('loaded results from ', data_file)
# for k,v in all_dict.items():
# 	print (k)
y2 = all_dict['all_train_bpd']



rows = 1
cols = 1
fig = plt.figure(figsize=(10+cols,4+rows), facecolor='white') #, dpi=150)

col =0
row = 0
ax = plt.subplot2grid((rows,cols), (row,col), frameon=False, colspan=1, rowspan=1)

ax.plot(x, y, label=r'$RELAX$')
ax.plot(x, y2, label=r'$SIMPLAX$')
# ax.plot(thetas, pz_grad_means, label='reinforce p(z)')
# ax.plot(thetas, np.ones(len(thetas))*dif, label='expected')

ax.legend()
ax.set_xlabel('Steps', size=8, family='serif')
ax.set_ylabel('BPD', size=8, family='serif')

ax.grid(True, alpha=.3)
ax.tick_params(labelsize=7)

save_dir = home+'/Documents/Grad_Estimators/new/'
plt_path = save_dir+'cifar_exp.png'
plt.savefig(plt_path)
print ('saved training plot', plt_path)
plt.close()









