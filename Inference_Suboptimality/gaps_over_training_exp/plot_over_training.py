



import time
import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import csv

epochs=['200','400','600']
bounds = ['logpx', 'L_q_star', 'L_q']


values = {}
values['training'] = {}
values['validation'] = {}
for epoch in epochs:
    values['training'][epoch] = {}
    values['validation'][epoch] = {}
# for bound in bounds:
#     for epoch in epochs:
#         values['training'][epoch][bound] = {}
#         values['validation'][epoch][bound] = {}


    
#read values
file_ = home+'/Documents/tmp/inference_suboptimality/over_training_exps/results1.txt'

with open(file_, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        if len(row) and row[0] in ['training','validation']: 
            print (row)
            dataset = row[0]
            epoch = row[1]
            bound = row[2]
            value = row[3]
            values[dataset][epoch][bound] = value

# print (values)

#convert to list
training_plot = {}
for bound in bounds:
    values_to_plot = []
    for epoch in epochs:
        values_to_plot.append(float(values['training'][epoch][bound]))
    training_plot[bound] = values_to_plot 
print (training_plot)


validation_plot = {}
for bound in bounds:
    values_to_plot = []
    for epoch in epochs:
        values_to_plot.append(float(values['training'][epoch][bound]))
    validation_plot[bound] = values_to_plot 
print (validation_plot)


epochs_float = [float(x) for x in epochs]


rows = 1
cols = 2

legend=False

fig = plt.figure(figsize=(8+cols,2+rows), facecolor='white')



# Training set
ax = plt.subplot2grid((rows,cols), (0,0), frameon=False)

ax.set_title('Training Set',family='serif')

# for bound in bounds:
#     ax.plot(epochs_float,training_plot[bound]) #, label=legends[i], c=colors[i], ls=line_styles[i])

ax.fill_between(epochs_float, training_plot['logpx'], training_plot['L_q_star'])
ax.fill_between(epochs_float, training_plot['L_q_star'], training_plot['L_q'])



# Validation set
ax = plt.subplot2grid((rows,cols), (0,1), frameon=False)

ax.set_title('Validation Set',family='serif')

# for bound in bounds:
#     ax.plot(epochs_float,validation_plot[bound]) #, label=legends[i], c=colors[i], ls=line_styles[i])

ax.fill_between(epochs_float, validation_plot['logpx'], validation_plot['L_q_star'])
ax.fill_between(epochs_float, validation_plot['L_q_star'], validation_plot['L_q'])





name_file = home+'/Documents/tmp/inference_suboptimality/over_training_exps/results1.png'
name_file2 = home+'/Documents/tmp/inference_suboptimality/over_training_exps/results1.pdf'
# name_file = home+'/Documents/tmp/plot.png'
plt.savefig(name_file)
plt.savefig(name_file2)
print ('Saved fig', name_file)
print ('Saved fig', name_file2)

fdsa



# # models = [standard,flow1,aux_nf]#,hnf]
# # models = [standard,standard_large_encoder]#, aux_nf aux_large_encoder]#,hnf]
# models = [standard,aux_nf]#, aux_nf aux_large_encoder]#,hnf]


# # model_names = ['standard','flow1','aux_nf','hnf']
# # model_names = ['VAE','NF','Aux+NF']#,'HNF']
# # model_names = ['FFG','Flow']#,'HNF']
# # model_names = ['FFG','Flow']#,'HNF']
# model_names = ['FFG','Flow']#  'aux_nf','aux_large_encoder']#,'HNF']





# # legends = ['IW train', 'IW test', 'AIS train', 'AIS test']
# # legends = ['VAE train', 'VAE test', 'IW train', 'IW test', 'AIS train', 'AIS test']

# legends = ['VAE train', 'VAE test', 'IW train', 'IW test', 'AIS train', 'AIS test']


# colors = ['blue', 'blue', 'green', 'green', 'red', 'red']

# line_styles = [':', '-', ':', '-', ':', '-']




rows = 1
cols = 1

legend=False

fig = plt.figure(figsize=(2+cols,2+rows), facecolor='white')

# Get y-axis limits
min_ = None
max_ = None
for m in range(len(models)):
    for i in range(len(legends)):
        if i == 1:
            continue
        this_min = np.min(models[m][i])
        this_max = np.max(models[m][i])
        if min_ ==None or this_min < min_:
            min_ = this_min
        if max_ ==None or this_max > max_:
            max_ = this_max

min_ -= .1
max_ += .1
# print (min_)
# print (max_)
ylimits = [min_, max_]
xlimits = [x[0], x[-1]]

# fasd

# ax.plot(x,hnf_ais, label='hnf_ais')
# ax.set_yticks([])
# ax.set_xticks([])
# if samp_i==0:  ax.annotate('Sample', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

for m in range(len(models)):
    ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
    for i in range(len(legends)):
        if i == 1:
            continue
        ax.set_title(model_names[m],family='serif')
        ax.plot(x,models[m][i], label=legends[i], c=colors[i], ls=line_styles[i])
        plt.legend(fontsize=6) 
        # ax.set(adjustable='box-forced', aspect='equal')
        plt.yticks(size=6)
        # plt.xticks(x,size=6)
        plt.xticks([400,1300,2200,3100],size=6)

        # ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)
        ax.set_xlim(xlimits)

        ax.set_xlabel('Epochs',size=6)
        if m==0:
          ax.set_ylabel('Log-Likelihood',size=6)


        ax.grid(True, alpha=.1)


# m+=1
# ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
# ax.set_title('AIS_test')
# for m in range(len(models)):
#     ax.plot(x,models[m][3], label=model_names[m])
#     plt.legend(fontsize=4) 
#     plt.yticks(size=6)





# plt.gca().set_aspect('equal', adjustable='box')
name_file = home+'/Documents/tmp/plot.png'
plt.savefig(name_file)
print ('Saved fig', name_file)

name_file = home+'/Documents/tmp/plot.eps'
plt.savefig(name_file)
print ('Saved fig', name_file)


name_file = home+'/Documents/tmp/plot.pdf'
plt.savefig(name_file)
print ('Saved fig', name_file)










