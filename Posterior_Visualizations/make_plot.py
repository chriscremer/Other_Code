








import time
import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# x = [1000,1900,2800]

# # Standard, k10 for AIS
# x = [100,1000,1900,2800]
# IW_train = [-102.99, -92.04, -90.98, -90.4949665833]
# IW_test = [-104.086235809,-93.0277107239,-92.2452793884,-91.9473944092 ]
# AIS_train =[-102.945,-92.2062,-91.219, -90.7987]
# AIS_test=[ -103.968,-92.906,-92.0348,-91.6747]


# # Standard, k100 for AIS
# x = [1000,1900,2800]
# IW_train = [-92.0210244751,-90.9609240723,-90.4828463745]
# IW_test = [-93.0489350891,-92.2423101807,-91.9248104858]
# AIS_train =[-91.9124,-90.8787,-90.4299]
# AIS_test=[-92.6238,-91.7527,-91.3643]




# #Flow1, k100 for AIS

# IW_train = [-91.5039059448,  -90.5030630493,  -89.9896644592]
# IW_test = [-92.4487319946, -91.6231837463, -91.2205012512]
# AIS_train =[-91.4004,  -90.4505, -89.9924]
# AIS_test=[-92.1416, -91.3033, -90.8853] 


# #Aux nf
# IW_train =[-91.187793273925777, -90.245821075439451, -89.789920806884766]
# IW_test =[-92.114274902343752, -91.29430206298828, -90.890433349609381]
# AIS_train =[-91.132156, -90.212952, -89.77301]
# AIS_test=[-91.857346, -90.957596, -90.491089]

 
# #HNF
# IW_train =[-92.031605682373041, -90.975176849365241, -90.529517211914069]
# IW_test =[-92.982969665527349, -92.126790466308591, -91.791965484619141]
# AIS_train =[-91.861977, -90.817436, -90.384499]
# AIS_test =[-92.572243, -91.632591, -91.204674]









# plt.plot(x,IW_train, label='IW_train')
# plt.plot(x,IW_test, label='IW_test')
# plt.plot(x,AIS_train, label='AIS_train')
# plt.plot(x,AIS_test, label='AIS_test')

# plt.legend()

# name_file = home+'/Documents/tmp/plot.png'
# plt.savefig(name_file)
# print ('Saved fig', name_file)









# # Plot AIS plot of each model


# x = [1000,1900,2800]
# standard_ais=[-92.6238,-91.7527,-91.3643]
# standard_iw=[-93.0489350891,-92.2423101807,-91.9248104858]

# flow1_ais = [-92.1416, -91.3033, -90.8853]
# flow1_iw = [-92.4487319946, -91.6231837463, -91.2205012512]

# aux_nf_iw=[-92.114274902343752, -91.29430206298828, -90.890433349609381]
# aux_nf_ais = [-91.857346, -90.957596, -90.491089]


# hnf_iw=[-92.982969665527349, -92.126790466308591, -91.791965484619141]
# hnf_ais = [-92.572243, -91.632591, -91.204674]

# plt.plot(x,standard_ais, label='standard_ais')
# plt.plot(x,flow1_ais, label='flow1_ais')
# # plt.plot(x,standard_iw, label='standard_iw')
# # plt.plot(x,flow1_iw, label='flow1_iw')
# # plt.plot(x,aux_nf_iw, label='aux_nf_iw')
# plt.plot(x,aux_nf_ais, label='aux_nf_ais')

# # plt.plot(x,hnf_iw, label='hnf_iw')
# plt.plot(x,hnf_ais, label='hnf_ais')

# plt.legend()

# name_file = home+'/Documents/tmp/plot.png'
# plt.savefig(name_file)
# print ('Saved fig', name_file)












# # Large N early experiment, lr was still .0001


# x = [1000,1900,2800]


# standard=[[-92.0210244751,-90.9609240723,-90.4828463745],
# [-93.0489350891,-92.2423101807,-91.9248104858],
# [-91.9124,-90.8787,-90.4299],
# [-92.6238,-91.7527,-91.3643]]


# flow1=[
# [-91.5039059448,  -90.5030630493,  -89.9896644592],
# [-92.4487319946, -91.6231837463, -91.2205012512],
# [-91.4004,  -90.4505, -89.9924],
# [-92.1416, -91.3033, -90.8853] ]


# aux_nf = [
# [-91.187793273925777, -90.245821075439451, -89.789920806884766],
# [-92.114274902343752, -91.29430206298828, -90.890433349609381],
# [-91.132156, -90.212952, -89.77301],
# [-91.857346, -90.957596, -90.491089]]



# models = [standard,flow1,aux_nf]#,hnf]
# # model_names = ['standard','flow1','aux_nf','hnf']
# model_names = ['VAE','NF','Aux+NF']#,'HNF']
# # model_names = ['FFG','Flow']#,'HNF']



# legends = ['IW_train', 'IW_test', 'AIS_train', 'AIS_test']



# rows = 1
# cols = 2

# legend=False

# fig = plt.figure(figsize=(8+cols,2+rows), facecolor='white')

# # Get y-axis limits
# min_ = None
# max_ = None
# for m in range(len(models)):
#     for i in range(len(legends)):
#         this_min = np.min(models[m][i])
#         this_max = np.max(models[m][i])
#         if min_ ==None or this_min < min_:
#             min_ = this_min
#         if max_ ==None or this_max > max_:
#             max_ = this_max

# min_ -= .1
# max_ += .1
# # print (min_)
# # print (max_)
# ylimits = [min_, max_]
# # fasd

# # ax.plot(x,hnf_ais, label='hnf_ais')
# # ax.set_yticks([])
# # ax.set_xticks([])
# # if samp_i==0:  ax.annotate('Sample', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

# for m in range(len(models)):
#     ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
#     for i in range(len(legends)):
#         ax.set_title(model_names[m])
#         ax.plot(x,models[m][i], label=legends[i])
#         plt.legend(fontsize=4) 
#         # ax.set(adjustable='box-forced', aspect='equal')
#         plt.yticks(size=6)
#         # ax.set_xlim(xlimits)
#         ax.set_ylim(ylimits)

# m+=1
# ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
# ax.set_title('AIS_test')
# for m in range(len(models)):
#     ax.plot(x,models[m][3], label=model_names[m])
#     plt.legend(fontsize=4) 
#     plt.yticks(size=6)





# # plt.gca().set_aspect('equal', adjustable='box')
# name_file = home+'/Documents/tmp/plot.png'
# plt.savefig(name_file)
# print ('Saved fig', name_file)















# # Large N early experiment, lr was still .0001
# #remove a couple plots


# x = [1000,1900,2800]


# standard=[[-92.0210244751,-90.9609240723,-90.4828463745],
# [-93.0489350891,-92.2423101807,-91.9248104858],
# [-91.9124,-90.8787,-90.4299],
# [-92.6238,-91.7527,-91.3643]]


# flow1=[
# [-91.5039059448,  -90.5030630493,  -89.9896644592],
# [-92.4487319946, -91.6231837463, -91.2205012512],
# [-91.4004,  -90.4505, -89.9924],
# [-92.1416, -91.3033, -90.8853] ]


# aux_nf = [
# [-91.187793273925777, -90.245821075439451, -89.789920806884766],
# [-92.114274902343752, -91.29430206298828, -90.890433349609381],
# [-91.132156, -90.212952, -89.77301],
# [-91.857346, -90.957596, -90.491089]]



# # models = [standard,flow1,aux_nf]#,hnf]
# models = [standard,aux_nf]#,hnf]

# # model_names = ['standard','flow1','aux_nf','hnf']
# # model_names = ['VAE','NF','Aux+NF']#,'HNF']
# model_names = ['FFG','Flow']#,'HNF']



# legends = ['IW_train', 'IW_test', 'AIS_train', 'AIS_test']



# rows = 1
# cols = 2

# legend=False

# fig = plt.figure(figsize=(2+cols,2+rows), facecolor='white')

# # Get y-axis limits
# min_ = None
# max_ = None
# for m in range(len(models)):
#     for i in range(len(legends)):
#         this_min = np.min(models[m][i])
#         this_max = np.max(models[m][i])
#         if min_ ==None or this_min < min_:
#             min_ = this_min
#         if max_ ==None or this_max > max_:
#             max_ = this_max

# min_ -= .1
# max_ += .1
# # print (min_)
# # print (max_)
# ylimits = [min_, max_]
# xlimits = [1000, 3000]

# # fasd

# # ax.plot(x,hnf_ais, label='hnf_ais')
# # ax.set_yticks([])
# # ax.set_xticks([])
# # if samp_i==0:  ax.annotate('Sample', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

# for m in range(len(models)):
#     ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
#     for i in range(len(legends)):
#         ax.set_title(model_names[m])
#         ax.plot(x,models[m][i], label=legends[i])
#         plt.legend(fontsize=5) 
#         # ax.set(adjustable='box-forced', aspect='equal')
#         plt.yticks(size=6)
#         plt.xticks(size=6)
#         # ax.set_xlim(xlimits)
#         ax.set_ylim(ylimits)
#         ax.set_xlim(xlimits)


# # m+=1
# # ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
# # ax.set_title('AIS_test')
# # for m in range(len(models)):
# #     ax.plot(x,models[m][3], label=model_names[m])
# #     plt.legend(fontsize=4) 
# #     plt.yticks(size=6)





# # plt.gca().set_aspect('equal', adjustable='box')
# name_file = home+'/Documents/tmp/plot.png'
# plt.savefig(name_file)
# print ('Saved fig', name_file)


























# # smae as above but small N 


# x = [1000,1900,2800]


# standard=[[-80.628628387451172, -74.229540557861327, -71.979493103027337],
# [-143.82347305297853, -157.52427764892579, -164.08638336181642],
# [-80.592911, -74.224045, -72.014435],
# [-128.38687, -134.6702, -137.37544]]


# flow1=[
# [-79.701681518554693, -74.258941497802738, -72.515675964355466],
# [-138.13904357910155, -149.23984283447265, -154.02070617675781],
# [-79.627747, -74.209457, -72.521675],
# [-125.6359, -130.40797, -132.47397]]


# aux_nf = [
# [-80.063534088134759, -74.791913452148435, -73.034408874511712],
# [-135.5944808959961, -146.04012008666993, -150.7372442626953],
# [-79.960373, -74.743088, -73.02948],
# [-124.04991, -128.41185, -130.45657]]


# hnf=[
# [-98.079388580322259, -79.584146118164057, -75.128403320312501],
# [-130.45183181762695, -144.81117645263672, -156.68782989501952],
# [-96.931862, -79.275116, -74.856613],
# [-125.4861, -128.58835, -131.9249]]


# # models = [standard,flow1,aux_nf]#,hnf]
# models = [standard,aux_nf]#,hnf]

# # model_names = ['standard','flow1','aux_nf','hnf']
# # model_names = ['VAE','NF','Aux+NF']#,'HNF']
# model_names = ['FFG','Flow']#,'HNF']



# legends = ['IW_train', 'IW_test', 'AIS_train', 'AIS_test']



# rows = 1
# cols = 2

# legend=False

# fig = plt.figure(figsize=(2+cols,2+rows), facecolor='white')

# # Get y-axis limits
# min_ = None
# max_ = None
# for m in range(len(models)):
#     for i in range(len(legends)):
#         this_min = np.min(models[m][i])
#         this_max = np.max(models[m][i])
#         if min_ ==None or this_min < min_:
#             min_ = this_min
#         if max_ ==None or this_max > max_:
#             max_ = this_max

# min_ -= .1
# max_ += .1
# # print (min_)
# # print (max_)
# ylimits = [min_, max_]
# xlimits = [1000, 3000]

# # fasd

# # ax.plot(x,hnf_ais, label='hnf_ais')
# # ax.set_yticks([])
# # ax.set_xticks([])
# # if samp_i==0:  ax.annotate('Sample', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

# for m in range(len(models)):
#     ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
#     for i in range(len(legends)):
#         ax.set_title(model_names[m])
#         ax.plot(x,models[m][i], label=legends[i])
#         plt.legend(fontsize=5) 
#         # ax.set(adjustable='box-forced', aspect='equal')
#         plt.yticks(size=6)
#         plt.xticks(size=6)
#         # ax.set_xlim(xlimits)
#         ax.set_ylim(ylimits)
#         ax.set_xlim(xlimits)


# # m+=1
# # ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
# # ax.set_title('AIS_test')
# # for m in range(len(models)):
# #     ax.plot(x,models[m][3], label=model_names[m])
# #     plt.legend(fontsize=4) 
# #     plt.yticks(size=6)





# # plt.gca().set_aspect('equal', adjustable='box')
# name_file = home+'/Documents/tmp/plot.png'
# plt.savefig(name_file)
# print ('Saved fig', name_file)












































# # 2D models


# x = [1000,1900,2800]


# standard=[[-131.5193441772461, -129.75819458007811, -128.81870803833007],
# [-139.47349960327148, -138.94495162963867, -139.06166854858398],
# [-130.28397, -128.3606, -127.40322],
# [-134.69041, -132.63577, -131.53952]]


# flow1=[
# [-131.3821598815918, -130.00654403686522, -129.3363656616211],
# [-139.83494216918945, -140.14312118530273, -140.27652770996093],
# [-130.10603, -128.64423, -127.78947],
# [-135.15401, -133.48634, -132.44528]]


# aux_nf = [
# [-131.13105239868165, -129.38401412963867, -128.27181106567383],
# [-137.33705673217773, -136.68337936401366, -135.93538650512696],
# [-129.73923, -127.75482, -126.6568],
# [-134.1727, -132.67764, -131.63309]]


# hnf=[
# [-130.98559112548827, -129.2897299194336, -128.43326797485352],
# [-138.1710707092285, -138.7076222229004, -139.34092864990234],
# [-129.90315, -127.98631, -126.99387],
# [-134.49527, -132.68562, -131.72183]]


# models = [standard,flow1,aux_nf,hnf]
# model_names = ['standard','flow1','aux_nf','hnf']

# legends = ['IW_train', 'IW_test', 'AIS_train', 'AIS_test']



# rows = 1
# cols = 5

# legend=False

# fig = plt.figure(figsize=(8+cols,2+rows), facecolor='white')




# # ax.plot(x,hnf_ais, label='hnf_ais')
# # ax.set_yticks([])
# # ax.set_xticks([])
# # if samp_i==0:  ax.annotate('Sample', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

# for m in range(len(models)):
#     ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
#     for i in range(len(legends)):
#         ax.set_title(model_names[m])
#         ax.plot(x,models[m][i], label=legends[i])
#         plt.legend(fontsize=4) 
#         # ax.set(adjustable='box-forced', aspect='equal')
#         plt.yticks(size=6)


# m+=1
# ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
# ax.set_title('AIS_test')
# for m in range(len(models)):
#     ax.plot(x,models[m][3], label=model_names[m])
#     plt.legend(fontsize=4) 
#     plt.yticks(size=6)





# # plt.gca().set_aspect('equal', adjustable='box')
# name_file = home+'/Documents/tmp/plot.png'
# plt.savefig(name_file)
# print ('Saved fig', name_file)

















































# # Small N 


# x = [1000,1900,2800]


# standard=[[-80.628628387451172, -74.229540557861327, -71.979493103027337],
# [-143.82347305297853, -157.52427764892579, -164.08638336181642],
# [-80.592911, -74.224045, -72.014435],
# [-128.38687, -134.6702, -137.37544]]


# flow1=[
# [-79.701681518554693, -74.258941497802738, -72.515675964355466],
# [-138.13904357910155, -149.23984283447265, -154.02070617675781],
# [-79.627747, -74.209457, -72.521675],
# [-125.6359, -130.40797, -132.47397]]


# aux_nf = [
# [-80.063534088134759, -74.791913452148435, -73.034408874511712],
# [-135.5944808959961, -146.04012008666993, -150.7372442626953],
# [-79.960373, -74.743088, -73.02948],
# [-124.04991, -128.41185, -130.45657]]


# hnf=[
# [-98.079388580322259, -79.584146118164057, -75.128403320312501],
# [-130.45183181762695, -144.81117645263672, -156.68782989501952],
# [-96.931862, -79.275116, -74.856613],
# [-125.4861, -128.58835, -131.9249]]


# models = [standard,flow1,aux_nf,hnf]
# # model_names = ['standard','flow1','aux_nf','hnf']
# model_names = ['VAE','NF','Aux+NF','HNF']


# legends = ['IW_train', 'IW_test', 'AIS_train', 'AIS_test']



# rows = 1
# cols = 5

# legend=False

# fig = plt.figure(figsize=(8+cols,2+rows), facecolor='white')

# # Get y-axis limits
# min_ = None
# max_ = None
# for m in range(len(models)):
#     for i in range(len(legends)):
#         this_min = np.min(models[m][i])
#         this_max = np.max(models[m][i])
#         if min_ ==None or this_min < min_:
#             min_ = this_min
#         if max_ ==None or this_max > max_:
#             max_ = this_max

# min_ -= 5
# max_ += 5
# # print (min_)
# # print (max_)
# ylimits = [min_, max_]
# # fasd

# # ax.plot(x,hnf_ais, label='hnf_ais')
# # ax.set_yticks([])
# # ax.set_xticks([])
# # if samp_i==0:  ax.annotate('Sample', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

# for m in range(len(models)):
#     ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
#     for i in range(len(legends)):
#         ax.set_title(model_names[m])
#         ax.plot(x,models[m][i], label=legends[i])
#         plt.legend(fontsize=4) 
#         # ax.set(adjustable='box-forced', aspect='equal')
#         plt.yticks(size=6)
#         # ax.set_xlim(xlimits)
#         ax.set_ylim(ylimits)

# m+=1
# ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
# ax.set_title('AIS_test')
# for m in range(len(models)):
#     ax.plot(x,models[m][3], label=model_names[m])
#     plt.legend(fontsize=4) 
#     plt.yticks(size=6)





# # plt.gca().set_aspect('equal', adjustable='box')
# name_file = home+'/Documents/tmp/plot.png'
# plt.savefig(name_file)
# print ('Saved fig', name_file)


























# # Large N 


# x = [1000,1900,2800]


# standard=[[-90.740479431152337, -90.537772521972656, -90.557544250488277],
# [-92.424965972900395, -92.400728302001951, -92.745819549560551],
# [-90.449623, -90.21714, -90.167946],
# [-91.3741, -91.128098, -91.161736]]


# flow1=[
# [-90.35064285278321, -90.096706237792972, -90.01254837036133],
# [-91.87166793823242, -91.854810943603511, -91.808141937255854],
# [-90.119041, -89.874161, -89.75338],
# [-90.934425, -90.763588, -90.679955]]


# aux_nf = [
# [-90.35673767089844, -90.110631713867193, -90.017584991455081],
# [-92.171059570312494, -92.092414550781257, -92.274104461669921],
# [-90.013298, -89.795433, -89.694679],
# [-90.901505, -90.772018, -90.670067]]

# nan = -90
# hnf=[
# [-91.166111450195316, nan, nan],
# [-92.968502349853509, nan, nan],
# [-90.599625, nan, nan],
# [-91.622414, nan, nan]]


# models = [standard,flow1,aux_nf,hnf]
# # model_names = ['standard','flow1','aux_nf','hnf']
# model_names = ['VAE','NF','Aux+NF','HNF']


# legends = ['IW_train', 'IW_test', 'AIS_train', 'AIS_test']



# rows = 1
# cols = 5

# legend=False

# fig = plt.figure(figsize=(8+cols,2+rows), facecolor='white')

# # Get y-axis limits
# min_ = None
# max_ = None
# for m in range(len(models)):
#     for i in range(len(legends)):
#         this_min = np.min(models[m][i])
#         this_max = np.max(models[m][i])
#         if min_ ==None or this_min < min_:
#             min_ = this_min
#         if max_ ==None or this_max > max_:
#             max_ = this_max

# min_ -= 5
# max_ += 5
# # print (min_)
# # print (max_)
# ylimits = [min_, max_]
# # fasd

# # ax.plot(x,hnf_ais, label='hnf_ais')
# # ax.set_yticks([])
# # ax.set_xticks([])
# # if samp_i==0:  ax.annotate('Sample', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

# for m in range(len(models)):
#     ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
#     for i in range(len(legends)):
#         ax.set_title(model_names[m])
#         ax.plot(x,models[m][i], label=legends[i])
#         plt.legend(fontsize=4) 
#         # ax.set(adjustable='box-forced', aspect='equal')
#         plt.yticks(size=6)
#         # ax.set_xlim(xlimits)
#         ax.set_ylim(ylimits)

# m+=1
# ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
# ax.set_title('AIS_test')
# for m in range(len(models)):
#     ax.plot(x,models[m][3], label=model_names[m])
#     plt.legend(fontsize=4) 
#     plt.yticks(size=6)





# # plt.gca().set_aspect('equal', adjustable='box')
# name_file = home+'/Documents/tmp/plot.png'
# plt.savefig(name_file)
# print ('Saved fig', name_file)



















# # this is the data for the 2800 epochs of .001 lr


# x = [1000,1900,2800]


# standard=[
# [-95.851856079101566, -95.577651367187499, -95.50925323486328],
# [-100.11830459594727, -100.5851774597168, -100.82129425048828],
# [-91.513867492675786, -91.195111999511724, -91.074837493896482],
# [-93.504496459960933, -93.493116912841799, -93.568391571044927],
# [-91.156891, -90.848351, -90.715424],
# [-92.259468, -91.969917, -91.882751]]


# flow1=[
# [-95.47870208740234, -94.827168273925778, -94.620732727050779],
# [-98.976015777587889, -98.868874206542969, -99.571325531005854],
# [-91.035421447753905, -90.60315811157227, -90.442083129882818],
# [-92.577168884277341, -92.306014709472663, -92.520370635986325],
# [-90.758804, -90.307571, -90.136314],
# [-91.588821, -91.118736, -91.099899]]


# aux_nf = [
# [-95.77624588012695, -95.400592651367191, -95.132877960205079],
# [-99.507552337646487, -99.908326721191401, -100.11862487792969],
# [-90.910116577148443, -90.565655670166009, -90.396984252929684],
# [-92.385527191162112, -92.281648254394526, -92.291657409667962],
# [-90.539856, -90.194038, -90.004097],
# [-91.341171, -91.067337, -90.807068]]





# # models = [standard,flow1,aux_nf]#,hnf]
# models = [standard,aux_nf]#,hnf]

# # model_names = ['standard','flow1','aux_nf','hnf']
# # model_names = ['VAE','NF','Aux+NF']#,'HNF']
# model_names = ['FFG','Flow']#,'HNF']



# # legends = ['IW train', 'IW test', 'AIS train', 'AIS test']
# legends = ['VAE train', 'VAE test', 'IW train', 'IW test', 'AIS train', 'AIS test']

# colors = ['blue', 'blue', 'green', 'green', 'red', 'red']

# line_styles = [':', '-', ':', '-', ':', '-']




# rows = 1
# cols = 2

# legend=False

# fig = plt.figure(figsize=(2+cols,2+rows), facecolor='white')

# # Get y-axis limits
# min_ = None
# max_ = None
# for m in range(len(models)):
#     for i in range(len(legends)):
#         if i == 1:
#             continue
#         this_min = np.min(models[m][i])
#         this_max = np.max(models[m][i])
#         if min_ ==None or this_min < min_:
#             min_ = this_min
#         if max_ ==None or this_max > max_:
#             max_ = this_max

# min_ -= .1
# max_ += .1
# # print (min_)
# # print (max_)
# ylimits = [min_, max_]
# xlimits = [1000, 3000]

# # fasd

# # ax.plot(x,hnf_ais, label='hnf_ais')
# # ax.set_yticks([])
# # ax.set_xticks([])
# # if samp_i==0:  ax.annotate('Sample', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

# for m in range(len(models)):
#     ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
#     for i in range(len(legends)):
#         if i == 1:
#             continue
#         ax.set_title(model_names[m])
#         ax.plot(x,models[m][i], label=legends[i], c=colors[i], ls=line_styles[i])
#         plt.legend(fontsize=5) 
#         # ax.set(adjustable='box-forced', aspect='equal')
#         plt.yticks(size=6)
#         plt.xticks(size=6)
#         # ax.set_xlim(xlimits)
#         ax.set_ylim(ylimits)
#         ax.set_xlim(xlimits)


# # m+=1
# # ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
# # ax.set_title('AIS_test')
# # for m in range(len(models)):
# #     ax.plot(x,models[m][3], label=model_names[m])
# #     plt.legend(fontsize=4) 
# #     plt.yticks(size=6)





# # plt.gca().set_aspect('equal', adjustable='box')
# name_file = home+'/Documents/tmp/plot.png'
# plt.savefig(name_file)
# print ('Saved fig', name_file)

















# same as above but for small N


x = [1000,1900,2800]


standard=[
[-82.854370727539063, -76.070108032226557, -73.729456939697272],
[-163.057890625, -182.79393188476561, -193.6818292236328],
[-80.628628387451172, -74.229540557861327, -71.979493103027337],
[-143.82347305297853, -157.52427764892579, -164.08638336181642],
[-80.592911, -74.224045, -72.014435],
[-128.38687, -134.6702, -137.37544]]


flow1=[
[-82.55953750610351, -76.815496215820318, -75.015984191894532],
[-157.82005401611329, -174.70116516113282, -183.27636962890625],
[-79.701681518554693, -74.258941497802738, -72.515675964355466],
[-138.13904357910155, -149.23984283447265, -154.02070617675781],
[-79.627747, -74.209457, -72.521675],
[-125.6359, -130.40797, -132.47397]]


aux_nf = [
[-83.103402557373045, -77.468452453613281, -75.410720062255862],
[-155.14535125732422, -171.71119354248046, -179.66849517822266],
[-80.063534088134759, -74.791913452148435, -73.034408874511712],
[-135.5944808959961, -146.04012008666993, -150.7372442626953],
[-79.960373, -74.743088, -73.02948],
[-124.04991, -128.41185, -130.45657]]





# models = [standard,flow1,aux_nf]#,hnf]
models = [standard,aux_nf]#,hnf]

# model_names = ['standard','flow1','aux_nf','hnf']
# model_names = ['VAE','NF','Aux+NF']#,'HNF']
model_names = ['FFG','Flow']#,'HNF']



# legends = ['IW train', 'IW test', 'AIS train', 'AIS test']
legends = ['VAE train', 'VAE test', 'IW train', 'IW test', 'AIS train', 'AIS test']

colors = ['blue', 'blue', 'green', 'green', 'red', 'red']

line_styles = [':', '-', ':', '-', ':', '-']




rows = 1
cols = 2

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
xlimits = [1000, 3000]

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
        ax.set_title(model_names[m])
        ax.plot(x,models[m][i], label=legends[i], c=colors[i], ls=line_styles[i])
        plt.legend(fontsize=5) 
        # ax.set(adjustable='box-forced', aspect='equal')
        plt.yticks(size=6)
        plt.xticks(size=6)
        # ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)
        ax.set_xlim(xlimits)


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





























