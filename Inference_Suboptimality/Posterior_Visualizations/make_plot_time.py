








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




























# # Large N time
 

# x = [5,4,3,2,1]


# standard=[[-90.702315826416012, -90.960945129394531, -91.179870147705074, -91.307066497802737, -91.541283416748044],
# [-92.829085235595699, -92.696218719482417, -92.845530853271484, -92.93976242065429, -93.093510894775392],
# [-90.295609, -90.575516, -90.793289, -90.928864, -91.13958],
# [-91.31221, -91.548203, -91.802658, -91.898094, -92.046806]]


# flow1=[
# [-90.650220947265623, -91.05579483032227, -91.573744201660162, -91.739595489501951, -91.889657592773432],
# [-92.378957519531255, -92.69502883911133, -92.973268890380865, -93.272323150634762, -93.358189849853517],
# [-90.309952, -90.780884, -91.281883, -91.422859, -91.600243],
# [-91.268311, -91.766647, -92.079201, -92.365234, -92.574211]]


# aux_nf = [
# [-90.918987426757809, -91.276008605957031, -91.658234100341801, -91.994766693115238, -92.16865295410156],
# [-92.661103820800776, -92.726740264892584, -93.02299407958985, -93.323872985839841, -93.435993652343754],
# [-90.553062, -90.913162, -91.319992, -91.596596, -91.829086],
# [-91.37339, -91.731262, -92.110397, -92.388535, -92.630745]]



# models = [standard,flow1,aux_nf]#,hnf]
# # model_names = ['standard','flow1','aux_nf','hnf']
# model_names = ['VAE','NF','Aux+NF']#,'HNF']


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

# min_ -= 2
# max_ += 2
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



















# Large N time, started at 3 hours
 

x = [5,4,3,2,1]


standard=[[-90.356217041015626, -90.680784301757811, -90.540812988281246, -90.587139282226559, -90.636305389404299],
[-92.610456390380861, -92.716960754394535, -92.693904876708984, -92.830287017822272, -92.77354675292969],
[-90.034859, -90.31353, -90.199432, -90.210648, -90.290329],
[-91.100952, -91.249527, -91.232422, -91.359818, -91.309135]]


flow1=[
[-90.271184082031255, -90.411840667724604, -90.525875854492185, -90.586142272949218, -90.5160971069336],
[-92.296845092773438, -92.239070434570309, -92.321230316162115, -92.415106201171881, -92.322035980224612],
[-89.973557, -90.179146, -90.283897, -90.282059, -90.281517],
[-90.983269, -91.187912, -91.268188, -91.260056, -91.231453]]


aux_nf = [
[-90.619600219726564, -90.864978485107429, -90.739211883544925, -90.871030578613286, -90.804663848876956],
[-92.693523559570309, -92.736759033203128, -92.500957794189446, -92.672872161865229, -92.511756896972656],
[-90.226097, -90.397934, -90.423958, -90.483292, -90.528809],
[-91.142136, -91.265198, -91.21122, -91.290939, -91.295013]]



models = [standard,flow1,aux_nf]#,hnf]
# model_names = ['standard','flow1','aux_nf','hnf']
model_names = ['VAE','NF','Aux+NF']#,'HNF']


legends = ['IW_train', 'IW_test', 'AIS_train', 'AIS_test']



rows = 1
cols = 5

legend=False

fig = plt.figure(figsize=(8+cols,2+rows), facecolor='white')

# Get y-axis limits
min_ = None
max_ = None
for m in range(len(models)):
    for i in range(len(legends)):
        this_min = np.min(models[m][i])
        this_max = np.max(models[m][i])
        if min_ ==None or this_min < min_:
            min_ = this_min
        if max_ ==None or this_max > max_:
            max_ = this_max

min_ -= 2
max_ += 2
# print (min_)
# print (max_)
ylimits = [min_, max_]
# fasd

# ax.plot(x,hnf_ais, label='hnf_ais')
# ax.set_yticks([])
# ax.set_xticks([])
# if samp_i==0:  ax.annotate('Sample', xytext=(.3, 1.1), xy=(0, 1), textcoords='axes fraction')

for m in range(len(models)):
    ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
    for i in range(len(legends)):
        ax.set_title(model_names[m])
        ax.plot(x,models[m][i], label=legends[i])
        plt.legend(fontsize=4) 
        # ax.set(adjustable='box-forced', aspect='equal')
        plt.yticks(size=6)
        # ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)

m+=1
ax = plt.subplot2grid((rows,cols), (0,m), frameon=False)
ax.set_title('AIS_test')
for m in range(len(models)):
    ax.plot(x,models[m][3], label=model_names[m])
    plt.legend(fontsize=4) 
    plt.yticks(size=6)





# plt.gca().set_aspect('equal', adjustable='box')
name_file = home+'/Documents/tmp/plot.png'
plt.savefig(name_file)
print ('Saved fig', name_file)



























