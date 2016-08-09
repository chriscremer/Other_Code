

import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,'../tools')
import protein_model_tools as tools

from os.path import expanduser
home = expanduser("~")


import pickle

from sklearn.decomposition import PCA
from sklearn.neighbors.kde import KernelDensity



rao = 0
mac = 1


if rao == 1:
	msa_file = home + '/protein_data/RASH_HUMAN2_RASH_HUMAN2_jackhmmer_e-10_m30_complete_run.fa'
if mac == 1:
	msa_file = home + '/Documents/Protein_data/RASH/RASH_HUMAN2_833a6535-26d0-4c47-8463-7970dae27a32_evfold_result/alignment/RASH_HUMAN2_RASH_HUMAN2_jackhmmer_e-10_m30_complete_run.fa'


#RASH
L = 166
msa, n_aa = tools.convert_msa(L, msa_file)
print len(msa), len(msa[0]), n_aa


#Convert to matrix
msa_vectors = []
for samp in range(2000):
	msa_vectors.append(np.ndarray.flatten(tools.convert_samp_to_one_hot(msa[samp], n_aa)))
msa_vectors = np.array(msa_vectors)
print msa_vectors.shape

#PCA
pca = PCA(n_components=20)
pca.fit(msa_vectors[1000:])
a_samps_pca = pca.transform(msa_vectors[1000:])
b_samps_pca = pca.transform(msa_vectors[:1000])
print a_samps_pca.shape

#KDE
# for bw in [.01, .1, 1., 10.]:
for bw in [ 1.]:

	kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(a_samps_pca)
	# density_train = kde.score_samples(msa_vectors)
	print bw, kde.score(b_samps_pca)

densities = kde.score_samples(b_samps_pca)
# densities = np.ones(1000)

#Scale densities to betw 0 and 1
min_density = np.min(densities)
densities = densities - min_density + 1.

weights = np.reciprocal(densities)

max_weights = np.max(weights)
weights = weights / max_weights

print np.max(weights)
print np.mean(weights)
print np.min(weights)

#MI
MI = np.zeros((L,L))

for i in range(L):

	for j in range(i+1,L):


		#Calc MI for columsn i and j
		MI_sum = 0
		for a in range(n_aa):

			for b in range(n_aa):

				fAB = 1.
				fA = 1.
				fB = 1.
				for samp in range(len(msa[:1000])):

					if msa[samp][i] == a:
						# fA += 1.
						fA += weights[samp]
					if msa[samp][j] == b:
						# fB += 1.
						fB += weights[samp]
					if msa[samp][i] == a and msa[samp][j] == b:
						# fAB += 1.
						fAB += weights[samp]



				MI_sum += fAB * np.log(fAB/(fA*fB))




		MI[i,j] = MI_sum
		print i, j, MI_sum


#save results
if rao == 1:
	with open(home + '/protein_data/MI_bw_1.pkl', "wb" ) as f:
		pickle.dump(MI, f)
		print 'saved results'
if mac == 1:
	with open(home + 'Documents/Protein_data/RASH/MI_bw_1.pkl', "wb" ) as f:
		pickle.dump(MI, f)
		print 'saved results'



