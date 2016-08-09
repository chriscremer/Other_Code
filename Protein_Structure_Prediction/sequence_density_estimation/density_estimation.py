
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,'../tools')
import protein_model_tools as tools

from os.path import expanduser
home = expanduser("~")

from sklearn.neighbors.kde import KernelDensity




#RASH
L = 166
msa_file = home + '/Documents/Protein_data/RASH/RASH_HUMAN2_833a6535-26d0-4c47-8463-7970dae27a32_evfold_result/alignment/RASH_HUMAN2_RASH_HUMAN2_jackhmmer_e-10_m30_complete_run.fa'
msa, n_aa = tools.convert_msa(L, msa_file)
print len(msa), len(msa[0]), n_aa



msa_vectors = []
for samp in range(2000):
	msa_vectors.append(np.ndarray.flatten(tools.convert_samp_to_one_hot(msa[samp], n_aa)))


msa_vectors = np.array(msa_vectors)
print msa_vectors.shape

for bw in [.01, .1, 1., 10.]:

	kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(msa_vectors[1000:])
	# density_train = kde.score_samples(msa_vectors)
	print bw, kde.score(msa_vectors[:1000])

