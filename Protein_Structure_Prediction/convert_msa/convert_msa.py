


import numpy as np
import csv
from os.path import expanduser
home = expanduser("~")
import json

msa_length = 166

msa_file = home + '/Downloads/RASH_HUMAN_bf7f9ca0-0462-4325-88d7-3e16eb7adfbc_evfold_result/alignment/RASH_HUMAN_RASH_HUMAN_jackhmmer_e-10_m30_complete_run.fa'

aa_letters = []
# get all possible AAs
with open(msa_file, 'rb') as f:
	for row in f:
		if row[0] == '>':
			continue
		else:
			for i in range(len(row)):
				if row[i].lower() not in aa_letters and row[i] != '\n' and row[i] != '.':

					aa_letters.append(row[i].lower())
print aa_letters
print len(aa_letters)


msa = []
# convert letters to numbers 
with open(msa_file, 'rb') as f:
	
	for row in f:
		temp_samp = []
		if row[0] == '>':
			continue
		else:
			for i in range(msa_length):
				aa = row[i].lower()
				if aa == '.':
					aa = '-'
				temp_samp.append(aa_letters.index(aa))
		msa.append(temp_samp)

print len(msa)
print len(msa[0])

#save a file of the AAs converted to numbers


