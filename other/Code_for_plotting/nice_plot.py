

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np


def plot_lines(x, ys, stds, labels, x_label, y_label, save_as):


	plt.figure(1)
	ax = plt.subplot(211)
	# ax = plt.subplot(111)
	for y in range(len(ys)):
		plt.plot(x, ys[y], label=labels[y])
		error = np.array(stds[y])
		plt.fill_between(x, np.array(ys[y])-error, np.array(ys[y])+error, alpha=.5)
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	# Shrink current axis by 20%
	# box = ax.get_position()
	# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':8})

	ax.legend(prop={'size':8})
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

	plt.savefig(save_as)
	print 'Saved plot'



def plot_lines_2_graphs(x, ys, stds, labels, x_label, y_label, save_as, top, done):


	plt.figure(1)

	if top:
		ax = plt.subplot(211)
	else:
		ax = plt.subplot(212)
	# ax = plt.subplot(111)
	for y in range(len(ys)):
		plt.plot(x, ys[y], label=labels[y])
		error = np.array(stds[y])
		plt.fill_between(x, np.array(ys[y])-error, np.array(ys[y])+error, alpha=.5)
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	# Shrink current axis by 20%
	# box = ax.get_position()
	# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':8})

	ax.legend(prop={'size':8})
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')	

	if done:
		plt.savefig(save_as)
		print 'Saved plot'


def plot_lines_2graphs_errorbars(x, ys, stds, labels, x_label, y_label, save_as, top, done):


	plt.figure(1)

	if top:
		ax = plt.subplot(211)
	else:
		ax = plt.subplot(212)
	# ax = plt.subplot(111)
	for y in range(len(ys)):
		plt.errorbar(x, ys[y], yerr=stds[y], label=labels[y])
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	ax.legend(prop={'size':8})
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')	

	if done:
		plt.savefig(save_as)
		print 'Saved plot'


def plot_lines_2graphs_errorbars2(x, ys, stds, labels, x_label, y_label, save_as, top, done):


	plt.figure(1)

	if top:
		ax = plt.subplot(211)
	else:
		ax = plt.subplot(212)
	# ax = plt.subplot(111)
	for y in range(len(ys)):
		plt.errorbar(x, ys[y], yerr=stds[y], label=labels[y])
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * .95])
	ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), prop={'size':8})

	# ax.legend(prop={'size':8})
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')	

	# plt.xlim(xmin=9.05)
	plt.ylim(ymin=0)
	
	# if done == 1:
		
	# 	plt.ylim(0, 2.01)

	if done:
		plt.savefig(save_as)
		print 'Saved plot'


def plot_lines_2graphs_errorbars3(x, ys, stds, labels, x_label, y_label, save_as, top, done, xlim=None):


	plt.figure(1)

	if top:
		ax = plt.subplot(211)
	else:
		ax = plt.subplot(212)
	# ax = plt.subplot(111)
	for y in range(len(ys)):
		plt.errorbar(x, ys[y], yerr=stds[y], label=labels[y])
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * .92])
	ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), prop={'size':8})

	# ax.legend(prop={'size':8})
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')	

	if xlim != None:
		plt.xlim(xlim[0],xlim[1])
	
	# if done == 1:
		
	# 	plt.ylim(0, 2.01)

	if done:
		plt.savefig(save_as)
		print 'Saved plot'


def plot_lines_2graphs_errorbars3_logscale(x, ys, stds, labels, x_label, y_label, save_as, top, done, xlim=None):


	plt.figure(1)

	if top:
		ax = plt.subplot(211)
	else:
		ax = plt.subplot(212)
	# ax = plt.subplot(111)
	for y in range(len(ys)):
		plt.errorbar(x, ys[y], yerr=stds[y], label=labels[y])
	plt.xlabel(x_label)
	plt.ylabel(y_label)



	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * .9])
	ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), prop={'size':8})

	# ax.legend(prop={'size':8})
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')	


	ax.set_xscale("log", nonposx='clip')


	if xlim != None:
		plt.xlim(xlim[0],xlim[1])
	
	# if done == 1:
		
	# 	plt.ylim(0, 2.01)

	if done:
		plt.savefig(save_as)
		print 'Saved plot'





if __name__ == "__main__":


	x = [1,2,3,4]
	ys = [[2,3,4,5],[4,1,2,5]]
	stds = [[.1,.2,.1,.1], [.1,.2,.1,.2]]
	labels = ['line1', 'line2']
	x_label = 'x1'
	y_label = 'y1'

	plot_lines(x, ys, stds, labels, x_label, y_label)

	plt.savefig('test1.png')
	print 'Saved plot'