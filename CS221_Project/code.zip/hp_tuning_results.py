"""
File: hp_tuning_results
Author: Irving Rodriguez
Description: File for processing and plotting the results of hyperparameter tuning.

Run using the following, changing the prefix variable below:
python hp_tuning.results.py
"""

import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['figure.figsize'] = (12,9)

prefix = "hoc4"

hp_names = ["layers", "hidden_units", "l_rate", "dropout"]
layers = [i for i in xrange(1, 6)]
units = [i for i in xrange(200, 500, 100)]
l_rate = [10**(-i) for i in xrange(1, 5)]
dropout = [False, True]

value_lists = [layers, units, l_rate, dropout]
hyperparams = {name:value_list for name, value_list in zip(hp_names, value_lists)}

for name, hp_value_list in hyperparams.items():
	
	#First, load the histories for this hyperparam
	name_loss_histories = []
	for value in hp_value_list:
		output_file = "%shp_results/loss_history%s=%g" % (prefix, name, value)

		with open(output_file, "r") as f:
			name_loss_histories.append(json.load(f))

	#Then plot them
	mean_histories = [np.mean(history) for history in name_loss_histories]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(hp_value_list, mean_histories, "bo")
	#plt.plot(opt_values[index], opt_loss, "ko") #for showing optimal loss
	ax.set_xlabel("%s" % name)
	ax.set_ylabel("Mean loss")

	#X-axis rescaling
	xticks, xticklabels = plt.xticks()
	# shift half a step to the left
	xmin = (3*xticks[0] - xticks[1])/2.
	# shift half a step to the right
	xmax = (3*xticks[-1] - xticks[-2])/2.
	plt.xlim(xmin, xmax)
	#plt.xticks(xticks)
	plt.xticks(hp_value_list)

	#Y-axis rescaling
	yticks, yticklabels = plt.yticks()
	# shift half a step to the left
	ymin = (3*yticks[0] - yticks[1])/2.
	# shift half a step to the right
	ymax = (3*yticks[-1] - yticks[-2])/2.
	plt.ylim(ymin, ymax)
	plt.yticks(yticks)
	
	plt.show()
