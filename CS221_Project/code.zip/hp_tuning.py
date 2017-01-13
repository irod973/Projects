"""
File: hp_tuning
Author: Irving Rodriguez
Description: File for tuning hyperparamters of lstm contained in lstm_model. 
 
Run using following command (after making hocXhp_results directory), changing between 4 and 18 for the two datasets:
python hp_tuning.py >> hoc4hp_results/hp_results.txt

python lstm_model.py sequence_len batch_size layers hidden lr dropout

default: 10 10 1 200 e-3 False

Hidden Units | Layers | Learning Rate | Dropout

Layers: [1, 5], +1
Units: [200, 500], +100
lr: [e-5, e-1], +e-1
dropout: false, true

OPT VALUES:
layers = 4
units = 300
lr = 0.1
"""

import json
import lstm_model
import numpy as np

prefix = "hoc18"

def set_defaults():
	config = lstm_model.Config()

	config.num_layers = 2
	config.num_hidden_units = 200
	config.lr = 10**(-3)
	config.dropout = False
	config.sequence_len = 10
	config.max_epochs = 200

	return config

hp_names = ["layers", "hidden_units", "l_rate", "dropout"]
layers = [i for i in xrange(1, 6)]
units = [i for i in xrange(200, 600, 100)]
l_rate = [10**(-i) for i in xrange(1, 6)]
dropout = [False, True]

value_lists = [layers, units, l_rate, dropout]
hyperparams = {name:value_list for name, value_list in zip(hp_names, value_lists)}

for name, hp_value_list in hyperparams.items():
	for value in hp_value_list:
		output_file = "%shp_results/loss_history%s=%g" % (prefix, name, value)

		config = set_defaults()

		if name == "layers":
			config.num_layers = value
		elif name == "hidden_units":
			config.num_hidden_units = value
		elif name == "l_rate":
			config.lr = value

		last_epoch_loss, loss_history = lstm_model.train(config)

		with open(output_file, "w") as f:
			json.dump(list(loss_history), f)
			print "Loss history written to %s" % output_file
			print "Last loss: %g" % last_epoch_loss
			print "Mean loss: %g" % np.mean(loss_history)

#OPT
#NASTA Visa
