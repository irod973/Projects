"""
File: opt_training.py
Author: Irving Rodriguez
Description: File for running final LSTM experiments on test set, when neural net fully tuned. 

Run with following, for both datasets:
python opt_training.py >> hocXexp_results/test_results.txt

OPT VALUES:
layers = 4
units = 300
lr = 0.1
"""

import json
import lstm_model
import numpy as np

PREFIX = "exp_results/"
TRAINING_LOSS_FILE = "%stest_opt_loss_history" % PREFIX

OPT_LAYERS = 4
OPT_UNITS = 300
OPT_LR = 0.1
DROPOUT = False

def set_defaults():
	config = lstm_model.Config()

	config.num_layers = 2
	config.num_hidden_units = 200
	config.lr = 10**(-3)
	config.dropout = False
	config.sequence_len = 10
	config.max_epochs = 20

	return config

def set_optimal():
	config = lstm_model.Config()

	config.num_layers = OPT_LAYERS
	config.num_hidden_units = OPT_UNITS
	config.lr = OPT_LR
	config.sequence_len = 10
	config.max_epochs = 200
	config.dropout = DROPOUT

	config.set_model_name(config.num_layers, config.num_hidden_units, config.lr, PREFIX)

	return config

def opt_training():
	config = set_optimal()

	last_epoch_loss, loss_history = lstm_model.train(config)

	with open(TRAINING_LOSS_FILE, "w") as f:
		json.dump(list(loss_history), f)
		print "Loss history written to %s" % TRAINING_LOSS_FILE
		print "Last loss: %g" % last_epoch_loss
		print "Mean loss: %g" % np.mean(loss_history)

if __name__ == "__main__":
	opt_training()
