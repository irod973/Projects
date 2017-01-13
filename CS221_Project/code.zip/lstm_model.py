"""
File: lstm_model
Author: Irving Rodriguez
Description: Architecture for training an LSTM for partial solution language task.

This file is called by opt_training.py. Directories below are set for results from data_preprocess.py.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn

prefix_num = "4"

TRAIN_DATA_DIR = "data/hoc%s_train_data/" % prefix_num
VAL_DATA_DIR = "data/hoc%s_val_data/" % prefix_num
TEST_DATA_DIR = "data/hoc%s_test_data/" % prefix_num

ACC_OUTPUT = "hoc%s_acc.json"
LOSS_OUTPUT = "hoc%s_loss.json"

ACC_EPOCH = 10 #calculate accuracy every x epochs

class Config(object):
	#Dummy initialization; use data to set
	input_size = 100 #ie num of features
	sequence_len = 10

	#Hyperparams
	batch_size = 20
	num_layers = 1
	num_hidden_units = 200 
	num_classes = 1

	#Optimizations
	lr = 10**(-3)
	num_iters = 10
	dropout = True
	dropout_keep_prob = 0.9

	forget_bias = 1.0
	max_epochs = 20
	early_stopping = 2 #accuracy cycles

	model_name = "dummy"

	def set_model_name(self, layers, units, lr, prefix=None):
		if prefix:
			self.model_name = "%slayers=%dunits=%dlr=%f.weights" % (prefix, layers, units, lr)
		else:
			self.model_name = "layers=%dunits=%dlr=%f.weights" % (layers, units, lr)


class RNN_LSTM(object):

	def add_model_vars(self):
		"""
		Add weights and biases for softmax layer.
		"""

		#TODO: ensure correct dimensionality
		with tf.variable_scope("LSTM", reuse = None) as scope:
			W_s = tf.get_variable("W_s", shape =(self.config.num_hidden_units, self.config.num_classes))
			b_s = tf.get_variable("b_s", shape = (self.config.num_classes,))

	def loss(self, logits, labels):
		"""
		Computes cross entropy loss on output of

		logits: size_batch x num_classes, probabilities over the classes BEFORE softmax
		labels: size_batch x num_classes, one-hot vectors denoting correct class
		"""

		loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
		return loss

	def add_train_op(self, loss):
		"""
		Add training operation to graph.
		"""

		optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.config.lr)
		train_op = optimizer.minimize(loss)
		return train_op

	def __init__(self, config, is_training = False):
		self.config = config
		self.add_model_vars()

		#Define placeholders for inputs and labels
		#TODO: adapt for mini-batch SGD
		#Make dimensions seq_len x input_size x 
		self.inputs = tf.placeholder(tf.float32, [self.config.sequence_len, self.config.input_size])
		#TODO: with above, make shape batch x num_classes
		#self.labels = tf.placeholder(tf.float32, [self.config.sequence_len, self.config.num_classes])
		self.labels = tf.placeholder(tf.float32, [self.config.num_classes,])
		initializer = tf.random_uniform_initializer(-0.01, 0.01)

		#Define cell using LSTMCell wrapper
		lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.num_hidden_units, forget_bias = self.config.forget_bias, initializer = initializer, state_is_tuple = True)

		if config.dropout:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = self.config.dropout_keep_prob)

		stacked_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.config.num_layers, state_is_tuple = True)
		self.initial_state = stacked_cell.zero_state(self.config.sequence_len, tf.float32)

		#Compute logits for sequence
		cell_outputs, state = rnn.rnn(stacked_cell, [tf.convert_to_tensor(self.inputs)], initial_state = self.initial_state)

		with tf.variable_scope("LSTM", reuse = True) as scope:
			W_s = tf.get_variable("W_s")
			b_s = tf.get_variable("b_s")
			# print "W shape | b shape | state shape"
			# print W_s.get_shape(), b_s.get_shape(), cell_outputs[0].get_shape()
			#logits = tf.matmul(W_s, tf.transpose(cell_outputs[0])) + b_s
			logits = tf.matmul(cell_outputs[0], W_s) + b_s
			# print "Logits shape (should be num_classes x batch): ", logits.get_shape()

		logits = tf.slice(logits, [self.config.sequence_len - 1, 0], size = [-1, self.config.num_classes])
		logits = tf.reshape(logits, [-1])

		if not is_training:
			self.predicted_vectors = tf.nn.softmax([logits])
			return

		self.model_loss = self.loss([logits], [self.labels])
		self.train_op = self.add_train_op(self.model_loss)

def data_stream(data_directory):
	path = os.path.join(os.getcwd(), data_directory)
	for filename in os.listdir(path):
		inputs, labels = [], []
		filepath = os.path.join(path, filename)
		with open(filepath, "r") as f:
			#Data is (sequence, label_vector) pairs
			data = json.load(f)
			for index in xrange(len(data)):
				inputs.append(data[index][0])
				labels.append(data[index][1])

		yield inputs, labels

def train(config):
	"""
	Data is split by inputs and labels.
	Inputs: 
		data = 
		[ seq1, 
			seq2, ...,
		]
		where seq = list of lists
	"""
	#Load data
	train_generator = data_stream(TRAIN_DATA_DIR)
	val_generator = data_stream(VAL_DATA_DIR)
	#test_generator = data_stream(TEST_DATA_DIR)
	train_inputs, train_labels = next(train_generator)

	num_iters = len(train_inputs) / config.batch_size
	config.input_size = len(train_inputs[0][0])
	config.num_classes = len(train_labels[0])

	print "Units | Layers | Learning Rate | Dropout"
	print "%d | %d | %g | %r" % (config.num_hidden_units, config.num_layers, config.lr, config.dropout)

	with tf.Graph().as_default(), tf.Session() as sess:
		loss_history = []
		train_acc_history = []
		val_acc_history = []
		prev_epoch_loss = float('inf')
		best_val_epoch = 0
		best_val_acc = -float('inf')
		stopped = -1

		with tf.variable_scope("LSTM", reuse = None) as scope:
			train_model = RNN_LSTM(config, is_training = True)
		with tf.variable_scope("LSTM", reuse = True) as scope:
			#prob don't need to change batch size
			eval_train_model = RNN_LSTM(config, is_training = False)
			eval_val_model = RNN_LSTM(config, is_training = False)

		tf.initialize_all_variables().run()

		for epoch in xrange(config.max_epochs):
			print "Epoch %d" % epoch

			try:
				train_inputs, train_labels = next(train_generator)
			except StopIteration:
				train_generator = data_stream(TRAIN_DATA_DIR)
				train_inputs, train_labels = next(train_generator)

			try:
				val_inputs, val_labels = next(val_generator)
			except StopIteration:
				val_generator = data_stream(VAL_DATA_DIR)
				val_inputs, val_labels = next(val_generator)

			#TODO: pad sequences if necessary!
			#Recall inputs is a list of sequences
			for index in xrange(len(train_inputs)):
				new_sequence = train_inputs[index]
				if len(new_sequence) > config.sequence_len:
					#Cut off the sequence at sequence_len solutions
					new_sequence = new_sequence[:config.sequence_len]
					train_inputs[index] = new_sequence

				elif len(new_sequence) < config.sequence_len:
					#Add blank solutions to sequence
					pad_difference = config.sequence_len - len(new_sequence)
					for i in xrange(pad_difference):
						new_sequence.append([0]*config.input_size)
		
					train_inputs[index] = new_sequence

			#Run Epoch!
				epoch_loss = []
			#TODO: use this for mini-batch SGD
			# for i in xrange(config.num_iters-1):
			# 	#Make dicts
			# 	lower_index = i*config.batch_size
			# 	upper_index = (i+1)*config.batch_size
			# 	train_dict = {train_model.inputs: np.asarray(train_inputs[lower_index:upper_index]), train_model.labels: np.asarray(train_labels[lower_index:upper_index])}
				
				train_dict = {train_model.inputs: np.asarray(train_inputs[index]), train_model.labels: np.asarray(train_labels[index])}

				feed_dict = train_dict
				loss, _ = sess.run([train_model.model_loss, train_model.train_op], feed_dict)
				epoch_loss.append(loss)
				loss_history.extend(loss.tolist())

			mean_loss = np.mean(epoch_loss)
			print "Epoch training loss: ", mean_loss
			prev_epoch_loss = mean_loss

			#Evaluate accuracy every 10 epochs
			if epoch != 0 and epoch % ACC_EPOCH == 0:
				print "Calculating accuracy."
				#Training Set
				train_predicted_vectors = []
				for index in xrange(len(train_inputs)):
					train_dict = {eval_train_model.inputs: np.asarray(train_inputs[index]), eval_train_model.labels: np.asarray(train_labels[index])}

					feed_dict = train_dict
					predictions = sess.run([eval_train_model.predicted_vectors], feed_dict)
					#TODO: verify this works
					#predictions = predictions.tolist()
					max_prob_index = predictions.index(max(predictions))
					train_predicted_vectors.append([1.0*(i==max_prob_index) for i in xrange(len(predictions))])

				train_accuracy = sum(1.0 for index, predicted_solution in enumerate(train_predicted_vectors) if predicted_solution == train_labels[index])/len(train_predicted_vectors)

				#Validation Set
				val_predicted_vectors = []
				for index in xrange(len(val_inputs)):
					new_sequence = val_inputs[index]
					if len(new_sequence) > config.sequence_len:
						#Cut off the sequence at sequence_len solutions
						new_sequence = new_sequence[:config.sequence_len]
						val_inputs[index] = new_sequence

					elif len(new_sequence) < config.sequence_len:
						#Add blank solutions to sequence
						pad_difference = config.sequence_len - len(new_sequence)
						for i in xrange(pad_difference):
							new_sequence.append([0]*config.input_size)
			
						val_inputs[index] = new_sequence

					val_dict = {eval_val_model.inputs: np.asarray(val_inputs[index]), eval_val_model.labels: np.asarray(val_labels[index])}

					feed_dict = val_dict
					predictions = sess.run([eval_val_model.predicted_vectors], feed_dict)
					#TODO: verify this works
					#predictions = predictions.tolist()
					max_prob_index = predictions.index(max(predictions))
					val_predicted_vectors.append([1.0*(i==max_prob_index) for i in xrange(len(predictions))])

				val_accuracy = sum(1.0 for index, predicted_solution in enumerate(val_predicted_vectors) if predicted_solution == val_labels[index])/len(val_predicted_vectors)

				train_acc_history.append(train_accuracy)
				val_acc_history.append(val_accuracy)

				print "Evaluated train | val inputs of size %d | %d" % (len(train_inputs), len(val_inputs))
				print ">> Train | Val: %f | %f" %(train_accuracy, val_accuracy)

				#Save best model
				if val_accuracy > best_val_acc:
					best_val_acc = val_accuracy
					best_epoch = epoch
					saver = tf.train.Saver()
					saver.save(sess, './%s' % train_model.config.model_name)

				#Early stopping 
				if epoch - best_val_epoch >= train_model.config.early_stopping * ACC_EPOCH:
					stopped = epoch
					print "Stopped at %d" % stopped
					break

		return prev_epoch_loss, loss_history

if __name__ == "__main__":
	"""
	Expecting python lstm_model.py 10 20 1 200 True
	"""
	config = Config()

	#Set hyperparameters
	#TODO: automatically parse max sequence len from somewhere
	config.sequence_len, config.batch_size, config.num_layers, config.num_hidden_units, config.dropout = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), bool(sys.argv[5])

	train(config)

	#use separate function for evaluating test accuracy
