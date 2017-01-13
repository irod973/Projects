"""
File: data_preprocess
Author: Irving Rodriguez
Description: File to preprocess raw HoC data for inputs  into LSTM.
Run using python data_preprocess.py
Change prefix_num below between 4 and 18 to process each dataset.
"""

import sys
import random
import numpy as np
from pprint import pprint

#TODO: use intermediate map between soln_id and class_id, to condense 10k->3k

prefix_num = "18"

def get_features(problem_dict):
	"""
	What we really want is for this to return every (type) and (type, child_type) pair. The existence of these is our feature vector.
	"""
	#features = {}
	features = []

	unrestricted = False
	def recurse(current_dict, current_chain = []):
		if 'children' not in current_dict:
			if unrestricted:
				return current_chain
			return

		curr_type = current_dict['type']
		curr_children = current_dict['children'] #this returns a list
		#add children that are nested
		children_types = [child['type'] for child in curr_children]
		#features[curr_type] = children_types
		
		pairs = [", ".join([curr_type, child_type]) for child_type in children_types]
		features.extend(pairs)

		#TODO: go back to do this
		if unrestricted:
			for child in curr_children:
				childrens_children = recurse(child)
				features[curr_type].extend(childrens_children)
		else:
			for child in curr_children:
				recurse(child)

	recurse(problem_dict)
	return set(features)

#First, loop through asts files with .json extension
prefix = "hoc%s_" % prefix_num
import json

AST_FILE = "%sasts.json" % prefix
with open(AST_FILE, "r") as f:
	lines = f.readlines()
	desired_files = [line.strip() for line in lines]

#Loop through solutions to create feature dicts
print ">>> Creating feature dictionary..."
feature_keys = []
all_features = {} #map of id:feature_dict pairs

for filename in desired_files:
	with open(filename, "r") as f:
		solution_dict = json.load(f)

	solution_id = filename.split("/")[-1].split('.')[0]
	features = get_features(solution_dict)
	#Add any unseen feature keys to the map
	for key in features:
		if key not in feature_keys:
			feature_keys.append(key)

	all_features[solution_id] = features

print "Done."

#Next, create feature:index mapping!
print ">>> Creating feature-to-index mapping..."
feature_keys = set(feature_keys)
num_features = len(feature_keys)
rand_indices = [i for i in xrange(len(feature_keys))]
random.shuffle(rand_indices)
feature_to_index = {feature:index for feature, index in zip(feature_keys, rand_indices)}

#Loop through features and turn them from dicts to vectors
all_vectors = {} #map of id:feature_vector pairs
max_soln_id = 0
for ID, feature_dict in all_features.items():
	feature_vector = [0]*num_features
	
	#Increment index of feature in the vector
	for feature in feature_dict:
		feature_index = feature_to_index[feature]
		feature_vector[feature_index] = 1

	#Add vector to all_vectors
	all_vectors[ID] = feature_vector
	#Grab highest ID
	if int(ID) > max_soln_id:
		max_soln_id = int(ID)
print "Done. Here is the map: "
pprint(feature_to_index)

#Finally, loop through trajectories to make (seq, next_soln) pairs.
print ">>> Reading trajectories..."
num_unique_solutions = len(all_vectors)
max_sequence_len = 0

TRAJECTORY = "%straj.json" % prefix

trajectories = []
with open(TRAJECTORY, "r") as f:
	for trajectory_file in f.readlines():
		trajectory_file = trajectory_file.strip()

		with open(trajectory_file, "r") as f2:
			trajectory = [soln_ID.strip() for soln_ID in f2.readlines()] #keep as strings, since keys are strings
			trajectories.append(trajectory)

#TODO: stream data to file. Otherwise too large and will segfault
num_samples = 0
for trajectory in trajectories:
	num_samples += len(trajectory)/2

#FINALLY, split data in train/dev/test split and write to respective files. let's do 80/10/10 split
num_val_samples = num_samples/10
num_test_samples = num_samples/10

#val
count = num_val_samples
val_indices = []
while count != 0:
	pot_index = random.randint(0, num_samples-1)
	#Append indices that have not been seen before
	if pot_index in val_indices:
		continue
	else:
		val_indices.append(pot_index)
		count -= 1

#test
count = num_test_samples
test_indices = []
while count != 0:
	pot_index = random.randint(0, num_samples-1)
	#Append indices that have not been seen before
	if pot_index in val_indices or pot_index in test_indices:
		continue
	else:
		test_indices.append(pot_index)
		count -= 1

print "Done."
print ">>> Creating data..."

TRAIN_DATA_PREFIX = "project/%strain_data/%strain" % (prefix, prefix)
VAL_DATA_PREFIX = "project/%sval_data/%sval" % (prefix, prefix)
TEST_DATA_PREFIX = "project/%stest_data/%stest" % (prefix, prefix)

train_file_count = 0
val_file_count = 0
test_file_count = 0

megabyte = 1000000
file_size_threshold = 125*megabyte
len_threshold = 1000 #This gives ~30MB files!

#TODO: Keep track of sub-traj looped through to dump all data when last sub-traj processed
soln_index = 0
train_data = []
val_data = []
test_data = []
for trajectory in trajectories:
	#Make (seq, next_soln) for sub-trajectories starting at half-index
	num_solutions = len(trajectory)
	for next_index in xrange(num_solutions/2, num_solutions-1):
		#Make list of feature vectors
		#sub_trajectory = trajectory[:next_index-1]
		sub_trajectory = trajectory[max(next_index-11,0):next_index-1] #can limit trajectory size

		#Make sequence of feature vectors
		sequence = [all_vectors[ID] for ID in sub_trajectory if ID in all_vectors]

		#Check sequence len
		if len(sequence) > max_sequence_len:
			max_sequence_len = len(sequence)

		#Make the next_label one-shot vector
		next_solution_ID = int(trajectory[next_index])
		next_solution_label = [0]*max_soln_id
		try:
			next_solution_label[next_solution_ID] = 1.
		except IndexError:
			print next_solution_ID

		#TODO: remove print statements
		if soln_index in val_indices:
			val_data.append([sequence, next_solution_label])
		elif soln_index in test_indices:
			test_data.append([sequence, next_solution_label])
		else:
			train_data.append([sequence, next_solution_label])
		soln_index += 1

		#Dump data to file if exceeding threshold
		#if val_data.nbytes > file_size_threshold:
		if len(val_data) > len_threshold:
			#write to val file
			val_file_count += 1
			filename = "%s%s%s" % (VAL_DATA_PREFIX, str(val_file_count), ".json")
			with open(filename, "w") as f:
					json.dump(val_data, f)
			print "Dumped data to %s" % filename
			val_data = []
		#if test_data.nbytes > file_size_threshold:
		if len(test_data) > len_threshold:
			#write to test file
			test_file_count += 1
			filename = "%s%s%s" % (TEST_DATA_PREFIX, str(test_file_count), ".json")
			with open(filename, "w") as f:
					json.dump(test_data, f)
			print "Dumped data to %s" % filename
			test_data = []
		#if sys.getsizeof(train_data) > file_size_threshold:
		if len(train_data) > len_threshold:
			#write to train file
			train_file_count += 1
			filename = "%s%s%s" % (TRAIN_DATA_PREFIX, str(train_file_count), ".json")
			with open(filename, "w") as f:
					json.dump(train_data, f)
			print "Dumped data to %s" % filename
			train_data = []
			
#Write feature_to_index dict to file, for reference
FEATURE_INDEX = "%sfeature_to_index.json" % prefix
with open(FEATURE_INDEX, "w") as f:
	print>>f, "Max seq len: %d" % max_sequence_len
	json.dump(feature_to_index, f)

print "Processed %d samples. Max seq len: %d. Max soln id: %d" % (num_samples, max_sequence_len, max_soln_id)

#DONE!
