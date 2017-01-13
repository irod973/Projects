"""
File:  data_explore
Author: Irving Rodriguez
Description: Preliminary study of data. Not used in final project pipeline.
"""

import os
import json
from pprint import pprint


# PROB_DIRS = ["data/hoc18/", "data/hoc4/"]

# SUBDIRS = ["asts", "graphs","groundTruth", "interpolated", "nextProblem", "trajectories", "unseen"]

# SUBDIRS = ["asts/", "trajectories/"]

suffix = "4"

OUTPUT = "hoc%s_asts.json" % suffix
#OUTPUT = "hoc%s_traj.json" % suffix

def file_list(directory):
	for filename in os.listdir(directory):
		yield filename

def main(problem_dirs, sub_dirs):
	files = {}

	current_dir = os.getcwd() 
	#between hoc18 and hoc4
	for data_dir in problem_dirs:
		count = 0
		
		subdir_files = {}
		#between the different subdirs
		for subdir in sub_dirs:
			filenames = []

			path = os.path.join(current_dir, data_dir+subdir)
			for filename in file_list(path):
				count += 1
				filenames.append(os.path.join(path, filename))
			subdir_files[subdir] = filenames

		files[data_dir] = subdir_files
		
		print "Looped through %d files for %s" % (count, data_dir)
	
	return files

if __name__ == "__main__":
	if "asts" in OUTPUT:
		subdirs = ["asts/"]
	else:
		subdirs = ["trajectories/"]
	files = main(["data/hoc%s/" % suffix], subdirs)

	examples = []

	for key in files:
		print "%s has %d subdirs." % (key, len(files[key]))

		print "The subdirs are:"
		for subdir_key in files[key]:
			print "%s which has %d items" % (subdir_key, len(files[key][subdir_key]))
			#print "Some of these include: ", files[key][subdir_key][-10:]
			if "asts" in OUTPUT:
				examples.append(files[key][subdir_key]) #for asts
			else:
				examples.append(files[key][subdir_key][:-2]) #for traj, exclude final counts and idMap file
		print

	#TODO: didn't need to do this, could just do what we did in prev comment
	#For asts, want .json extension files
	desired_files = []
	if "asts" in OUTPUT:
		for lst in examples:
			for filepath in lst:

				#For asts
				filename = filepath.split("/")[-1]
				if filename.split(".")[-1] == "json":
					desired_files.append(filepath)
	else:
		#For traj, already excluded final two files
		for lst in examples:
			for filepath in lst:
				desired_files.append(filepath)

	with open(OUTPUT, "w") as f:
		for filepath in desired_files:
			print>>f, filepath
	"""
	path1 = ["/mnt/c/Users/Irving/Documents/Dropbox/CS221/data/hoc4/asts/997.json"]
	#for json_file in json_examples[-1]:
	for json_file in path1:
		count = 10
		with open(json_file, "r") as f:
			print "Opened %s." % json_file
			file_read = json.load(f)
			print file_read.keys()[:5]
			#keys are 'type', 'id', 'children'
			pprint(file_read['type'])
			pprint(file_read['id'])
			pprint(file_read['children'])

			count -= 1
			if count == 0:
				break
	
	Some of these include:  ['/mnt/c/Users/Irving/Documents/Dropbox/CS221/data/hoc4/asts/997.json', '/mnt/c/Users/Irving/Documents/Dropbox/CS221/data/hoc4/asts/9977.json', '/mnt/c/Users/Irving/Documents/Dropbox/CS221/data/hoc4/asts/9983.json', '/mnt/c/Users/Irving/Documents/Dropbox/CS221/data/hoc4/asts/9992.json', '/mnt/c/Users/Irving/Documents/Dropbox/CS221/data/hoc4/asts/9993.json', '/mnt/c/Users/Irving/Documents/Dropbox/CS221/data/hoc4/asts/9994.json', '/mnt/c/Users/Irving/Documents/Dropbox/CS221/data/hoc4/asts/9998.json', '/mnt/c/Users/Irving/Documents/Dropbox/CS221/data/hoc4/asts/9999.json', '/mnt/c/Users/Irving/Documents/Dropbox/CS221/data/hoc4/asts/counts.txt', '/mnt/c/Users/Irving/Documents/Dropbox/CS221/data/hoc4/asts/unitTestResults.txt']
	"""
