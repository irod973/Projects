#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File: pageRank.py
Author: Irving Rodriguez
Description: Determines the pageRank of a link matrix using the power method.
Derivation and explanation at: http://www.ams.org/samplings/feature-column/fcarc-pagerank
"""

import numpy as np

'''
Function: setup
Description: Transposes the link matrix, finds the dangling nodes, and counts the number of targets for each page.
Input: outLinks, array of arrays. pages should be coded by index: page1 = 1, page2 = 2, etc. thus
	outLinks[1] = [4, 6, 9]
	means that page1 targets page4, page6, page9
Output: inlinks, array of arrays. pages will be coded by index:
	page1 = 1, page2 = 2, etc. thus,
	inlinks[4] = [1]
	means that page4 is targeted by page1
'''
def setup(outLinks):
	#Total number of pages
	numPages = len(outLinks)

	#inLinks will contain the pages that target the given page.
	inLinks = [ [] for i in range(numPages)]

	#numLinks will contain the number of targets for each page
	numLinks = np.zeros(numPages, np.int32)

	#the pages with no targets
	danglingNodes = []

	for pageIndex in range(numPages):
		#Check for dangling node
		if len(outLinks[pageIndex]) == 0:
			danglingNodes.append(pageIndex)

		else:
			#Count this page's number of targets
			numLinks[pageIndex] = len(outLinks[pageIndex])
			#For each target, 
			for targetIndex in outLinks[pageIndex]:
				inLinks[targetIndex].append(pageIndex)

	#Convert to numpy arrays
	inLinks = [np.array(pageIndex) for pageIndex in inLinks]
	numLinks = np.array(numLinks)
	danglingNodes = np.array(danglingNodes)

	return inLinks, numLinks, danglingNodes

'''
Function: vectorGenerator
Description: A generator wrapper for calculating the principal eigenvector of the Google matrix through the power method.
	The generator allows us to iterate through the power method while only holding the current version of the eigenvector in memory
Inputs:
	inLinks: array of arrays. pages will be coded by index:
	page1 = 1, page2 = 2, etc. thus,
	inlinks[4] = [1]
	means that page4 is targeted by page1
	numLinks: numLinks[i] contains the number of targets that page i contains
danglingNodes: an array of indices of pages with no targets
Output: newVector, the current iteration of the eigenvector
'''
def vectorGenerator(inLinks, numLinks, danglingNodes, alpha=0.85, convergeFactor = 0.001, convergeSteps=10):

	numPages = len(inLinks)
	numDanglingPages = danglingNodes.shape[0]

	#The vectors calculated in the power method. Initial guess: 1/number of links
	newVector = np.ones((numPages,), np.float32) / numPages
	oldVector = np.ones((numPages,), np.float32) / numPages

	#add failsafe? for breaking after certain number of iterations?

	converged = False
	while not converged:
		#suggested that the vector be normalized, for stability
		newVector /= sum(newVector)

		#run through certain number of iterations before checking for sufficient 
		for step in range(convergeSteps):
			#swap the new and old vector after an iteration
			oldVector, newVector = newVector, oldVector

			#calculate the N piece of the Google matrix; note that its rows are all identical (all ones, with a cofactor)
			elemOfN = (1 - alpha) * sum(oldVector) / numPages

			#calculate the A piece of the Google matrix; note that its rows are all identical (zeros except for indices of dangling nodes)
			elemOfA = 0.0
			if numDanglingPages > 0:
				elemOfA = alpha * sum(oldVector.take(danglingNodes, axis=0)) / numPages

			#calculate the H piece of the Google matrix; note that it should be sparse (small number of targets per index, on average)
			pageIndex = 0
			while pageIndex < numPages:
				page = inLinks[pageIndex]
				h = 0.0

				#If the page is not empty
				if page.shape[0]:
					h = alpha * np.dot(oldVector.take(page, axis=0), 1.0 / numLinks.take(page, axis=0))
				newVector[pageIndex] = h + elemOfA + elemOfN
				pageIndex += 1

			#Break once the vector stops 'converging' sufficiently
			diff = newVector - oldVector

			converged = np.sqrt(np.dot(diff, diff)) / numPages < convergeFactor
			#Otherwise, yield the iterator
			yield newVector

'''
Function: pageRank
Description: Calculates the pageRank of a specified matrix, outLinks, using the power method, calling the generator specified in vectorGenerator
'''
def pageRank(outLinks, alpha = 0.85, convergeFactor = 0.001, convergeSteps = 10):

	#Transpose the link matrix (for easier calculation) 
	inLinks, numLinks, danglingNodes = setup(outLinks)

	for genVector in vectorGenerator(inLinks, numLinks, danglingNodes, alpha = alpha, convergeFactor = convergeFactor, convergeSteps = convergeSteps):

		eigenvector = genVector

	return eigenvector 
