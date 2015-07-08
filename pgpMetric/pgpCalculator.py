'''
File: pgpCalculator
Author: Irving Rodriguez
Description: This program takes a player's name from the 2013-2014 NBA season as an input and calculates their points generated per possession as outlined by the accompanying pgpExplained.docx file.
'''

import json
from nbaPlayer import Player
from pprint import pprint

#JSON files containing player statistics for the 2013-2014 NBA season in Python dictionaries.

PASSING_DATA = "passingData.json"
TOUCHES_DATA = "touchesData.json"
CATCH_SHOOT_DATA = "catchShootData.json"
PULL_UP_DATA = "pullUpShootData.json"
DRIVES_DATA = "drivesData.json"

def main():
	statFiles = [PASSING_DATA, TOUCHES_DATA, CATCH_SHOOT_DATA, PULL_UP_DATA, DRIVES_DATA]

	##Loop through the stat files to create the master dictionary containing every player and their stats.
	masterData = makeMasterData(statFiles)

	while True:
	##Prompt the user for a player's name
		name = raw_input("Enter a player's name (Enter 0 to exit): ")

		if name == "0": break
		if name.upper() not in masterData:
			print "Uh oh, that person did not play in the NBA in 2013-2014"
		else:
			player = Player(masterData[name.upper()], name.upper())
			print name, "'s points generated per possession: ", player.calcPGP()

'''
Function: makeMasterData
Input: A list containing data file names.
Output: A dictionary whose keys are player names and values are a second dictionary. The second dictionary has strings describing stats as keys with those stats as values.
'''	
def makeMasterData(fileList):
	lst = []
	for fil in fileList:
		json_data=open(fil)
		lst.append(json.load(json_data))
		json_data.close()

	playerMap = {}
	for currData in lst:
		
		##Each data file is structured so that 0th element contains a legend equating stats abbreviations with stat names (i.e. 'AST':Assist)
		legend = currData[0]
		for i in range(1, len(currData)):

			##Player's name
			player = currData[i][1].upper()
			##If the player's name is not already in the dictionary, create a new dict for the player
			if player not in playerMap: playerMap[player] = {}

			##Loop through each stat and append it accordingly in the player's dicitonary.
			for j in range(1, len(legend)):
				playerMap[player][legend[j]] = currData[i][j]
	
	return playerMap

if __name__ == "__main__":
		main()