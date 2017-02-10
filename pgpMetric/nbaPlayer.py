'''
Filename: nbaPlayer
Author: Irving Rodriguez
Description: This file outlines a class for an NBA Player. The four calculating functions are explained in the accompanying pgpExplained.docx file.
'''

class Player():

	##The player's properties are name and a dictionary containing that player's statistics. 
	def __init__(self, statsMap, name):
		self.name = name + " is name"
		self.data = statsMap

	def calcPGP(self):
		psa = self.calcPSA() ##Points from Successful Attempts
		ap = self.calcAP() ##Total Attempts at Points
		sap = self.calcSAP() ##Successful Attempts at Points
		pgp = (psa/self.data['FC_TCH']) * (sap/ap)
		return pgp

###Points from Successful Attempts
	def calcPSA(self):
		assistPoints = self.data['AST']*2.0 + self.data['AS_TFT'] + self.data['AST_SEC']
		catchShootPoints = self.data['CSFGM']*2 + self.data['CSFG3M']*3
		pullUpPoints = self.data['PUFGM']*2 + self.data['PUFG3M']*3
		drivePoints = self.data['DPP']
		psa = assistPoints + catchShootPoints + pullUpPoints + drivePoints
		return psa

##AP = Total Attempts at Points
	def calcAP(self):
		assistAttempts = self.data['AST_POT']
		catchShootAttempts = self.data['CSFGA'] + self.data['CSFG3A']
		pullUpAttempts = self.data['PUFGA'] + self.data['PUFG3A']
		driveAttempts = (self.data['DPP']/2.0) * (self.data['DFG_PCT'])
		ap = assistAttempts + catchShootAttempts + pullUpAttempts + driveAttempts
		return ap

###SAP = Successful Attempts at Points
	def calcSAP(self):
		assistsMade = self.data['AST'] + self.data['AS_TFT'] + self.data['AST_SEC']/2.0
		catchShootMade = self.data['CSFGM'] + self.data['CSFG3M']
		pullUpMade = self.data['PUFGM'] + self.data['PUFG3M']
		drivesMade = self.data['DPP']/2.0
		sap = assistsMade + catchShootMade + pullUpMade + drivesMade
		return sap
