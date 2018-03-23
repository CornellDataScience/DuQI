#eval.py
#23 March 2018
#Arnav Ghosh

import numpy as np

def getMetrics(yPred, yTe):
	predDups = np.where(yPred == 1)
	predUnDups = np.where(yPred == 0)

	dupUnique, dupCounts = yTe[predDups].unique()
	unDupUnique, unDupCounts = yTe[predUnDups].unique()

	precision = dupCounts[1] / (dupCounts[1] + dupCounts[0])
	recall = dupCounts[1] / (dupCounts[1] + unDupCounts[1])
	f1 = 2 * ((precision * recall)/(precision + recall))

	return precision, recall, f1
