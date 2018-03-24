#eval.py
#23 March 2018
#Arnav Ghosh

import numpy as np
import sklearn.metrics

def getScores(yTe, yPred):
	precision = sklearn.metrics.precision_score(yTe, yPred)
	recall = sklearn.metrics.recall_score(yTe, yPred)
	f1 = sklearn.metrics.f1_score(yTe, yPred)

	return precision, recall, f1

################ CUSTOM EVAL ################
def custGetScore(yPred, yTe):
	predDups = np.where(yPred == 1)
	predUnDups = np.where(yPred == 0)

	dupUnique, dupCounts = np.unique(yTe[predDups], return_counts = True)
	unDupUnique, unDupCounts = np.unique(yTe[predUnDups], return_counts = True)

	precision = dupCounts[1] / (dupCounts[1] + dupCounts[0])
	recall = dupCounts[1] / (dupCounts[1] + unDupCounts[1])
	f1 = 2 * ((precision * recall)/(precision + recall))

	return precision, recall, f1
