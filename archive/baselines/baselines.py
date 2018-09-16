#baseline.py
#23rd March 2018
#Arnav Ghosh

import eval
import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

SVM_C_INIT = [1, 10, 100, 1000, 10000]
SVM_MAX_ITER = [1, 50, 100, 200, 500, 1000]

def createBayesClassifier(dataMatrix, yTr, dupRatio):
	clf = MultinomialNB(class_prior = [1 - dupRatio, dupRatio])
	clf.fit(dataMatrix, yTr)
	return clf

def createSVMClassifier(xTrain, yTrain, C, maxIter):
	svmCLF = svm.LinearSVC(penalty = "l1", loss = "squared_hinge", dual = False, C = C, max_iter = maxIter)
	svmCLF.fit(xTrain, yTrain)
	return svmCLF

def scoreClassifier(classifier, xTe, yTe):
	preds = classifier.predict(xTe)
	precision, recall, f1 = eval.getScores(yTe, preds)

	return precision, recall, f1

def svmParaSearch(xTrain, yTrain, xTe, yTe):
	results = {"C" : [], "iter": [], "precision": [], "recall" : [], "f1": []}

	for c in SVM_C_INIT:
		for i in SVM_MAX_ITER:
			clf = createSVMClassifier(xTrain, yTrain, c, i)
			precision, recall, f1 = scoreClassifier(clf, xTe, yTe)
			results["C"] = results["C"] + [c]
			results["iter"] = results["iter"] + [i]
			results["precision"] = results["precision"] + [precision]
			results["recall"] = results["recall"] + [recall]
			results["f1"] = results["f1"] + [f1]

	return pd.DataFrame(data = results)



