#baseline.py
#23rd March 2018
#Arnav Ghosh

import eval
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

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
	precision, recall, f1 = eval.getScores(preds, yTe)

	return precision, recall, f1