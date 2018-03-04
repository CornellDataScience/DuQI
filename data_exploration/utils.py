import pandas as pd
from nltk.util import ngrams
from nltk.corpus import stopwords

#COLUMN NAMES
QUESTION_1 = "question1"
QUESTION_2 = "question2"
IS_DUP = "is_duplicate"

#CONSTANTS
STOPWORDS = set(stopwords.words('english'))
PUNC_TABLE = str.maketrans("","",".,?")

"""
"""
def readData(fileName):
	return pd.read_csv(fileName)

def appendUnsharedWords(questionFrame):
	newFrame = questionFrame.copy(deep =  True)
	newFrame["words_not_in_q2"] = newFrame.apply(lambda x : differenceInWords(x[QUESTION_1], x[QUESTION_2]), axis = 1)
	newFrame["words_not_in_q1"] = newFrame.apply(lambda x : differenceInWords(x[QUESTION_2], x[QUESTION_1]), axis = 1)
	return newFrame

def differenceInWords(q1, q2):
	try:
		q1Shingles = set(q1.lower().translate(PUNC_TABLE).split()).difference(STOPWORDS)
		q2Shingles = set(q2.lower().translate(PUNC_TABLE).split()).difference(STOPWORDS)
		return (q1Shingles.difference(q2Shingles))
	except:
		None

def appendJaccardSimilarity(questionFrame, k):
	newFrame = questionFrame.copy(deep = True)
	newFrame["jaccard"] = newFrame.apply(lambda x : jaccardSimilarity(x[QUESTION_1], x[QUESTION_2], k), axis = 1)
	return newFrame

def jaccardSimilarity(q1, q2, k):
	try:
		q1Shingles = set(ngrams(q1.lower().translate(PUNC_TABLE).split(), k))
		q2Shingles = set(ngrams(q2.lower().translate(PUNC_TABLE).split(), k))
		return len(q1Shingles.intersection(q2Shingles)) / len(q1Shingles.union(q2Shingles))
	except:
		return None

# how many questions are the same
# how many are the same if we use the basic forms of the words
def basicStats(questionFrame):
	pass

