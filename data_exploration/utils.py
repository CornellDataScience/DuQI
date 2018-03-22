import pandas as pd
from nltk.util import ngrams
from nltk.corpus import stopwords
import spacy
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import timeit
import textacy

#COLUMN NAMES
QUESTION_1 	= "question1"
QUESTION_2 	= "question2"
IS_DUP 		= "is_duplicate"
WORDS_NOT_IN_Q1 = "words_not_in_q1"
WORDS_NOT_IN_Q2 = "words_not_in_q2"
POS_NOT_IN_Q1 	= "pos_not_in_q1"
POS_NOT_IN_Q2 	= "pos_not_in_q2"

#CONSTANTS
STOPWORDS = set(stopwords.words('english'))
PUNC_TABLE = str.maketrans("","",".,?")
LANG_MODEL = spacy.load("en_core_web_sm")

def normalize_text(text):
	text = textacy.preprocess.normalize_whitespace(textacy.preprocess.transliterate_unicode(str(text)))
	return textacy.preprocess_text(text, fix_unicode=False, lowercase=False, transliterate=False,
        							no_urls=True, no_emails=True, no_phone_numbers=True,
        							no_numbers=False, no_currency_symbols=True, no_punct=False,
        							no_contractions=True, no_accents=True)

"""
"""
def readData(fileName, normalize = False):
	df = pd.read_csv(fileName)
	if not normalize:
		df[QUESTION_1] = df[QUESTION_1].apply(lambda x : normalize_text(x))
		df[QUESTION_2] = df[QUESTION_2].apply(lambda x : normalize_text(x))
	return df

def appendUnsharedWords(questionFrame):
	start_time = timeit.default_timer()
	newFrame = questionFrame.copy(deep =  True)
	newFrame[WORDS_NOT_IN_Q2] = newFrame.apply(lambda x : differenceInWords(x[QUESTION_1], x[QUESTION_2]), axis = 1)
	print(timeit.default_timer() - start_time)
	newFrame[WORDS_NOT_IN_Q1] = newFrame.apply(lambda x : differenceInWords(x[QUESTION_2], x[QUESTION_1]), axis = 1)
	print(timeit.default_timer() - start_time)
	return newFrame

def appendJaccardSimilarity(questionFrame, k):
	newFrame = questionFrame.copy(deep = True)
	newFrame["jaccard"] = newFrame.apply(lambda x : jaccardSimilarity(x[QUESTION_1], x[QUESTION_2], k), axis = 1)
	return newFrame

def appendUnSharedPOS(questionFrame):
	start_time = timeit.default_timer()
	newFrame = questionFrame.copy(deep =  True)
	newFrame[POS_NOT_IN_Q1] = newFrame.apply(lambda x : sentencePOS(x[WORDS_NOT_IN_Q1]) , axis = 1)
	newFrame[POS_NOT_IN_Q2] = newFrame.apply(lambda x : sentencePOS(x[WORDS_NOT_IN_Q2]) , axis = 1)
	print(timeit.default_timer() - start_time)
	return newFrame

####################

def differenceInWords(q1, q2):
	try:
		q1Shingles = set(q1.lower().translate(PUNC_TABLE).split()).difference(STOPWORDS)
		q2Shingles = set(q2.lower().translate(PUNC_TABLE).split()).difference(STOPWORDS)
		return " ".join(q1Shingles.difference(q2Shingles))
	except:
		None

def jaccardSimilarity(q1, q2, k):
	try:
		q1Shingles = set(ngrams(q1.lower().translate(PUNC_TABLE).split(), k))
		q2Shingles = set(ngrams(q2.lower().translate(PUNC_TABLE).split(), k))
		return len(q1Shingles.intersection(q2Shingles)) / len(q1Shingles.union(q2Shingles))
	except:
		return None

def sentencePOS(sentence):
	try:
		doc = LANG_MODEL(sentence)
		posTags = set(map(lambda x : x.pos_, doc))
		return posTags
	except:
		return set()

def createDataMatrix(dataFrame, addCols = True, semToIndices = None):
	nextIndex = 1
	if semToIndices is None:
		semToIndices = {"jaccard" : 0}
	xMatrix = np.zeros((len(dataFrame), len(semToIndices)))

	for index, row in dataFrame.iterrows():
		xMatrix[index, semToIndices["jaccard"]] = row["jaccard"]

		for tag in list(row[POS_NOT_IN_Q1]):
			if tag not in semToIndices and addCols:
				semToIndices[tag] = nextIndex
				nextIndex = nextIndex + 1
				xMatrix = np.hstack((xMatrix, np.zeros((len(dataFrame), 1))))

			xMatrix[index, semToIndices[tag]] += 1

		for tag in list(row[POS_NOT_IN_Q2]):
			if tag not in semToIndices and addCols:
				semToIndices[tag] = nextIndex
				nextIndex = nextIndex + 1
				xMatrix = np.hstack((xMatrix, np.zeros((len(dataFrame), 1))))
			
			try:
				xMatrix[index, semToIndices[tag]] = np.abs(xMatrix[index, semToIndices[tag]] + 1)
			except:
				pass
		
	return semToIndices, xMatrix

def createBayesClassifier(dataMatrix, y, dupRatio):
	clf = MultinomialNB(class_prior = [1 - dupRatio, dupRatio])
	clf.fit(dataMatrix, y)
	return clf


#jaccard similarity
#best relation is with single jaccard, have better histogram

#naive bayes
#features include jaccard, use of POS
#IDEA
#- usuallly stuff like nouns, you have non-duplicates
#- people have different ways of saying a sentence but that usually
#involves non - content words
#but obviously doesn't take into account the structure or the exact meaning of words
#doesn't assign POS based on location in the sentence which technically affects the
#POS accuracy



























