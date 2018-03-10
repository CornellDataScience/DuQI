import re
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer

WNL = WordNetLemmatizer()

def lemmatizer(word):
    """Returns: lemmatized word if word >= length 5
    """
    if len(word)<4:
        return word
    return WNL.lemmatize(WNL.lemmatize(word, "n"), "v")

def preprocess(string): # From kaggle-quora-dup submission
    """Pipeline: string -> processed -> lemmatized -> output
    """
    string = string.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    string = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)
    string = re.sub(r"([0-9]+)000000", r"\1m", string)
    string = re.sub(r"([0-9]+)000", r"\1k", string)
    string = ' '.join([lemmatizer(w) for w in string.split()])
    return string

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train["question1"] = train["question1"].fillna("").apply(preprocess)
train["question2"] = train["question2"].fillna("").apply(preprocess)

#TODO: Tokenize the sentences