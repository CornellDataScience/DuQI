import re
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt

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

def save_clean_data():
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    print ('Preprocessing train Q1s...')
    train["question1"] = train["question1"].fillna("").apply(preprocess)
    print ('Preprocessing train Q2s...')
    train["question2"] = train["question2"].fillna("").apply(preprocess)
    print ('Preprocessing test Q1s...')
    test["question1"] = test["question1"].fillna("").apply(preprocess)
    print ('Preprocessing test Q2s...')
    test["question2"] = test["question2"].fillna("").apply(preprocess)
    print ('Storing data in CSV format...')
    train.to_csv('../data/train_clean.csv')
    test.to_csv('../data/test_clean.csv')
    print ('Data cleaned.')

def visualize_data(csvfilepath):
    data = pd.read_csv(csvfilepath)
    all_questions = data['question1'].append(data['question2'])
    all_split = all_questions.fillna("").apply(lambda x: x.split() if isinstance(x,str) else print(x))
    all_lens = all_split.str.len()
    plt.hist(all_lens,bins=200)
    plt.xlabel('Length (in words)')
    plt.ylabel('Frequency')
    plt.show()

def get_example_sents(csvfilepath,length):
    data = pd.read_csv(csvfilepath)
    all_questions = data['question1'].append(data['question2'])
    all_split = all_questions.fillna("").apply(lambda x: x.split() if isinstance(x,str) else print(x))
    all_lens = all_split.str.len()
    all_questions = all_questions[all_lens==length]
    chosen = np.random.choice(all_questions.index.values, 5)
    chosen_sents = all_questions.ix[chosen]
    for _, val in chosen_sents.iteritems():
        print (val)
        print()

if __name__=="__main__":
    get_example_sents('../data/train_clean.csv',50)