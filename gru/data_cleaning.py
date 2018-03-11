import re
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.stanford import StanfordTokenizer
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
    print('Loading training data...')
    train = pd.read_csv("../data/train.csv")
    print('Loading test data...')
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
    tokenizer = StanfordTokenizer()
    all_tokenized = all_questions.apply(tokenizer.tokenize)
    print(all_tokenized.shape)
    return
    all_lens = all_questions.str.len()
    plt.hist(all_lens)
    plt.show()

    # # Generate a normal distribution, center at x=0 and y=5
    # x = np.random.randn(N_points)
    # y = .4 * x + np.random.randn(100000) + 5

    # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # # We can set the number of bins with the `bins` kwarg
    # axs[0].hist(x, bins=n_bins)
    # axs[1].hist(y, bins=n_bins)

if __name__=="__main__":
    visualize_data('../data/train_clean.csv')

#TODO: Analyze the data, decide on best fixed SENT_LEN, maybe cut out questions that are too long