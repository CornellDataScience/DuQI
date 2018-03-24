import numpy as np
import pandas as pd
import gensim as gs
from data_cleaning import split_and_exclude
import constants as c

class Word2VecModel:

    def __init__(self,use_pretrained=True):
        if not use_pretrained:
            train = pd.read_csv("../data/train_clean.csv")
            train = split_and_exclude(train)
            train['question1'] = train['question1'].apply(' '.join)
            train['question2'] = train['question2'].apply(' '.join)
            #TODO: create embedding for unknown word
            all_questions = train['question1'].append(train['question2'])
            sentences = all_questions.fillna('').apply(lambda x: x.split())
            print('Training model...')
            self.model = gs.models.Word2Vec(sentences, 
                                            size=c.WORD_EMBED_SIZE, 
                                            window=5, 
                                            min_count=c.UNKNOWN_MIN_COUNT, 
                                            workers=4)
            print('Done.')
            
            self.model.save(c.WORD2VEC_FILEPATH)
        else:
            self.model = gs.models.Word2Vec.load(c.WORD2VEC_FILEPATH)

if __name__=='__main__':
    m = Word2VecModel(False)