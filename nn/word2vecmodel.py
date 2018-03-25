import numpy as np
import pandas as pd
import gensim as gs

import constants as c
from preprocessing import exclude_sents, train_val_split

class Word2VecModel:

    def __init__(self,*,use_pretrained=True):
        if not use_pretrained:
            data = pd.read_csv("../data/train_clean.csv")
            data = exclude_sents(data)
            train, _ = train_val_split(data)
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
    m = Word2VecModel(use_pretrained=False)