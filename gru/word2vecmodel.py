import numpy as np
import pandas as pd
import gensim as gs
from preprocessing import exclude_sents
import constants as c

class Word2VecModel:

    def __init__(self,use_pretrained=True):
        if not use_pretrained:
            train = pd.read_csv("../data/train_clean.csv")
            train = exclude_sents(train)
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