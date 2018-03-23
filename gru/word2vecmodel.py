import numpy as np
import pandas as pd
import gensim as gs

class Word2VecModel:
    
    EMBEDDING_SIZE = 100
    WORD2VEC_FILEPATH = '../models/word2vecmodel-'+str(EMBEDDING_SIZE)+'d'

    def __init__(self,use_pretrained=True):
        if not use_pretrained:
            train = pd.read_csv("../data/train_clean.csv")
            all_questions = train['question1'].append(train['question2'])
            sentences = all_questions.fillna('').apply(lambda x: x.split())
            print('Training model...')
            self.model = gs.models.Word2Vec(sentences, 
                                            size=self.EMBEDDING_SIZE, 
                                            window=5, 
                                            min_count=100, 
                                            workers=4)
            print('Done.')
            self.model.save(self.WORD2VEC_FILEPATH)
        else:
            self.model = gs.models.Word2Vec.load(self.WORD2VEC_FILEPATH)

if __name__=='__main__':
    m = Word2VecModel(False)