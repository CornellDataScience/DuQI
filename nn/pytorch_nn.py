import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

torch.manual_seed(1)

NUM_EPOCHS = 10
BATCH_SIZE = 256
WORD_EMBED_SIZE = 200           # size of word embedding
SENT_EMBED_SIZE = 300           # size of sentence embedding
SENT_LEN = 50                   # fixed length of sentence
SENT_INCLUSION_MIN = 3          # shorter sentences removed from training data
SENT_INCLUSION_MAX = SENT_LEN   # longer sentences removed from training data
NUM_FOLDS = 5

CSV_FILEPATH = '../data/train_clean.csv'
GLOVE_FILEPATH = '../lang_models/glove.6B.'+str(WORD_EMBED_SIZE)+'d.txt'

class Model(nn.Module):
    def __init__(self, data):
        raise NotImplementedError
    # TODO: forward
    # TODO: train model
    # TODO: evaluate model

def glove_dict():
        """Creates the GloVe dictionary from pretrained GloVe embeddings.
        """
        print('Creating GloVe dict...')
        embeddings_dict = {}
        with io.open(GLOVE_FILEPATH,'r',encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                embeddings_dict[word] = np.asarray(values[1:], dtype='float32')
        return embeddings_dict

def 

if __name__ == '__main__':
    data = pd.read_csv(CSV_FILEPATH)
    # TODO: exclude high, low length sentences
    
    # TODO: lemmatize
    # TODO: preprocess tokens
    # TODO: save to csv
    # TODO: augment data
    # TODO: train/val split based on fold
    # TODO: turn training data into embedding data (batch size, sentence len, embedding size)
    # TODO: initialize model with training data embeddings
    # TODO: evaluate model on validation data