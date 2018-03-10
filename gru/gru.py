import keras
import numpy as np
import pandas as pd

# input tensor is (batch_size, timesteps, input_dim)
class Model:

    EMBEDDING_SIZE = 100            # size of word embedding
    VECTOR_DIM_OUT = 50             # size of output vector
    SENT_LEN = 0                    # fixed length of sentence

    def __init__(self,is_pretrained):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.GRU(VECTOR_DIM_OUT,
                                        input_shape = (SENT_LEN, EMBEDDING_SIZE),
                                        activation='relu',
                                        dropout=0.0, 
                                        recurrent_dropout=0.0))


    #TODO: Build full GRU model