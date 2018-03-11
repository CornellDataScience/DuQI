import keras as k
import numpy as np
import pandas as pd
import gensim as gs

# input tensor is (batch_size, timesteps, input_dim)
class Model:

    WORD_EMBED_SIZE = 50            # size of word embedding
    SENT_LEN = 50                   # fixed length of sentence
    SENT_EMBED_SIZE = 50            # size of output vector
    TOP_WORD_THRESHOLD = 50         # words with at least this frequency are considered "top" words
    NUM_EPOCHS = 10
    BATCH_SIZE = 128
    MODEL_NAME = 'gru_v1.h5'
    WORD2VEC_FILEPATH = '../data/word2vecmodel'

    def __init__(self, have_model):

        data = pd.read_csv('../data/')
        x_train = #TODO: shape (traindatasize,WORD_EMBED_SIZE,SENT_LEN)x2 for two questions?
        x_train_q1 = #TODO: get Q1 features only
        x_train_q2 = #TODO: get Q2 features only
        x_val = #TODO: shape (testdatasize,WORD_EMBED_SIZE,SENT_LEN)x2 for two questions?
        x_val_q1 = #TODO: get Q1 features only
        x_val_q2 = #TODO: get Q2 features only
        y_train = #TODO: shape (traindatasize,)
        y_val = #TODO: shape (testdatasize,)

        if have_model:
            print('Loading model...')
            self.model = k.models.load_model('../data/'+self.MODEL_NAME)
            print('Model loaded from data/'+self.MODEL_NAME)
        else:
            self.model = self.similarity_model()
            self.model.compile(loss='mean_squared_error', optimizer='adam')
            print('Training model...')
            self.model.fit([x_train_q1, x_train_q2], y_train,
                        validation_data=([x_val_q1, x_val_q2], y_val),
                        batch_size=self.BATCH_SIZE,
                        nb_epoch=self.NUM_EPOCHS)
            print('Model trained.\nSaving model...')
            self.model.save('../data/'+self.MODEL_NAME)
            print('Model saved to data/'+self.MODEL_NAME)

        pred_train = self.model.predict([x_train_q1, x_train_q2])
        acc_train = self.compute_accuracy(pred_train, y_train)
        print('* Accuracy on training set: %0.2f%%' % (100 * acc_train))

        pred_val = self.model.predict([x_val_q1, x_val_q2])
        acc_val = self.compute_accuracy(pred_val, y_val)
        print('* Accuracy on validation set: %0.2f%%' % (100 * acc_val))

    def gru_embedding(self):
        """Returns: GRU model for sentence embedding, applied to each question input.
        """
        gru = k.layers.GRU(self.SENT_EMBED_SIZE,
                            input_shape = (self.SENT_LEN,self.WORD_EMBED_SIZE),
                            activation='relu', #TODO: test tanh
                            dropout=0.0) #TODO: test 0.2
        return gru

    def eucl_dist(self, vects):
        x, y = vects
        return k.backend.sqrt(k.backend.sum(k.backend.square(x - y), axis=1, keepdims=True))

    def eucl_dist_shape(self, shapes):
        shape1, _ = shapes
        return (shape1[0], 1)

    def similarity_model(self):
        """Returns: Full similarity model between two sentences, based on Euclidean distance.
        """
        input1 = k.layers.Input(shape=(self.SENT_LEN,self.WORD_EMBED_SIZE))
        input2 = k.layers.Input(shape=(self.SENT_LEN,self.WORD_EMBED_SIZE))
        gru1_out = self.gru_embedding()(input1)
        gru2_out = self.gru_embedding()(input2)
        distance = k.layers.Lambda(self.eucl_dist, output_shape=self.eucl_dist_shape)([gru1_out,gru2_out])
        model = k.models.Model(inputs=[input1, input2], outputs=[distance])
        return model

    def compute_accuracy(self, preds, labels):
        return labels[preds.ravel() < 0.5].mean()

    def create_word2vec(self, use_pretrained):
        """Returns: word2vec model
        """
        if not use_pretrained:
            #TODO: get sentences, should be iterable of lists of words per sentence
            model = gs.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4) #TODO: adjust parameters
            model.save(self.WORD2VEC_FILEPATH)
        else:
            model = gs.models.Word2Vec.load(self.WORD2VEC_FILEPATH)
        return model

if __name__=="__main__":
    m = Model(False)