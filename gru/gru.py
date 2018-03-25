# packages
import keras as k
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import gensim as gs
from sklearn.model_selection import train_test_split
# files
from word2vecmodel import Word2VecModel
from preprocessing import exclude_sents
import constants as c

#TODO: Don't train tokenizer on validation data

class Model:
    def __init__(self, use_pretrained=True):
        data = pd.read_csv('../data/train_clean.csv')
        # dropping low-length and high-length sentences
        data = exclude_sents(data)
        # pd.Series to ndarray
        q1_strings = data['question1'].values
        q2_strings = data['question2'].values

        print('Fitting tokenizer...')
        self.tokenizer = Tokenizer(filters="")
        self.tokenizer.fit_on_texts(np.append(q1_strings, q2_strings))
        self.w2v = Word2VecModel()
        unk_embed = self.produce_unk_embed()
        
        print('Converting strings to int arrays...')
        q1_intarr = pad_sequences(self.tokenizer.texts_to_sequences(q1_strings), maxlen=c.SENT_LEN)
        q2_intarr = pad_sequences(self.tokenizer.texts_to_sequences(q2_strings), maxlen=c.SENT_LEN)
        labels_arr = np.array(data["is_duplicate"])
        # train-val shuffle and split
        x_train_q1, x_val_q1, x_train_q2, x_val_q2, y_train, y_val = \
            train_test_split(q1_intarr, q2_intarr, labels_arr, test_size=0.2, random_state=27)
        num_words = len(self.tokenizer.word_index)+1

        print('Creating embeddings matrix...')
        self.embedding_matrix = np.zeros((num_words, c.WORD_EMBED_SIZE))
        for word, i in self.tokenizer.word_index.items():
            try:
                embedding_vector = self.w2v.model.wv[word]
            except KeyError:
                embedding_vector = unk_embed
            self.embedding_matrix[i] = embedding_vector

        if use_pretrained:
            print('Loading model...')
            self.model = k.models.load_model('../models/'+c.MODEL_NAME)
            print('Model loaded from models/'+c.MODEL_NAME)
        else:
            print('Training model...')
            self.model = self.similarity_model()
            self.model.compile(loss='mean_squared_error', optimizer='adam')
            print('Training model...')
            self.model.fit([x_train_q1, x_train_q2], y_train,
                        validation_data=([x_val_q1, x_val_q2], y_val),
                        batch_size=c.BATCH_SIZE,
                        epochs=c.NUM_EPOCHS)
            print('Model trained.\nSaving model...')
            self.model.save('../models/'+c.MODEL_NAME)
            print('Model saved to models/'+c.MODEL_NAME)
        # predictions
        pred_train = self.model.predict([x_train_q1, x_train_q2])
        acc_train = self.compute_accuracy(pred_train, y_train)
        print('* Accuracy on training set: %0.2f%%' % (100 * acc_train))

        pred_val = self.model.predict([x_val_q1, x_val_q2])
        acc_val = self.compute_accuracy(pred_val, y_val)
        print('* Accuracy on validation set: %0.2f%%' % (100 * acc_val))

    def produce_unk_embed(self):
        """Returns: unknown token embedding - the average embedding of semi-infrequent words
        """
        unknown_embedding = np.zeros((c.WORD_EMBED_SIZE),dtype='float32')
        unknown_count = 0
        for word, vocab_obj in self.w2v.model.wv.vocab.items():
            if vocab_obj.count<c.MIN_COUNT:
                unknown_embedding += self.w2v.model.wv[word]
                unknown_count += 1
        unknown_embedding = unknown_embedding/unknown_count
        return unknown_embedding

    def gru_embedding(self):
        """Returns: GRU model for sentence embedding, applied to each question input.
        """        
        gru = k.models.Sequential()
        num_words = len(self.tokenizer.word_index.items())
        embed_matrix_init = lambda shape, dtype=None: self.embedding_matrix
        # the model will take as input an integer matrix of size (batch, input_length).
        gru.add(k.layers.Embedding(num_words,
                                   c.WORD_EMBED_SIZE,
                                   embeddings_initializer=embed_matrix_init,
                                   input_length=c.SENT_LEN))
        # now output shape is (None, SENT_LEN, WORD_EMBED_SIZE), where None is the batch dimension.
        gru.add(k.layers.GRU(c.SENT_EMBED_SIZE,
                             activation='relu', #TODO: test tanh
        return gru

    def eucl_dist(self, vects):
        x, y = vects
        return k.backend.sqrt(k.backend.sum(k.backend.square(x - y), axis=1, keepdims=True))

    def eucl_dist_shape(self, shapes):
        shape1, _ = shapes
        return (shape1[0], 1)

    def similarity_model(self):
        """Returns: Full similarity model between two sentences
        """
        input1 = k.layers.Input(shape=(c.SENT_LEN,))
        input2 = k.layers.Input(shape=(c.SENT_LEN,))
        gru1_out = self.gru_embedding()(input1)
        gru2_out = self.gru_embedding()(input2)
        distance = k.layers.Lambda(self.eucl_dist,
                                   output_shape=self.eucl_dist_shape)([gru1_out,gru2_out])
        out = k.layers.Dense(1, activation="sigmoid")(distance)
        model = k.models.Model(inputs=[input1, input2], outputs=[out])
        return model

    def compute_accuracy(self, preds, labels):
        return labels[preds.ravel() >= 0.5].mean()

if __name__=="__main__":
    m = Model(False)