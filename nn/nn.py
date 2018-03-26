# packages
import keras as k
import numpy as np
import pandas as pd
import gensim as gs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# files
import constants as c
from word2vecmodel import Word2VecModel
from preprocessing import exclude_sents, train_val_split

class Model:
    def __init__(self):
        """Arguments: 
           - model_name: name of the model to load/save to
           - use_pretrained: if True, then loading model from model_name, else training
        """
        data = pd.read_csv('../data/train_clean.csv')
        # dropping low-length and high-length sentences
        data = exclude_sents(data)
        # randomly shuffling data and separating into train/val data
        train_data, val_data = train_val_split(data)
        # pd.Series to ndarray
        train_q1_str, train_q2_str = train_data['question1'].values, train_data['question2'].values
        val_q1_str, val_q2_str = val_data['question1'].values, val_data['question2'].values

        print('Fitting tokenizer...')
        self.tokenizer = Tokenizer(filters="", oov_token='!UNK!')
        self.tokenizer.fit_on_texts(np.append(train_q1_str, train_q2_str))
        self.w2v = Word2VecModel()
        unk_embed = self.produce_unk_embed()

        print('Converting strings to int arrays...')
        self.x_train_q1 = pad_sequences(self.tokenizer.texts_to_sequences(train_q1_str), maxlen=c.SENT_LEN)
        self.x_train_q2 = pad_sequences(self.tokenizer.texts_to_sequences(train_q2_str), maxlen=c.SENT_LEN)
        self.y_train = train_data['is_duplicate'].values
        self.x_val_q1 = pad_sequences(self.tokenizer.texts_to_sequences(val_q1_str), maxlen=c.SENT_LEN)
        self.x_val_q2 = pad_sequences(self.tokenizer.texts_to_sequences(val_q2_str), maxlen=c.SENT_LEN)
        self.y_val = val_data['is_duplicate'].values

        print('Creating embeddings matrix...')
        num_words = len(self.tokenizer.word_index)+2    # 0 index is reserved, oov token appended
        self.embedding_matrix = np.zeros((num_words, c.WORD_EMBED_SIZE))
        for word, i in self.tokenizer.word_index.items():
            try:
                embedding_vector = self.w2v.model.wv[word]
            except KeyError:
                embedding_vector = unk_embed
            self.embedding_matrix[i] = embedding_vector

    def load_pretrained(self, *, model_name):
        print('Loading model...')
        self.model = k.models.load_model('../models/'+model_name)
        print('Model loaded from models/'+model_name)

    def train_model(self, *, model_name, model_func):
        print('Training '+model_name+' model...')
        self.model = model_func()
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit([self.x_train_q1, self.x_train_q2], self.y_train,
                    validation_data=([self.x_val_q1, self.x_val_q2], self.y_val),
                    batch_size=c.BATCH_SIZE,
                    epochs=c.NUM_EPOCHS)
        print('Model trained.\nSaving model...')
        self.model.save('../models/'+model_name)
        print('Model saved to models/'+model_name)

    def evaluate_preds(self):
        """Prints: accuracy of evaluation on training and validation data.
        """
        pred_train = self.model.predict([self.x_train_q1, self.x_train_q2])
        acc_train = self.compute_accuracy(pred_train, self.y_train)
        print('* Accuracy on training set: %0.2f%%' % (100 * acc_train))

        pred_val = self.model.predict([self.x_val_q1, self.x_val_q2])
        acc_val = self.compute_accuracy(pred_val, self.y_val)
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
                             activation='relu'))    #TODO: test tanh
        return gru

    def lstm_embedding(self):
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
        gru.add(k.layers.LSTM(c.SENT_EMBED_SIZE,
                             activation='relu'))    #TODO: test tanh
        return gru

    def eucl_dist(self, vects):
        x, y = vects
        return k.backend.sqrt(k.backend.sum(k.backend.square(x - y), axis=1, keepdims=True))

    def eucl_dist_shape(self, shapes):
        shape1, _ = shapes
        return (shape1[0], 1)

    def gru_similarity_model(self):
        """GRU embedding -> Euclidean distance -> sigmoid activation
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

    def lstm_similarity_model(self):
        """LSTM embedding -> Euclidean distance -> sigmoid activation
        """
        input1 = k.layers.Input(shape=(c.SENT_LEN,))
        input2 = k.layers.Input(shape=(c.SENT_LEN,))
        lstm1_out = self.lstm_embedding()(input1)
        lstm2_out = self.lstm_embedding()(input2)
        distance = k.layers.Lambda(self.eucl_dist,
                                   output_shape=self.eucl_dist_shape)([lstm1_out,lstm2_out])
        out = k.layers.Dense(1, activation="sigmoid")(distance)
        model = k.models.Model(inputs=[input1, input2], outputs=[out])
        return model

    def compute_accuracy(self, preds, labels):
        return labels[preds.ravel() >= 0.5].mean()

if __name__=="__main__":
    m = Model()
    m.train_model(model_name='lstm_v1.h5',model_func=m.lstm_similarity_model)
    # m.load_pretrained(model_name='')
    m.evaluate_preds()