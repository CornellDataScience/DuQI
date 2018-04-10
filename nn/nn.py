# packages
import io
import tensorflow as tf
import keras as k
import numpy as np
import pandas as pd
import gensim as gs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score
# files
import constants as c
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
        self.glove = self.glove_dict()
        unk_embed = self.produce_unk_embed()

        print('Converting training strings to int arrays...')
        self.x_train_q1 = pad_sequences(self.tokenizer.texts_to_sequences(train_q1_str), maxlen=c.SENT_LEN)
        self.x_train_q2 = pad_sequences(self.tokenizer.texts_to_sequences(train_q2_str), maxlen=c.SENT_LEN)
        # one-hotting the labels
        self.y_train = np.zeros((len(train_q1_str),2))
        self.y_train[:,1] = 1
        y_train_labels = train_data['is_duplicate'].values
        self.y_train[y_train_labels==0] = np.array([1,0])

        print('Converting validation strings to int arrays...')
        self.x_val_q1 = pad_sequences(self.tokenizer.texts_to_sequences(val_q1_str), maxlen=c.SENT_LEN)
        self.x_val_q2 = pad_sequences(self.tokenizer.texts_to_sequences(val_q2_str), maxlen=c.SENT_LEN)
        # one-hotting the labels
        self.y_val = np.zeros((len(val_q1_str),2))
        self.y_val[:,1] = 1
        y_val_labels = val_data['is_duplicate'].values
        self.y_val[y_val_labels==0] = np.array([1,0])

        print('Creating embeddings matrix...')
        num_words = len(self.tokenizer.word_index)+2    # 0 index is reserved, oov token appended
        self.embedding_matrix = np.zeros((num_words, c.WORD_EMBED_SIZE))
        for word, i in self.tokenizer.word_index.items():
            try:
                embedding_vector = self.glove[word]
            except KeyError:
                embedding_vector = unk_embed
            self.embedding_matrix[i] = embedding_vector

    def glove_dict(self):
        """Creates the GloVe dictionary from pretrained GloVe embeddings.
        """
        print('Creating GloVe dict...')
        embeddings_dict = {}
        with io.open(c.GLOVE_FILEPATH,'r',encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                embeddings_dict[word] = np.asarray(values[1:], dtype='float32')
        return embeddings_dict

    def load_pretrained(self, *, model_name, model_func):
        """Loads pretrained model weights.
        """
        print('Loading model...')
        self.model = model_func()
        self.model.load_weights('../models/'+model_name)
        print('Model loaded from models/'+model_name)

    def train_model(self, *, model_name, model_func):
        """Trains the model, saves the model weights.
        """
        print('Training '+model_name+' model...')
        self.model = model_func()
        self.model.compile(loss='binary_crossentropy', optimizer='adam')

        # TensorBoard callback
        tboard = k.callbacks.TensorBoard(log_dir='logs/'+model_name[:-3],
                                         write_graph=True,
                                         write_images=True)
        
        self.model.fit([self.x_train_q1, self.x_train_q2], self.y_train,
                    validation_data=([self.x_val_q1, self.x_val_q2], self.y_val),
                    batch_size=c.BATCH_SIZE,
                    epochs=c.NUM_EPOCHS,
                    callbacks=[tboard])

        print('Model trained.\nSaving model...')
        self.model.save_weights('../models/'+model_name)
        print('Model saved to models/'+model_name)

    def evaluate_preds(self):
        """Prints: accuracy and f1 of evaluation on training and validation data.
        """
        print('Generating predictions...')
        pred_train = self.model.predict([self.x_train_q1, self.x_train_q2])
        print('Evaluating predictions...')
        acc_train, f1_train = self.compute_accuracy(pred_train, self.y_train)
        print('* Accuracy on training set: %0.4f' % acc_train)
        print('* F1 score on training set: %0.4f' % f1_train)

        print('Generating predictions...')
        pred_val = self.model.predict([self.x_val_q1, self.x_val_q2])
        print('Evaluating predictions...')
        acc_val, f1_val = self.compute_accuracy(pred_val, self.y_val)
        print('* Accuracy on validation set: %0.4f' % acc_val)
        print('* F1 score on validation set: %0.4f' % f1_val)

    def produce_unk_embed(self):
        """Returns: unknown token embedding - the average embedding of every GloVe word
        """
        unknown_embedding = sum(self.glove.values())/len(self.glove.values())
        return unknown_embedding

    def gru_similarity_model(self):
        """GRU embedding -> Euclidean distance -> sigmoid activation
        """
        input1 = k.layers.Input(shape=(c.SENT_LEN,))
        input2 = k.layers.Input(shape=(c.SENT_LEN,))

        gru = k.models.Sequential()
        num_words = len(self.tokenizer.word_index.items())
        embed_matrix_init = lambda shape, dtype=None: self.embedding_matrix
        # input is integer matrix of size (None, SENT_LEN).
        gru.add(k.layers.Embedding(num_words,
                                   c.WORD_EMBED_SIZE,
                                   embeddings_initializer=embed_matrix_init,
                                   input_length=c.SENT_LEN))
        # shape = (None, SENT_LEN, WORD_EMBED_SIZE)
        gru.add(k.layers.GRU(c.SENT_EMBED_SIZE,
                             activation='tanh',         # relu explodes
                             implementation=2))         # better GPU performance
        gru1_out = gru(input1)
        gru2_out = gru(input2)

        #TODO: add additional features to concatenated feature vector
        grus_out = k.layers.concatenate([gru1_out, gru2_out])               # concatenate
        dense1_out = k.layers.Dense(100)(grus_out)                          # dense
        norm1_out = k.layers.BatchNormalization()(dense1_out)               # batch norm
        active1_out = k.layers.Activation('relu')(norm1_out)                # relu
        out = k.layers.Dense(2, activation="softmax")(active1_out)          # dense softmax
        model = k.models.Model(inputs=[input1, input2], outputs=[out])
        return model

    def compute_accuracy(self, preds, labels):
        """Returns: accuracy, f1 score
        """
        # preds, labels are nx2 - time to turn to 0s and 1s
        binarize = lambda x: 1-x.transpose().flatten()[:x.size//2]
        preds = np.round(preds)
        accuracy = accuracy_score(binarize(labels),binarize(preds))
        f1 = f1_score(binarize(labels),binarize(preds))
        return accuracy, f1

if __name__=="__main__":
    m = Model()
    m.train_model(model_name='glove_gru4.h5',model_func=m.gru_similarity_model)
    # m.load_pretrained(model_name='glove_gru2.h5',model_func=m.gru_similarity_model)
    m.evaluate_preds()