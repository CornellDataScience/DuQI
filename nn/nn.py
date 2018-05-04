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
from abs_layer import AbsLayer
from preprocessing import augmented, clean_string

import pdb

class Model:
    def __init__(self,*,fold_num):
        """Arguments: 
           - model_name: name of the model to load/save to
           - use_pretrained: if True, then loading model from model_name, else training
        """
        csv_file = '../data/train_clean.csv'
        # data augmentation
        train_data, val_data = augmented(csv_file, method='AUG_TRAIN',fold_num=fold_num)
        # pd.Series to ndarray
        train_q1_str, train_q2_str = train_data['question1'].values, train_data['question2'].values
        train_labels = train_data['is_duplicate'].values
        val_q1_str, val_q2_str = val_data['question1'].values, val_data['question2'].values
        val_labels = val_data['is_duplicate'].values

        print('Fitting tokenizer...')
        self.tokenizer = Tokenizer(filters="", oov_token='!UNK!')
        self.tokenizer.fit_on_texts(np.concatenate([train_q1_str,train_q2_str]))
        self.glove = self.glove_dict()
        unk_embed = self.produce_unk_embed()

        print('Converting training strings to int arrays...')
        self.x_train_q1 = pad_sequences(self.tokenizer.texts_to_sequences(train_q1_str), maxlen=c.SENT_LEN)
        self.x_train_q2 = pad_sequences(self.tokenizer.texts_to_sequences(train_q2_str), maxlen=c.SENT_LEN)
        # one-hotting the labels
        self.y_train = np.zeros((len(train_q1_str),2))
        self.y_train[:,1] = 1
        self.y_train[train_labels==0] = np.array([1,0])

        print('Converting validation strings to int arrays...')
        self.x_val_q1 = pad_sequences(self.tokenizer.texts_to_sequences(val_q1_str), maxlen=c.SENT_LEN)
        self.x_val_q2 = pad_sequences(self.tokenizer.texts_to_sequences(val_q2_str), maxlen=c.SENT_LEN)
        # one-hotting the labels
        self.y_val = np.zeros((len(val_q1_str),2))
        self.y_val[:,1] = 1
        self.y_val[val_labels==0] = np.array([1,0])

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
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

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
    
    def is_dup(self, q1, q2):
        """Returns: Probability that questions are duplicates
        """
        q1 = pad_sequences(self.tokenizer.texts_to_sequences([clean_string(q1)]), maxlen=c.SENT_LEN)
        q2 = pad_sequences(self.tokenizer.texts_to_sequences([clean_string(q2)]), maxlen=c.SENT_LEN)
        pred = self.model.predict([q1,q2])
        return pred[0][1]

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
        return np.array([acc_train,f1_train,acc_val,f1_val])

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
        gru.add(k.layers.Masking(mask_value=0., input_shape=(c.SENT_LEN, c.WORD_EMBED_SIZE)))
        gru.add(k.layers.GRU(c.SENT_EMBED_SIZE,
                             dropout=0.2,
                             activation='tanh', # relu explodes, maybe test grad clipping/elu?
                             kernel_regularizer=k.regularizers.l2(0.0001),
                             recurrent_regularizer=k.regularizers.l2(0.0001),
                             bias_regularizer=k.regularizers.l2(0.0001),
                             implementation=2)) # better GPU performance
        gru1_out = gru(input1)
        gru2_out = gru(input2)

        synth_sub = k.layers.subtract([gru1_out, gru2_out])
        synth_feat1 = AbsLayer()(synth_sub)
        synth_feat2 = k.layers.multiply([gru1_out, gru2_out])
        grus_out = k.layers.concatenate([gru1_out, gru2_out, synth_feat1, synth_feat2])
        dense1_out = k.layers.Dense(100,
                                    kernel_regularizer=k.regularizers.l2(0.0001),
                                    bias_regularizer=k.regularizers.l2(0.0001))(grus_out)
        norm1_out = k.layers.BatchNormalization()(dense1_out)
        active1_out = k.layers.Activation('relu')(norm1_out)
        out = k.layers.Dense(2, activation="softmax",
                                kernel_regularizer=k.regularizers.l2(0.0001),
                                bias_regularizer=k.regularizers.l2(0.0001))(active1_out)
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
    MODEL_NAME = 'gru_v6_augtrain'
    summary = np.zeros((c.NUM_FOLDS,4))
    for i in range(c.NUM_FOLDS):
        m = Model(fold_num=i)
        m.train_model(model_name=MODEL_NAME+'_fold'+str(i)+'.h5',model_func=m.gru_similarity_model)
        summary[i] = m.evaluate_preds()
    pd_sum_cols = ['train_acc','train_f1','val_acc','val_f1']
    pd_summary = pd.DataFrame(data=summary,index=np.arange(c.NUM_FOLDS),columns=pd_sum_cols)
    pd_means = pd_summary.mean(axis=0)
    pd_summary.append(pd_means, ignore_index=True)
    print(pd_summary)
    pd_summary.to_csv('../models/'+MODEL_NAME+'_stats.csv')

    # m = Model(fold_num=0)
    # m.load_pretrained(model_name='gru_v5_fold0.h5',model_func=m.gru_similarity_model)
    # summary = m.evaluate_preds()
    # print(summary)
    