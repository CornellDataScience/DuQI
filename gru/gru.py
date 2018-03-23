import keras as k
import numpy as np
import pandas as pd
import gensim as gs
from word2vecmodel import Word2VecModel

# input tensor is (batch_size, timesteps, input_dim)
class Model:

    WORD_EMBED_SIZE = 100           # size of word embedding
    SENT_LEN = 50                   # fixed length of sentence
    SENT_EMBED_SIZE = 100           # size of output vector
    TOP_WORD_THRESHOLD = 50         # words with at least this frequency are considered "top" words
    SENT_INCLUSION_MIN = 3          # any sentence of lesser length will be removed from the data
    SENT_INCLUSION_MAX = SENT_LEN   # any sentence of greater length will be removed from the data
    NUM_EPOCHS = 10
    BATCH_SIZE = 128
    MODEL_NAME = 'gru_v1.h5'

    def __init__(self, use_pretrained=True):

        # TODO: separate data stuff to separate function
        data = pd.read_csv('../data/train_clean.csv')
        # splitting sentence strings
        data['question1'] = data['question1'].str.split()
        data['question2'] = data['question2'].str.split()
        # removing floats (NaN?)
        data = data.drop(data[data['question1'].apply(type)==float].index)  
        data = data.drop(data[data['question2'].apply(type)==float].index)
        # removing sentences that are too short/long
        data = data.drop(data[data['question1'].apply(len)>self.SENT_INCLUSION_MAX].index)
        data = data.drop(data[data['question1'].apply(len)<self.SENT_INCLUSION_MIN].index)
        data = data.drop(data[data['question2'].apply(len)>self.SENT_INCLUSION_MAX].index)
        data = data.drop(data[data['question2'].apply(len)<self.SENT_INCLUSION_MIN].index)
        shuffled_data = data.sample(frac=1,random_state=2727)

        self.w2v = Word2VecModel()

        q1_split = shuffled_data['question1']
        # padding front of questions
        pad_front = lambda lst: ['!EMPTY!']*(self.WORD_EMBED_SIZE-len(lst))+lst
        q1_split = q1_split.apply(pad_front)
        q1_split = q1_split.apply(np.asarray)

        q1_vectors = q1_split.apply(self.words_to_embeds)
        
        
        #TODO #TODO #TODO: finish self.words_to_embeds
        print(q1_vectors)
        # words_to_embeds = np.vectorize(word_to_embed)
        # q1_vectors = q1_split.apply(words_to_embeds)
        # print (q1_vectors)
        #TODO: Finish data processing
        # the numpy array embedding for a word is w2vmodel.wv['someword']

        return
        
        y_data = shuffled_data['is_duplicate'].values    # ndarray (n,)
        
        # x_train = #TODO: shape (traindatasize,WORD_EMBED_SIZE,SENT_LEN)x2 for two questions?
        # x_train_q1 = #TODO: get Q1 features only
        # x_train_q2 = #TODO: get Q2 features only
        # y_train = #TODO: shape (traindatasize,)
        # x_val = #TODO: shape (valdatasize,WORD_EMBED_SIZE,SENT_LEN)x2 for two questions?
        # x_val_q1 = #TODO: get Q1 features only
        # x_val_q2 = #TODO: get Q2 features only
        # y_val = #TODO: shape (valdatasize,)

        if use_pretrained:
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

    def words_to_embeds(self, words_array):
            """Returns: vector that is flattened SENT_LEN x WORD_EMBED_SIZE matrix
            """
            words_series = pd.Series(data=words_array)
            zero_embed = np.array([0]*self.WORD_EMBED_SIZE,dtype='float32')
            word_to_embed = lambda word: zero_embed.copy() if word=='!EMPTY!' else pd.Series(data=self.w2v.model.wv[word])
            split_embeds = words_series.apply(word_to_embed)    # dataframe
            flattened = split_embeds.values.flatten()

            return flattened

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

if __name__=="__main__":
    m = Model(False)