import keras as k
import numpy as np
import pandas as pd
import gensim as gs
from word2vecmodel import Word2VecModel
from data_cleaning import split_and_exclude
import constants as c

# input tensor is (batch_size, timesteps, input_dim)
class Model:

    def __init__(self, use_pretrained=True):
        data = pd.read_csv('../data/train_clean.csv')
        data = split_and_exclude(data)
        shuffled_data = data.sample(frac=1,random_state=2727)
        self.w2v = Word2VecModel()
        self.unk_embed = self.produce_unk_embed()

        #TODO #TODO #TODO: Train with full code (unrestricted datasets)
        # q_vectors_list = [None,None]
        # for question_num in [1,2]:
        #     q_split = shuffled_data['question'+str(question_num)]
        #     # padding front of questions
        #     pad_front = lambda lst: ['!EMPTY!']*(c.WORD_EMBED_SIZE-len(lst))+lst
        #     q_split = q_split.apply(pad_front)
        #     q_split = q_split.apply(np.asarray)
        #     self.iters_count = 0
        #     q_vectors = q_split.apply(self.words_to_embeds)
        #     q_vectors_list[question_num-1] = q_vectors
        #
        # y_data = shuffled_data['is_duplicate']

        q_vectors_list = [None,None]
        for question_num in [1,2]:
            q_split = shuffled_data['question'+str(question_num)].head(n=1000)
            # padding front of questions
            pad_front = lambda lst: ['!EMPTY!']*(c.SENT_LEN-len(lst))+lst
            q_split = q_split.apply(pad_front)
            q_split = q_split.apply(np.asarray)
            self.iters_count = 0
            q_vectors = q_split.apply(self.words_to_embeds)
            q_vectors_list[question_num-1] = q_vectors
        
        y_data = shuffled_data['is_duplicate'].head(n=1000)
        print(q_vectors_list[0].iloc[0].shape) # (1250,) because 25-dim, 50 words

        return
        
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
            self.model = k.models.load_model('../data/'+c.MODEL_NAME)
            print('Model loaded from data/'+c.MODEL_NAME)
        else:
            self.model = self.similarity_model()
            self.model.compile(loss='mean_squared_error', optimizer='adam')
            print('Training model...')
            self.model.fit([x_train_q1, x_train_q2], y_train,
                        validation_data=([x_val_q1, x_val_q2], y_val),
                        batch_size=c.BATCH_SIZE,
                        nb_epoch=c.NUM_EPOCHS)
            print('Model trained.\nSaving model...')
            self.model.save('../data/'+c.MODEL_NAME)
            print('Model saved to data/'+c.MODEL_NAME)

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

    def word_to_embed(self, word):
        if word=='!EMPTY!':
            return pd.Series(data=[0]*c.WORD_EMBED_SIZE,dtype='float32')
        else:
            try:
                return pd.Series(data=self.w2v.model.wv[word],dtype='float32')
            except KeyError:
                return pd.Series(data=self.unk_embed,dtype='float32')

    def words_to_embeds(self, words_array):
            """Returns: vector that is flattened SENT_LEN x WORD_EMBED_SIZE matrix
            """
            words_series = pd.Series(data=words_array)
            split_embeds = words_series.apply(self.word_to_embed)    # dataframe
            flattened = split_embeds.values.flatten()
            self.iters_count+=1
            if self.iters_count%1000 == 0:
                print('Questions processed: '+str(self.iters_count))
            return flattened

    def gru_embedding(self):
        """Returns: GRU model for sentence embedding, applied to each question input.
        """
        gru = k.layers.GRU(c.SENT_EMBED_SIZE,
                            input_shape = (c.SENT_LEN,c.WORD_EMBED_SIZE),
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
        input1 = k.layers.Input(shape=(c.SENT_LEN,c.WORD_EMBED_SIZE))
        input2 = k.layers.Input(shape=(c.SENT_LEN,c.WORD_EMBED_SIZE))
        gru1_out = self.gru_embedding()(input1)
        gru2_out = self.gru_embedding()(input2)
        distance = k.layers.Lambda(self.eucl_dist, output_shape=self.eucl_dist_shape)([gru1_out,gru2_out])
        model = k.models.Model(inputs=[input1, input2], outputs=[distance])
        return model

    def compute_accuracy(self, preds, labels):
        return labels[preds.ravel() < 0.5].mean()

if __name__=="__main__":
    m = Model(False)