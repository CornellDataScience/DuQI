import keras as k
import numpy as np
import pandas as pd

# input tensor is (batch_size, timesteps, input_dim)
class Model:

    WORD_EMBED_SIZE = 100            # size of word embedding
    SENT_LEN = 100                   # fixed length of sentence
    SENT_EMBED_SIZE = 50             # size of output vector
    NUM_EPOCHS = 10
    BATCH_SIZE = 128
    TRAIN_DAT_SIZE = 0
    MODEL_NAME = 'gru_v1.h5'

    def __init__(self, have_model):
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
            model = k.models.load_model('../data/'+self.MODEL_NAME)
            print('Model loaded from data/'+self.MODEL_NAME)
        else:
            model = self.similarity_model()
            model.compile(loss='mean_squared_error', optimizer='adam')
            print('Training model...')
            model.fit([x_train_q1, x_train_q2], y_train,
                        validation_data=([x_val_q1, x_val_q2], y_val),
                        batch_size=self.BATCH_SIZE,
                        nb_epoch=self.NUM_EPOCHS)
            print('Model trained.\nSaving model...')
            model.save('../data/'+self.MODEL_NAME)
            print('Model saved to data/'+self.MODEL_NAME)

        pred_train = model.predict([x_train_q1, x_train_q2])
        acc_train = self.compute_accuracy(pred_train, y_train)
        print('* Accuracy on training set: %0.2f%%' % (100 * acc_train))

        pred_val = model.predict([x_val_q1, x_val_q2])
        acc_val = self.compute_accuracy(pred_val, y_val)
        print('* Accuracy on validation set: %0.2f%%' % (100 * acc_val))

    def gru_embedding(self):
        """GRU model for sentence embedding, applied to each question input.
        """
        gru = k.layers.GRU(self.SENT_EMBED_SIZE,
                            input_shape = (self.SENT_LEN,self.WORD_EMBED_SIZE),
                            activation='relu', #tanh
                            dropout=0.0) #0.2
        return gru

    def eucl_dist(self, vects):
        x, y = vects
        return k.backend.sqrt(k.backend.sum(k.backend.square(x - y), axis=1, keepdims=True))

    def eucl_dist_shape(self, shapes):
        shape1, _ = shapes
        return (shape1[0], 1)

    def similarity_model(self):
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