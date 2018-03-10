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

    def __init__(self, is_pretrained):
        x_train = #TODO: shape (traindatasize,WORD_EMBED_SIZE,SENT_LEN)x2 for two questions?
        x_train_q1 = #TODO: get Q1 features only
        x_train_q2 = #TODO: get Q2 features only
        x_val = #TODO: shape (testdatasize,WORD_EMBED_SIZE,SENT_LEN)x2 for two questions?
        x_val_q1 = #TODO: get Q1 features only
        x_val_q2 = #TODO: get Q2 features only
        y_train = #TODO: shape (traindatasize,)
        y_val = #TODO: shape (testdatasize,)

        model = self.similarity_model()
        if is_pretrained:
            #TODO: load weights
        else:
            model.compile(loss=contrastive_loss, optimizer=rms)
            model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                        validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
                        batch_size=BATCH_SIZE,
                        nb_epoch=NUM_EPOCHS)
            #TODO: save weights

        pred_train = model.predict([x_train_q1, x_train_q2])
        acc_train = self.compute_accuracy(pred_train, y_train)
        print('* Accuracy on training set: %0.2f%%' % (100 * acc_train))

        pred_val = model.predict([x_val_q1, x_val_q2])

        acc_val = self.compute_accuracy(pred_val, y_val)
        print('* Accuracy on validation set: %0.2f%%' % (100 * acc_val))

    def gru_embedding(self):
        """GRU model for sentence embedding, applied to each question input.
        """
        gru = k.models.Sequential()
        gru.add(k.layers.GRU(SENT_EMBED_SIZE,
                            input_shape = (SENT_LEN,WORD_EMBED_SIZE),
                            activation='relu', #tanh
                            dropout=0.0)) #0.2
        return gru

    def similarity_model(self):
        input1 = k.layers.Input(shape=(SENT_LEN,WORD_EMBED_SIZE))
        input2 = k.layers.Input(shape=(SENT_LEN,WORD_EMBED_SIZE))
        gru1_out = self.gru_embedding()(input1)
        gru2_out = self.gru_embedding()(input2)
        similarity = k.layers.Lambda(1-np.linalg.norm)(gru1_out - gru2_out)
        model = k.models.Model(input=[input1, input2], output=similarity)
        return model

    def compute_accuracy(self, preds, labels):
        return labels[preds.ravel() > 0.5].mean()

    
        