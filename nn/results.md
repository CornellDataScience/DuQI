(3-24) gru_v1.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- accidentally trained tokenizer and word2vec on both training and validation
- Training accuracy: 96.41%
- Validation accuracy: 85.92%

(3-25) gru_v2.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- Training loss: [0.2088,0.1715,0.1558,0.1432,0.1311,0.1198,0.1092,0.0996,0.0907,0.0826]
- Validation loss: [0.1814,0.1679,0.1640,0.1589,0.1601,0.1590,0.1613,0.1604,0.1612,0.1638]
- Training accuracy: 90.09%
- Validation accuracy: 71.26%

(3-25) gru_v3.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- word embedding size 100, sentence embedding size 100
- Training loss: [.1833,.1501,.1299,.1141,.1004,.0880,.0777,.0685,.0607,.0537]
- Validation loss: [.1641,.1503,.1460,.1435,.1441,.1442,.1463,.1473,.1520,.1513]
- Training accuracy: 93.34%
- Validation accuracy: 72.65%

(3-25) lstm_v1.h5
- LSTM embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- Training accuracy: 91.27%
- Validation accuracy: 69.54%
322455/322455 [==============================] - 332s 1ms/step - loss: 0.2001 - val_loss: 0.1782
Epoch 2/10
322455/322455 [==============================] - 330s 1ms/step - loss: 0.1674 - val_loss: 0.1647
Epoch 3/10
322455/322455 [==============================] - 330s 1ms/step - loss: 0.1507 - val_loss: 0.1586
Epoch 4/10
322455/322455 [==============================] - 330s 1ms/step - loss: 0.1355 - val_loss: 0.1556
Epoch 5/10
322455/322455 [==============================] - 331s 1ms/step - loss: 0.1219 - val_loss: 0.1558
Epoch 6/10
322455/322455 [==============================] - 329s 1ms/step - loss: 0.1093 - val_loss: 0.1544
Epoch 7/10
322455/322455 [==============================] - 331s 1ms/step - loss: 0.0977 - val_loss: 0.1570
Epoch 8/10
322455/322455 [==============================] - 329s 1ms/step - loss: 0.0873 - val_loss: 0.1594
Epoch 9/10
322455/322455 [==============================] - 331s 1ms/step - loss: 0.0777 - val_loss: 0.1619
Epoch 10/10
322455/322455 [==============================] - 330s 1ms/step - loss: 0.0694 - val_loss: 0.1676

(3-25) gru_v4.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- dropout 0.3
- Training accuracy: 85.21%
- Validation accuracy: 72.77%
322455/322455 [==============================] - 291s 904us/step - loss: 0.2067 - val_loss: 0.1831
Epoch 2/10
322455/322455 [==============================] - 288s 893us/step - loss: 0.1759 - val_loss: 0.1686
Epoch 3/10
322455/322455 [==============================] - 291s 901us/step - loss: 0.1619 - val_loss: 0.1626
Epoch 4/10
322455/322455 [==============================] - 288s 894us/step - loss: 0.1519 - val_loss: 0.1601
Epoch 5/10
322455/322455 [==============================] - 289s 897us/step - loss: 0.1438 - val_loss: 0.1578
Epoch 6/10
322455/322455 [==============================] - 288s 893us/step - loss: 0.1367 - val_loss: 0.1568
Epoch 7/10
322455/322455 [==============================] - 289s 895us/step - loss: 0.1307 - val_loss: 0.1557
Epoch 8/10
322455/322455 [==============================] - 289s 895us/step - loss: 0.1248 - val_loss: 0.1566
Epoch 9/10
322455/322455 [==============================] - 289s 895us/step - loss: 0.1190 - val_loss: 0.1569
Epoch 10/10
322455/322455 [==============================] - 288s 894us/step - loss: 0.1148 - val_loss: 0.1566

TODO: TEST PRE- and POST- DROPOUT, CODE ALREADY MODIFIED
