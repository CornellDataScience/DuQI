### (3-24) gru_v1.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- *accidentally trained tokenizer and word2vec on both training and validation
- Training accuracy: 96.41%
- Validation accuracy: 85.92%

### (3-25) gru_v2.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- Training loss: [0.2088,0.1715,0.1558,0.1432,0.1311,0.1198,0.1092,0.0996,0.0907,0.0826]
- Validation loss: [0.1814,0.1679,0.1640,0.1589,0.1601,0.1590,0.1613,0.1604,0.1612,0.1638]
- Training accuracy: 90.09%
- Validation accuracy: 71.26%

### (3-25) gru_v3.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- *word embedding size 100, sentence embedding size 100
- Training loss: [.1833,.1501,.1299,.1141,.1004,.0880,.0777,.0685,.0607,.0537]
- Validation loss: [.1641,.1503,.1460,.1435,.1441,.1442,.1463,.1473,.1520,.1513]
- Training accuracy: 93.34%
- Validation accuracy: 72.65%

### (3-25) lstm_v1.h5
- *LSTM embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- Training accuracy: 91.27%
- Validation accuracy: 69.54%
loss: 0.2001 - val_loss: 0.1782
loss: 0.1674 - val_loss: 0.1647
loss: 0.1507 - val_loss: 0.1586
loss: 0.1355 - val_loss: 0.1556
loss: 0.1219 - val_loss: 0.1558
loss: 0.1093 - val_loss: 0.1544
loss: 0.0977 - val_loss: 0.1570
loss: 0.0873 - val_loss: 0.1594
loss: 0.0777 - val_loss: 0.1619
loss: 0.0694 - val_loss: 0.1676

### (3-25) gru_v4.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- *dropout 0.3
- Training accuracy: 85.21%
- Validation accuracy: 72.77%
loss: 0.2067 - val_loss: 0.1831
loss: 0.1759 - val_loss: 0.1686
loss: 0.1619 - val_loss: 0.1626
loss: 0.1519 - val_loss: 0.1601
loss: 0.1438 - val_loss: 0.1578
loss: 0.1367 - val_loss: 0.1568
loss: 0.1307 - val_loss: 0.1557
loss: 0.1248 - val_loss: 0.1566
loss: 0.1190 - val_loss: 0.1569
loss: 0.1148 - val_loss: 0.1566

### (3-26) gru_v5.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- *pre- and post- dropout 0.2
- Training accuracy: 74.27%
- Validation accuracy: 61.14%
loss: 0.1931 - val_loss: 0.1741
loss: 0.1654 - val_loss: 0.1664
loss: 0.1523 - val_loss: 0.1621
loss: 0.1428 - val_loss: 0.1643
loss: 0.1352 - val_loss: 0.1673
loss: 0.1282 - val_loss: 0.1642
loss: 0.1226 - val_loss: 0.1728
loss: 0.1167 - val_loss: 0.1736
loss: 0.1122 - val_loss: 0.1770
loss: 0.1077 - val_loss: 0.1750

### (3-26) gru_v1_re.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- *trained tokenizer and word2vec on both training and validation
- *dropout = 0.3
- Training accuracy: 78.06%
- Validation accuracy: 68.55%
loss: 0.1970 - val_loss: 0.1735
loss: 0.1691 - val_loss: 0.1605
loss: 0.1543 - val_loss: 0.1539
loss: 0.1440 - val_loss: 0.1527
loss: 0.1368 - val_loss: 0.1490
loss: 0.1303 - val_loss: 0.1499
loss: 0.1255 - val_loss: 0.1473
loss: 0.1216 - val_loss: 0.1494
loss: 0.1174 - val_loss: 0.1514
loss: 0.1138 - val_loss: 0.1493

### (4-3) glove_gru1.h5
- GRU embeddng, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- Training Euclidean GRU on GloVe.
- **Hypothesis:** Better performance than Word2Vec due to size and breadth of language model training sources, resulting in more generalized embeddings rather than overfitting to training data.
- **Concerns:**
    - Unknown token embedding was taken to be average of all GloVe embeddings. Should find a method of weighting the embeddings to have a more accurate unknown embedding.
    - Not using any techniques to reduce overfitting (dropout, regularization, etc). Paper mentioned data augmentation being the most effective method to reduce overfitting, mentioned L2 loss being very ineffective for various values of lambda. Dropout not mentioned.
- Training
    - Accuracy: .9379
    - F1: .9502
- Validation
    - Accuracy: .7484
    - F1: .7591
- Loss per epoch
    - loss: 0.1774 - val_loss: 0.1541
    - loss: 0.1370 - val_loss: 0.1388
    - loss: 0.1151 - val_loss: 0.1328
    - loss: 0.0989 - val_loss: 0.1298
    - loss: 0.0860 - val_loss: 0.1291
    - loss: 0.0751 - val_loss: 0.1287
    - loss: 0.0658 - val_loss: 0.1304
    - loss: 0.0577 - val_loss: 0.1310
    - loss: 0.0507 - val_loss: 0.1313
    - loss: 0.0445 - val_loss: 0.1335
- **Observations:**
    - Definitely overfitting, though the disparity in accuracy between validation and training is more pronounced than that of the loss. Means there is much room to improve the loss metric. Fully connected similarity network next?