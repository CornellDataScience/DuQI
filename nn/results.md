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
- Loss per epoch
    - loss: 0.2001 - val_loss: 0.1782
    - loss: 0.1674 - val_loss: 0.1647
    - loss: 0.1507 - val_loss: 0.1586
    - loss: 0.1355 - val_loss: 0.1556
    - loss: 0.1219 - val_loss: 0.1558
    - loss: 0.1093 - val_loss: 0.1544
    - loss: 0.0977 - val_loss: 0.1570
    - loss: 0.0873 - val_loss: 0.1594
    - loss: 0.0777 - val_loss: 0.1619
    - loss: 0.0694 - val_loss: 0.1676

### (3-25) gru_v4.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- *dropout 0.3
- Training accuracy: 85.21%
- Validation accuracy: 72.77%
- Loss per epoch
    - loss: 0.2067 - val_loss: 0.1831
    - loss: 0.1759 - val_loss: 0.1686
    - loss: 0.1619 - val_loss: 0.1626
    - loss: 0.1519 - val_loss: 0.1601
    - loss: 0.1438 - val_loss: 0.1578
    - loss: 0.1367 - val_loss: 0.1568
    - loss: 0.1307 - val_loss: 0.1557
    - loss: 0.1248 - val_loss: 0.1566
    - loss: 0.1190 - val_loss: 0.1569
    - loss: 0.1148 - val_loss: 0.1566

### (3-26) gru_v5.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- *pre- and post- dropout 0.2
- Training accuracy: 74.27%
- Validation accuracy: 61.14%
- Loss per epoch
    - loss: 0.1931 - val_loss: 0.1741
    - loss: 0.1654 - val_loss: 0.1664
    - loss: 0.1523 - val_loss: 0.1621
    - loss: 0.1428 - val_loss: 0.1643
    - loss: 0.1352 - val_loss: 0.1673
    - loss: 0.1282 - val_loss: 0.1642
    - loss: 0.1226 - val_loss: 0.1728
    - loss: 0.1167 - val_loss: 0.1736
    - loss: 0.1122 - val_loss: 0.1770
    - loss: 0.1077 - val_loss: 0.1750

### (3-26) gru_v1_re.h5
- GRU embedding, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- *trained tokenizer and word2vec on both training and validation
- *dropout = 0.3
- Training accuracy: 78.06%
- Validation accuracy: 68.55%
- Loss per epoch
    - loss: 0.1970 - val_loss: 0.1735
    - loss: 0.1691 - val_loss: 0.1605
    - loss: 0.1543 - val_loss: 0.1539
    - loss: 0.1440 - val_loss: 0.1527
    - loss: 0.1368 - val_loss: 0.1490
    - loss: 0.1303 - val_loss: 0.1499
    - loss: 0.1255 - val_loss: 0.1473
    - loss: 0.1216 - val_loss: 0.1494
    - loss: 0.1174 - val_loss: 0.1514
    - loss: 0.1138 - val_loss: 0.1493

### (4-3) glove_gru1.h5
- GRU embeddng, Euclidean similarity, sigmoid activation
- word embedding size 50, sentence embedding size 100
- Training Euclidean GRU on GloVe.
- **Hypothesis:** Better performance than Word2Vec due to size and breadth of language model training sources, resulting in more generalized embeddings rather than overfitting to training data.
- **Concerns:**
    - Unknown token embedding was taken to be average of all GloVe embeddings. Should find a method of weighting the embeddings to have a more accurate unknown embedding.
    - Not using any techniques to reduce overfitting (dropout, regularization, etc). Paper mentioned data augmentation being the most effective method to reduce overfitting, included L2 regularization in loss function, dropout not mentioned.
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

### (4-5) glove_gru1_siamfix.h5
- Found that previous network was non-Siamese, so trained this Siamese network.
- Upon closer inspection, the output is the exact same for every pair of inputs.
    - Perhaps this is due to `gru1_out` and `gru2_out` having the same output?
- Loss per epoch
    - loss: 0.2347 - val_loss: 0.2333
    - loss: 0.2331 - val_loss: 0.2333
    - loss: 0.2331 - val_loss: 0.2333
    - loss: 0.2331 - val_loss: 0.2333
    - loss: 0.2331 - val_loss: 0.2333
    - loss: 0.2331 - val_loss: 0.2333
    - loss: 0.2331 - val_loss: 0.2333
    - loss: 0.2331 - val_loss: 0.2333
    - loss: 0.2331 - val_loss: 0.2333
    - loss: 0.2331 - val_loss: 0.2333

### (4-5) glove_gru2_v1.h5
- Siamese GRU encoding, 2 layer similarity network
- Similarity network input is concatenation of 2 sentence vectors
- Training
    - Accuracy: 0.9633
    - F1 score: 0.9514
- Validation
    - Accuracy: 0.8384
    - F1 score: 0.7895
- Loss per epoch
    - loss: 0.4880 - val_loss: 0.4251
    - loss: 0.3796 - val_loss: 0.3858
    - loss: 0.3197 - val_loss: 0.3776
    - loss: 0.2763 - val_loss: 0.3815
    - loss: 0.2428 - val_loss: 0.4000
    - loss: 0.2151 - val_loss: 0.4117
    - loss: 0.1908 - val_loss: 0.4398
    - loss: 0.1682 - val_loss: 0.4701
    - loss: 0.1483 - val_loss: 0.5147
    - loss: 0.1294 - val_loss: 0.5444
- **Observations:**
    - Extreme overfitting. Should test dropout, L2 regularization, etc.

### (4-10) glove_gru2_v2.h5
- Vanilla glove_gru2 with:
    - Pre-activation batch norm to first dense layer
    - 0.4 dropout in GRU
- Training
    - Accuracy: 0.8623
    - F1 score: 0.8317
- Validation
    - Accuracy: 0.8169
    - F1 score: 0.7766
- Loss per epoch
    - loss: 0.5572 - val_loss: 0.4892
    - loss: 0.4916 - val_loss: 0.4720
    - loss: 0.4598 - val_loss: 0.4595
    - loss: 0.4362 - val_loss: 0.4803
    - loss: 0.4147 - val_loss: 0.4446
    - loss: 0.3974 - val_loss: 0.4515
    - loss: 0.3833 - val_loss: 0.4277
    - loss: 0.3696 - val_loss: 0.4692
    - loss: 0.3580 - val_loss: 0.4239
    - loss: 0.3476 - val_loss: 0.4095
- **Observations:**
    - Perplexing lack of generalization
    - Erratic validation loss behavior
    - Maybe dropout was overtuned?

### (4-10) glove_gru2_v3.h5
- Vanilla glove_gru2 with:
    - Pre-activation batch norm to first dense layer
- Training
    - Accuracy: 0.9560
    - F1 score: 0.9420
- Validation
    - Accuracy: 0.8417
    - F1 score: 0.7929
- Loss per epoch
    - loss: 0.4820 - val_loss: 0.4299
    - loss: 0.3788 - val_loss: 0.4327
    - loss: 0.3244 - val_loss: 0.3787
    - loss: 0.2848 - val_loss: 0.3797
    - loss: 0.2525 - val_loss: 0.3880
    - loss: 0.2252 - val_loss: 0.3998
    - loss: 0.2006 - val_loss: 0.4152
    - loss: 0.1790 - val_loss: 0.4448
    - loss: 0.1596 - val_loss: 0.4603
    - loss: 0.1425 - val_loss: 0.4987
- **Observations:**
    - still slightly inferior to vanilla glove_gru2 model

### (4-10) glove_gru2_v1_100d.h5
- Training
    - Accuracy: 0.9799
    - F1 score: 0.9729
    - 100-dimensional
- Validation
    - Accuracy: 0.8422
    - F1 score: 0.7875
- Loss per epoch
    - loss: 0.4761 - val_loss: 0.4099
    - loss: 0.3591 - val_loss: 0.3750
    - loss: 0.2892 - val_loss: 0.3786
    - loss: 0.2394 - val_loss: 0.3810
    - loss: 0.1997 - val_loss: 0.4073
    - loss: 0.1664 - val_loss: 0.4410
    - loss: 0.1383 - val_loss: 0.5007
    - loss: 0.1140 - val_loss: 0.5487
    - loss: 0.0943 - val_loss: 0.6000
    - loss: 0.0774 - val_loss: 0.6710
- **Observations:**
    - More evidence that glove_gru2_v1 was overfitting

### (4-10) glove_gru2_v2_300d.h5
- Vanilla glove_gru2 with:
    - Pre-activation batch norm to first dense layer
    - 0.4 dropout in GRU
    - 300-dimensional
- Training
    - Accuracy: 0.9339
    - F1 score: 0.9165
- Validation
    - Accuracy: 0.8346
    - F1 score: 0.7960
- Loss per epoch
    - loss: 0.5176 - val_loss: 0.4413
    - loss: 0.4242 - val_loss: 0.4006
    - loss: 0.3720 - val_loss: 0.3907
    - loss: 0.3317 - val_loss: 0.3879
    - loss: 0.3017 - val_loss: 0.4115
    - loss: 0.2768 - val_loss: 0.3935
    - loss: 0.2568 - val_loss: 0.4127
    - loss: 0.2388 - val_loss: 0.4079
    - loss: 0.2226 - val_loss: 0.4241
    - loss: 0.2100 - val_loss: 0.4375
- **Observations:**
    - Overfitting still not ideal
    - 300d doesn't take much more training time than 50d.

### (4-12) glove_gru3_v1.h5
- Notes
    - Changed binary CE to categorical CE
    - Batch norm for dense layer
    - L2 regularization (lambda = 0.0001) on GRU weights only
    - No dropout
- Training
    - Accuracy: 0.9731
    - F1 score: 0.9640
- Validation
    - Accuracy: 0.8448
    - F1 score: 0.7934
- Loss per epoch
    - loss: 0.4884 - acc: 0.7758 - val_loss: 0.4296 - val_acc: 0.8089
    - loss: 0.3597 - acc: 0.8495 - val_loss: 0.4195 - val_acc: 0.8214
    - loss: 0.2889 - acc: 0.8862 - val_loss: 0.4024 - val_acc: 0.8380
    - loss: 0.2431 - acc: 0.9090 - val_loss: 0.4130 - val_acc: 0.8399
    - loss: 0.2098 - acc: 0.9249 - val_loss: 0.4515 - val_acc: 0.8414
    - loss: 0.1855 - acc: 0.9368 - val_loss: 0.4636 - val_acc: 0.8418
    - loss: 0.1646 - acc: 0.9463 - val_loss: 0.5135 - val_acc: 0.8371
    - loss: 0.1498 - acc: 0.9525 - val_loss: 0.5258 - val_acc: 0.8436
    - loss: 0.1364 - acc: 0.9584 - val_loss: 0.5403 - val_acc: 0.8445
    - loss: 0.1257 - acc: 0.9632 - val_loss: 0.5800 - val_acc: 0.8448
- **Observations:**
    - Would like to observe more epochs, more regularization. Dropout?
    - Need to start augmenting data, playing around with dense layer inputs

### (4-16) glove_gru3_v2.h5
- Notes
    - Added L2 reg to dense layers
    - Added q1-q2, q1*q2 similarity features
- tr_acc: 0.9704
- tr_f1: 0.9610
- val_acc: 0.8484
- val_f1: 0.8038
- Loss per epoch
    - loss: 0.4910 - acc: 0.7830 - val_loss: 0.4265 - val_acc: 0.8196
    - loss: 0.3619 - acc: 0.8543 - val_loss: 0.3907 - val_acc: 0.8414
    - loss: 0.2924 - acc: 0.8899 - val_loss: 0.3949 - val_acc: 0.8486
    - loss: 0.2483 - acc: 0.9111 - val_loss: 0.4132 - val_acc: 0.8482
    - loss: 0.2159 - acc: 0.9268 - val_loss: 0.4438 - val_acc: 0.8514
    - loss: 0.1920 - acc: 0.9379 - val_loss: 0.4698 - val_acc: 0.8532
    - loss: 0.1735 - acc: 0.9465 - val_loss: 0.5016 - val_acc: 0.8537
    - loss: 0.1584 - acc: 0.9536 - val_loss: 0.5451 - val_acc: 0.8508
    - loss: 0.1466 - acc: 0.9592 - val_loss: 0.5804 - val_acc: 0.8506
    - loss: 0.1366 - acc: 0.9635 - val_loss: 0.6143 - val_acc: 0.8484
- **Observations:**
    - Not a huge improvement, expect that similarity features will show use after data augmentation

### (4-18) glove_gru4_v1.h5
- Notes
    - Added masking layer before GRU
    - Augmented data by flipping question order, matching unique questions with themselves
        - 403,069 question pairs -> 1,388,474 question pairs
    - Sentence embedding size 300 (from 100)
    - dropout = 0.2 (recurrent dropout = 0)
- tr_acc: 0.9736
- tr_f1: 0.9793
- val_acc: 0.9125
- val_f1: 0.9356
- Loss per epoch
    - loss: 0.3686 - acc: 0.8696 - val_loss: 0.2996 - val_acc: 0.8978
    - loss: 0.2534 - acc: 0.9176 - val_loss: 0.2708 - val_acc: 0.9115
    - loss: 0.2101 - acc: 0.9344 - val_loss: 0.2855 - val_acc: 0.9077
    - loss: 0.1844 - acc: 0.9445 - val_loss: 0.2972 - val_acc: 0.9148
    - loss: 0.1670 - acc: 0.9515 - val_loss: 0.2908 - val_acc: 0.9166
    - loss: 0.1539 - acc: 0.9568 - val_loss: 0.3369 - val_acc: 0.9126
    - loss: 0.1434 - acc: 0.9609 - val_loss: 0.3279 - val_acc: 0.9118
    - loss: 0.1353 - acc: 0.9640 - val_loss: 0.3129 - val_acc: 0.9145
    - loss: 0.1285 - acc: 0.9667 - val_loss: 0.3149 - val_acc: 0.9185
    - loss: 0.1222 - acc: 0.9689 - val_loss: 0.3489 - val_acc: 0.9125

## gru_v5
- 

### Experiments TODO:
- Handle long sentences
- Use excluded sents as additional validation metric
- Attention mechanism
