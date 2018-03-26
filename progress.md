# Progress Report

Pre-03/11 (Zhao):
* Readings: [CNN/LSTM](https://web.stanford.edu/class/cs224n/reports/2759336.pdf), 
           [GRU/distance](https://web.stanford.edu/class/cs224n/reports/2748045.pdf)
* Created `gru/gru.py`
    * Wrote baseline Siamese NN with GRU sentence encoding, Euclidean distance similarity
* Created `gru/data_cleaning.py`
    * Wrote preprocessing script (from kaggle-quora-dup), saved processed data to `data/train_clean.csv`
* Downloaded GloVe embeddings to `data/`, added Data Requirements section to `README.md`

03/11 (Zhao):
* Tentatively `SENT_LEN = 50` while excluding questions less than 4 or greater than 50 in length.
    * Added sentence length visualization and example printing to `gru/data_cleaning.py`
* Created `GRU.md` for logging/presentation.
* Began working on Word2Vec model in `gru/gru.py`

03/12 (Zhao):
* Migrated Word2Vec to `word2vecmodel.py`
    * Trained 50D, 100D models from training data located in `models/`
  
03/12 (Arnav):
* Added Pre-Processed Questions (via textacy)
* Generated Naive Bayes Model
  * Features: Jaccard Similarity, Dissimilar word POS overlap
  * Precision: 0.608
  * Intention:
           * Non-overlapping content words --> more likely to be non-duplicates
  * Evaluation:
           * Guided by Jacc. Similarity

03/26 (Zhao):
* Finished `nn.py` (previous `gru.py`) - basic framework for training neural nets
* Trained a variety of models, later deleted them due to Keras loading issue with Lambda layers.
    * Collected training/validation accuracy and loss data