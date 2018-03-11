# Progress Report

Pre-03/11 (Zhao):
* Readings: [CNN/LSTM](https://web.stanford.edu/class/cs224n/reports/2759336.pdf), 
           [GRU/distance](https://web.stanford.edu/class/cs224n/reports/2748045.pdf)
* Created `gru/gru.py`
    * Wrote baseline Siamese NN with GRU sentence encoding, Euclidean distance similarity
* Created `gru/data_cleaning.py`
    * Wrote preprocessing script, saved processed data to `data/train_clean.csv`
* Downloaded GloVe embeddings to `data/`, added Data Requirements section to `README.md`