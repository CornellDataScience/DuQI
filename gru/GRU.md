# Data Exploration

![hist](https://github.com/CornellDataScience/DuQI/raw/master/images/gru_questions_clean_hist.png)

![zoomed](https://github.com/CornellDataScience/DuQI/raw/master/images/gru_questions_clean_hist_zoom.png)

* Processed sentences:
    * Length 1 - no
        * "ok"
        * "o"
        * "lol"
        * "delete"
        * "aaa"
    * Length 2 - not great
        * "what empathy"
        * "personality development"
        * "fore play"
        * "debit question"
        * "learn python"
    * Length 3 - barely serviceable
        * "did christ exist"
        * "is bloodborne hard"
        * "neuroscience amity university"
        * "what is industry"
        * "doe belief matter"
    * Length 4 - generally reasonable
        * "do dog eat cat"
        * "what are hydrogen bond"
        * "how is temperature measure"
        * "how safe is boston"
        * "how do girl pee"
    * Length 50 - especially detailed
        * "i am interest in chemistry and biology but i am unable to choose them for my bachelor 
        own degree due to some personal problem am i leave with any option regard further study
        base on biology or chemistry or should i have to continue in the same field mechanical
        engineer"

# GRU

![Intro](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)

![Model](https://github.com/CornellDataScience/DuQI/raw/master/images/gru_model.png)

* Challenges:
    * Unknown words in training data
        * We could assign embeddings to every single word, but don't want to overfit
        * Train model on data with infrequent as UNK, preprocess test data prior to running model