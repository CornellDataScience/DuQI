# Constants
# For use across files

NUM_EPOCHS = 10
BATCH_SIZE = 256
WORD_EMBED_SIZE = 50            # size of word embedding
SENT_EMBED_SIZE = 100           # size of sentence embedding
SENT_LEN = 50                   # fixed length of sentence
SENT_INCLUSION_MIN = 3          # shorter sentences removed from training data
SENT_INCLUSION_MAX = SENT_LEN   # longer sentences removed from training data

# GloVe
GLOVE_FILEPATH = '../lang_models/glove.6B.'+str(WORD_EMBED_SIZE)+'d.txt'

# Word2Vec
UNKNOWN_MIN_COUNT = 50          # min frequency of words that get trained for w2v model
MIN_COUNT = 100                 # min frequency of words that don't count as !UNK!
WORD2VEC_FILEPATH = '../lang_models/word2vecmodel-'+str(WORD_EMBED_SIZE)+'d_val_included'