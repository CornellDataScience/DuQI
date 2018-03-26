# Constants
# For use across files

WORD_EMBED_SIZE = 50           # size of word embedding
SENT_LEN = 50                   # fixed length of sentence
SENT_EMBED_SIZE = 100           # size of sentence embedding
TOP_WORD_THRESHOLD = 50         # minimum frequency for "top" words
SENT_INCLUSION_MIN = 3          # shorter sentences removed from training data
SENT_INCLUSION_MAX = SENT_LEN   # longer sentences removed from training data
UNKNOWN_MIN_COUNT = 50          # min frequency of words that get trained for w2v model
MIN_COUNT = 100                 # min frequency of words that don't count as !UNK!
NUM_EPOCHS = 10
BATCH_SIZE = 128
WORD2VEC_FILEPATH = '../lang_models/word2vecmodel-'+str(WORD_EMBED_SIZE)+'d_val_included'