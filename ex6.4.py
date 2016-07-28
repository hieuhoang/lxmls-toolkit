# Embeddings Path
import lxmls.readers.pos_corpus as pcc
import lxmls.deep_learning.embeddings as emb
import os
import lxmls.deep_learning.rnn as rnns
import theano
import theano.tensor as T

corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("data/train-02-21.conll", max_sent_len=15, max_nr_sent=1000)
test_seq = corpus.read_sequence_list_conll("data/test-23.conll", max_sent_len=15, max_nr_sent=1000)
dev_seq = corpus.read_sequence_list_conll("data/dev-22.conll", max_sent_len=15, max_nr_sent=1000)

# Redo indices
train_seq, test_seq, dev_seq = pcc.compacify(train_seq, test_seq, dev_seq, theano=True)
print "train_seq=", len(train_seq)
print "test_seq=", len(test_seq)

# Get number of words and tags in the corpus
nr_words = len(train_seq.x_dict)
nr_tags = len(train_seq.y_dict)
print "nr_words=", nr_words, "nr_tags=", nr_tags
#print "x_dict=", train_seq.x_dict \
#print "y_dict=", train_seq.y_dict

SEED = 1234 # Random seed to initialize weigths
emb_size = 50 # Size of word embeddings
hidden_size = 20 # size of hidden layer

# Instantiate the class
rnn = rnns.RNN(nr_words, emb_size, hidden_size, nr_tags, seed=SEED)

# EMBEDDINGS
reload(emb)
EMBEDDINGS = "data/senna_50"
if not os.path.isfile(EMBEDDINGS):
    emb.download_embeddings('senna_50', "data/senna_50")
E = emb.extract_embeddings("data/senna_50", train_seq.x_dict)
# Reset model to remove the effect of training
rnn = rnns.reset_model(rnn, seed=SEED)
# Set the embedding layer to the pre-trained values
rnn.param[0].set_value(E.astype(theano.config.floatX))

