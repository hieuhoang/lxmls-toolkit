import lxmls.readers.pos_corpus as pcc
import lxmls.deep_learning.rnn as rnns
import theano
import theano.tensor as T

corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("data/train-02-21.conll", max_sent_len=15, max_nr_sent=1000)
test_seq = corpus.read_sequence_list_conll("data/test-23.conll", max_sent_len=15, max_nr_sent=1000)
dev_seq = corpus.read_sequence_list_conll("data/dev-22.conll", max_sent_len=15, max_nr_sent=1000)

# Redo indices
train_seq, test_seq, dev_seq = pcc.compacify(train_seq, test_seq, dev_seq, theano=True)
#print "train_seq=", train_seq
#print "test_seq=", test_seq

# Get number of words and tags in the corpus
nr_words = len(train_seq.x_dict)
nr_tags = len(train_seq.y_dict)
print "nr_words=", nr_words, "nr_tags=", nr_tags

SEED = 1234 # Random seed to initialize weigths
emb_size = 50 # Size of word embeddings
hidden_size = 20 # size of hidden layer

# Instantiate the class
rnn = rnns.RNN(nr_words, emb_size, hidden_size, nr_tags, seed=SEED)
# Compile the forward pass function
x = T.ivector('x')
print "x=", x
th_forward = theano.function([x], rnn._forward(x).T)
print "th_forward=", th_forward("x0")