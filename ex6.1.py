import lxmls.readers.pos_corpus as pcc

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

import lxmls.deep_learning.rnn as rnns
#reload(rnns)
# RNN configuration
SEED = 1234 # Random seed to initialize weigths
emb_size = 50 # Size of word embeddings
hidden_size = 20 # size of hidden layer
# RNN
np_rnn = rnns.NumpyRNN(nr_words, emb_size, hidden_size, nr_tags, seed=SEED)
# Example sentence
x0 = train_seq[0].x
y0 = train_seq[0].y
print "x0=", x0
print "y0=", y0

# Forward pass
p_y, y_rnn, h, z1, x = np_rnn.forward(x0, all_outputs=True) # Gradients
numpy_rnn_gradients = np_rnn.grads(x0, y0)

