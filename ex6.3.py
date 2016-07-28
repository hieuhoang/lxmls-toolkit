import numpy as np
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

# RNN
x0 = train_seq[0].x
y0 = train_seq[0].y

np_rnn = rnns.NumpyRNN(nr_words, emb_size, hidden_size, nr_tags, seed=SEED)
p_y, y_rnn, h, z1, x = np_rnn.forward(x0, all_outputs=True) # Gradients
numpy_rnn_gradients = np_rnn.grads(x0, y0)

# THEANO
# Compile the forward pass function
x = T.ivector('x')
#print "x=", x
th_forward = theano.function([x], rnn._forward(x).T)
#print "th_forward=", th_forward

# Compile function returning the list of gradients
x = T.ivector('x')     # Input words
y = T.ivector('y')     # gold tags
p_y = rnn._forward(x)
cost = -T.mean(T.log(p_y)[T.arange(y.shape[0]), y])
grads_fun = theano.function([x, y], [T.grad(cost, par) for par in rnn.param])

# Compare numpy and theano gradients
theano_rnn_gradients = grads_fun(x0, y0)
for n in range(len(theano_rnn_gradients)):
    assert np.allclose(numpy_rnn_gradients[n], theano_rnn_gradients[n]), "Numpy and Theano gradients differ in step n"
print "theano_rnn_gradients=", len(theano_rnn_gradients)

for obj in theano_rnn_gradients:
    print obj.shape

# Parameters
lrate = 0.5 # Learning rate
n_iter = 5 # Number of iterations
# Get list of SGD batch update rule for each parameter
updates = [(par, par - lrate*T.grad(cost, par)) for par in rnn.param]
# compile
rnn_prediction = theano.function([x], T.argmax(p_y, 1))
rnn_batch_update = theano.function([x, y], cost, updates=updates)
print "rnn_prediction=", rnn_prediction
print "rnn_batch_update=", rnn_batch_update

nr_words = sum([len(seq.x) for seq in train_seq])
print "nr_words=", nr_words


