import numpy as np
import lxmls.readers.sentiment_reader as srs

scr = srs.SentimentCorpus("books")
train_x = scr.train_X.T
train_y = scr.train_y[:, 0]
test_x = scr.test_X.T
test_y = scr.test_y[:, 0]
print "train_x=", train_x.shape, train_x

x = test_x
print "x=", x.shape

# Neural network modules
import lxmls.deep_learning.mlp as dl
import lxmls.deep_learning.sgd as sgd # Model parameters

geometry = [train_x.shape[0], 20, 2]
actvfunc = ['sigmoid', 'softmax']

#geometry = [train_x.shape[0], 20, 5, 2]
#actvfunc = ['sigmoid', 'sigmoid', 'softmax']

print "geometry=", geometry

# Instantiate model
mlp = dl.NumpyMLP(geometry, actvfunc)
print "params=", len(mlp.params)
for i in xrange(len(mlp.params)):
    obj = mlp.params[i]
    print i, "=", obj.shape #, obj

# ex 5.2
W1, b1 = mlp.params[:2]
#print "W1=", W1
#print "b1=", b1

z1 = np.dot(W1, x) + b1
tilde_z1 = 1/(1+np.exp(-z1))
print "z1=", z1.shape
print "tilde_z1=", tilde_z1.shape, tilde_z1

# Theano code.
# NOTE: We use undescore to denote symbolic equivalents to Numpy variables. # This is no Python convention!.
import theano
import theano.tensor as T
_x = T.matrix('x')

_W1 = theano.shared(value=W1, name='W1', borrow=True)
_b1 = theano.shared(value=b1, name='b1', borrow=True, broadcastable=(False, True))

_z1            = T.dot(_W1, _x) + _b1
_tilde_z1      = T.nnet.sigmoid(_z1)
# Keep in mind that naming variables is useful when debugging
_z1.name       = 'z1'
_tilde_z1.name = 'tilde_z1'

print "_W1=", _W1
print "_z1=", _z1

theano.printing.debugprint(_tilde_z1)

# evaluate test set
layer1 = theano.function([_x], _tilde_z1)
aa = layer1(x.astype(theano.config.floatX))
print "aa=", aa.shape, aa

# Check Numpy and Theano match
assert np.allclose(tilde_z1, layer1(x.astype(theano.config.floatX))), "Numpy and Theano Perceptrons are different"

# this should be before we train???
train_x = train_x.astype(theano.config.floatX)
train_y = train_y.astype('int32')

