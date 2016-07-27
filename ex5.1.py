import numpy as np
import lxmls.readers.sentiment_reader as srs

scr = srs.SentimentCorpus("books")
train_x = scr.train_X.T
train_y = scr.train_y[:, 0]
test_x = scr.test_X.T
test_y = scr.test_y[:, 0]
print "train_x=", train_x.shape, train_x

# Neural network modules
import lxmls.deep_learning.mlp as dl
import lxmls.deep_learning.sgd as sgd # Model parameters
#geometry = [train_x.shape[0], 20, 2]
geometry = [train_x.shape[0], 20,10, 3]
print "geometry=", geometry

actvfunc = ['sigmoid', 'sigmoid', 'softmax']
# Instantiate model
mlp = dl.NumpyMLP(geometry, actvfunc)

# Model parameters
n_iter = 5
bsize  = 5
lrate  = 0.01

# Train
sgd.SGD_train(mlp, n_iter, bsize=bsize, lrate=lrate, train_set=(train_x, train_y))

acc_train = sgd.class_acc(mlp.forward(train_x), train_y)[0]
acc_test = sgd.class_acc(mlp.forward(test_x), test_y)[0]
print "MLP (%s) Amazon Sentiment Accuracy train: %f test: %f" % (geometry, acc_train,
acc_test)


