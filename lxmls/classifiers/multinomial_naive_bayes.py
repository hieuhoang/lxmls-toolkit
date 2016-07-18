import numpy as np
import scipy as scipy
import lxmls.classifiers.linear_classifier as lc
import sys
from lxmls.distributions.gaussian import *


class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self, xtype="gaussian"):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words
        n_docs, n_words = x.shape
        print "n_docs=" + str(n_docs) + " n_words=" + str(n_words)
        print "x=" + str(x.shape)
        print "y=" + str(y.shape)
         
        # classes = a list of possible classes
        classes = np.unique(y)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        print "classes=" + str(classes)
        print "n_classes=" + str(n_classes)

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
        # prior[0] is the prior probability of a document being of class 0
        # likelihood[4, 0] is the likelihood of the fifth(*) feature being
        # active, given that the document is of class 0
        # (*) recall that Python starts indices at 0, so an index of 4
        # corresponds to the fifth feature!

        # Complete Exercise 1.1 
        #raise NotImplementedError("Complete Exercise 1.1")
        for i in xrange(n_classes):
          print "i=%s class=%s" % (i, classes[i])
          docs_in_class, _ = np.nonzero(y == classes[i])  # docs_in_class = indices of documents in class i
          print "docs_in_class=" + str(len(docs_in_class))
          prior[i] = 1.0 * len(docs_in_class) / n_docs  # prior = fraction of documents with this class
          print "prior=" + str(prior)
          
          # word_count_in_class = count of word occurrences in documents of class i
          temp = x[docs_in_class, :]
          print "temp=" + str(temp.shape)
          
          word_count_in_class = temp.sum(0)
          print "word_count_in_class=" + str(word_count_in_class.shape) + " " + str(word_count_in_class)

          total_words_in_class = word_count_in_class.sum()  # total_words_in_class = total number of words in documents of class i
          print "total_words_in_class=" + str(total_words_in_class)

          if not self.smooth:
            # likelihood = count of occurrences of a word in a class
            likelihood[:, i] = word_count_in_class / total_words_in_class
          else:
            likelihood[:, i] = (word_count_in_class+self.smooth_param) / (total_words_in_class + self.smooth_param*n_words)
          print "likelihood=" + str(likelihood.shape) + str(likelihood)
          
          
        params = np.zeros((n_words+1, n_classes))
        for i in xrange(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
