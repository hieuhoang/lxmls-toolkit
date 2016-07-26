import sys
import numpy as np
import lxmls.classifiers.linear_classifier as lc


class Perceptron(lc.LinearClassifier):

    def __init__(self, nr_epochs=10, learning_rate=1, averaged=True):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.nr_epochs = nr_epochs
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.params_per_round = []

    def train(self, x, y, seed=1):
        self.params_per_round = []
        x_orig = x[:, :]
        x = self.add_intercept_term(x)

        nr_x, nr_f = x.shape
        nr_c = np.unique(y).shape[0]
        w = np.zeros((nr_f, nr_c))
        #print "w=", w
        #print "nr_x=", nr_x
        #print "nr_f=", nr_f
        #print "nr_c=", nr_c
        #print "y=", y

        for epoch_nr in xrange(self.nr_epochs):
            #print "learning_rate=", self.learning_rate

            # use seed to generate permutation
            np.random.seed(seed)
            perm = np.random.permutation(nr_x)

            # change the seed so next epoch we don't get the same permutation
            seed += 1

            for nr in xrange(nr_x):
                # print "iter %i" %( epoch_nr*nr_x + nr)
                inst = perm[nr]
                sample = x[inst:inst+1, :]
                #print "inst=", inst, "sample=", sample

                y_hat = self.get_label(sample, w)
                y_truth = y[inst:inst+1, 0]
                #print "y_truth=", y_truth, "y_hat=", y_hat

                if y_truth != y_hat:
                    # Increase features of th e truth
                    #print "BEFORE w", w
                    w[:, y_truth] += self.learning_rate * sample.transpose()

                    # Decrease features of the prediction
                    w[:, y_hat] += -1 * self.learning_rate * sample.transpose()
                    #print "AFTER w", w

            self.params_per_round.append(w.copy())
            self.trained = True
            y_pred = self.test(x_orig, w)
            acc = self.evaluate(y, y_pred)
            self.trained = False
            print "Rounds: %i Accuracy: %f" % (epoch_nr, acc)
        self.trained = True

        print "w=", w
        if self.averaged:
            new_w = 0
            for old_w in self.params_per_round:
                new_w += old_w
            new_w /= len(self.params_per_round)
            print "new_w=", new_w
            return new_w
        return w
