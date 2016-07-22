import lxmls.readers.sentiment_reader as srs 

scr = srs.SentimentCorpus("books")

import lxmls.classifiers.multinomial_naive_bayes as mnbb
mnb = mnbb.MultinomialNaiveBayes()

params_nb_sc = mnb.train(scr.train_X,scr.train_y)
#params_nb_sc = mnb.train(scr.train_X[0:10,:], scr.train_y[0:10])

y_pred_train = mnb.test(scr.train_X,params_nb_sc)
acc_train = mnb.evaluate(scr.train_y, y_pred_train)
y_pred_test = mnb.test(scr.test_X,params_nb_sc)
acc_test = mnb.evaluate(scr.test_y, y_pred_test)
print "Multinomial Naive Bayes Amazon Sentiment Accuracy train: %f test: %f"%(
    acc_train,acc_test)   

#REF
#Multinomial Naive Bayes Amazon Sentiment Accuracy train: 0.987500 test: 0.635000

#MINE
#sumCounts= [ 123848.  134370.]
#Multinomial Naive Bayes Amazon Sentiment Accuracy train: 0.987500 test: 0.635000

