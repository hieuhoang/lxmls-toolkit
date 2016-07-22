import lxmls.readers.sentiment_reader as srs
import lxmls.classifiers.multinomial_naive_bayes as mnbb

scr = srs.SentimentCorpus("books")

mnb = mnbb.MultinomialNaiveBayes()
params_nb_sc = mnb.train(scr.train_X,scr.train_y)
print "params_nb_sc=" + str(params_nb_sc.shape)

y_pred_train = mnb.test(scr.train_X,params_nb_sc)
print "y_pred_train=" + str(y_pred_train.shape) + str(y_pred_train)

acc_train = mnb.evaluate(scr.train_y, y_pred_train)
print "acc_train=" + str(acc_train)

y_pred_test = mnb.test(scr.test_X,params_nb_sc)
print "y_pred_test=" + str(y_pred_test)

acc_test = mnb.evaluate(scr.test_y, y_pred_test)
print "acc_test=" + str(acc_test)

print "Multinomial Naive Bayes Amazon Sentiment Accuracy train: %f test: %f"%(
    acc_train,acc_test)
