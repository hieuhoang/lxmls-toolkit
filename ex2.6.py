import lxmls.sequences.hmm as hmmc
import lxmls.readers.pos_corpus as pcc

corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("data/train-02-21.conll",max_sent_len=15,
    max_nr_sent=1000)
test_seq = corpus.read_sequence_list_conll("data/test-23.conll",max_sent_len=15,
    max_nr_sent=1000)
dev_seq = corpus.read_sequence_list_conll("data/dev-22.conll",max_sent_len=15,max_nr_sent
    =1000)

# ex 2.8
hmm = hmmc.HMM(corpus.word_dict, corpus.tag_dict)
hmm.train_supervised(train_seq)
hmm.print_transition_matrix()

# ex 2.9
viterbi_pred_train = hmm.viterbi_decode_corpus(train_seq)
#posterior_pred_train = hmm.posterior_decode_corpus(train_seq)
# eval_viterbi_train = hmm.evaluate_corpus(train_seq, viterbi_pred_train)
# eval_posterior_train = hmm.evaluate_corpus(train_seq, posterior_pred_train)
# print "Train Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(eval_posterior_train,eval_viterbi_train)
# #Train Set Accuracy: Posterior Decode 0.985, Viterbi Decode: 0.985
#
# viterbi_pred_test = hmm.viterbi_decode_corpus(test_seq)
# posterior_pred_test = hmm.posterior_decode_corpus(test_seq)
# eval_viterbi_test = hmm.evaluate_corpus(test_seq,viterbi_pred_test)
# eval_posterior_test = hmm.evaluate_corpus(test_seq,posterior_pred_test)
# print "Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(eval_posterior_test,eval_viterbi_test)
#Test Set Accuracy: Posterior Decode 0.350, Viterbi Decode: 0.509
