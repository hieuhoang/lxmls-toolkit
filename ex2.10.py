import lxmls.sequences.hmm as hmmc
import lxmls.readers.pos_corpus as pcc

corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("data/train-02-21.conll",max_sent_len=15,
    max_nr_sent=1000)
test_seq = corpus.read_sequence_list_conll("data/test-23.conll",max_sent_len=15,
    max_nr_sent=1000)
dev_seq = corpus.read_sequence_list_conll("data/dev-22.conll",max_sent_len=15,max_nr_sent
    =1000)

hmm = hmmc.HMM(corpus.word_dict, corpus.tag_dict)
hmm.train_EM(train_seq, 0.1, 20, evaluate=True)

