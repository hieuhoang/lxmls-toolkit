import lxmls.sequences.hmm as hmmc
import lxmls.readers.pos_corpus as pcc
import lxmls.readers.simple_sequence as ssr

simple = ssr.SimpleSequence()
hmm = hmmc.HMM(simple.x_dict, simple.y_dict)

hmm.train_supervised(simple.train, smoothing=0.1)

print "simple.test.seq_list[0]=", simple.test.seq_list[0]
print "x=", simple.test.seq_list[0].x
print "y=", simple.test.seq_list[0].y

y_pred, score = hmm.viterbi_decode(simple.test.seq_list[0])

print "Viterbi decoding Prediction test 0 with smoothing:", y_pred, score

