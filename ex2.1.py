import lxmls.readers.simple_sequence as ssr 

simple = ssr.SimpleSequence()
print simple.train

print simple.test

for sequence in simple.train.seq_list:
    print sequence

print "observed:"
for sequence in simple.train.seq_list:
    print sequence.x

print "hidden"
for sequence in simple.train.seq_list:
    print sequence.y
import lxmls.sequences.hmm as hmmc
hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)

print "Initial Probabilities:", hmm.initial_probs
print "Transition Probabilities:\n", hmm.transition_probs
print "Final Probabilities:", hmm.final_probs
print "Emission Probabilities\n", hmm.emission_probs

initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])

print initial_scores
print transition_scores
print final_scores
print emission_scores

import numpy as np
a = np.random.rand(10)
np.log(sum(np.exp(a)))

#ex 2.5
log_likelihood, forward = hmm.decoder.run_forward(initial_scores, transition_scores, final_scores, emission_scores)
print 'Log-Likelihood =', log_likelihood

log_likelihood, backward = hmm.decoder.run_backward(initial_scores, transition_scores,
    final_scores, emission_scores)

#ex 2.6
initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])
state_posteriors, _, _ = hmm.compute_posteriors(initial_scores, transition_scores, final_scores, emission_scores)

print "state_posteriors"
print state_posteriors

y_pred = hmm.posterior_decode(simple.test.seq_list[0])

print "Prediction test 0:", y_pred
print "Truth test 0:", simple.test.seq_list[0]

y_pred = hmm.posterior_decode(simple.test.seq_list[1])

print "Prediction test 1:", y_pred
print "Truth test 1:", simple.test.seq_list[1]

hmm.train_supervised(simple.train, smoothing=0.1)
y_pred = hmm.posterior_decode(simple.test.seq_list[0])
print "Prediction test 0 with smoothing:", y_pred
print "Truth test 0:", simple.test.seq_list[0]

y_pred = hmm.posterior_decode(simple.test.seq_list[1])
print "Prediction test 1 with smoothing:", y_pred
print "Truth test 1:", simple.test.seq_list[1]

# ex 2.8
hmm.train_supervised(simple.train, smoothing=0.1)

y_pred, score = hmm.viterbi_decode(simple.test.seq_list[0])
print "Viterbi decoding Prediction test 0 with smoothing:", y_pred, score
print "Truth test 0:", simple.test.seq_list[0]

y_pred, score = hmm.viterbi_decode(simple.test.seq_list[1])
print "Viterbi decoding Prediction test 1 with smoothing:", y_pred, score
print "Truth test 1:", simple.test.seq_list[1]
