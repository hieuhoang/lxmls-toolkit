import numpy as np
from lxmls.sequences.log_domain import *
import pdb


class SequenceClassificationDecoder:
    """ Implements a sequence classification decoder."""

    def __init__(self):
        pass

    # ----------
    # Computes the forward trellis for a given sequence.
    # Receives:
    #
    # Initial scores: (num_states) array
    # Transition scores: (length-1, num_states, num_states) array
    # Final scores: (num_states) array
    # Emission scoress: (length, num_states) array
    # ----------
    def run_forward(self, initial_scores, transition_scores, final_scores, emission_scores):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Forward variables.
        forward = np.zeros([length, num_states]) + logzero()

        # Initialization.
        forward[0, :] = emission_scores[0, :] + initial_scores

        # Forward loop.
        for pos in xrange(1, length):
            for current_state in xrange(num_states):
                # Note the fact that multiplication in log domain turns a sum and sum turns a logsum
                forward[pos, current_state] = logsum(forward[pos-1, :] + transition_scores[pos-1, current_state, :])
                forward[pos, current_state] += emission_scores[pos, current_state]

        # Termination.
        log_likelihood = logsum(forward[length-1, :] + final_scores)

        return log_likelihood, forward

    # ----------
    # Computes the backward trellis for a given sequence.
    # Receives:
    #
    # Initial scores: (num_states) array
    # Transition scores: (length-1, num_states, num_states) array
    # Final scores: (num_states) array
    # Emission scoress: (length, num_states) array
    # ----------
    def run_backward(self, initial_scores, transition_scores, final_scores, emission_scores):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Backward variables.
        backward = np.zeros([length, num_states]) + logzero()

        # Initialization.
        backward[length-1, :] = final_scores

        # Backward loop.
        for pos in xrange(length-2, -1, -1):
            for current_state in xrange(num_states):
                backward[pos, current_state] = \
                    logsum(backward[pos+1, :] +
                           transition_scores[pos, :, current_state] +
                           emission_scores[pos+1, :])

        # Termination.
        log_likelihood = logsum(backward[0, :] + initial_scores + emission_scores[0, :])

        return log_likelihood, backward

    # ----------
    # Computes the viterbi trellis for a given sequence.
    # Receives:
    #
    # Initial scores: (num_states) array
    # Transition scores: (length-1, num_states, num_states) array
    # Final scores: (num_states) array
    # Emission scoress: (length, num_states) array
    # ----------
    def run_viterbi(self, initial_scores, transition_scores, final_scores, emission_scores):
        print "initial_scores=", initial_scores
        print "transition_scores=", transition_scores.shape, transition_scores
        print "final_scores=", final_scores.shape, final_scores
        print "emission_scores=", emission_scores.shape, emission_scores

        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Variables storing the Viterbi scores.
        viterbi_scores = np.zeros([length, num_states]) + logzero()
        print "viterbi_scores=", viterbi_scores.shape, viterbi_scores

        # Variables storing the paths to backtrack.
        viterbi_paths = -np.ones([length, num_states], dtype=int)
        print "viterbi_paths=", viterbi_paths

        # Most likely sequence.
        best_path = -np.ones(length, dtype=int)

        # Complete Exercise 2.8 
        #raise NotImplementedError("Complete Exercise 2.8")

        #### Little guide of the implementation ####################################
        # Initializatize the viterbi scores
        #
        # Do the double of the viterbi loop (lines 7 to 12 in the guide pdf)
        # from 1 to length
        #     from 0 to num_states
        #       ...
        #
        # define the best_path and best_score
        #
        # backtrack the best_path using the viterbi paths (lines 17-18 pseudocode in the guide pdf)
        #
        # return best_path and best_score
        ############################################################################

        # Initialization.
        viterbi_scores[0, :] = emission_scores[0, :] + initial_scores
        print "viterbi_scores=", viterbi_scores

        for pos in xrange(1, length):
            print "pos=", pos
            for current_state in xrange(num_states):
                print "current_state=", current_state

                viterbi_scores[pos, current_state] = max(viterbi_scores[pos-1, :] + transition_scores[pos-1, current_state, :])
                viterbi_scores[pos, current_state] += emission_scores[pos, current_state]

                # emissScore = emission_scores[pos, current_state]
                # maxTrans = -np.inf
                # for prev_state in xrange(num_states):
                #     prevViterbi = viterbi_scores[pos - 1, prev_state]
                #     transScore = transition_scores[pos, prev_state, current_state]
                #
                #     if maxTrans < transScore + prevViterbi + emissScore:
                #         maxTrans = transScore + prevViterbi + emissScore

            #viterbi_scores[pos, current_state] = maxTrans

        print "viterbi_scores=", viterbi_scores.shape, viterbi_scores
        best_score = max(viterbi_scores[length - 1, :])

        return best_path, best_score

    def run_forward_backward(self, initial_scores, transition_scores, final_scores, emission_scores):
        log_likelihood, forward = self.run_forward(initial_scores, transition_scores, final_scores, emission_scores)
        print 'Log-Likelihood =', log_likelihood

        log_likelihood, backward = self.run_backward(initial_scores, transition_scores, final_scores, emission_scores)
        print 'Log-Likelihood =', log_likelihood

        return forward, backward
