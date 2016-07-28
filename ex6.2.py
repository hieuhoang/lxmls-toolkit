import numpy as np
import theano
import theano.tensor as T
theano.config.optimizer='None'

def square(x):
    return x**2

# Python
def np_square_n_steps(nr_steps):
    out = []
    for n in np.arange(nr_steps):
        out.append(square(n))
    return np.array(out)

# Theano
nr_steps = T.lscalar('nr_steps')
print "nr_steps=", nr_steps
print "T.arange=", T.arange(nr_steps)

h, _ = theano.scan(fn=square, sequences=T.arange(nr_steps))
#print "h=", h

th_square_n_steps = theano.function([nr_steps], h)

# Compare both
print np_square_n_steps(10)
print th_square_n_steps(10)

# Configuration
print "MATRICES:"
nr_states = 3
nr_steps = 5
# Transition matrix
A = np.abs(np.random.randn(nr_states, nr_states))
A = A/A.sum(0, keepdims=True)
print "A=", A

# Initial state
s0 = np.zeros(nr_states)
s0[0] = 1
print "s0=", s0

# Numpy version
def np_markov_step(s_tm1):
    s_t = np.dot(s_tm1, A.T)
    return s_t

def np_markov_chain(nr_steps, A, s0):
    # Pre-allocate space
    s = np.zeros((nr_steps+1, nr_states))
    s[0, :] = s0
    for t in np.arange(nr_steps):
        s[t+1, :] = np_markov_step(s[t, :])
    return s

sEnd = np_markov_chain(nr_steps, A, s0)
print "sEnd=", sEnd

# Theano version
print "THEANO"
# Store variables as shared variables
th_A = theano.shared(A, name='A', borrow=True)
th_s0 = theano.shared(s0, name='s0', borrow=True)
print "th_A=", th_A

# Symbolic variable for the number of steps
th_nr_steps = T.lscalar('nr_steps')
print "nr_steps=", nr_steps, "th_nr_steps=", th_nr_steps

def th_markov_step(s_tm1):
    s_t = T.dot(s_tm1, th_A.T)
    # Remember to name variables s_t.name = 's_t'
    return s_t

s, _ = theano.scan(th_markov_step, outputs_info=[dict(initial=th_s0)],n_steps=th_nr_steps)
th_markov_chain = theano.function([th_nr_steps], T.concatenate((th_s0[None, :], s), 0))
th_sEnd = th_markov_chain(nr_steps)
print "th_sEnd=", th_sEnd