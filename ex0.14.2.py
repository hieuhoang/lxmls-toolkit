import lxmls.readers.galton as galton
import numpy as np
import matplotlib.pyplot as plt

def get_err(f, s, w0, w1) :
   ret = s - np.dot(f, w1) - w0
   ret *= ret
   ret = ret.sum()
   return ret

def compute_grad(f, s, w0, w1) :
  epsilon = 0.05
  
  errOrig = get_err(f, s, w0, w1)
  
  w0New = w0 + epsilon
  w1New = w1 + epsilon
  
  err0 = get_err(f, s, w0New, w1)
  err1 = get_err(f, s, w0, w1New)
  
  grad0 = (err0 - errOrig) / epsilon
  grad1 = (err1 - errOrig) / epsilon

  return np.array([grad0, grad1])
  
def optimize(data):
  f = data[:,0]
  s = data[:, 1]
  maxIter = 50
  oldWeights = np.array([0.1, 0.2])
  
  for iter in xrange(maxIter):
     grads = compute_grad(f, s, oldWeights[0], oldWeights[1])
     newWeights = oldWeights + grads
     print iter, oldWeights, grads, newWeights, "\n"
     
     oldWeights = newWeights
     
  return oldWeights
  
galton_data = galton.load()
   

