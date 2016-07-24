import numpy as np
import matplotlib.pyplot as plt

def get_err(f, s, w0, w1) :
   #ret = s - np.dot(f, w1) - w0
   ret = np.dot(f, w0) + w1 - s
   ret *= ret
   ret = ret.sum()
   return ret

def compute_grad(f, s, w0, w1) :
    grad0 = 2 * (f * w1 + w0 - s)
    grad1 = 2 * f * (f * w1 + w0 - s)
    return np.array([grad0.sum(), grad1.sum()])

# def compute_grad(f, s, w0, w1) :
#   epsilon = 0.0005
#
#   fOrig = get_err(f, s, w0, w1)
#
#   w0New = w0 + epsilon
#   w1New = w1 + epsilon
#
#   f0 = get_err(f, s, w0New, w1)
#   f1 = get_err(f, s, w0, w1New)
#   print "f=", fOrig, f0, f1
#
#   grad0 = (f0 - fOrig) / epsilon
#   grad1 = (f1 - fOrig) / epsilon
#
#   return np.array([grad0, grad1])
  
def optimize(data):
  f = data[:,0]
  s = data[:, 1]
  maxIter = 10
  precision = 0.00001
  stepSizes = np.array([0.0001, 0.0000001])
  oldWeights = np.array([0.0, 0.0])

  for iter in xrange(maxIter):
     grads = compute_grad(f, s, oldWeights[0], oldWeights[1])
     newWeights = oldWeights - grads * stepSizes

     errOld = get_err(f, s, oldWeights[0], oldWeights[1])
     errNew = get_err(f, s, newWeights[0], newWeights[1])

     print iter, "weights=", oldWeights, newWeights, "grads=", grads, "err=", errOld, errNew


     oldWeights = newWeights
     
  return oldWeights
  
import lxmls.readers.galton as galton
galton_data = galton.load()

optimize(galton_data)

#a=np.vstack(galton_data[:,0])
#b=np.vstack(galton_data[:,1])

#c = np.linalg.lstsq(a,b)
#(array([[ 0.99654386]]), array([ 5003.81262763]), 1, array([ 2081.59013737]))

#a2 = np.array( [galton_data[:,0], np.ones(len(galton_data))] )
#c2 = np.linalg.lstsq(a2.T,b)
#(array([[  0.64629058],
#        [ 23.94153018]]),
# array([ 4640.27261443]),
# 2,
# array([  2.08181288e+03,   7.96301870e-01]))