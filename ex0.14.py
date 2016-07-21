import lxmls.readers.galton as galton
import numpy as np
import matplotlib.pyplot as plt

def err(x, y, w) :
   temp = x * w - y
   temp *= temp
   
   return temp.sum()

galton_data = galton.load()
   
w = np.arange(-1, 2, 0.1)
e = map(lambda u: err(galton_data[:,0], galton_data[:,1], u), w)

plt.plot(w,e)

