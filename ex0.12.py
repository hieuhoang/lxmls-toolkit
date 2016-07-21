import numpy as np

def get_y(x):
    return (x+2)**2 - 16*np.exp(-((x-2)**2))
