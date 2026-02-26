import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    if x.all() >= 0:
        return 1 / (1 + np.exp(-x))
    
    else:
        return np.exp(x) / (1 + np.exp(x))