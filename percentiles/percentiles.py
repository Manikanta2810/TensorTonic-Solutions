import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x = np.asarray(x,dtype=float)
    q = np.asarray(q,dtype=float)
    x = np.sort(x)
    n =len(x)
    position = (q/100.0)*(n-1)
    low = np.floor(position).astype(int)
    high = np.ceil(position).astype(int)
    fr = position - low 
    result = x[low]+fr*(x[high]-x[low])
    return result