import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x)
    m = np.mean(x)
    n = len(x)
    s = (1/(n-1))*np.sum((x-m)**2)
    return s,np.sqrt(s)