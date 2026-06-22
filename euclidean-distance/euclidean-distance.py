import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x = np.array(x)
    y = np.array(y)
    if x.shape != y.shape:
        raise ValueError
    dist = np.sqrt(np.sum((y-x)**2))
    return dist
        