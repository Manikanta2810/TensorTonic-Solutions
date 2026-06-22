import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.array(y)
    values,counts=np.unique(y,return_counts=True)
    prob = counts/len(y)
    return -np.sum(prob*np.log2(prob))
    