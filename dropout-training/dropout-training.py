import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x,dtype=float)
    if rng==None:
        rng = np.random.default_rng()
    mask = (rng.random(x.shape)>p).astype(float)
    drp = mask/(1-p)
    output = x * drp
    return output , drp