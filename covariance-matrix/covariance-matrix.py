import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if X.ndim != 2:
        return None
    m,n = X.shape
    if m==1:
        return None
    mean_m = np.mean(X,axis=0)
    Xc = X - mean_m
    ans = (Xc.T@Xc)/(m-1)
    return ans