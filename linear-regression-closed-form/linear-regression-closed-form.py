import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)
    Xt = X.T@X
    Xty = X.T@y
    Xt_inverse=np.linalg.inv(Xt)
    w = (Xt_inverse)@Xty
    return w