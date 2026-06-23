def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X = np.asarray(X)
    n=X.shape[1]
    y = np.asarray(y)
    w = np.linalg.inv(X.T @ X + lam * np.eye(n)) @ X.T @ y
    return w