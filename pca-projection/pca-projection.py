import numpy as n


def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.array(X, dtype=float)
    mean = np.mean(X, axis=0)
    Xc = X - mean
    n = X.shape[0]
    C = (Xc.T @ Xc)/(n-1)
    eigenvalues, eigenvectors = np.linalg.eig(C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    W = eigenvectors[:, :k]
    X_proj = Xc @ W
    return X_proj



