import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        A = np.asarray(matrix, dtype=float)
    except:
        return None
    if A.ndim != 2:
        return None
    if A.size == 0:
        return None
    if A.shape[0] != A.shape[1]:
        return None
    eigvals = np.linalg.eigvals(A)
    order = np.lexsort((eigvals.imag, eigvals.real))
    eigvals = eigvals[order]

    return eigvals