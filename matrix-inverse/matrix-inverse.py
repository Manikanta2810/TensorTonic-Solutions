import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    
    A = np.array(A)
    m,n = A.shape
    if m!=n:
        return None
    det_A = np.linalg.det(A)
    if det_A==0:
        return None
    cof_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            minor = np.delete(A,i,axis=0)
            minor = np.delete(minor,j,axis=1)
            minor_det = np.linalg.det(minor)
            cof_mat[i][j] = ((-1) ** (i + j)) * minor_det
    adj = cof_mat.T
    inverse = adj/det_A
    return inverse
