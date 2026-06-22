import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    A = np.array(A)
    m,n =A.shape
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
            cof_mat[i][j]= ((-1)**(i+j))*np.linalg.det(minor)
    adj = cof_mat.T
    inverse = adj/det_A
    return inverse

